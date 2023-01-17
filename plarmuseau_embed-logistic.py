import os
import gensim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import PassiveAggressiveClassifier,Perceptron,SGDClassifier

import warnings
warnings.filterwarnings('ignore')

print(os.listdir("../input"))
%matplotlib inline
data = pd.read_json('../input/train.json')
test = pd.read_json('../input/test.json')
print('Training data shape: {}'.format(data.shape))
print('Test data shape: {}'.format(test.shape))
# Target variable 
target = data.cuisine
data['ingredient_count'] = data.ingredients.apply(lambda x: len(x))
def flatten_lists(lst):
    """Remove nested lists."""
    return [item for sublist in lst for item in sublist]
f = plt.figure(figsize=(14,8))
gs = gridspec.GridSpec(2, 2)

ax1 = plt.subplot(gs[0, :])
data.ingredient_count.value_counts().hist(ax=ax1)
ax1.set_title('Recipe richness', fontsize=12)

ax2 = plt.subplot(gs[1, 0])
pd.Series(flatten_lists(list(data['ingredients']))).value_counts()[:20].plot(kind='barh', ax=ax2)
ax2.set_title('Most popular ingredients', fontsize=12)

ax3 = plt.subplot(gs[1, 1])
data.groupby('cuisine').mean()['ingredient_count'].sort_values(ascending=False).plot(kind='barh', ax=ax3)
ax3.set_title('Average number of ingredients in cuisines', fontsize=12)

plt.show()
# Feed a word2vec with the ingredients
w2v = gensim.models.Word2Vec(list(data.append(test).ingredients), size=1000, window=10, min_count=1, iter=20)  #cont min should be 1 since you want all unique words to embed, size should be as large as pissblie
len(w2v.wv['onions'])
w2v.most_similar(['green onions'])
w2v.most_similar(['kosher salt'])
def document_vector(doc):
    """Create document vectors by averaging word vectors. Remove out-of-vocabulary words."""
    doc = [word for word in doc if word in w2v.wv.vocab]
    return np.mean(w2v[doc], axis=0)
data['doc_vector'] = data.ingredients.apply(document_vector)
test['doc_vector'] = test.ingredients.apply(document_vector)
lb = LabelEncoder()
y = lb.fit_transform(target)
print ("TF-IDF on text data ... ")
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

tfidf = CountVectorizer(ngram_range=(1,1)) #binary=True)
print(tfidf)
data['ingredients']= data['ingredients'].map(", ".join)
test['ingredients']= test['ingredients'].map(", ".join)
tfidf.fit_transform(data['ingredients'].append(test['ingredients'])).astype(np.float32)

X =  tfidf.transform(data['ingredients']).astype(np.float32)
X_test = tfidf.transform(test['ingredients']).astype(np.float32)


Xv = list(data['doc_vector'])
X_testv = list(test['doc_vector'])

from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import PassiveAggressiveClassifier,Perceptron,SGDClassifier

temp=np.concatenate( (X.todense(), Xv ), axis=1 )
U1=pd.DataFrame( temp , index=data.index )
print(X.shape,U1.shape )
classifier = SGDClassifier(max_iter=20,n_jobs=4,fit_intercept=True)  #0.77

model = OneVsRestClassifier(classifier)
model.fit(U1,y)
print( (model.predict(U1)==y).mean() ) 

print( (model.predict(U1)==y).mean() ) 

temp=np.concatenate( (X_test.todense(), X_testv ), axis=1 )
U2=pd.DataFrame( temp , index=test.index )

# Predictions 
print ("Predict on test data ... ")
y_test = model.predict(U2)
y_pred = lb.inverse_transform(y_test)

# Submission
print ("Generate Submission File ... ")
test_id = test.id
sub = pd.DataFrame({'id': test_id, 'cuisine': y_pred}, columns=['id', 'cuisine'])
sub.to_csv('svm_output.csv', index=False)
sub
