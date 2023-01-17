# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import pandas as pd  #data processing and data level operation
import numpy as np  # Linear Algebra
import os, re
import string
import nltk 
from nltk.corpus import stopwords

#Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud,STOPWORDS

%matplotlib inline
###For downlaod the nltk
########nltk.download()
#Current working directory
print('current workind directory ==== ',os.getcwd())

#Loading data
train = pd.read_csv('/kaggle/input/word2vec-nlp-tutorial/labeledTrainData.tsv.zip',delimiter = '\t')
test = pd.read_csv('/kaggle/input/word2vec-nlp-tutorial/testData.tsv.zip',delimiter = '\t')

train.shape, test.shape
train.head()
test.head()
print ("number of rows for sentiment 1: {}".format(len(train[train.sentiment == 1])))
print ( "number of rows for sentiment 0: {}".format(len(train[train.sentiment == 0])))
train.groupby('sentiment').describe().transpose()
#Creating a new col
train['length'] = train['review'].apply(len)
train.head()
#Histogram of count of letters
train['length'].plot.hist(bins = 100)
train.length.describe()
train.hist(column='length', by='sentiment', bins=100,figsize=(12,4))
from bs4 import BeautifulSoup

#Creating a function for cleaning of data
def clean_text(raw_text):
    # 1. remove HTML tags
    raw_text = BeautifulSoup(raw_text).get_text() 
    
    # 2. removing all non letters from text
    letters_only = re.sub("[^a-zA-Z]", " ", raw_text) 
    
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()                           
    
    # 4. Create variable which contain set of stopwords
    stops = set(stopwords.words("english"))                  
    
    # 5. Remove stop word & returning   
    return [w for w in words if not w in stops]
#Cleaning review and also adding a new col as its len count of words
train['clean_review'] = train['review'].apply(clean_text)
train['length_clean_review'] = train['clean_review'].apply(len)
train.head()
train.describe()
#Checking the smallest review
print(train[train['length_clean_review'] == 4]['review'].iloc[0])
print('------After Cleaning------')
print(train[train['length_clean_review'] == 4]['clean_review'].iloc[0])
#Plot wordcloud
word_cloud = WordCloud(width = 1000, height = 500, stopwords = STOPWORDS, background_color = 'red').generate(
                        ''.join(train['review']))

plt.figure(figsize = (15,8))
plt.imshow(word_cloud)
plt.axis('off')
plt.show()

#word_cloud.to_file('aa.png')   #for saving file
from sklearn.feature_extraction.text import CountVectorizer
# Might take awhile...
bow_transform = CountVectorizer(analyzer=clean_text).fit(train['review'])  #bow = bag of word

# Print total number of vocab words
print(len(bow_transform.vocabulary_))
review1 = train['review'][1]
print(review1)
bow1 = bow_transform.transform([review1])
print(bow1)
print(bow1.shape)
print(bow_transform.get_feature_names()[71821])
print(bow_transform.get_feature_names()[72911])
#Creating bag of words for our review variable
review_bow = bow_transform.transform(train['review'])
print('Shape of Sparse Matrix: ', review_bow.shape)
print('Amount of Non-Zero occurences: ', review_bow.nnz)
sparsity = (100.0 * review_bow.nnz / (review_bow.shape[0] * review_bow.shape[1]))
print('sparsity: {}'.format(sparsity))
from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer().fit(review_bow)
tfidf1 = tfidf_transformer.transform(bow1)
print(tfidf1)
print(tfidf_transformer.idf_[bow_transform.vocabulary_['war']])
print(tfidf_transformer.idf_[bow_transform.vocabulary_['book']])
review_tfidf = tfidf_transformer.transform(review_bow)
print(review_tfidf.shape)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train['review'], train['sentiment'], test_size=0.22, random_state=101)

len(X_train), len(X_test), len(X_train) + len(X_test)
from sklearn.metrics import classification_report
#Predicting & Stats Function
def pred(predicted,compare):
    cm = pd.crosstab(compare,predicted)
    TN = cm.iloc[0,0]
    FN = cm.iloc[1,0]
    TP = cm.iloc[1,1]
    FP = cm.iloc[0,1]
    print("CONFUSION MATRIX ------->> ")
    print(cm)
    print()
    
    ##check accuracy of model
    print('Classification paradox :------->>')
    print('Accuracy :- ', round(((TP+TN)*100)/(TP+TN+FP+FN),2))
    print()
    print('False Negative Rate :- ',round((FN*100)/(FN+TP),2))
    print()
    print('False Postive Rate :- ',round((FP*100)/(FP+TN),2))
    print()
    print(classification_report(compare,predicted))
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=clean_text)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', LogisticRegression(random_state=101)),  # train on TF-IDF vectors w/ Naive Bayes classifier
])

pipeline.fit(X_train,y_train)
predictions = pipeline.predict(X_train)
pred(predictions,y_train)
#Test Set Result
predictions = pipeline.predict(X_test)
pred(predictions,y_test)
#Saving Output
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pipeline_logit = Pipeline([
    ('bow', CountVectorizer(analyzer=clean_text)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', LogisticRegression(random_state=101)),  # train on TF-IDF vectors w/ Naive Bayes classifier
])

pipeline_logit.fit(train['review'],train['sentiment'])
test['sentiment'] = pipeline_logit.predict(test['review'])
test.head(5)
output = test[['id','sentiment']]
print(output)
output.to_csv( "sentiment.csv", index=False, quoting=3 )
