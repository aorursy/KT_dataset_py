# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import re
from collections import Counter
import nltk
#from nltk.util import ngrams
import collections
import spacy
import matplotlib
import matplotlib.pyplot as plt # for plotting
import seaborn as sns
# Any results you write to the current directory are saved as output.
train_data = pd.read_csv("../input/train.tsv",sep='\t',nrows =10000) ##Load full data
test_data = pd.read_csv("../input/test.tsv",sep='\t',nrows=10000) ##Load full data
print("Train",train_data.shape,"Test",test_data.shape)
train_data.head()
test_data.head()
count = train_data['Sentiment'].value_counts().plot(kind="pie",shadow=True,startangle=90,autopct='%1.1f%%',figsize=(7,7))
train_word_counter = collections.Counter([word for sentence in train_data['Phrase'] for word in sentence.split()])
test_word_counter = collections.Counter([word for sentence in test_data['Phrase'] for word in sentence.split()])
print('{} Words in Training dataset.'.format(len([word for sentence in train_data['Phrase'] for word in sentence.split()])))
print('{} unique words in Training dataset.'.format(len(train_word_counter)))
print('20 Most common words in the Training dataset:')
print('"' + '" "'.join(list(zip(*train_word_counter.most_common(30)))[0]) + '"')

print('{} Words in Training dataset.'.format(len([word for sentence in test_data['Phrase'] for word in sentence.split()])))
print('{} unique words in Training dataset.'.format(len(test_word_counter)))
print('20 Most common words in the Training dataset:')
print('"' + '" "'.join(list(zip(*train_word_counter.most_common(30)))[0]) + '"')

print('Average count of phrases per review in train is {0:.0f}.'.format(train_data.groupby('SentenceId')['Phrase'].count().mean()))
print('Average count of phrases per review in test is {0:.0f}.'.format(test_data.groupby('SentenceId')['Phrase'].count().mean()))
import spacy #load spacy
nlp = spacy.load("en", disable=['parser', 'tagger', 'ner'])
from nltk.corpus import stopwords
stops = stopwords.words("english")



def normalize(comment, lowercase, remove_stopwords):
    if lowercase:
        comment = comment.lower()
    comment = nlp(comment)
    lemmatized = list()
    for word in comment:
        lemma = word.lemma_.strip()
        if lemma:
            if not remove_stopwords or (remove_stopwords and lemma not in stops):
                lemmatized.append(lemma)
    return " ".join(lemmatized)
train_data['Phrase_Clean'] = train_data['Phrase'].apply(normalize, lowercase=True, remove_stopwords=True)
test_data['Phrase_Clean'] = test_data['Phrase'].apply(normalize, lowercase=True, remove_stopwords=True)

train_data.head(10)
def cleaning(s):
    
    s = str(s)
    #s = s.split(s)
    s = s.lower()
    s = re.sub('\s\W',' ',s)
    s = re.sub('\W,\s',' ',s)
    s = re.sub(r'[^\w]', ' ', s)
    #s = re.sub("\d+", "", s)
    s = re.sub('\s+',' ',s)
    s = re.sub('[!@#$_]', '', s)
    s = s.replace(",","")
    s = s.replace("[\w*"," ")
    s = re.sub(r'https?:\/\/.*[\r\n]*', '', s, flags=re.MULTILINE)
    s = re.sub(r'\<a href', ' ', s)
    s = re.sub(r'&amp;', '', s) 
    s = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', s)
    s = re.sub(r'[^\x00-\x7f]',r'',s) #removes arabic
    s = re.sub(r'<br />', ' ', s)
    s = re.sub(r'\'', ' ', s)
    s = re.sub(r"agh","are not gonna happen",s)
    s = re.sub(r"wtf","what the fuck",s)
    s = re.sub(r"asap","as soon as possible",s)
    s = re.sub(r"lol","lots of laughs",s)
    s = re.sub(r"[0-9]+", '',s)
    s = re.sub(r" s", '',s)
    
    return s
train_data['Phrase_Clean'] = [cleaning(s) for s in train_data['Phrase_Clean']]
train_data['Phrase_Clean'].isnull().sum()
train_data.head()
from nltk import ngrams
text = ' '.join(train_data.loc[train_data.Sentiment == 0, 'Phrase_Clean'].values)
NegativeSentence = [i for i in ngrams(text.split(), 1)]
print("Top NegativeSentence Words in Training Dataset",Counter(NegativeSentence).most_common(10))
text = ' '.join(train_data.loc[train_data.Sentiment == 1, 'Phrase_Clean'].values)
Somewhatnegative = [i for i in ngrams(text.split(), 1)]
print("Top Somewhatnegative Words in Training Dataset",Counter(Somewhatnegative).most_common(10))
text = ' '.join(train_data.loc[train_data.Sentiment == 2, 'Phrase_Clean'].values)
Neutral = [i for i in ngrams(text.split(), 1)]
print("Top Neutral Words in Training Dataset",Counter(Neutral).most_common(10))
text = ' '.join(train_data.loc[train_data.Sentiment == 3, 'Phrase_Clean'].values)
Somewhatpositive = [i for i in ngrams(text.split(), 1)]
print("Top Somewhatpositive Words in Training Dataset",Counter(Somewhatpositive).most_common(10))
text = ' '.join(train_data.loc[train_data.Sentiment == 4, 'Phrase_Clean'].values)
Postive = [i for i in ngrams(text.split(), 1)]
print("Top Postive Words in Training Dataset",Counter(Postive).most_common(10))
#SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
import tqdm
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.metrics import accuracy_score

from sklearn.pipeline import FeatureUnion
vectorizer = FeatureUnion([
('word_vectorizer',  TfidfVectorizer(
sublinear_tf=True,
stop_words = 'english',
analyzer='word',
token_pattern=r'\w{1,}',
ngram_range =(1,3),
max_features=35000)),

('char_vectorizer', TfidfVectorizer(
sublinear_tf=True,
stop_words = 'english',
analyzer='char',
ngram_range=(1,3),
max_features=80000))
 ])
vectorizer.fit(train_data['Phrase_Clean'])
train_features = vectorizer.fit_transform(train_data['Phrase_Clean'])
test_features = vectorizer.transform(test_data['Phrase_Clean'])
Target = train_data["Sentiment"]
seed = 101 
np.random.seed(seed)
#Data Split
X_train_tfidf, X_test_tfidf, y_train_tfidf, y_test_tfidf = train_test_split(train_features, Target,stratify=Target,random_state=seed)

from sklearn.svm import SVC # "Support vector classifier"
model_svc_linear = SVC(kernel='linear', C=1,class_weight='balanced')
model_svc_linear.fit(X_train_tfidf, y_train_tfidf)
predictions_tfidf = model_svc_linear.predict(X_test_tfidf)
accuracy_tfidf = accuracy_score(y_test_tfidf, predictions_tfidf)
print(accuracy_tfidf)
from sklearn.metrics import log_loss,confusion_matrix,classification_report,roc_curve,auc

target_name = train_data['Sentiment'].unique()

confusion_matrix(y_test_tfidf,predictions_tfidf,labels=target_name)
print(classification_report(y_test_tfidf,predictions_tfidf))
###Test
#test_fit = model_svc_linear.predict(test_features)

#sub = pd.read_csv('../input/sampleSubmission.csv')
#sub['Sentiment'] =  test_fit
#sub.to_csv("SVC.csv", index=False)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import cross_val_score

models = [
    RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
    LinearSVC(multi_class='ovr',class_weight='balanced'),
    MultinomialNB(),
    LogisticRegression(solver="sag", max_iter=500,multi_class = 'ovr'),
]
CV = 5
cv_df = pd.DataFrame(index=range(CV * len(models)))
entries = []
for model in models:
  model_name = model.__class__.__name__
  accuracies = cross_val_score(model, train_features, Target, scoring='accuracy', cv=CV)
  for fold_idx, accuracy in enumerate(accuracies):
    entries.append((model_name, fold_idx, accuracy))
cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])


import seaborn as sns

sns.boxplot(x='model_name', y='accuracy', data=cv_df)
sns.stripplot(x='model_name', y='accuracy', data=cv_df, 
              size=15, linewidth=2)
plt.show()

cv_df.groupby('model_name').accuracy.mean()
