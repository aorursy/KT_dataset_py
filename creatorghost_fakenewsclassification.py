import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#1: unreliable
#0: reliable
train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')

test['label']='t'

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer

#data prep
test=test.fillna(' ')
train=train.fillna(' ')
test['total']=test['title']+' '+test['author']+test['text']
train['total']=train['title']+' '+train['author']+train['text']




train.columns
train.label.value_counts(normalize=True)
train.columns
sorted_data=train.sort_values('id', axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last')
#Deduplication of entries
final=sorted_data.drop_duplicates(subset={'title', 'author', 'text'}, keep='first', inplace=False)
final.shape
# Checking data loss due to duplication
(final['id'].size*1.0)/(train['id'].size*1.0)*100
from bs4 import BeautifulSoup
from tqdm import tqdm
import re
from nltk.stem import SnowballStemmer

snow=SnowballStemmer('english')
## function to remove URl
def removeUrl(text):
  pr=re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
  return pr
## we also use Beautifull Soup to remove Html
def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

stopwords= set(['br', 'the', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\
            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \
            'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \
            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\
            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\
            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\
            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \
            's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \
            've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',\
            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',\
            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", \
            'won', "won't", 'wouldn', "wouldn't"])
final.head(2)
def removeStopWord(word):
  token=word.split(" ")   ## coverting string to token (list of word) \\ like ["this","is","token"]
  removestop=[snow.stem(x) for x in token if x not in stopwords]   ##removing stopwords and also doing Stemming
  removed=" ".join(removestop)  ##joing back the list into sentence
  return removed
from tqdm import tqdm
preprocessed_reviews = []
for line in tqdm(final.total.values):
  line= BeautifulSoup(line, 'lxml').get_text() ## Remove Html Tags
  line=removeUrl(line) #removing url
  line=decontracted(line)    #Coverting word like { are't -> are not}
  line = re.sub(r'[0-9]+', '', line)   ## To Remove Numbers from the string
  line=line.lower()   ## Converting every word to lower case
  line = re.sub(r'[^a-z0-9\s]', '', line)   ## To clean all special Charaters
  line=removeStopWord(line)    ## Removing Stop Words And doing Steaming
  preprocessed_reviews.append(line.strip()) ## ading cleaned word into a list after removing spaces {By using strip()}
from tqdm import tqdm
preprocessed_test = []
for line in tqdm(test.total.values):
  line= BeautifulSoup(line, 'lxml').get_text() ## Remove Html Tags
  line=removeUrl(line) #removing url
  line=decontracted(line)    #Coverting word like { are't -> are not}
  line = re.sub(r'[0-9]+', '', line)   ## To Remove Numbers from the string
  line=line.lower()   ## Converting every word to lower case
  line = re.sub(r'[^a-z0-9\s]', '', line)   ## To clean all special Charaters
  line=removeStopWord(line)    ## Removing Stop Words And doing Steaming
  preprocessed_test.append(line.strip()) ## ading cleaned word into a list after removing spaces {By using strip()}
#tfidf
transformer = TfidfTransformer(smooth_idf=False)
count_vectorizer = CountVectorizer(ngram_range=(1, 2))
counts = count_vectorizer.fit_transform(preprocessed_reviews)
tfidf = transformer.fit_transform(counts)

targets = final['label'].values
test_counts = count_vectorizer.transform(preprocessed_test)
test_tfidf = transformer.fit_transform(test_counts)

#split in samples
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(tfidf, targets, random_state=0)


from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              AdaBoostClassifier)

Extr = ExtraTreesClassifier(n_estimators=5,n_jobs=4)
Extr.fit(X_train, y_train)
print('Accuracy of ExtrTrees classifier on training set: {:.2f}'
     .format(Extr.score(X_train, y_train)))
print('Accuracy of Extratrees classifier on test set: {:.2f}'
     .format(Extr.score(X_test, y_test)))
from sklearn.tree import DecisionTreeClassifier

Adab= AdaBoostClassifier(DecisionTreeClassifier(max_depth=3),n_estimators=5)
Adab.fit(X_train, y_train)
print('Accuracy of Adaboost classifier on training set: {:.2f}'
     .format(Adab.score(X_train, y_train)))
print('Accuracy of Adaboost classifier on test set: {:.2f}'
     .format(Adab.score(X_test, y_test)))
Rando= RandomForestClassifier(n_estimators=5)

Rando.fit(X_train, y_train)
print('Accuracy of randomforest classifier on training set: {:.2f}'
     .format(Rando.score(X_train, y_train)))
print('Accuracy of randomforest classifier on test set: {:.2f}'
     .format(Rando.score(X_test, y_test)))
from sklearn.naive_bayes import MultinomialNB

NB = MultinomialNB()
NB.fit(X_train, y_train)
print('Accuracy of NB  classifier on training set: {:.2f}'
     .format(NB.score(X_train, y_train)))
print('Accuracy of NB classifier on test set: {:.2f}'
     .format(NB.score(X_test, y_test)))

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(C=1e5)

logreg.fit(X_train, y_train)
print('Accuracy of Lasso classifier on training set: {:.2f}'
     .format(logreg.score(X_train, y_train)))
print('Accuracy of Lasso classifier on test set: {:.2f}'
     .format(logreg.score(X_test, y_test)))
targets = final['label'].values
logreg = LogisticRegression()
logreg.fit(counts, targets)


# Vectorizing Our Test Data
example_counts = count_vectorizer.transform(preprocessed_test)

predictions = logreg.predict(example_counts)

pred=pd.DataFrame(predictions,columns=['label'])
pred['id']=test['id']
pred.groupby('label').count()
pred.to_csv('submition.csv', index=False)

pred.head(5)
