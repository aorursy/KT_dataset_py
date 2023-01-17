import math
import pandas as pd
import numpy as np
import random as rnd
import matplotlib.pyplot as plt
import time
import pickle
import string
import sys
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import re
import os
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import RidgeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
# df = pd.read_csv('/content/drive/My Drive/Colab_Data_set/bgg-13m-reviews.csv')
# df.head()

df =  pd.read_csv('../input/boardgamegeek-reviews/bgg-13m-reviews.csv', index_col=0)
df.head()
dataset = df.drop(columns="ID")
dataset = dataset.drop(columns="user")
dataset = dataset.drop(columns="name")


dataset.head()
#Removing Empty Comments
dataset = dataset.dropna(subset=['comment'])
dataset.head()

#plot histogram of ratings
num_bins = 500
plt.hist(dataset.rating, num_bins, facecolor='blue', alpha=10)

#plt.xticks(range(9000))
plt.title('Histogram of Ratings')
plt.xlabel('Ratings')
plt.ylabel('Count')
plt.show()
from sklearn.model_selection import train_test_split
train,test = train_test_split(dataset,test_size = 0.25,random_state=0)
train.head()
temp_train= train.copy()
temp_train['rating'].max()
temp_train['rating'].min()
temp_train.head()
sample_1,sample_2 = train_test_split(temp_train,test_size=0.5,random_state = 2)
sample_1.head()
sample_2.head()
#lowercase and remove punctuation
sample_1['comment'] = sample_1['comment'].str.lower().apply(lambda x:''.join([i for i in x if i not in string.punctuation]))

# stopword list to use
stopwords_list = stopwords.words('english')

"""
Since this is a game dataset review, and since i ran the data cleaning process once before,
i identified a few extra words which can also be added to our stopword list

"""
stopwords_list.extend(('game','play','played','players','player','people','really','board','games','one','plays','cards','would')) 
#remove stopwords
sample_1['comment'] = sample_1['comment'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords_list)]))

sample_1.head()
#lowercase and remove punctuation
sample_2['comment'] = sample_2['comment'].str.lower().apply(lambda x:''.join([i for i in x if i not in string.punctuation]))

#remove stopwords
sample_2['comment'] = sample_2['comment'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords_list)]))

sample_2.head()
#Splitting Sample 1 
sample_1_train, sample_1_dev = train_test_split(sample_1,test_size=0.2,random_state = 0)

#Splitting Sample 2
sample_2_train, sample_2_dev = train_test_split(sample_2,test_size=0.2,random_state = 0)
sample_1_train.head()
sample_1_dev.head()
sample_2_train.head()
sample_2_dev.head()
#Training X and Y for Sample 1
train_1x=[]
for i in sample_1_train['comment']:
    train_1x.append(i)

train_1y=[]
for i in sample_1_train['rating']:
    train_1y.append(i)

dev_1x=[]
for i in sample_1_dev['comment']:
    dev_1x.append(i)

dev_1y=[]
for i in sample_1_dev['rating']:
    dev_1y.append(i)
#Training X and Y for Sample 2
train_2x=[]
for i in sample_2_train['comment']:
    train_2x.append(i)

train_2y=[]
for i in sample_2_train['rating']:
    train_2y.append(i)

dev_2x=[]
for i in sample_2_dev['comment']:
    dev_2x.append(i)

dev_2y=[]
for i in sample_2_dev['rating']:
    dev_2y.append(i)
train_1x[:5]
train_1y[:5]
dev_1x[:5]
dev_1y[:5]
train_2x[:5]
train_2y[:5]
dev_2x[:5]
dev_2y[:5]
#Text Cleaning
def TextClean(data):
    
    txt = []
    for T in data:
        T = re.sub(r'@[A-Za-z0-9_]+','',T)
        T = re.sub(r"http\S+", "", T)
        T = T.replace('<br />', '')
        T = T.replace("\'","")
        T = T.replace("?'","")
        T = T.replace("*", "")
        T = T.replace("/", "")
        T = T.replace("\ ", "")
        T = T.replace(".", "")
        T = T.replace("(", "")
        T = T.replace(")", "")
        T = T.replace(":", "")
        T = T.replace('"', "")
        T = T.replace(",", "")
        T = T.replace("!", "")
        T = T.replace("'", "")
        T = T.replace("&", "")
        T = re.sub(r"[0-9]*", "", T)
        T = re.sub(r"(”|“|-|\+|`|#|,|;|\|/|\\|)*","", T)
        T = re.sub(r"&amp","", T)
        T = T.lower()
        txt.append(T)
    return txt


#Removing Special Characters
def Remove_SC(text):
    alphabet = []
    alpha = 'a'
    for i in range(0, 26): 
        alphabet.append(alpha) 
        alpha = chr(ord(alpha) + 1)
    l = []
    for i in text:
        txt = []
        t = i.split(' ')
        for j in t:
            m = j
            for k in m:
                if k not in alphabet:
                    m = m.replace(k, '')
            if m != '':
                txt.append(m)
        #l.append(txt)
        s = ''
        for j in txt:
            s = s + j + ' '
        l.append(s)
    return l
#Cleaning Sample 1 Train Sets and Development Sets
#Execution takes 120 seconds

#Clean Text and Remove Numerical Values 
train_1x = TextClean(train_1x)
dev_1x   = TextClean(dev_1x)
#Remove Special Characters
train_1x = Remove_SC(train_1x)
dev_1x = Remove_SC(dev_1x)
train_1x[:5]
dev_1x[:5]
#Cleaning Sample 2 Train Sets and Development Sets
#Execution takes 120 seconds
#Clean Text
train_2x = TextClean(train_2x)
dev_2x   = TextClean(dev_2x)
#Remove Special Characters
train_2x = Remove_SC(train_2x)
dev_2x = Remove_SC(dev_2x)
train_2x[:5]
dev_2x[:5]
#Rounding Rating Values for : 
#Training And Development Set for Sample 1

train_1y  = [round(num) for num in train_1y]
dev_1y    = [round(num) for num in dev_1y]

#Training And Development Set for Sample 2

train_2y  = [round(num) for num in train_2y]
dev_2y    = [round(num) for num in dev_2y]
#Takes 30 seconds to execute
nb_1 = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', MultinomialNB()),])
#Fitting Training Set to Model
nb_1.fit(train_1x,train_1y)
#Prediction
y_pred = nb_1.predict(dev_1x)
#Predicting For Development Set 1
print("Accuracy of Multinomial Naive-Bayes Classifier for Sample 1 :", accuracy_score(dev_1y, y_pred)*100," %")

#Takes 30 seconds to execute
nb_2 = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', MultinomialNB()),])
#Fitting Training Set to Model
nb_2.fit(train_2x,train_2y)
#Prediction
y_pred = nb_2.predict(dev_2x)
#Predicting For Development Set 2
print("Accuracy of Multinomial Naive-Bayes Classifier for Sample 2 :", accuracy_score(dev_2y, y_pred)*100," %")
#Takes 1 minute to execute
sgd_clf_1 = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf',
  SGDClassifier(penalty='l2',alpha=1e-3, random_state=42,max_iter=5, tol=None)),])
#Fitting Model 
sgd_clf_1.fit(train_1x,train_1y)
#Prediction
y_pred = sgd_clf_1.predict(dev_1x)
#Predicting For Development Set 1
print("Accuracy of Linear SVM Classifier for Sample 1 :", accuracy_score(dev_1y, y_pred)*100," %")
#Takes 1 minute to execute
sgd_clf_2 = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', 
  SGDClassifier(penalty='l2',alpha=1e-3, random_state=42,max_iter=5, tol=None)),])
#Fitting Model 
sgd_clf_2.fit(train_2x,train_2y)
#Prediction
y_pred = sgd_clf_2.predict(dev_2x)
#Predicting For Development Set 2
print("Accuracy of Linear SVM Classifier for Sample 2 :", accuracy_score(dev_2y, y_pred)*100," %")
#This Code takes 230 seconds to execute
tridge_clf_1 = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf',RidgeClassifier()),])
tridge_clf_1.fit(train_1x,train_1y)

y_pred = tridge_clf_1.predict(dev_1x)
#Predicting For Development Set 1
print("Accuracy of Ridge Classifier for Sample 1  :", accuracy_score(dev_1y, y_pred)*100," %")
#This Code takes 230 seconds to execute
tridge_clf_2 = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf',RidgeClassifier()),])
tridge_clf_2.fit(train_2x,train_2y)

y_pred = tridge_clf_2.predict(dev_2x)
#Predicting For Development Set 2
print("Accuracy of Ridge Classifier for Sample 2  :", accuracy_score(dev_2y, y_pred)*100," %")
#Function for calculating accuracy with a smoothing factor
def smooth_acc(yr,yp):
  c=0
  for i in range(len(yr)):
    if(yr[i] == yp[i] or yr[i]==(yp[i]+1) or yr[i]==(yp[i]-1) ):
      c=c+1
    
  return c/len(yr)
nb_1 = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', MultinomialNB()),])
#Fitting Training Set to Model
nb_1.fit(train_1x,train_1y)
#Prediction
y_pred = nb_1.predict(dev_1x)
#Predicting For Development Set 1
print("Accuracy of Multinomial Naive-Bayes Classifier for Sample 1 :", smooth_acc(dev_1y, y_pred)*100," %")
nb_2 = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', MultinomialNB()),])
#Fitting Training Set to Model
nb_2.fit(train_2x,train_2y)
#Prediction
y_pred = nb_2.predict(dev_2x)
#Predicting For Development Set 2
print("Accuracy of Multinomial Naive-Bayes Classifier for Sample 1 :", smooth_acc(dev_2y, y_pred)*100," %")
sgd_clf_1 = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', 
  SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42,max_iter=5, tol=None)),])
#Fitting Model 
sgd_clf_1.fit(train_1x,train_1y)
#Prediction
y_pred = sgd_clf_1.predict(dev_1x)
#Predicting For Development Set 1
print("Accuracy of Linear SVM Classifier for Sample 1 :", smooth_acc(dev_1y, y_pred)*100," %")
sgd_clf_2 = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', 
  SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42,max_iter=5, tol=None)),])
#Fitting Model 
sgd_clf_2.fit(train_2x,train_2y)
#Prediction
y_pred = sgd_clf_2.predict(dev_2x)
#Predicting For Development Set 2
print("Accuracy of Linear SVM Classifier for Sample 2 :", smooth_acc(dev_2y, y_pred)*100," %")
#This Code takes 230 seconds to execute
tridge_clf_1 = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf',RidgeClassifier()),])
tridge_clf_1.fit(train_1x,train_1y)

y_pred = tridge_clf_1.predict(dev_1x)
#Predicting For Development Set 1
print("Accuracy of Ridge Classifier for Sample 1  :", smooth_acc(dev_1y, y_pred)*100," %")
#This Code takes 230 seconds to execute
tridge_clf_2 = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf',RidgeClassifier()),])
tridge_clf_2.fit(train_2x,train_2y)

y_pred = tridge_clf_2.predict(dev_2x)
#Predicting For Development Set 2
print("Accuracy of Ridge Classifier for Sample 2  :", smooth_acc(dev_2y, y_pred)*100," %")
train_1x[:5]
dev_1x[:5]
train_2x[:5]
dev_2x[:5]
print("For Training Set ( Sample 1 ) ")
print("Max value : ",max(train_1y), " Min Rating : ",min(train_1y))

print("For Development Set ( Sample 1 ) ")
print("Max value : ",max(dev_1y), " Min Rating : ",min(dev_1y))
print("For Training Set ( Sample 2 ) ")
print("Max value : ",max(train_2y), " Min Rating : ",min(train_2y))

print("For Development Set ( Sample 2 ) ")
print("Max value : ",max(dev_2y), " Min Rating : ",min(dev_2y))
#Train and Development for Sample 1
new_train_1y=[]
for i in sample_1_train['rating']:
    new_train_1y.append(i)

new_dev_1y=[]
for i in sample_1_dev['rating']:
    new_dev_1y.append(i)

#Train and Development for Sample 2
new_train_2y=[]
for i in sample_2_train['rating']:
    new_train_2y.append(i)

new_dev_2y=[]
for i in sample_2_dev['rating']:
    new_dev_2y.append(i)

#Re-Scaling Values
#Training And Development Set for Sample 1
new_train_1y  = [round(num/2) for num in new_train_1y]
new_dev_1y    = [round(num/2) for num in new_dev_1y]

print("For Training Set ( Sample 1 ) ")
print("Max value : ",max(new_train_1y), " Min Rating : ",min(new_train_1y))

print("For Development Set ( Sample 1 ) ")
print("Max value : ",max(new_dev_1y), " Min Rating : ",min(new_dev_1y))
#Re-Scaling Values
#Training And Development Set for Sample 2
new_train_2y  = [round(num/2) for num in new_train_2y]
new_dev_2y    = [round(num/2) for num in new_dev_2y]

print("For Training Set ( Sample 2 ) ")
print("Max value : ",max(new_train_2y), " Min Rating : ",min(new_train_2y))

print("For Development Set ( Sample 2 ) ")
print("Max value : ",max(new_dev_2y), " Min Rating : ",min(new_dev_2y))
nb_n1 = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', MultinomialNB()),])
#Fitting Training Set to Model
nb_n1.fit(train_1x,new_train_1y)
#Prediction
y_pred = nb_n1.predict(dev_1x)
#Predicting For Development Set 1
print("Accuracy of Multinomial Naive-Bayes Classifier for Sample 1 :", accuracy_score(new_dev_1y, y_pred)*100," %")

nb_n2 = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', MultinomialNB()),])
#Fitting Training Set to Model
nb_n2.fit(train_2x,new_train_2y)
#Prediction
y_pred = nb_n2.predict(dev_2x)
#Predicting For Development Set 2
print("Accuracy of Multinomial Naive-Bayes Classifier for Sample 2 :", accuracy_score(new_dev_2y, y_pred)*100," %")
sgd_clf_n1 = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', 
  SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42,max_iter=5, tol=None)),])
#Fitting Model 
sgd_clf_n1.fit(train_1x,new_train_1y)
#Prediction
y_pred = sgd_clf_n1.predict(dev_1x)
#Predicting For Development Set 1
print("Accuracy of Linear SVM Classifier for Sample 1 :", accuracy_score(new_dev_1y, y_pred)*100," %")

sgd_clf_n2 = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', 
  SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42,max_iter=5, tol=None)),])
#Fitting Model 
sgd_clf_n2.fit(train_2x,new_train_2y)
#Prediction
y_pred = sgd_clf_n2.predict(dev_2x)
#Predicting For Development Set 2
print("Accuracy of Linear SVM Classifier for Sample 2 :", accuracy_score(new_dev_2y, y_pred)*100," %")
#This Code takes 230 seconds to execute
tridge_clf_n1 = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf',RidgeClassifier()),])
tridge_clf_n1.fit(train_1x,new_train_1y)

y_pred = tridge_clf_n1.predict(dev_1x)
#Predicting For Development Set 1
print("Accuracy of Ridge Classifier for Sample 1  :", accuracy_score(new_dev_1y, y_pred)*100," %")

#This Code takes 230 seconds to execute
tridge_clf_n2 = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf',RidgeClassifier()),])
tridge_clf_n2.fit(train_2x,new_train_2y)

y_pred = tridge_clf_n2.predict(dev_2x)
#Predicting For Development Set 2
print("Accuracy of Ridge Classifier for Sample 2  :", accuracy_score(new_dev_2y, y_pred)*100," %")
nb_n1 = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', MultinomialNB()),])
#Fitting Training Set to Model
nb_n1.fit(train_1x,new_train_1y)
#Prediction
y_pred = nb_n1.predict(dev_1x)
#Predicting For Development Set 1
print("Accuracy of Multinomial Naive-Bayes Classifier for Sample 1 :", smooth_acc(new_dev_1y, y_pred)*100," %")

nb_n2 = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', MultinomialNB()),])
#Fitting Training Set to Model
nb_n2.fit(train_2x,new_train_2y)
#Prediction
y_pred = nb_n2.predict(dev_2x)
#Predicting For Development Set 2
print("Accuracy of Multinomial Naive-Bayes Classifier for Sample 2 :", smooth_acc(new_dev_2y, y_pred)*100," %")
sgd_clf_n1 = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', 
  SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42,max_iter=5, tol=None)),])
#Fitting Model 
sgd_clf_n1.fit(train_1x,new_train_1y)
#Prediction
y_pred = sgd_clf_n1.predict(dev_1x)
#Predicting For Development Set 1
print("Accuracy of Linear SVM Classifier for Sample 1 :", smooth_acc(new_dev_1y, y_pred)*100," %")

sgd_clf_n2 = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', 
  SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42,max_iter=5, tol=None)),])
#Fitting Model 
sgd_clf_n2.fit(train_2x,new_train_2y)
#Prediction
y_pred = sgd_clf_n2.predict(dev_2x)
#Predicting For Development Set 2
print("Accuracy of Linear SVM Classifier for Sample 2 :", smooth_acc(new_dev_2y, y_pred)*100," %")
#This Code takes 230 seconds to execute
tridge_clf_n1 = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf',RidgeClassifier()),])
tridge_clf_n1.fit(train_1x,new_train_1y)

y_pred = tridge_clf_n1.predict(dev_1x)
#Predicting For Development Set 1
print("Accuracy of Ridge Classifier for Sample 1  :", smooth_acc(new_dev_1y, y_pred)*100," %")

#This Code takes 230 seconds to execute
tridge_clf_n2 = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf',RidgeClassifier()),])
tridge_clf_n2.fit(train_2x,new_train_2y)

y_pred = tridge_clf_n2.predict(dev_2x)
#Predicting For Development Set 2
print("Accuracy of Ridge Classifier for Sample 2  :", smooth_acc(new_dev_2y, y_pred)*100," %")
#lowercase and remove punctuation
train['comment'] = train['comment'].str.lower().apply(lambda x:''.join([i for i in x if i not in string.punctuation]))

#remove stopwords
test['comment'] = test['comment'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords_list)]))
#Training X and Y
x_train=[]
for i in train['comment']:
    x_train.append(i)

y_train=[]
for i in train['rating']:
  y_train.append(i)

#Test X and Y
x_test=[]
for i in test['comment']:
    x_test.append(i)

y_test=[]
for i in test['rating']:
    y_test.append(i)
#Cleaning Train Sets and Test Sets

#Execution takes 300 seconds
#Clean Text
x_train = TextClean(x_train)
x_test   = TextClean(x_test)
#Remove Special Characters
x_train = Remove_SC(x_train)
x_test = Remove_SC(x_test)
#Training And Test Set for Sample 1
y_train  = [round(num/2) for num in y_train]
y_test   = [round(num/2) for num in y_test]

#This Code takes 230 seconds to execute
tridge_clf_n1 = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf',RidgeClassifier()),])
#Fitting Training Set to Model
tridge_clf_n1.fit(train_1x,new_train_1y)

#Prediction
y_pred = tridge_clf_n1.predict(x_test)
#Predicting For Development Set 1
print("Accuracy of Ridge Classifier for Test Set  :", accuracy_score(y_test, y_pred)*100," %")

#Prediction
y_pred = tridge_clf_n1.predict(x_test)
#Predicting For Development Set 1
print("Accuracy of Ridge Classifier for Test Set  :", smooth_acc(y_test, y_pred)*100," %")
