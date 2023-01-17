#Loading essential packages

import numpy as np

import pandas as pd
# Loading the dataset

df = pd.read_csv('../input/fakenews2/kaggle_fake_train.csv')
df.shape
df.columns
df.head()
#Dropping the 'id' column

df.drop('id', axis=1, inplace=True)
df.columns
#Importing packages

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
#Visualizing the count of 'fake news' in the dataset

plt.figure(figsize=(10,7))

sns.countplot(x='label', data=df)

plt.xlabel('News Classification')

plt.ylabel('Count')
print(df.shape)
#Finding any NaN values

df.isna().any()
#Dropping NaN values

df.dropna(inplace=True)

print(df.shape)
news = df.copy()
news.reset_index(inplace=True)
#Importing essential libraries for performing Natural Language Processing on 'kaggle_fake_train' dataset

import nltk

import re

nltk.download('stopwords')

from nltk.corpus import stopwords

from nltk.stem import PorterStemmer
#Cleaning the news

corpus = []

ps = PorterStemmer()



for i in range(0,news.shape[0]):



  #Cleaning special character from the news-title

  title = re.sub(pattern='[^a-zA-Z]', repl=' ', string=news.title[i])



  #Converting the entire news-title to lower case

  title = title.lower()



  #Tokenizing the news-title by words

  words = title.split()



  #Removing the stopwords

  words = [word for word in words if word not in set(stopwords.words('english'))]



  #Stemming the words

  words = [ps.stem(word) for word in words]



  #Joining the stemmed words

  title = ' '.join(words)



  #Building a corpus of news-title

  corpus.append(title)
corpus[0:10]
#Creating the Bag of Words model

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=5000, ngram_range=(1,3))

X = cv.fit_transform(corpus).toarray()
X.shape
X[0:10]
#Extracting dependent variable from the dataset

y = news['label']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
from sklearn.linear_model import LogisticRegression

lr_classifier = LogisticRegression(random_state=0)

lr_classifier.fit(X_train, y_train)
# Predicting the Test set results

lr_y_pred = lr_classifier.predict(X_test)
# Accuracy, Precision and Recall

from sklearn.metrics import accuracy_score, precision_score, recall_score

score1 = accuracy_score(y_test, lr_y_pred)

score2 = precision_score(y_test, lr_y_pred)

score3 = recall_score(y_test, lr_y_pred)

print("---- Scores ----")

print("Accuracy score is: {}%".format(round(score1*100,2)))

print("Precision score is: {}".format(round(score2,2)))

print("Recall score is: {}".format(round(score3,2)))
# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

lr_cm = confusion_matrix(y_test, lr_y_pred)
lr_cm
# Plotting the confusion matrix

plt.figure(figsize=(10,7))

sns.heatmap(data=lr_cm, annot=True, cmap="Blues", xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])

plt.xlabel('Predicted values')

plt.ylabel('Actual values')

plt.title('Confusion Matrix for Logistic Regression Algorithm')

plt.show()
# Hyperparameter tuning the Logistic Regression Classifier

best_accuracy = 0.0

c_val = 0.0

for i in np.arange(0.1,1.1,0.1):

  temp_classifier = LogisticRegression(C=i, random_state=0)

  temp_classifier.fit(X_train, y_train)

  temp_y_pred = temp_classifier.predict(X_test)

  score = accuracy_score(y_test, temp_y_pred)

  print("Accuracy score for C={} is: {}%".format(round(i,1), round(score*100,2)))

  if score>best_accuracy:

    best_accuracy = score

    c_val = i

print('--------------------------------------------')

print('The best accuracy is {}% with C value as {}'.format(round(best_accuracy*100, 2), round(c_val,1)))
classifier = LogisticRegression(C=0.8, random_state=0)

classifier.fit(X_train, y_train)
def fake_news(sample_news):

  sample_news = re.sub(pattern='[^a-zA-Z]',repl=' ', string=sample_news)

  sample_news = sample_news.lower()

  sample_news_words = sample_news.split()

  sample_news_words = [word for word in sample_news_words if not word in set(stopwords.words('english'))]

  ps = PorterStemmer()

  final_news = [ps.stem(word) for word in sample_news_words]

  final_news = ' '.join(final_news)



  temp = cv.transform([final_news]).toarray()

  return classifier.predict(temp)
# Importing test dataset

df_test = pd.read_csv('../input/fakenews1/kaggle_fake_test.csv')
df_test.columns
news_title = df_test['title']
news_title.shape
#For generating random integer

from random import randint
#Predicting values

row = randint(0,news_title.shape[0]-1)

sample_news = news_title[row]



print('News: {}'.format(sample_news))

if fake_news(sample_news):

  print('Prediction: This is a FAKE news!')

else:

  print('Prediction: This is a REAL news.')
#Predicting values

row = randint(0,news_title.shape[0]-1)

sample_news = news_title[row]



print('News: {}'.format(sample_news))

if fake_news(sample_news):

  print('Prediction: This is a FAKE news!')

else:

  print('Prediction: This is a REAL news.')
#Predicting values

row = randint(0,news_title.shape[0]-1)

sample_news = news_title[row]



print('News: {}'.format(sample_news))

if fake_news(sample_news):

  print('Prediction: This is a FAKE news!')

else:

  print('Prediction: This is a REAL news.')
#Predicting values

row = randint(0,news_title.shape[0]-1)

sample_news = news_title[row]



print('News: {}'.format(sample_news))

if fake_news(sample_news):

  print('Prediction: This is a FAKE news!')

else:

  print('Prediction: This is a REAL news.')