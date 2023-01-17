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
#Importing Libraries

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
#Get Data

train_data=pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

test_data=pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
#Check snapshot of data

train_data.head()
#Check snapshot of data

test_data.head()
train_data.shape
test_data.shape
#Creating a column named target for test data with NA values

test_data['target']=np.nan

#Checking snapshot of test data

test_data.head()
#Concat train and test data set 

dataframe=pd.concat([train_data,test_data],ignore_index=True)

#Check head of dataframe

dataframe.head()
dataframe.shape
#top 10 keywords from tweets in data

dataframe['keyword'].value_counts(dropna=False).head(10)
#Top 10 Locations of the Tweets in data

dataframe.location.value_counts(dropna=False).head(10)
#Top 10 locations from where real Disaster tweets have received

temp4=dataframe.groupby('target')['location'].value_counts()[1][:10]

temp4
#Check Target Variable

dataframe.target.value_counts(dropna=False)
#Here we remove leading and ending spaces of string

dataframe['text']=dataframe['text'].apply(lambda x: x.strip())
#Convert text to lowercase

dataframe['text']=dataframe['text'].apply(lambda x: " ".join(x.lower() for x in x.split()))

dataframe.head()
#Removing Punctuations

import string

punct_dict=dict((ord(punct),None) for punct in string.punctuation)

print(string.punctuation)

print(punct_dict)
#Removing Stop Words

from nltk.corpus import stopwords

stop=stopwords.words('english')

len(stop)
dataframe['text']=dataframe['text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

dataframe['text'].head()
#Removing Numbers

dataframe['text']=dataframe['text'].apply(lambda x: ''.join([x for x in x if not x.isdigit()]))

dataframe['text']
#Tokenization

from nltk.tokenize import word_tokenize

for i in range(0,len(dataframe['text'])):

    dataframe['text'][i]=word_tokenize(dataframe['text'][i])

dataframe['text']   
#Lemmatization on sentences

from nltk.stem import WordNetLemmatizer

lem=WordNetLemmatizer()

dataframe['text']=dataframe['text'].apply(lambda x: ' '.join(lem.lemmatize(term) for term in x))

dataframe['text']
#Converting datatype of target variable from float to int

dataframe['target']=dataframe['target'][:7613].astype('int')

dataframe.head()
#Removing URLs

dataframe['text'] = dataframe['text'].str.replace('http\S+|www.\S+','',case=False)

dataframe['text']
#Get Frequency of Words

all_words=[]

for msg in dataframe['text']:

    words=word_tokenize(msg)

    for w in words:

        all_words.append(w)

import nltk

frequency_dist=nltk.FreqDist(all_words)

print('Length of the words',len(frequency_dist))

print('Most Common Words',frequency_dist.most_common(100))
#Feature extraction using Tfidf Vectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

tfidfVect=TfidfVectorizer(max_features=5000,stop_words='english')



tfidf=tfidfVect.fit_transform(dataframe['text'])

tfidf.shape
#Splitting data into train and test

train=tfidf[:7613]

test=tfidf[7613:]

#splitting train data into training and validation sets 

X_train=train[:5330]

X_valid=train[5330:]

y_train=dataframe['target'][:5330]

y_valid=dataframe['target'][5330:7613]
#Importing Classification algorithms

from sklearn.naive_bayes import MultinomialNB

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier
#Naive Bayes Model

naive=MultinomialNB()

naive_model=naive.fit(X_train,y_train)

naive_model
pred=naive_model.predict(X_valid)

pred
from sklearn.metrics import classification_report,confusion_matrix,precision_score,recall_score,f1_score,accuracy_score

print('Classification Report',classification_report(y_valid,pred))

print('Confusion Matrix',confusion_matrix(y_valid,pred))

print('Accuracy Score',accuracy_score(y_valid,pred))

print('Precision Score',precision_score(y_valid,pred))

print('Recall Score',recall_score(y_valid,pred))

print('F1 Score',f1_score(y_valid,pred))
#Predicting final test data

final_naive_pred=naive_model.predict(test)

final_naive_pred=pd.Series(final_naive_pred)

final_naive_pred.value_counts()
#Logistic Regression Results

log=LogisticRegression()

log_model=log.fit(X_train,y_train)

log_model
#Prediction

from sklearn.preprocessing import binarize

pred2=log_model.predict_proba(X_valid)

pred3=log_model.predict(X_valid)

#Important Metrics used to know the performance of the model

print('Classification Report',classification_report(y_valid,pred3))

print('Confusion Matrix',confusion_matrix(y_valid,pred3))

print('Accuracy Score',accuracy_score(y_valid,pred3))

print('Precision Score',precision_score(y_valid,pred3))

print('Recall Score',recall_score(y_valid,pred3))

print('F1 Score',f1_score(y_valid,pred3))
#ROC AUC Curve plot

from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve







fpr,tpr,thresholds=roc_curve(y_valid,pred2[:,1])

plt.figure(figsize=(10,8))

plt.plot(fpr,tpr)

plt.xlim([0.0,1.0])

plt.ylim([0.0,1.0])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate ')

plt.legend()

plt.title('ROC Curve')
#Final Prediction of test data 

final_log_pred=log_model.predict(test)

final_log_pred=pd.Series(final_log_pred)

final_log_pred.value_counts()
#Decision Tree Model

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(random_state=101)

dt_model=dt.fit(X_train,y_train)

dt_model
#prediction

pred4=dt_model.predict(X_valid)

pred4
from sklearn.metrics import classification_report,confusion_matrix,precision_score,recall_score,f1_score,accuracy_score

print('Classification Report',classification_report(y_valid,pred4))

print('Confusion Matrix',confusion_matrix(y_valid,pred4))

print('Accuracy Score',accuracy_score(y_valid,pred4))

print('Precision Score',precision_score(y_valid,pred4))

print('Recall Score',recall_score(y_valid,pred4))

print('F1 Score',f1_score(y_valid,pred4))

#Predicting final test data

final_dt_pred=dt_model.predict(test)

final_dt_pred=pd.Series(final_dt_pred)

final_dt_pred.value_counts()
#Random Forest Model

rf=RandomForestClassifier(n_estimators=100,max_features='sqrt')

rf_model=rf.fit(X_train,y_train)

rf_model
#Prediction

pred5=rf_model.predict(X_valid)

pred5
#Important Metrics used to know the Performance of Model

print('Classification Report',classification_report(y_valid,pred5))

print('Confusion Matrix',confusion_matrix(y_valid,pred5))

print('Accuracy Score',accuracy_score(y_valid,pred5))

print('Precision Score',precision_score(y_valid,pred5))

print('Recall Score',recall_score(y_valid,pred5))

print('F1 Score',f1_score(y_valid,pred5))
final_rf_pred=rf_model.predict(test)

final_rf_pred=pd.Series(final_rf_pred)

final_rf_pred.value_counts()
#Final sumbmission Best accuracy of 0.76

naive_pred=pd.DataFrame(final_naive_pred, columns=['target'])

naive_pred

test_data1=pd.concat([test_data['id'],naive_pred], axis=1)

final_sub1=test_data1.to_csv('naive.csv', index=False, header=True)
#LogReg Final

log_pred=pd.DataFrame(final_log_pred, columns=['target'])

log_pred

test_data2=pd.concat([test_data['id'],log_pred], axis=1)

final_sub2=test_data2.to_csv('LogReg.csv', index=False, header=True)
#DT Final

dt_pred=pd.DataFrame(final_dt_pred, columns=['target'])

dt_pred

test_data3=pd.concat([test_data['id'],dt_pred], axis=1)

final_sub3=test_data3.to_csv('DT.csv', index=False, header=True)
#Random Forest Final

rf_pred=pd.DataFrame(final_rf_pred, columns=['target'])

rf_pred

test_data4=pd.concat([test_data['id'],rf_pred], axis=1)

final_sub4=test_data4.to_csv('RF.csv', index=False, header=True)