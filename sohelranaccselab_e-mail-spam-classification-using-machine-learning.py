# ****-------Notebook Summary----***



#Data Science, Machine Learning



#Data Visualization,EDA Analysis, Data Pre-processing,Data Cleaning,Data Split

#-------------------------------------------------------------------------------------------------

#Machine Learning Algorithm:



#Best Model accuracy:model_3: 99.31%

#Visualize output at graph
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
#Enivornment Setup
#Import libraries



import matplotlib.pyplot as plt 

import seaborn as sns



from nltk.stem.porter import PorterStemmer

from nltk.stem import WordNetLemmatizer

from nltk.sentiment.vader import SentimentIntensityAnalyzer

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize, sent_tokenize





from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import accuracy_score, precision_score, recall_score

from sklearn.model_selection import train_test_split



from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import MultinomialNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier



from collections import Counter

import string



import warnings

warnings.filterwarnings('ignore')

import warnings

warnings.filterwarnings('ignore')
#Data Read, Data Visualization,EDA Analysis,Data Pre-Processing,Data Splitting
#Data Read

file_path = '/kaggle/input/lingspam-classification'

df = pd.read_csv(f'{file_path}/messages.csv')
df.describe()
df.info()
import seaborn; seaborn.set()

df.plot();
#checking the target variable countplot

sns.countplot(data=df,x = 'label',palette='plasma')
df.head()
# converting all messages to lower case



df['message'] = df['message'].str.lower()
# check data once 

df.head()
df.apply(lambda x: sum(x.isnull()),axis=0)
df.fillna(df['subject'].mode().values[0],inplace=True)
df.apply(lambda x: sum(x.isnull()),axis=0)
df['sub_mssg']=df['subject']+df['message']

df.head()
df['sub_mssg'].describe()
df['length']=df['sub_mssg'].apply(len)

df.head()
#now i'm going to drop un-necessary features 

df.drop('subject',axis=1,inplace=True)
# check it once 

df.head()
lb=df['label'].value_counts().index.tolist()

val=df['label'].value_counts().values.tolist()

exp=(0.025,0)

clr=('orange','blue')

plt.figure(figsize=(10,8),dpi=140)

plt.pie(x=val,explode=exp,labels=lb,colors=clr,autopct='%2.0f%%',pctdistance=0.5, shadow=True,radius=0.9)

plt.legend(["0 = NO SPAM",'1 = SPAM'])

plt.show()
df['message'][0]
import re
def decontact(phrase):

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
mssg=decontact(df['message'][70])

mssg
#REPLACING NUMBERS

df['sub_mssg']=df['sub_mssg'].str.replace(r'\d+(\.\d+)?', 'numbers')

df['sub_mssg'][0]
#CONVRTING EVERYTHING TO LOWERCASE

df['sub_mssg']=df['sub_mssg'].str.lower()

#REPLACING NEXT LINES BY 'WHITE SPACE'

df['sub_mssg']=df['sub_mssg'].str.replace(r'\n'," ") 

# REPLACING EMAIL IDs BY 'MAILID'

df['sub_mssg']=df['sub_mssg'].str.replace(r'^.+@[^\.].*\.[a-z]{2,}$','MailID')

# REPLACING URLs  BY 'Links'

df['sub_mssg']=df['sub_mssg'].str.replace(r'^http\://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(/\S*)?$','Links')

# REPLACING CURRENCY SIGNS BY 'MONEY'

df['sub_mssg']=df['sub_mssg'].str.replace(r'£|\$', 'Money')

# REPLACING LARGE WHITE SPACE BY SINGLE WHITE SPACE

df['sub_mssg']=df['sub_mssg'].str.replace(r'\s+', ' ')



# REPLACING LEADING AND TRAILING WHITE SPACE BY SINGLE WHITE SPACE

df['sub_mssg']=df['sub_mssg'].str.replace(r'^\s+|\s+?$', '')

#REPLACING CONTACT NUMBERS

df['sub_mssg']=df['sub_mssg'].str.replace(r'^\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{4}$','contact number')

#REPLACING SPECIAL CHARACTERS  BY WHITE SPACE 

df['sub_mssg']=df['sub_mssg'].str.replace(r"[^a-zA-Z0-9]+", " ")
#CONVRTING EVERYTHING TO LOWERCASE

df['message']=df['message'].str.lower()

#REPLACING NEXT LINES BY 'WHITE SPACE'

df['message']=df['message'].str.replace(r'\n'," ") 

# REPLACING EMAIL IDs BY 'MAILID'

df['message']=df['message'].str.replace(r'^.+@[^\.].*\.[a-z]{2,}$','MailID')

# REPLACING URLs  BY 'Links'

df['message']=df['message'].str.replace(r'^http\://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(/\S*)?$','Links')

# REPLACING CURRENCY SIGNS BY 'MONEY'

df['message']=df['message'].str.replace(r'£|\$', 'Money')

# REPLACING LARGE WHITE SPACE BY SINGLE WHITE SPACE

df['message']=df['message'].str.replace(r'\s+', ' ')



# REPLACING LEADING AND TRAILING WHITE SPACE BY SINGLE WHITE SPACE

df['message']=df['message'].str.replace(r'^\s+|\s+?$', '')

#REPLACING CONTACT NUMBERS

df['message']=df['message'].str.replace(r'^\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{4}$','contact number')

#REPLACING SPECIAL CHARACTERS  BY WHITE SPACE 

df['message']=df['message'].str.replace(r"[^a-zA-Z0-9]+", " ")
df['sub_mssg'][0]
df.head()
from tqdm import tqdm

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize, sent_tokenize

# removing stopwords 

stop = stopwords.words('english')

df['Cleaned_Text'] = df['sub_mssg'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
df.head()
df.drop('message',axis=1,inplace=True)
df.drop('sub_mssg',axis=1,inplace=True)
df.head()
df.apply(lambda x: sum(x.isnull()),axis=0)
df['lgth_clean']=df['Cleaned_Text'].apply(len)

df.head()
original_length=sum(df['length'])

after_cleaning=sum(df['lgth_clean'])
print("original_length",original_length)

print('after_cleaning',after_cleaning)
# 1. Convert text into vectors using TF-IDF

# 2. Instantiate MultinomialNB classifier

# 3. Split feature and label

from sklearn.feature_extraction.text import TfidfVectorizer



from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, precision_score, recall_score

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score

import warnings

from sklearn.pipeline import Pipeline
tvec = TfidfVectorizer()

lr = LogisticRegression(solver = "lbfgs")
X = df.Cleaned_Text

Y = df.label



X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, random_state = 42,stratify=Y)
model = Pipeline([('vectorizer',tvec),('classifier',lr)])



model.fit(X_train,Y_train)
from sklearn.metrics import confusion_matrix



y_pred = model.predict(X_test)



confusion_matrix(y_pred,Y_test)
print("Accuracy : ", accuracy_score(y_pred,Y_test))

print("Precision : ", precision_score(y_pred,Y_test, average = 'weighted'))

print("Recall : ", recall_score(y_pred,Y_test, average = 'weighted'))
knc = KNeighborsClassifier()

model_1 = Pipeline([('vectorizer',tvec),('classifier',knc)])

model_1.fit(X_train,Y_train)
y_pred = model_1.predict(X_test)



print(confusion_matrix(y_pred,Y_test))

print("Accuracy : ", accuracy_score(y_pred,Y_test))

print("Precision : ", precision_score(y_pred,Y_test, average = 'weighted'))

print("Recall : ", recall_score(y_pred,Y_test, average = 'weighted'))
abc = AdaBoostClassifier()

model_3 = Pipeline([('vectorizer',tvec),('classifier',abc)])

model_3.fit(X_train,Y_train)

y_pred = model_3.predict(X_test)



print(confusion_matrix(y_pred,Y_test))

print("Accuracy : ", accuracy_score(y_pred,Y_test))

print("Precision : ", precision_score(y_pred,Y_test, average = 'weighted'))

print("Recall : ", recall_score(y_pred,Y_test, average = 'weighted'))
mnb = MultinomialNB()

model_5 = Pipeline([('vectorizer',tvec),('classifier',mnb)])

model_5.fit(X_train,Y_train)
y_pred = model_5.predict(X_test)



print(confusion_matrix(y_pred,Y_test))

print("Accuracy : ", accuracy_score(y_pred,Y_test))

print("Precision : ", precision_score(y_pred,Y_test, average = 'weighted'))

print("Recall : ", recall_score(y_pred,Y_test, average = 'weighted'))
gbc = GradientBoostingClassifier()

model_6= Pipeline([('vectorizer',tvec),('classifier',gbc)])

model_6.fit(X_train,Y_train)





y_pred = model_6.predict(X_test)

print(confusion_matrix(y_pred,Y_test))

print("Accuracy : ", accuracy_score(y_pred,Y_test))

print("Precision : ", precision_score(y_pred,Y_test, average = 'weighted'))

print("Recall : ", recall_score(y_pred,Y_test, average = 'weighted'))
from sklearn.ensemble import RandomForestClassifier as RFC

rfc = RFC(random_state=42)

model_7 = Pipeline([('vectorizer',tvec),('classifier',rfc)])



model_7.fit(X_train,Y_train)



y_pred = model_7.predict(X_test)

print(confusion_matrix(y_pred,Y_test))

print("Accuracy : ", accuracy_score(y_pred,Y_test))

print("Precision : ", precision_score(y_pred,Y_test, average = 'weighted'))

print("Recall : ", recall_score(y_pred,Y_test, average = 'weighted'))
#Using RandomForestRegressor

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=1000,random_state=2520)


model_8 = Pipeline([('vectorizer',tvec),('classifier',rfc)])



model_8.fit(X_train,Y_train)



y_pred = model_8.predict(X_test)

print(confusion_matrix(y_pred,Y_test))

print("Accuracy : ", accuracy_score(y_pred,Y_test))

print("Precision : ", precision_score(y_pred,Y_test, average = 'weighted'))

print("Recall : ", recall_score(y_pred,Y_test, average = 'weighted'))
from xgboost import XGBClassifier
xgb = XGBClassifier(base_score=0.5, booster='dart', colsample_bylevel=1,

              colsample_bynode=1, colsample_bytree=1, gamma=0.7795918367346939,

              learning_rate=0.325, max_delta_step=0, max_depth=22,

              min_child_weight=1, missing=None, n_estimators=833, n_jobs=1,

              nthread=None, objective='binary:logistic', random_state=0,

              reg_alpha=0.25, reg_lambda=2, scale_pos_weight=1, seed=None,

              silent=None, subsample=1, verbosity=1)


model_9 = Pipeline([('vectorizer',tvec),('classifier',xgb)])



model_9.fit(X_train,Y_train)



y_pred = model_9.predict(X_test)

print(confusion_matrix(y_pred,Y_test))

print("Accuracy : ", accuracy_score(y_pred,Y_test))

print("Precision : ", precision_score(y_pred,Y_test, average = 'weighted'))

print("Recall : ", recall_score(y_pred,Y_test, average = 'weighted'))
result=model_9.predict(['your microsoft account has been compromised ,you must update before or else your account going to close click to update'])

result
result=model_9.predict(['Today we want to inform you that the application period for 15.000 free Udacity Scholarships in Data Science is now open! Please apply by November 16th, 2020 via https://www.udacity.com/bertelsmann-tech-scholarships.'])

result
#Here 0 is spam and 1 is normal message.
result=model_3.predict(['your microsoft account has been compromised ,you must update before or else your account going to close click to update'])

result
result=model_3.predict(['Today we want to inform you that the application period for 15.000 free Udacity Scholarships in Data Science is now open! Please apply by November 16th, 2020 via https://www.udacity.com/bertelsmann-tech-scholarships.'])

result