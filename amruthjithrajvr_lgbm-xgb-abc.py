# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
raw_data = pd.read_csv("../input/nlp-getting-started/train.csv")
raw_data
import pandas as pd

import numpy as np 

import matplotlib.pyplot as plt

import seaborn as sns

import re

import nltk

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

from sklearn.model_selection import cross_val_score

from nltk.stem import WordNetLemmatizer

import pickle

nltk.download('stopwords')

nltk.download('wordnet')
raw_data['target'] = raw_data['target'].replace(0, 'Zero')

raw_data['target'] = raw_data['target'].replace(1, 'One')
# Shuffle the Dataset.

shuffled_data = raw_data.sample(frac=1,random_state=4)



# Put all the fraud class in a separate dataset.

Zero = shuffled_data.loc[shuffled_data['target'] == 'Zero'].sample(n=3271,random_state=42)



#Randomly select 866 observations from the non-fraud (majority class)

One = shuffled_data.loc[shuffled_data['target'] == 'One'].sample(n=3271,random_state=42)



# Concatenate both dataframes again

data = pd.concat([Zero, One])



plt.figure(figsize=(8, 8))

sns.countplot('target', data=data)

plt.title('Balanced Classes')

plt.show()
#droping unwanted coulmns 

data = data.drop(["id", "keyword", "location"],axis = 1)

#lable encoding for catagorical data

from sklearn.preprocessing import LabelEncoder

#lable encoding for catagorical data

bin_cols = data.nunique()[data.nunique() == 2].keys().tolist()

le = LabelEncoder()

for i in bin_cols :

    data[i] = le.fit_transform(data[i])



data.reset_index(inplace = True)
# Cleaning the texts

corpus = []

i = 0

review = ""

for i in range(0, 6542):

    text = re.sub('[^a-zA-Z]', ' ', data['text'][i])

    text = text.lower()

    text = text.split()

    wl = WordNetLemmatizer()

    text = [wl.lemmatize(word) for word in text if not word in set(stopwords.words('english'))]

    text = ' '.join(text)

    corpus.append(text)
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(ngram_range=(1,1))

X = cv.fit_transform(corpus).toarray()

y = data.iloc[:, -1].values
from sklearn.feature_selection import SelectKBest 

from sklearn.feature_selection import chi2 

# 15000 features with highest chi-squared statistics are selected 

chi2_features = SelectKBest(chi2, k = 14000)

X = chi2_features.fit_transform(X, y)
# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 50)
import lightgbm as lgb

gbm = lgb.LGBMClassifier(objective='binary')

gbm.fit(X_train, y_train,

        eval_set=[(X_test, y_test)],

        eval_metric=['auc', 'binary_logloss'],

early_stopping_rounds=5)

y_pred = gbm.predict(X_test)
from sklearn.metrics import f1_score

f1 = f1_score(y_test, y_pred)

print("F1 Score :", f1)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_pred,y_test)

print("Accuracy :", accuracy)

from sklearn.metrics import cohen_kappa_score

cohen = cohen_kappa_score(y_pred,y_test)

print("Cohens Kappa :", cohen)
from xgboost import XGBClassifier

xgb = XGBClassifier()

xgb.fit(X_train, y_train)   

# Predicting the Test set results

y_pred = xgb.predict(X_test)

cross = cross_val_score(estimator = xgb, X = X_train, y = y_train, cv =2)

cross = cross.mean()

print("10-Fold Cross  Validation Score:", cross)

from sklearn.metrics import f1_score

f1 = f1_score(y_test, y_pred)

print("F1 Score :", f1)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_pred,y_test)

print("Accuracy :", accuracy)

from sklearn.metrics import cohen_kappa_score

cohen = cohen_kappa_score(y_pred,y_test)

print("Cohens Kappa :", cohen)
from sklearn.ensemble import AdaBoostClassifier

abc = AdaBoostClassifier()

# Train Adaboost Classifer

model = abc.fit(X_train, y_train)

y_pred = abc.predict(X_test)

cross = cross_val_score(estimator = model, X = X_train, y = y_train, cv =2)

cross = cross.mean()

print("10-Fold Cross  Validation Score", cross)

from sklearn.metrics import f1_score

f1 = f1_score(y_test, y_pred)

print("F1 Score :", f1)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_pred,y_test)

print("Accuracy :", accuracy)

from sklearn.metrics import cohen_kappa_score

cohen = cohen_kappa_score(y_pred,y_test)

print("Cohens Kappa :", cohen)