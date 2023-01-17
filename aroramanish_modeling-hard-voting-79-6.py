# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv(r"../input/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df.head()
df.info()
df.isnull().sum()
df.describe() 
import matplotlib.pyplot as plt
import seaborn as sns
corr = df.corr()

corr.style.background_gradient(cmap='coolwarm')
df['PaymentMethod'].value_counts()
df_cat = df[['gender','SeniorCitizen','Partner','Dependents','PhoneService','MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract','PaperlessBilling','PaymentMethod','Churn']]
df_cat.head()
df.head()
df = df.iloc[:,1:]
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'],errors = 'coerce')

df = df.dropna()
df.head()

df['Churn'].replace(to_replace = 'No', value = 0, inplace = True)

df['Churn'].replace(to_replace = 'Yes', value = 1, inplace = True)
df = pd.get_dummies(df)



plt.figure(figsize=(20,8))

corr = df.corr()['Churn'].sort_values(ascending = False).plot(kind='bar')



X = df.drop(['Churn'],1)

X.head()
y = df['Churn']

y.head()
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier

from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score



X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.30, random_state = 40)

classifiers = [ ['LogisticRegression :', LogisticRegression()],

                 ['DecisionTree :',DecisionTreeClassifier()],

               ['RandomForest :',RandomForestClassifier()], 

               ['AdaBoostClassifier :', AdaBoostClassifier()],

               ['XGB :', XGBClassifier()]]

predictions_df = pd.DataFrame()

predictions_df['actual_labels'] = y_test



for name,classifier in classifiers:

    classifier = classifier

    classifier.fit(X_train, y_train)

    predictions = classifier.predict(X_test)

    predictions_df[name.strip(" :")] = predictions

    print(name, accuracy_score(y_test, predictions))
from sklearn.ensemble import VotingClassifier

clf1 = AdaBoostClassifier()

clf2 = LogisticRegression()

clf3 = XGBClassifier()

clf4 = RandomForestClassifier()

vclf = VotingClassifier(estimators=[('adab', clf1), ('lr', clf2), ('xgb', clf3),('rf', clf4)], voting='hard')

vclf.fit(X_train, y_train)

predictions = vclf.predict(X_test)

print(accuracy_score(y_test, predictions))