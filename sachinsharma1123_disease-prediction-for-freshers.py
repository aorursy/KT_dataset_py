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
df=pd.read_csv('/kaggle/input/heart-disease-prediction/Heart_Disease_Prediction.csv')
df
df.info()
df.isnull().sum()
import seaborn as sns

import matplotlib.pyplot as plt
#lets try to get some insight info from the dataset

sns.countplot(x=df['Heart Disease'],hue='Sex',data=df)
#here males are more prone to heart diseases as compare to women
sns.countplot(x=df['Heart Disease'],hue='Chest pain type',data=df)
#here it seems that people who suffered from pain type 4 have more chances of heart disease
sns.countplot(x=df['Sex'],hue='Chest pain type',data=df)
# as we can see from here that more males suffers from chest pain type 4 which is dangerous
sns.barplot(x=df['Sex'],y=df['BP'],data=df)
#there bp rates are almost equal
sns.barplot(x=df['Sex'],y=df['Cholesterol'],data=df)
#females have little bit of higher cholesterol than males
sns.barplot(x=df['Heart Disease'],y=df['Cholesterol'],data=df)
#here we can observe that higher cholesterol level results in chances of heart disease
sns.barplot(x=df['Heart Disease'],y=df['BP'],data=df)
#similary here too,high bp results in more chances of disease

sns.lineplot(x=df['Age'],y=df['BP'],data=df)
#here we can observe that bp increases at the age of 50-60
sns.lineplot(x=df['Age'],y=df['Cholesterol'],data=df)
#similarly here the cholesterol level increases at the age group of 50-65
sns.lineplot(x=df['Age'],y=df['ST depression'],data=df)
#we can observe from here that depression mostly increases bw the age group of 30-40
sns.barplot(x=df['Sex'],y=df['ST depression'],data=df)
#more males are prone to depression as compare to females
sns.barplot(x=df['Heart Disease'],y=df['Exercise angina'],data=df)
#person with high exercise angina has more chances of heart disease
sns.barplot(x=df['Sex'],y=df['Exercise angina'],data=df)
# males have have high exercise angina
sns.barplot(x=df['Heart Disease'],y=df['Number of vessels fluro'],data=df)
#people having high Number of vessels fluro have high chances of heart disease

sns.barplot(x=df['Heart Disease'],y=df['Thallium'],data=df)
#highn thallium count may lead to heart disease
sns.barplot(x=df['Sex'],y=df['FBS over 120'],data=df)
#males have high no of FBS over 120
sns.heatmap(df.corr())
#now we are done with the data visualisations

from sklearn.preprocessing import LabelEncoder,StandardScaler

le=LabelEncoder()

df['Heart Disease']=le.fit_transform(df['Heart Disease'])
y=df['Heart Disease']

x=df.drop(['Heart Disease'],axis=1)
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.2)

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

lr=LogisticRegression(max_iter=10000)

lr.fit(x_train,y_train)

pred_1=lr.predict(x_test)

score_1=accuracy_score(y_test,pred_1)
score_1
from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier()

rfc.fit(x_train,y_train)

pred_2=rfc.predict(x_test)

score_2=accuracy_score(y_test,pred_2)
score_2
from xgboost import XGBClassifier

xgb=XGBClassifier()

xgb.fit(x_train,y_train)

pred_3=xgb.predict(x_test)

score_3=accuracy_score(y_test,pred_3)

score_3
from sklearn.neighbors import KNeighborsClassifier

list_1=[]

for i in range(1,21):

    knn=KNeighborsClassifier(n_neighbors=i)

    knn.fit(x_train,y_train)

    preds=knn.predict(x_test)

    scores=accuracy_score(y_test,preds)

    list_1.append(scores)
max(list_1)
#from all the classifiers used randomforest gives the best accuracy score