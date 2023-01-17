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
df=pd.read_csv('/kaggle/input/german-credit-data-with-risk/german_credit_data.csv')
df
df.info()
df.isnull().sum()
#fill the nan values in both columns
df['Saving accounts']=df['Saving accounts'].fillna(df['Saving accounts'].mode()[0])
df['Checking account']=df['Checking account'].fillna(df['Checking account'].mode()[0])
import seaborn as sns

import matplotlib.pyplot as plt

sns.heatmap(df.isnull())
#now remove the unnamed column as it is unnecessary

df=df.drop(['Unnamed: 0'],axis=1)
df
sns.countplot(x=df['Risk'],hue='Sex',data=df)
sns.countplot(x=df['Risk'],hue='Housing',data=df)
#there is  good risk at own house
sns.countplot(x=df['Risk'],hue='Saving accounts',data=df)
#high risk at little savings
sns.countplot(x=df['Risk'],hue='Checking account',data=df)
sns.scatterplot(x=df['Age'],y=df['Credit amount'],data=df)
sns.distplot(df['Credit amount'])
sns.countplot(x=df['Risk'],hue='Job',data=df)
#now dealing with the categorical features
from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

for i in list(df.columns):

    if df[i].dtype=='object':

        df[i]=le.fit_transform(df[i])
df
y=df['Risk']

x=df.drop(['Risk'],axis=1)
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.2)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

list_scores=[]

list_models=[]

lr=LogisticRegression(max_iter=10000)

lr.fit(x_train,y_train)

pred_1=lr.predict(x_test)

score_1=accuracy_score(y_test,pred_1)

list_scores.append(score_1)

list_models.append('logistic regression')

score_1
from sklearn.neighbors import KNeighborsClassifier

list_1=[]

for i in range(1,21):

    knn=KNeighborsClassifier(n_neighbors=i)

    knn.fit(x_train,y_train)

    pred_s=knn.predict(x_test)

    scores=accuracy_score(y_test,pred_s)

    list_1.append(scores)

    
sns.set_style('whitegrid')

sns.lineplot(x=list(range(1,21)),y=list_1)
list_scores.append(max(list_1))

list_models.append('kneighbors classifier')
from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier()

rfc.fit(x_train,y_train)

pred_2=rfc.predict(x_test)

score_2=accuracy_score(y_test,pred_2)
list_scores.append(score_2)

list_models.append('random forest classifier')
score_2
from sklearn.svm import SVC

svm=SVC()

svm.fit(x_train,y_train)

pred_3=svm.predict(x_test)

score_3=accuracy_score(y_test,pred_3)

list_scores.append(score_3)

list_models.append('support vector machines')
score_3
from sklearn.ensemble import GradientBoostingClassifier

gbr=GradientBoostingClassifier()

gbr.fit(x_train,y_train)

pred_4=gbr.predict(x_test)

score_4=accuracy_score(y_test,pred_4)

list_scores.append(score_4)

list_models.append('gradient boosting classifier')
score_4
plt.figure(figsize=(10,3))

plt.bar(list_models,list_scores,width=0.1)

plt.show()