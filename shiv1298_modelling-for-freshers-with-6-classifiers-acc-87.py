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
df=pd.read_csv('/kaggle/input/churn-modelling/Churn_Modelling.csv')
df
df.info()
import seaborn as sns

import matplotlib.pyplot as plt

#some visualization to get some insight info from thet data
sns.countplot(df['Geography'])
#most people belongs to france in this dataset
sns.countplot(x=df['Geography'],hue='Gender',data=df)
#here the no of males is greater than females
sns.countplot(x=df['Exited'],hue='Geography',data=df)
sns.countplot(x=df['Exited'],hue='Gender',data=df)
sns.countplot(x=df['HasCrCard'],hue='Geography',data=df)
#more people in france have credit card as comparison to others
sns.countplot(x=df['HasCrCard'],hue='Gender',data=df)
#more males carry credit card as compare to females
sns.lineplot(x=df['CreditScore'],y=df['EstimatedSalary'],data=df)
sns.distplot(df['EstimatedSalary'])
#next step is model building

#drop the unnecessary columns'

df=df.drop(['RowNumber','CustomerId','Surname'],axis=1)
from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

for i in list(df.columns):

    if df[i].dtype=='object':

        df[i]=le.fit_transform(df[i])
y=df['Exited']

x=df.drop(['Exited'],axis=1)
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.2)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

list_scores=[]

lr=LogisticRegression(max_iter=10000)

lr.fit(x_train,y_train)

pred_1=lr.predict(x_test)

score_1=accuracy_score(y_test,pred_1)

list_scores.append(score_1)
score_1
from sklearn.ensemble import RandomForestClassifier

rfr=RandomForestClassifier()

rfr.fit(x_train,y_train)

pred_2=rfr.predict(x_test)

score_2=accuracy_score(y_test,pred_2)

list_scores.append(score_2)
score_2
from sklearn.neighbors import KNeighborsClassifier

list_1=[]

for i in range(1,21):

    knn=KNeighborsClassifier(n_neighbors=i)

    knn.fit(x_train,y_train)

    preds=knn.predict(x_test)

    scores=accuracy_score(y_test,preds)

    list_1.append(scores)
list_scores.append(max(list_1))

sns.set_style('whitegrid')

sns.lineplot(x=list(range(1,21)),y=list_1)
print(max(list_1))
from sklearn.ensemble import GradientBoostingClassifier

gbc=GradientBoostingClassifier()

gbc.fit(x_train,y_train)

pred_3=gbc.predict(x_test)

score_3=accuracy_score(y_test,pred_3)

list_scores.append(score_3)
score_3
from sklearn.svm import SVC

svm=SVC()

svm.fit(x_train,y_train)

pred_4=svm.predict(x_test)

score_4=accuracy_score(y_test,pred_4)

list_scores.append(score_4)
score_4
from xgboost import XGBClassifier

xgb=XGBClassifier()

xgb.fit(x_train,y_train)

pred_5=xgb.predict(x_test)

score_5=accuracy_score(y_test,pred_5)

list_scores.append(score_5)
score_5
list_models=['logistic regression','randomforest classifier','kneighbors classifier','gradientboosting','svm','xgboost']

list_scores

plt.figure(figsize=(20,5))

plt.bar(list_models,list_scores,width=0.7)

plt.xlabel('classifiers')

plt.ylabel('accuracy scores')

plt.show()