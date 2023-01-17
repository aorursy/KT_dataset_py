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
train=pd.read_csv('/kaggle/input/udacity-mlcharity-competition/census.csv')
train
test=pd.read_csv('/kaggle/input/udacity-mlcharity-competition/test_census.csv')
test
train.isnull().sum()
test.isnull().sum()
test=test.drop(['Unnamed: 0'],axis=1)
#lets separate out the categorical and numerical features

list_cate=[]

list_num=[]
for i in list(test.columns):

    

    if test[i].dtype=='object':

        

        list_cate.append(i)

    else:

        list_num.append(i)

        
import matplotlib.pyplot as plt

import seaborn as sns

sns.countplot(x='income',hue='workclass',data=train)
#here we can see that private employees earn more than others
sns.countplot(x='income',hue='sex',data=train)
#here the earnings of amle are greater in number than females
sns.countplot(x='income',hue='race',data=train)
#white people have the majority of earnings
sns.countplot(x='income',hue='relationship',data=train)
#husbands i.e male earns more in this section
sns.countplot(x='income',hue='occupation',data=train)
sns.countplot(x='income',hue='education_level',data=train)
sns.countplot(x='income',hue='marital-status',data=train)
# most unmarried people fall in the category of <=50k where as Married people are mostly in category 0f income >=50k
#now fill the missing values in test dataset

from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

for i in list(train.columns):

    if train[i].dtype=='object':

        train[i]=le.fit_transform(train[i])
train
#fill the missing values in test dataset

#for categorical features

for i in list_cate:

    test[i]=test[i].fillna(test[i].mode()[0])



test.isnull().sum()
#filling missing values in numerical features

for i in list_num:

    test[i]=test[i].fillna(test[i].mean())
test.isnull().sum()
test
#similarly preprocess the categorical features of the test set aslo
for i in list(test.columns):

    if test[i].dtype=='object':

        test[i]=le.fit_transform(test[i])
test
y=train['income']

x=train.drop(['income'],axis=1)
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.2)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

lr=LogisticRegression(max_iter=10000)

lr.fit(x_train,y_train)

pred_1=lr.predict(x_test)

score_1=accuracy_score(y_test,pred_1)
score_1
from sklearn.neighbors import KNeighborsClassifier

list_score=[]

for i in range(1,21):

    knn=KNeighborsClassifier(n_neighbors=i)

    knn.fit(x_train,y_train)

    pred_s=knn.predict(x_test)

    scores=accuracy_score(y_test,pred_s)

    list_score.append(scores)
sns.barplot(x=list(range(1,21)),y=list_score)
plt.scatter(range(1,21),list_score)

plt.xlabel('k values')

plt.ylabel('accuracy score')

plt.show()
print(max(list_score))
from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier()

rfc.fit(x_train,y_train)

pred_2=rfc.predict(x_test)

score_2=accuracy_score(y_test,pred_2)
score_2
from sklearn.ensemble import GradientBoostingClassifier

gbc=GradientBoostingClassifier()

gbc.fit(x_train,y_train)

pred_3=gbc.predict(x_test)

score_3=accuracy_score(y_test,pred_3)
score_3
predictions=gbc.predict(test)
sub=pd.read_csv('/kaggle/input/udacity-mlcharity-competition/example_submission.csv')
sub=sub.drop(['income'],axis=1)
sub['income']=predictions
sub.to_csv('submission.csv',index=False)