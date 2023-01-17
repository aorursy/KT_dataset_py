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
df=pd.read_csv('/kaggle/input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')
df
df.info()
df.isnull().sum()
import matplotlib.pyplot as plt

import seaborn as sns
sns.countplot(x=df['DEATH_EVENT'],hue='anaemia',data=df)
#here we can say that people without anaemia has less chances of death
sns.countplot(x=df['DEATH_EVENT'],hue='diabetes',data=df)
sns.countplot(x=df['DEATH_EVENT'],hue='high_blood_pressure',data=df)
sns.countplot(x=df['DEATH_EVENT'],hue='sex',data=df)
sns.countplot(x=df['DEATH_EVENT'],hue='smoking',data=df)
sns.lineplot(x=df['age'],y=df['platelets'],data=df)
sns.lineplot(x=df['age'],y=df['serum_creatinine'],data=df)
y=df['DEATH_EVENT']

x=df.drop(['DEATH_EVENT'],axis=1)
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.2)

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

lr=LogisticRegression(max_iter=10000)

lr.fit(x_train,y_train)

pred_1=lr.predict(x_test)

score_1=accuracy_score(y_test,pred_1)
list_models=['LinearRegression']

list_score=[]

list_score.append(score_1)
list_score
from sklearn.neighbors import KNeighborsClassifier

list_1=[]

for i in range(1,21):

    knn=KNeighborsClassifier(n_neighbors=i)

    knn.fit(x_train,y_train)

    pred_s=knn.predict(x_test)

    scores=accuracy_score(y_test,pred_s)

    list_1.append(scores)

    
sns.barplot(x=list(range(1,21)),y=list_1)
plt.scatter(range(1,21),list_1)

plt.xlabel('k values')

plt.ylabel('accuracy scores')

plt.show()
score_2=max(list_1)

list_score.append(score_2)

list_models.append('Knearest neighbors')
from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier()

rfc.fit(x_train,y_train)

pred_2=rfc.predict(x_test)

score_3=accuracy_score(y_test,pred_2)
score_2
list_score.append(score_3)

list_models.append('random forest classifier')
list_score
from sklearn.ensemble import GradientBoostingClassifier

gbc=GradientBoostingClassifier()

gbc.fit(x_train,y_train)

pred_4=gbc.predict(x_test)

score_4=accuracy_score(y_test,pred_4)
score_4
list_score.append(score_4)

list_models.append('gradient boosting classifier')
from sklearn.svm import SVC

svm=SVC()

svm.fit(x_train,y_train)

pred_5=svm.predict(x_test)

score_5=accuracy_score(y_test,pred_5)
score_5
list_score.append(score_5)

list_models.append('support vector machines')
list_score,list_models
sns.set_style('darkgrid')

sns.barplot(x=list_models,y=list_score)
plt.figure(figsize=(12,4))

plt.bar(list_models,list_score,width=0.3,align='edge')

plt.xlabel('models')

plt.ylabel('accuracy scores')

plt.show()
# from all the above models randomforest classifier gives the best accuracy score