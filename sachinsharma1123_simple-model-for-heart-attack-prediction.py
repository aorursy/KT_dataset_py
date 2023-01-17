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
df=pd.read_csv('/kaggle/input/health-care-data-set-on-heart-attack-possibility/heart.csv')
df
df.info()
df.isnull().sum()
df['cp'].unique()
df['fbs'].unique()
df['ca'].unique()
df.corr()
import seaborn as sns

import matplotlib.pyplot as plt

sns.heatmap(df.corr())
list_1=list(df.columns)
list_2=[]

list_3=[]

for i in list_1:

    x=abs(df['target'].corr(df[i]))

    if x>0.2:

        

        list_2.append(x)

        list_3.append(i)
list_2,list_3
df[df['target']==1]['age'].sort_values()
#we can say that people above the age group of 29 has more chances of heart attack
sns.countplot(x='sex',hue='target',data=df)
#mostly women got heart attacks than mens
sns.countplot(x='fbs',hue='target',data=df)
sns.countplot(x='restecg',hue='target',data=df)
sns.countplot(x='exang',hue='target',data=df)
df
#we have taken the features having correlation greater than 0.2

x=df[['age',

  'sex',

  'cp',

  'thalach',

  'exang',

  'oldpeak',

  'slope',

  'ca',

  'thal']]
y=df['target']
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.2)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

list_1=[]

for i in range(1,11):

    knn=KNeighborsClassifier(n_neighbors=i)

    knn.fit(x_train,y_train)

    pred_y=knn.predict(x_test)

    score_1=accuracy_score(y_test,pred_y)

    list_1.append(score_1)
plt.plot(range(1,11),list_1)

plt.show()
print(max(list_1))
from sklearn.linear_model import LogisticRegression

lr=LogisticRegression(max_iter=500)

lr.fit(x_train,y_train)

pred_1=lr.predict(x_test)

score_2=accuracy_score(y_test,pred_1)
score_2
from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier()
from sklearn.model_selection import GridSearchCV

params={'n_estimators':[100,200,300,400,500,600,700],

        'max_features':['auto','sqrt','log2']}
clf=GridSearchCV(rfc,params,cv=5,verbose=0)

clf.fit(x_train,y_train)
print(clf.best_params_)
rfc=RandomForestClassifier(max_features='log2',n_estimators=300)

rfc.fit(x_train,y_train)

pred_4=rfc.predict(x_test)

score_4=accuracy_score(y_test,pred_4)
score_4
from sklearn import metrics

print(metrics.confusion_matrix(y_test,pred_4))
from sklearn.svm import SVC

clf_2=SVC()
params_2={'C':[0.001,0.01,0.1,1,10],

         'gamma':[0.001,0.01,0.1,1]}

clf_3=GridSearchCV(clf_2,params_2,cv=5,verbose=0)

clf_3.fit(x_train,y_train)
print(clf_3.best_params_)
clf_2=SVC(C=10,gamma=0.001)

clf_2.fit(x_train,y_train)

pred_5=clf_2.predict(x_test)

score_5=accuracy_score(y_test,pred_5)
score_5
#so out of all classifiers random forest gives the best accuracy score
new_df=pd.DataFrame({'actual':y_test,

                   'predicted':pred_4})
new_df