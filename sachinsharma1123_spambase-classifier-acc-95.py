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
df=pd.read_csv('/kaggle/input/spambase/realspambase.data')
df
df.info()
import seaborn as sns

sns.heatmap(df.isnull())
sns.heatmap(df.corr())
dict_1={}



dict_1=dict(df.corr()['1'])
list_features=[]

for key,values in dict_1.items():

    if abs(values)<0.2:

        list_features.append(key)
#lets drop the features which have coorelation less than <0.2

df=df.drop(list_features,axis=1)
df
y=df['1']

x=df.drop(['1'],axis=1)
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

list_models.append('logisticRegression')

score_1
from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier()

rfc.fit(x_train,y_train)

pred_2=rfc.predict(x_test)

score_2=accuracy_score(y_test,pred_2)

list_scores.append(score_2)

list_models.append('random forest')
score_2
from xgboost import XGBClassifier

xgb=XGBClassifier()

xgb.fit(x_train,y_train)

pred_3=xgb.predict(x_test)

score_3=accuracy_score(y_test,pred_3)

list_scores.append(score_3)

list_models.append('xgb')
score_3
from sklearn.neighbors import KNeighborsClassifier

list_1=[]

for i in range(1,21):

    knn=KNeighborsClassifier(n_neighbors=i)

    knn.fit(x_train,y_train)

    preds=knn.predict(x_test)

    scores=accuracy_score(y_test,preds)

    list_1.append(scores)

sns.lineplot(x=list(range(1,21)),y=list_1)

list_scores.append(max(list_1))

list_models.append('kneighbors')
from sklearn.svm import SVC

svm=SVC()

svm.fit(x_train,y_train)

pred_4=svm.predict(x_test)

score_4=accuracy_score(y_test,pred_4)

list_scores.append(score_4)

list_models.append('svm')
score_4
import matplotlib.pyplot as plt

plt.figure(figsize=(12,4))

plt.bar(list_models,list_scores,width=0.3)

plt.xlabel('classifiers')

plt.ylabel('accuracy scores')