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
df=pd.read_csv('/kaggle/input/breast-cancer-wisconsin-data/data.csv')

print(df)
df.columns
df.isnull().sum()
df=df.drop(['Unnamed: 32'],axis=1)
df
import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
sns.heatmap(df.corr())

df.corr()
df
from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

for i in list(df.columns):

    if(df[i].dtype=='object'):

        df[i]=le.fit_transform(df[i])
df
y=df['diagnosis']

x=df.drop('diagnosis',axis=1)
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.2)
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score

list_1=[]

for i in range(1,21):

    knn=KNeighborsClassifier(n_neighbors=i)

    knn.fit(x_train,y_train)

    pred_s=knn.predict(x_test)

    scores=accuracy_score(y_test,pred_s)

    list_1.append(scores)  
plt.scatter(range(1,21),list_1)
from sklearn.linear_model import LogisticRegression

lr=LogisticRegression(max_iter=100000)

lr.fit(x_train,y_train)

pred_1=lr.predict(x_test)

score_1=accuracy_score(pred_1,y_test)

score_1
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