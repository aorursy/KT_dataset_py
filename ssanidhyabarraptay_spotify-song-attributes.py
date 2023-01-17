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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import seaborn as sb
df=pd.read_csv('../input/spotifyclassification/data.csv')
df.head()
df.info()
train,test=train_test_split(df,test_size=0.25)
corr=df.corr()
plt.figure(figsize=(14,12))
sb.heatmap(corr,annot=True)
plt.show()
pos_temp=df[df['target']==1]['tempo']
neg_temp=df[df['target']==0]['tempo']
plt.title("Song Like/Dislike")
pos_temp.hist(alpha=0.7,bins=30,label='Positive')
neg_temp.hist(alpha=0.7,bins=30,label='Negative')
plt.legend(loc ='upper right')


pos_ene=df[df['target']==1]['energy']
neg_ene=df[df['target']==0]['energy']
fig=plt.figure(figsize=(14,12))
plt.title("Song Like/Dislike")
pos_ene.hist(alpha=0.7,bins=30,label='Positive')
neg_ene.hist(alpha=0.7,bins=30,label='Negative')
plt.legend(loc ='upper right')
pos_dance=df[df['target']==1]['danceability']
neg_dance=df[df['target']==0]['danceability']
fig=plt.figure(figsize=(12,10))
plt.title("Song Like/Dislike")
pos_dance.hist(alpha=0.7,bins=30,label='Positive')
neg_dance.hist(alpha=0.7,bins=30,label='Negative')
plt.legend(loc ='upper right')
pos_dance=df[df['target']==1]['danceability']
neg_dance=df[df['target']==0]['danceability']
fig=plt.figure(figsize=(12,10))
plt.title("Song Like/Dislike")
pos_dance.hist(alpha=0.7,bins=30,label='Positive')
neg_dance.hist(alpha=0.7,bins=30,label='Negative')
plt.legend(loc ='upper right')
features=df.select_dtypes(exclude='object').columns

c=DecisionTreeClassifier(min_samples_split=2)
X_train=train[features]
y_train=train['target']
X_test=test[features]
y_test=test['target']
dt=c.fit(X_train,y_train)
dt=c.score(X_train,y_train)
c.score(X_train,y_train)
y_pred=c.predict(X_test)
y_pred
from sklearn.metrics import accuracy_score
score=accuracy_score(y_test,y_pred)*100
score
