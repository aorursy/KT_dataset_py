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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
df=pd.read_csv("../input/train.csv")
dft=pd.read_csv("../input/test.csv")
df.tail()
dft.head()
df.describe()
df.Embarked.value_counts()
df[df.Embarked=='Q'][df.Survived==1].shape[0]

df[df.Sex=='female'][df.Survived==1].shape[0]
df['gender']=df['Sex'].map({'male':1,'female':0}).astype(int)
df.Embarked.fillna('S',inplace=True)
dft['gender']=dft['Sex'].map({'male':1,'female':0}).astype(int)
dft.Embarked.fillna('S',inplace=True)
df['emb']=df['Embarked'].map({'S':1,'C':2,'Q':3}).astype(int)
del df['Sex']
del df['Embarked']
df.rename(columns={'gender':'Sex'},inplace=True)
df.rename(columns={'emb':'Embarked'},inplace=True)
dft['emb']=dft['Embarked'].map({'S':1,'C':2,'Q':3}).astype(int)
del dft['Sex']
del dft['Embarked']
dft.rename(columns={'gender':'Sex'},inplace=True)
dft.rename(columns={'emb':'Embarked'},inplace=True)
df.isna().sum()
del df['Name']
del df['Ticket']
del dft['Name']
del dft['Ticket']
df.head()
df.isna().sum()
df[df.Cabin.isna()][df.Survived==1].shape[0]
dft.isna().sum()
def con(str):
    if str=='':
        return 0
    else:
        return 1
df['cabin']=df['Cabin'].apply(con)
del df['Cabin']
dft['cabin']=dft['Cabin'].apply(con)
del dft['Cabin']
df.rename(columns={'cabin':'Cabin'},inplace=True)
dft.rename(columns={'cabin':'Cabin'},inplace=True)
del df['SibSp']
del df['Parch']
del dft['SibSp']
del dft['Parch']
dft.isna().sum()
meanS=df[df.Survived==1].Age.mean()
df["Age"]=np.where(df.Age.isna() & df.Survived==1 , meanS,df["Age"])
df.isna().sum()
meanSt=dft.Age.mean()
dft["Age"]=np.where(dft.Age.isna()  , meanSt,dft["Age"])
dft.isna().sum()
meanNS=df[df.Survived==0].Age.mean()
df["Age"].fillna(meanNS,inplace=True)
dft.isna().sum()
del df['Fare']
del dft['Fare']
dft.head()
x_train=np.array(df.iloc[:,2:9])
y_train=np.array(df.iloc[:,1])
x_test=np.array(dft.iloc[:,1:8])
clf=LogisticRegression(C=0.3,max_iter=1000000)
#clf=SVC()
#clf = RandomForestClassifier(n_estimators=100)
#clf=KNeighborsClassifier()
clf.fit(x_train,y_train)
clf.score(x_train,y_train)
y_pred=clf.predict(x_test)
f=np.c_[np.array(dft.PassengerId),y_pred]
d=pd.DataFrame(f)
d.to_csv('pred.csv',index=False,header=['PassengerId','Survived'])
