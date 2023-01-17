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
df=pd.read_csv("../input/titanic/train.csv")

df
Null=[]

for i in df:

    Null.append((i,df[i].isna().mean()*100))

Null=pd.DataFrame(Null,columns=['class','per'])

Null
df=df.drop(["Cabin","Name","Ticket"],axis=1)

df.dtypes
df=df.fillna(df.mode)

df.isna().sum()
df.Sex.unique()
df.Sex=df.Sex.replace("male",1)

df.Sex=df.Sex.replace("female",2)

df.Sex.unique()
df.Embarked = df.Embarked.apply(str)

k=df.Embarked.unique()

k[-1]
df.Embarked=df.Embarked.replace('S',1)

df.Embarked=df.Embarked.replace('C',2)

df.Embarked=df.Embarked.replace('Q',3)

df.Embarked=df.Embarked.replace(k[-1],1)
df.Age=df.Age.apply(str)

k=df.Age.unique()

df.Age=df.Age.replace(k[4],"28.0")

df.Age=df.Age.apply(float)
import seaborn as sns

import matplotlib.pyplot as plt

plt.figure(figsize=(30,20))

sns.heatmap(df.corr(),annot = True,cmap="jet")
cols = [col for col in df.columns if col not in ["Survived"]]

X = df[cols]

y = df["Survived"]
l=list(X)
from sklearn.linear_model import LogisticRegression as LR

model = LR()

model.fit(X,y)

model.score(X,y)
df=pd.read_csv("../input/titanic/test.csv")

l

df=df[l]
df.Sex=df.Sex.replace("male",1)

df.Sex=df.Sex.replace("female",2)

df.Sex.unique()
df.Embarked = df.Embarked.apply(str)

k=df.Embarked.unique()
df.Embarked=df.Embarked.replace('S',1)

df.Embarked=df.Embarked.replace('C',2)

df.Embarked=df.Embarked.replace('Q',3)
df=df.fillna(df.mode)

df.isna().sum()
df.dtypes
df.Fare=df.Fare.apply(str)

df.Fare.unique()
s='<bound method DataFrame.mode of      PassengerId  Pclass  Sex   Age  SibSp  Parch      Fare  Embarked\n0            892       3    1  34.5      0      0    7.8292         3\n1            893       3    2  47.0      1      0    7.0000         1\n2            894       2    1  62.0      0      0    9.6875         3\n3            895       3    1  27.0      0      0    8.6625         1\n4            896       3    2  22.0      1      1   12.2875         1\n..           ...     ...  ...   ...    ...    ...       ...       ...\n413         1305       3    1   NaN      0      0    8.0500         1\n414         1306       1    2  39.0      0      0  108.9000         2\n415         1307       3    1  38.5      0      0    7.2500         1\n416         1308       3    1   NaN      0      0    8.0500         1\n417         1309       3    1   NaN      1      1   22.3583         2\n\n[418 rows x 8 columns]>'
df.Fare=df.Fare.replace(s,'7.75')
df.Fare=df.Fare.apply(float)
df.Age=df.Age.apply(str)

k=df.Age.unique()[10]

df.Age=df.Age.replace(k,"34.5")

df.Age=df.Age.apply(float)
pred=model.predict(df)
Id=df.PassengerId.values

l=[]

for i ,j in zip(Id,pred):

    l.append([i,j])
sub=pd.DataFrame(l,columns=["PassengerId","Survived"])
sub
filename = 'Titanic Predictions 1.csv'



sub.to_csv(filename,index=False)



print('Saved file: ' + filename)