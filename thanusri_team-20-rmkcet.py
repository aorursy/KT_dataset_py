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
import pandas as pd

import seaborn as sns

import numpy as np

import matplotlib.pyplot as plt
df=pd.read_csv('../input/heart.csv')
df.info()
df.describe()
df.head(10)
plt.figure(figsize=[12,12])

sns.countplot(x="age", data=df)
sns.countplot(x='cp', data=df)
sns.countplot(x='thal',data=df)
sns.countplot(x='sex',data=df)
sns.countplot(x='target',data=df)
min_age=min(df.age)

max_age=max(df.age)

mean_age=df.age.mean()

print("MINIMUM AGE:",min_age)

print("MAXIMUM AGE:",max_age)

print("MEAN AGE:",mean_age)
age29_38 = df[(df.age>=29)&(df.age<39)]

print("age29_38:",len(age29_38))
age39_48 = df[(df.age>=39)&(df.age<49)]

print("age39_48:",len(age39_48))
age49_58 = df[(df.age>=49)&(df.age<59)]

age59_68 = df[(df.age>=59)&(df.age<69)]

age69_77 = df[(df.age>=69)&(df.age<78)]

print("age49_58:",len(age49_58))

print("age59_68:",len(age59_68))

print("age69_77:",len(age69_77))
sns.scatterplot(x='sex',y='age',data=df,hue='target')
sns.scatterplot(x='chol',y='sex',data=df,hue='target')
df.head()
from sklearn.model_selection import train_test_split
x=df.drop('target',axis=1).values
y=df['target'].values
x
y
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25)
x_train.shape
x_test.shape
y_train.shape
y_test.shape
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(x_train,y_train)
y_predict_test=classifier.predict(x_test)
y_predict_test
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_predict_test)
sns.heatmap(cm,annot=True,fmt='d')
from sklearn.metrics import classification_report
print(classification_report(y_test,y_predict_test))
64/76