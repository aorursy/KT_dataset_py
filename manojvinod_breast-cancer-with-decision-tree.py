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

df= pd.read_csv("../input/breastCancer.csv")
df.info()

df.drop(['Unnamed: 32'],axis=1)
df['diagnosis']=df.diagnosis.map({'B':0, 'M':1})
x=df.iloc[:,2:32]

y=df[['diagnosis']]
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=0)
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(criterion='entropy',max_depth=1)

model.fit(x_train,y_train)
print("The test score is ",model.score(x_test,y_test))

print("The train score is ",model.score(x_train,y_train))
from sklearn.model_selection import cross_val_score,KFold

kfold = KFold(n_splits=10, random_state=10) 

score = cross_val_score(model, x, y, cv=kfold, scoring='accuracy').mean()

score
model1 = DecisionTreeClassifier(criterion="entropy",max_depth=None)

model1.fit(x_train,y_train)
print("The test score is ",model1.score(x_test,y_test))

print("The train score is ",model1.score(x_train,y_train))
from sklearn.model_selection import cross_val_score,KFold

kfold = KFold(n_splits=10, random_state=10) 

score = cross_val_score(model1, x, y, cv=kfold, scoring='accuracy').mean()

score
model2= DecisionTreeClassifier(criterion="entropy",max_depth=None,min_samples_leaf=200)

model2.fit(x_train,y_train)
print("The test score is ",model2.score(x_test,y_test))

print("The train score is ",model2.score(x_train,y_train))
from sklearn.model_selection import cross_val_score,KFold

kfold = KFold(n_splits=10, random_state=10) 

score = cross_val_score(model2, x, y, cv=kfold, scoring='accuracy').mean()

score
train_score=[]

for leaf in range(1,len(df)):

    model3 = DecisionTreeClassifier(criterion="entropy",max_depth=None,min_samples_leaf=leaf)

    model3.fit(x_train,y_train)

    print("The depth is ",leaf)

    print(model.score(x_test,y_test))

    train_score.append(model.score(x_test,y_test))
kfold_score=[]

for leaf in range(1,len(df)):

    model3 = DecisionTreeClassifier(criterion="entropy",max_depth=None,min_samples_leaf=leaf)

    from sklearn.model_selection import cross_val_score,KFold

    kfold = KFold(n_splits=10, random_state=10)

    score = cross_val_score(model3, x, y, cv=kfold, scoring='accuracy').mean()

    print("The depth is ",leaf)

    print(score)

    kfold_score.append(score)
import matplotlib.pyplot as plt

plt.scatter(range(1,len(df)),kfold_score)
import matplotlib.pyplot as plt

plt.scatter(range(1,len(df)),train_score)