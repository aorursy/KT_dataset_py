import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import linear_model
df = pd.read_csv('train.csv')

test = pd.read_csv('test.csv')

df.head()

df.isnull().sum()
df.drop(['id'],axis=1)


from sklearn.model_selection import train_test_split

y = df['class']

X = df.drop(['class'],axis=1)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=42) 

# from sklearn.ensemble import ExtraTreesClassifier

# et = ExtraTreesClassifier()

from sklearn.ensemble import RandomForestClassifier

et = RandomForestClassifier()

et.fit(X,y)
ids = test['id']

test.drop(['id'],axis=1)
u=et.predict(test)
print(u)
finan =[]

for i in range(len(u)):

  finan.append([ids[i],u[i]])
print(finan)
print(len(u))
finans = pd.DataFrame(data=finan,columns=['id','class'])
finans.to_csv('submission.csv',index=False)