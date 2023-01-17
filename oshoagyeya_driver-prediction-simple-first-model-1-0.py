# Ignore warnings

import warnings

warnings.filterwarnings('ignore')



# Handle table-like data and matrices

import numpy as np

import pandas as pd



# Modelling Algorithms

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split



# Visualisation

%matplotlib inline

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

import seaborn as sns
#getting data as a data frame

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
#train.info()
#train.head()
#train.describe()
#test.info()
#test.head()
#test.describe()
corr=train.corr()

#corr
sns.heatmap(corr)
corrt=corr[["target"]].drop("target")

corrt=corrt.reset_index().rename(columns={'index': 'x', 'target': 'y'})

corrt=corrt.sort_values("y", ascending=False)

#corrt
sns.barplot(x="y", y="x", data=corrt)
corrf=corrt[corrt.y>0]

#corrf
lcn=corrf['x'].tolist()

testf=test[lcn]

lcn.append("target")

trainf=train[lcn]
mc=trainf.mode().transpose().reset_index().rename(columns={'index': 'colu', 0: 'modeofc'})

for index,row in mc.iterrows():

    if(row['modeofc']==-1):

        trainf=trainf.drop(row['colu'],axis=1)

        testf=testf.drop(row['colu'],axis=1)

mc=trainf.mode().transpose().reset_index().rename(columns={'index': 'colu', 0: 'modeofc'})

mc2=testf.mode().transpose().reset_index().rename(columns={'index': 'colu', 0: 'modeofc'})
for index,row in mc.iterrows():

    c=row['colu']

    val=row['modeofc']

    if(trainf.dtypes[c]==np.int64):

        val=np.int64(val)

    trainf.loc[trainf[c] == -1, c] = val

for index,row in mc2.iterrows():

    c=row['colu']

    val=row['modeofc']

    if(testf.dtypes[c]==np.int64):

        val=np.int64(val)

    testf.loc[testf[c] == -1, c] = val
x=trainf.drop("target",axis=1)

y=trainf["target"]

X_test=pd.DataFrame(testf)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=42)
clf=LogisticRegression()

clf.fit(x_train,y_train)

Y_pred=clf.predict(X_test)

acc = round(clf.score(x_test, y_test) * 100, 2)

acc
submission = pd.DataFrame({"id": test["id"],"target": Y_pred})

submission.to_csv('mysubmission.csv', index=False)