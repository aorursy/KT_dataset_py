import sklearn

import numpy as np

import pandas as pd

from sklearn import tree
df=pd.read_csv("haberman.csv",header=None)
df.columns=["Age","Year","Aux","Survival"]
df.to_csv("h2.csv",index=True,inplace=True)
test_ind=[7,8,178,300]
test_data=pd.DataFrame(index=test_ind,columns=["Age","Year","Aux","Survival"])

print(test_data)
test_data.ix[test_ind]=df.ix[test_ind]
print(test_data)
test_data.drop("Survival",axis=1,inplace=True)
df.drop(df.index[test_ind])
labels=df["Survival"].values
print(labels)
##1 : Survived for 5 or more than 5 years.

##2 : died earlier
names=["Age","Year","Aux"]
features=df[names].values
print(features)
clf=tree.DecisionTreeClassifier()
clf.fit(features,labels)
clf.predict(test_data)
clf.predict([[64,87,2]])