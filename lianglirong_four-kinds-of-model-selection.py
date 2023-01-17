import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
import os
print(os.listdir("../input"))
names = ['preg','plas','pres','skin','test','mass','pedi','age','class']
data = pd.read_csv("../input/pima_data.csv")
array = data.values
X = array[:,0:8]
Y = array[:,8]
test_size = 0.33
speed = 4
#train_test_split accurate
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=test_size,random_state=speed)
model = LogisticRegression()
model.fit(X_train,Y_train)
result = model.score(X_test,Y_test)
print("accurate:",result)
#KFold accurate
kfold = KFold(n_splits=10,random_state=speed)
model = LogisticRegression()
result = cross_val_score(model,X,Y,cv=kfold)
print(result.mean(),result.std())
#LeaveOneOut accurate
loo = LeaveOneOut()
model = LogisticRegression()
result = cross_val_score(model,X,Y,cv=loo)
print(result.mean(),result.std())
kShuffe = ShuffleSplit(n_splits=10,random_state=speed)
model = LogisticRegression()
result = cross_val_score(model,X,Y,cv=kShuffe)
print(result.mean(),result.std())
