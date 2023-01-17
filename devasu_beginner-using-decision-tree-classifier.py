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
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
train = pd.read_csv('../input/train.csv')
train.head()
y = train['Survived'].copy()
features = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']  #All the features we might need in the model
X = train[features].copy()
X = pd.get_dummies(X) #one hot code encoding
X.head()
first_imputer = SimpleImputer(missing_values=np.nan,strategy='mean')
X = pd.DataFrame(first_imputer.fit_transform(X))
Xtrain, Xtest, ytrain, ytest = train_test_split(X,y,random_state=1,test_size=0.33)
def AccuracyTracker(Xtrain,Xtest,ytrain,ytest,n):
    model = DecisionTreeClassifier(max_leaf_nodes=n,random_state=1)
    model.fit(Xtrain,ytrain)
    print(n,accuracy_score(ytest,model.predict(Xtest)))
for i in range(2,50):
    AccuracyTracker(Xtrain,Xtest,ytrain,ytest,i)
model = RandomForestClassifier(n_estimators=100)
model.fit(Xtrain,ytrain)
accuracy_score(ytest,model.predict(Xtest))

modeltree = DecisionTreeClassifier(max_leaf_nodes=12,random_state=1)  #hollow tree created now let's put the data
modeltree.fit(X,y)
test = pd.read_csv("../input/test.csv")
pretest = test[features].copy()
pretest = pd.get_dummies(pretest)
imputedpretest = pd.DataFrame(first_imputer.fit_transform(pretest))
imputedpretest
res = modeltree.predict(imputedpretest)
ansdic = {'PassengerId':test['PassengerId'],'Survived':res}
ans = pd.DataFrame(ansdic)
ans.head()
ans.to_csv("answer.csv",index=False)

