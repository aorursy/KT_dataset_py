import numpy as np 
import pandas as pd 
import seaborn as sns
import os
print(os.listdir("../input"))
Train = pd.read_csv('../input/train.csv',index_col=0)
train_y = Train.Survived
pd.concat([Train.dtypes,(Train.isna().sum()),Train.nunique()],1)
from sklearn import impute
from sklearn.ensemble import RandomForestRegressor

train_x = Train.drop(['Survived','Name','Ticket'],1)
z = impute.SimpleImputer(strategy='most_frequent')
z.fit(train_x[train_x.Embarked.notnull()].values)
train_x[train_x.Embarked.isnull()] = z.transform(train_x[train_x.Embarked.isnull()].values)

train_x.Cabin = train_x.Cabin.str.slice(0,1)
for name in 'Pclass Sex Embarked'.split(' '):
    train_x = train_x.join(pd.get_dummies(train_x[name],prefix=name).iloc[:,:-1]).drop(name,1)
    
forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_x[train_x.Age.notnull()].drop(['Age','Cabin'],1), train_x[train_x.Age.notnull()].Age)
melb_preds = forest_model.predict(train_x[train_x.Age.isnull()].drop(['Age','Cabin'],1))
train_x.loc[train_x.Age.isnull(),'Age'] = melb_preds

# from sklearn import svm
# clf = svm.SVC()
# clf.fit(train_x[train_x.Cabin.notnull()].drop('Cabin',1), train_x[train_x.Cabin.notnull()].Cabin)
# train_x.loc[train_x.Cabin.isnull(),'Cabin'] = clf.predict(train_x[train_x.Cabin.isnull()].
#                                                           drop(['Cabin'],1))
# name = 'Cabin'
# train_x = train_x.join(pd.get_dummies(train_x[name],prefix=name).iloc[:,:-1]).drop(name,1)
train_x.drop('Cabin',1,inplace=True)
test_x = pd.read_csv('../input/test.csv',index_col=0).drop(['Name','Ticket'],1)
test_x.Fare = test_x.Fare.fillna(test_x.Fare.mean())

for name in 'Pclass Sex Embarked'.split(' '):
    test_x = test_x.join(pd.get_dummies(test_x[name],prefix=name).iloc[:,:-1]).drop(name,1)
test_x.loc[test_x.Age.isnull(),'Age'] = forest_model.predict(test_x[test_x.Age.isnull()]
                                                       .drop(['Age','Cabin'],1))
test_x.drop('Cabin',1,inplace=True)
from sklearn import svm
clf = svm.SVC()
clf.fit(train_x,train_y)
preds = clf.predict(test_x)
pre = pd.DataFrame(preds,index=test_x.index,columns=['Survived'])
pre.to_csv('pre.csv')
from sklearn import svm
clf = svm.SVC()
clf.fit(train_x.drop('Age',1),train_y)
preds = clf.predict(test_x.drop('Age',1))
pre = pd.DataFrame(preds,index=test_x.index,columns=['Survived'])
pre.to_csv('pre1.csv')