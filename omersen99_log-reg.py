import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_auc_score,accuracy_score
from sklearn import linear_model

trainData=pd.read_csv('../input/train.csv')
testData=pd.read_csv('../input/test.csv')
gender_submission=pd.read_csv('../input/gender_submission.csv')
train=trainData.copy()
test=testData.copy()
train.info()
train.describe()
train['Age']=train['Age'].fillna(train["Age"].median())
test['Age']=test['Age'].fillna(test["Age"].median())

train['Embarked']=train['Embarked'].fillna('C')

test['Fare']=test['Fare'].fillna(test["Fare"].median())
train.head()
#female=0 male=1
train.loc[train['Sex']=='female','Sex']=0
train.loc[train['Sex']=='male','Sex']=1
test.loc[test['Sex']=='female','Sex']=0
test.loc[test['Sex']=='male','Sex']=1
#Embarked C=1 Q=2 S=3
train.loc[train['Embarked']=='C','Embarked']=1
train.loc[train['Embarked']=='Q','Embarked']=2
train.loc[train['Embarked']=='S','Embarked']=3
test.loc[test['Embarked']=='C','Embarked']=1
test.loc[test['Embarked']=='Q','Embarked']=2
test.loc[test['Embarked']=='S','Embarked']=3
train_x=train.loc[:,['Pclass','Age','Sex','SibSp','Parch','Fare','Embarked']]
train_y=train['Survived']
test_x=test.loc[:,['Pclass','Age','Sex','SibSp','Parch','Fare','Embarked']]
test_y=gender_submission['Survived']
min_max_scaler = preprocessing.MaxAbsScaler()
train_x_minmax = min_max_scaler.fit_transform(train_x)
test_x_minmax = min_max_scaler.fit_transform(test_x)
# using Logistic regression
from sklearn import linear_model
logis = linear_model.LogisticRegression()
logis.fit(train_x_minmax, train_y)
test_predict=logis.predict(test_x_minmax)
#result
print("Accuracy=",accuracy_score(test_y,test_predict))
print("AUC=",roc_auc_score(test_y, test_predict))


