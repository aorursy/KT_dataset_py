import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np 

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

print(train.shape)
print(test.shape)

train = train.drop('Name',axis=1)
train = train.drop('Ticket',axis=1)

train.head()


train.loc[train['Sex'] == 'male',"Sex"] = 0
train.loc[train['Sex'] == 'female',"Sex"] = 1
train['Age'] = train.Age.fillna( train.Age.mean() )
train['Cabin'] = train.Cabin.fillna('U')
train['Cabin'] = train['Cabin'].map( lambda c : c[0] )
cabin = pd.get_dummies(train['Cabin'],prefix='C')
embark = pd.get_dummies(train['Embarked'],prefix='E')
embark
full = pd.concat([train,cabin,embark],axis=1)
target = full.Survived
full = full.drop('Survived',axis=1)
full = full.drop('Cabin',axis=1)
full = full.drop('Embarked',axis=1)

from sklearn.model_selection import train_test_split
data_train, data_test, target_train, target_test = train_test_split(full, target, random_state=0)
from sklearn.svm import SVC
svc = SVC()
svc.fit(data_train, target_train)
print('Train score: {:.3f}'.format(svc.score(data_train, target_train)))
print('Test score: {:.3f}'.format(svc.score(data_test, target_test)))
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(data_train, target_train)
print('Train score: {:.3f}'.format(rfc.score(data_train, target_train)))
print('Test score: {:.3f}'.format(rfc.score(data_test, target_test)))
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
dtc = DecisionTreeClassifier()
dtc.fit(data_train, target_train)
print('Train score: {:.3f}'.format(dtc.score(data_train, target_train)))
print('Test score: {:.3f}'.format(dtc.score(data_test, target_test)))
lgr = LogisticRegression()
lgr.fit(data_train, target_train)
print('Train score: {:.3f}'.format(lgr.score(data_train, target_train)))
print('Test score: {:.3f}'.format(lgr.score(data_test, target_test)))
gbc = GradientBoostingClassifier()
gbc.fit(data_train, target_train)
print('Train score: {:.3f}'.format(gbc.score(data_train, target_train)))
print('Test score: {:.3f}'.format(gbc.score(data_test, target_test)))
svc2 = SVC(kernel='sigmoid')
svc2.fit(data_train, target_train)
print('Train score: {:.3f}'.format(svc2.score(data_train, target_train)))
print('Test score: {:.3f}'.format(svc2.score(data_test, target_test)))
