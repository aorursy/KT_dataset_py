# pandas
import pandas as pd
from pandas import Series,DataFrame

# numpy, matplotlib, seaborn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
%matplotlib inline

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import sklearn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
main_file_path = '../input/train.csv' # this is the path to the Iowa data that you will use
data = pd.read_csv(main_file_path)
test  = pd.read_csv("../input/test.csv")
#drop unncecessary columns
data = data.drop(['PassengerId','Name','Ticket','Embarked','Cabin'], axis=1)
test = test.drop(['Name','Ticket','Embarked','Cabin'], axis=1)
#Fare
test['Fare'].fillna(test['Fare'].median(),inplace=True)


data.loc[ data['Fare'] <= 7.91, 'Fare'] = 0
data.loc[ (data['Fare'] > 7.91 ) & (data['Fare'] <= 14.454) , 'Fare'] = 1
data.loc[(data['Fare'] > 14.454) & (data['Fare'] <= 31), 'Fare'] = 2
data.loc[ data['Fare'] > 31, 'Fare'] = 3

test.loc[ test['Fare'] <= 7.91, 'Fare'] = 0
test.loc[ (test['Fare'] > 7.91) & (test['Fare'] <= 14.454) , 'Fare'] = 1
test.loc[(test['Fare'] > 14.454) & (test['Fare'] <=31), 'Fare'] = 2
test.loc[ test['Fare'] > 31, 'Fare'] = 3

#float to int
data['Fare'] = data['Fare'].astype(int)
test['Fare'] = test['Fare'].astype(int)
#Age
#fill missing values
data['Age'] = data.groupby(['Pclass'])['Age'].transform(lambda x: x.fillna(x.mean()))
test['Age'] = test.groupby(['Pclass'])['Age'].transform(lambda x: x.fillna(x.mean()))
data['Age'] = data['Age'].astype(int)
test['Age'] = test['Age'].astype(int)

data.loc[ data['Age'] <= 16 , 'Age'] = 0
data.loc[(data['Age'] > 16) & (data['Age'] <= 32), 'Age'] = 1
data.loc[(data['Age'] > 32) & (data['Age'] <= 48), 'Age'] = 2
data.loc[(data['Age'] > 48) & (data['Age'] <= 64), 'Age'] = 3
data.loc[data['Age'] > 64 , 'Age'] = 4

test.loc[ test['Age'] <= 16 , 'Age'] = 0
test.loc[(test['Age'] > 16) & (test['Age'] <= 32), 'Age'] = 1
test.loc[(test['Age'] > 32) & (test['Age'] <= 48), 'Age'] = 2
test.loc[(test['Age'] > 48) & (test['Age'] <= 64), 'Age'] = 3
test.loc[test['Age'] > 64 , 'Age'] = 4
# New feature Family from Parch and SibSp
data['Family'] = data['Parch'] + data['SibSp']

data['Family'].loc[data['Family'] > 0] = 1
data['Family'].loc[data['Family'] == 0] = 0

test['Family'] = test['Parch'] + test['SibSp']

test['Family'].loc[data['Family'] > 0 ] = 1
test['Family'].loc[data['Family'] == 0] = 0

#drop Parch and SibSp
data = data.drop(['Parch','SibSp'], axis = 1)
test = test.drop(['Parch','SibSp'], axis = 1)
#sex
sexes = sorted(data['Sex'].unique())
gender_mapping = dict(zip(sexes, range(0, len(sexes) + 1 )))
data['Sex'] = data['Sex'].map(gender_mapping).astype(int)
test['Sex'] = test['Sex'].map(gender_mapping).astype(int)
X_train = data.drop("Survived",axis=1)
Y_train = data["Survived"]
X_test  = test.drop('PassengerId', axis = 1).copy()

X_train.head()
train_X, val_X, train_y, val_y = train_test_split(X_train, Y_train,random_state = 0)
model = DecisionTreeRegressor()
model.fit(train_X,train_y)
survival = model.predict(val_X)
surv1 = survival.astype(int)
print('Accuracy is ',accuracy_score(val_y,surv1))
#accuracy_score(val_y,surv)
model = RandomForestRegressor()
model.fit(train_X,train_y)
survival = model.predict(val_X)
surv2 = survival.astype(int)
print('Accuracy is ',accuracy_score(val_y,surv2))
model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
model.fit(train_X,train_y)
survival = model.predict(val_X)
surv3 = survival.astype(int)
print('Accuracy is ',accuracy_score(val_y,surv3))
model = GradientBoostingRegressor()
model.fit(train_X,train_y)
survival = model.predict(val_X)
surv4 = survival.astype(int)
print('Accuracy is ',accuracy_score(val_y,surv4))
model = RandomForestClassifier(n_estimators=100)
model.fit(train_X,train_y)
survival = model.predict(val_X)
surv5 = survival.astype(int)
print('Accuracy is ',accuracy_score(val_y,surv5))
model = GradientBoostingClassifier(n_estimators=100)
model.fit(train_X,train_y)
survival = model.predict(val_X)
surv6 = survival.astype(int)
print('Accuracy is ',accuracy_score(val_y,surv6))
model = ExtraTreesClassifier(n_estimators=100)
model.fit(train_X,train_y)
survival = model.predict(val_X)
surv7 = survival.astype(int)
print('Accuracy is ',accuracy_score(val_y,surv7))
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, Y_train)
survival = model.predict(X_test)
final_survival = survival.astype(int)
final_survival
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": final_survival
    })
submission.to_csv('titanic.csv', index=False)