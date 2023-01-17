# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy # linear algebra

import pandas # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
titanic_train = pandas.read_csv("../input/train.csv")

titanic_test = pandas.read_csv('../input/test.csv')
titanic_train.head()
titanic_train = titanic_train.drop(['Name','Ticket','Embarked'],axis=1)

titanic_test = titanic_test.drop(['Name','Ticket','Embarked'],axis=1)
titanic_train.info()

titanic_test.info()
titanic_train.describe()
titanic_train.Age = titanic_train.Age.fillna(29.69)

titanic_test.Age = titanic_test.Age.fillna(29.69)

titanic_test.Fare = titanic_test.Fare.fillna(32.20)
#Now if cabin nan fill with 0 otherwise fill with 1

titanic_train.Cabin = titanic_train.Cabin.fillna(0)

titanic_test.Cabin = titanic_test.Cabin.fillna(0)



titanic_train.Cabin[titanic_train.Cabin != 0] = 1

titanic_test.Cabin[titanic_test.Cabin != 0] = 1
print(titanic_train.head())

print(titanic_test.head())
titanic_train.drop('PassengerId',axis=1,inplace=True)

passId = titanic_test.PassengerId

titanic_test.drop('PassengerId',axis=1,inplace=True)
titanic_train['AgeBand'] = pandas.cut(titanic_train['Age'], 5)

titanic_train[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)
titanic_train.head()
titanic_train.loc[ titanic_train['Age'] <= 16, 'Age'] = 0

titanic_train.loc[(titanic_train['Age'] > 16) & (titanic_train['Age'] <= 32), 'Age'] = 1

titanic_train.loc[(titanic_train['Age'] > 32) & (titanic_train['Age'] <= 48), 'Age'] = 2

titanic_train.loc[(titanic_train['Age'] > 48) & (titanic_train['Age'] <= 64), 'Age'] = 3

titanic_train.loc[ titanic_train['Age'] > 64, 'Age'] = 4
titanic_train.Age =titanic_train.Age.astype(int)

titanic_train.head()
titanic_train['FareBand'] = pandas.cut(titanic_train['Fare'], 4)

titanic_train[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)
titanic_train.loc[titanic_train.Fare<=128,'Fare'] = 0

titanic_train.loc[(titanic_train.Fare>128)&(titanic_train.Fare<=256),'Fare'] = 1

titanic_train.loc[(titanic_train.Fare>256)&(titanic_train.Fare<=384),'Fare'] = 2

titanic_train.loc[(titanic_train.Fare>384)&(titanic_train.Fare<=513),'Fare'] = 3
titanic_train.Fare = titanic_train.Fare.astype(int)
#Sex male = 0, female = 1

titanic_train.loc[titanic_train.Sex=='male','Sex'] = 0

titanic_train.loc[titanic_train.Sex == 'female','Sex'] = 1
#Now drop bands

titanic_train.drop(['AgeBand','FareBand'],axis=1,inplace=True)
titanic_train.info()
#After we changed cabin's type our trainset will be ready

titanic_train.Cabin = titanic_train.Cabin.astype(int)

from sklearn.model_selection import train_test_split

#lets split our data set 



train_X,test_X,train_y,test_y = train_test_split(titanic_train.drop(['Survived','Cabin'],axis=1),titanic_train.Survived,test_size =0.3)



from sklearn.linear_model import LogisticRegression



lr = LogisticRegression(C= 0.2,solver = "liblinear",max_iter=150,multi_class = "ovr")

lr.fit(train_X,train_y)
print('Test DataSet Score : ',lr.score(test_X,test_y))

print('Train DataSet Score : ',lr.score(train_X,train_y))

# not bad score
from sklearn.ensemble import RandomForestClassifier



rf = RandomForestClassifier()

rf.fit(train_X,train_y)
print('Test DataSet Score: ',rf.score(test_X,test_y))

print('Train DataSet Score: ',rf.score(train_X,train_y))
titanic_test.head()
titanic_test.Cabin = titanic_test.Cabin.astype(int)

titanic_test.loc[titanic_test.Fare<=128,'Fare'] = 0

titanic_test.loc[(titanic_test.Fare>128)&(titanic_test.Fare<=256),'Fare'] = 1

titanic_test.loc[(titanic_test.Fare>256)&(titanic_test.Fare<=384),'Fare'] = 2

titanic_test.loc[(titanic_test.Fare>384)&(titanic_test.Fare<=513),'Fare'] = 3

titanic_test.loc[ titanic_test['Age'] <= 16, 'Age'] = 0

titanic_test.loc[(titanic_test['Age'] > 16) & (titanic_test['Age'] <= 32), 'Age'] = 1

titanic_test.loc[(titanic_test['Age'] > 32) & (titanic_test['Age'] <= 48), 'Age'] = 2

titanic_test.loc[(titanic_test['Age'] > 48) & (titanic_test['Age'] <= 64), 'Age'] = 3

titanic_test.loc[ titanic_test['Age'] > 64, 'Age'] = 4
titanic_test.Age = titanic_test.Age.astype(int)

titanic_test.Fare = titanic_test.Fare.astype(int)

titanic_test.loc[titanic_test.Sex == 'male','Sex'] = 0

titanic_test.loc[titanic_test.Sex=='female','Sex'] = 1
#now we can train our model with full trainset

random_forest = RandomForestClassifier(n_estimators = 40)

random_forest.fit(titanic_train.drop(['Survived','Cabin','SibSp','Parch'],axis=1),titanic_train.Survived)
log_reg = LogisticRegression()

log_reg.fit(titanic_train.drop(['Survived','Cabin','SibSp','Parch'],axis=1),titanic_train.Survived)
rf_predicts = random_forest.predict(titanic_test.drop(['Cabin','SibSp','Parch'],axis=1))

lg_predicts = log_reg.predict(titanic_test.drop(['Cabin','SibSp','Parch'],axis=1))

print('Random Forest Predicts :\n ',rf_predicts)

print('Logistic Regression Predicts :\n',lg_predicts)
rf_solution = pandas.DataFrame({'PassengerId':passId,'Survived':rf_predicts})

lg_solution = pandas.DataFrame({'PassengerId':passId,'Survived':lg_predicts})
rf_solution.to_csv('RandomForest.csv',index=False)

lg_solution.to_csv('LogisticRegression.csv',index=False)