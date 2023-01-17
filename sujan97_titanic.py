import pandas as pd

from sklearn.ensemble import RandomForestClassifier



train=pd.read_csv('../input/train.csv')

test=pd.read_csv('../input/test.csv')
train.head()
#Mapping male to 0 and female to 1

train['Sex']=train['Sex'].map({'male':0,'female':1})

test['Sex']=test['Sex'].map({'male':0,'female':1})

test.head()
#Filling missing values of age with mean age value

medianAge=train.Age.median()

train['Age'].fillna(medianAge, inplace=True)

train['Age']=train['Age'].astype(int)



medianAge=test.Age.median()

test['Age'].fillna(medianAge, inplace=True)

test['Age']=test['Age'].astype(int)

test.head()
#Filling missing values of embarked with S and mapping

train['Embarked'].fillna(train['Embarked'].mode()[0], inplace=True)

train['Embarked']=train['Embarked'].map({'S':0,'Q':1,'C':2})



test['Embarked'].fillna(test['Embarked'].mode()[0], inplace=True)

test['Embarked']=test['Embarked'].map({'S':0,'Q':1,'C':2})
#We drop cabin because more than 70% of the data are missing

print('Percentage of missing values in cabin in train data:',(train['Cabin'].isnull().sum()/len(train['Cabin']))*100)

print('Percentage of missing values in cabin in test data:',(test['Cabin'].isnull().sum()/len(test['Cabin']))*100)

train.drop(['Cabin'], axis=1, inplace = True)

test.drop(['Cabin'],axis=1,inplace=True)
#combining sibsp and parch to single family column

train['fam']=train['SibSp']+train['Parch']+1

test['fam']=test['SibSp']+test['Parch']+1
#dropping unnecessary features

train=train.drop('Ticket',axis=1)

train=train.drop('SibSp',axis=1)

train=train.drop('Parch',axis=1)



test=test.drop('Ticket',axis=1)

test=test.drop('SibSp',axis=1)

test=test.drop('Parch',axis=1)
test.head()
#Extracting name titles and mapping

train['Name']=train['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, 

                 "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3,"Countess": 3,

                 "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona" : 3, "Mme": 3,"Capt": 3,"Sir": 3 }

train['Name']=train['Name'].map(title_mapping)



test['Name']=test['Name'].map(title_mapping)

test['Name']=test['Name'].fillna(0)
train['Fare']=train['Fare'].astype(int)



#filling missing values with median

avgFare=test['Fare'].median()

test['Fare']=test['Fare'].fillna(avgFare)

test['Fare']=test['Fare'].astype(int)
test.isnull().sum()
train.isnull().sum()
from sklearn.model_selection import train_test_split



X_train, X_test, Y_train, Y_test = train_test_split(train.drop(['Survived','PassengerId'], axis=1), 

                                                    train['Survived'], test_size = 0.2, 

                                                    random_state = 0)
random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)



Y_prediction = random_forest.predict(X_test)



random_forest.score(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)



print('The accuracy of the Random Forest Classifier is',acc_random_forest)
ids = test['PassengerId']

predictions = random_forest.predict(test.drop('PassengerId', axis=1))





output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })

output.to_csv('submission.csv', index=False)
submission = pd.read_csv('submission.csv')

submission.head(20)