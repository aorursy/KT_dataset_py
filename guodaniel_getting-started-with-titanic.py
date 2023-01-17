#Load data

import pandas as pd



train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')



#Drop features we are not going to use

train = train.drop(['Name','SibSp','Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],axis=1)

test = test.drop(['Name','SibSp','Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],axis=1)



#Look at the first 3 rows of our training data

train.head(3)
for df in [train,test]:

    df['Sex_binary']=df['Sex'].map({'male':1,'female':0})
df.head()
df.isnull().sum()

df[df.Age.isnull()]
train['Age'] = train['Age'].fillna(0)

test['Age'] = test['Age'].fillna(0)
features = ['Pclass','Age','Sex_binary']

target = 'Survived'
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=200)

rf.fit(train[features],train[target])

#predictions = rf.predict(test[features])

#rf.score(test[features],test[target])
submission = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':predictions})



#Visualize the first 5 rows

submission.head()
filename = 'submission.csv'



submission.to_csv(filename,index=False)



print('Saved file: ' + filename)