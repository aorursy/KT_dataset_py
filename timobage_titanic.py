#My first competition submission

import numpy as np
import pandas as pd
test_data = pd.read_csv('/kaggle/input/titanic/test.csv')
test_data.head()
test_data.info()
data = pd.read_csv('/kaggle/input/titanic/train.csv')
data.head()
data.info()
data.describe()
data.corr()
import seaborn as sns
sns.distplot(data['Age'],bins=30)
data_survived = data[data['Survived']==1]
data_survived.head()
data_survived.info()
sns.distplot(data_survived['Age'],bins=30)
sns.countplot(data['Pclass'], data=data)
sns.countplot(data_survived['Pclass'], data=data_survived)
sns.distplot(data['Fare'],bins=30)
data['Fare'].value_counts(ascending=False)
#kick out three passengers that are above Fare 300
data=data[data['Fare']<300]
sns.countplot(data['SibSp'], data=data)
sns.countplot(data['Parch'], data=data)
#data = data.drop(['PassengerId'],axis=1)
data.info()
data.head()
data['Cabin'].isnull().sum()
data['Cabin']
#Cabin could be split into level and room number. but NaN numbers are hughe, therefore probably not worth it
#Ticket seems to be a weird number
#Name not needed
data = data.drop(['Name'],axis=1)
data = data.drop(['Ticket'],axis=1)
data.info()
subs = pd.get_dummies(data['Sex'],drop_first=True)
subs
data = pd.concat([data,subs],axis=1)
data = data.drop(['Sex'],axis=1)
data = data.drop(['Cabin'],axis=1)
subs = pd.get_dummies(data['Embarked'],drop_first=True)
subs
data = pd.concat([data,subs],axis=1)
data = data.drop(['Embarked'],axis=1)
data.info()
sns.boxplot(x='Pclass',y='Age',data=data)
def ager(cols):
    agecol = cols[0]
    classcol = cols[1]
    
    if pd.isnull(agecol):
        if classcol == 1:
            return 37
        elif classcol == 2:
            return 29
        else:
            return 24
    else:
        return agecol
        
data['Age'] = data[['Age','Pclass']].apply(ager,axis=1)
data['Age Group'] = pd.cut(data['Age'],5)
data[['Age Group','Survived']].groupby(['Age Group']).mean()
data.loc[data['Age'] <= 16,'Age'] = 0
data.loc[(data['Age'] > 16) & (data['Age'] <= 32),'Age'] = 1
data.loc[(data['Age'] > 32) & (data['Age'] <= 48),'Age'] = 2
data.loc[(data['Age'] > 48) & (data['Age'] <= 64),'Age'] = 3
data.loc[data['Age'] > 64,'Age'] = 4
pd.value_counts(data['Age'])
data = data.drop(['Age Group'],axis=1)
data.head()
test_data = pd.read_csv('/kaggle/input/titanic/test.csv')
test_data.head()
pd.isnull(test_data['Fare']).sum()
np.argwhere(pd.isnull(test_data['Fare']))
test_data.loc[152]
sns.boxplot(x='Pclass',y='Fare',data=data)
test_data[test_data['Pclass']==3]['Fare'].mean()
test_data = test_data.drop(['Name'],axis=1)
test_data = test_data.drop(['Cabin'],axis=1)
test_data = test_data.drop(['Ticket'],axis=1)
test_data['Age'] = test_data[['Age','Pclass']].apply(ager,axis=1)
test_data.loc[152,'Fare']=12.46

feats = ['Sex','Embarked']
subs2 = pd.get_dummies(test_data[feats],drop_first=True)
test_data = pd.concat([test_data,subs2],axis=1)
test_data = test_data.drop(['Sex'],axis=1)
test_data = test_data.drop(['Embarked'],axis=1)
test_data.info()

test_data.loc[test_data['Age'] <= 16,'Age'] = 0
test_data.loc[(test_data['Age'] > 16) & (test_data['Age'] <= 32),'Age'] = 1
test_data.loc[(test_data['Age'] > 32) & (test_data['Age'] <= 48),'Age'] = 2
test_data.loc[(test_data['Age'] > 48) & (test_data['Age'] <= 64),'Age'] = 3
test_data.loc[test_data['Age'] > 64,'Age'] = 4
data.info()
#from sklearn.model_selection import train_test_split
y = data['Survived']
X = data.drop(['Survived'],axis=1)
X_test = test_data
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
from sklearn.tree import DecisionTreeClassifier
classi = DecisionTreeClassifier()
classi.fit(X,y)
predictions = classi.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
#print(classification_report(y_test,predictions))
#print(confusion_matrix(y_test,predictions))
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X,y)
rfcpredictions = rfc.predict(X_test)
#print(classification_report(y_test,rfcpredictions))
#print(confusion_matrix(y_test,rfcpredictions))
from sklearn.linear_model import LogisticRegression
log = LogisticRegression(max_iter=600)
log.fit(X,y)
logpredictions = log.predict(X_test)
#print(classification_report(y_test,logpredictions))
#print(confusion_matrix(y_test,logpredictions))
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': rfcpredictions})
output.to_csv('my_submission.csv', index=False)
output.info()
