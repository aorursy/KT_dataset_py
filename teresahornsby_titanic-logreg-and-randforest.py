#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#importing all the required machine learning packages
from sklearn.linear_model import LogisticRegression #logistic regression
from sklearn.model_selection import train_test_split #training and testing data split
from sklearn.metrics import confusion_matrix, classification_report #for confusion matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
#import data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.tail()
#creat a column that identifies those who were in C Cabins; very high survival rate
train['good_cabin'] = train['Cabin'].astype(str).str[0]
train['good_cabin'] = train['good_cabin'].map({'C':1})
train['good_cabin'][train['good_cabin'] != 1] = 0
train['good_cabin'] = train['good_cabin'].astype(np.int64)

test['good_cabin'] = test['Cabin'].astype(str).str[0]
test['good_cabin'] = test['good_cabin'].map({'C':1})
test['good_cabin'][test['good_cabin'] != 1] = 0
test['good_cabin'] = test['good_cabin'].astype(np.int64)
#created a column that assigns #s to titles. 
data = [train, test]
titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for dataset in data:
    # extract titles
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    # replace titles with a more common title or as Rare
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr',\
                                            'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    # convert titles into numbers
    dataset['Title'] = dataset['Title'].map(titles)
    # filling NaN with 0, to get safe
    dataset['Title'] = dataset['Title'].fillna(0)
#turn the Sex column into an int binary df
train['sex'] = train['Sex'].astype(str).str[0]
train['sex'] = train['Sex'].map({'female':1})
train['sex'][train['sex'] != 1] = 0
train['sex'] = train['sex'].astype(np.int64)

test['sex'] = test['Sex'].astype(str).str[0]
test['sex'] = test['Sex'].map({'female':1})
test['sex'][test['sex'] != 1] = 0
test['sex'] = test['sex'].astype(np.int64)

#sns.countplot('Title',data=traindf2, hue='Survived')
#this graph shows that title 1 does not have a high survival rate.
#train.head()
train['embark'] = train['Embarked'].astype(str).str[0]
train['embark'] = train['embark'].map({'S':0})
train['embark'][train['embark'] != 0] = 1
train['embark'] = train['embark'].astype(np.int64)

test['embark'] = test['Embarked'].astype(str).str[0]
test['embark'] = test['embark'].map({'S':0})
test['embark'][test['embark'] != 0] = 1
test['embark'] = test['embark'].astype(np.int64)
#I turned the title into a binary; title 1 is a 0 and all others are a 1. 
train['title'] = train['Title'].map({1:0})
train['title'][train['title'] != 0] = 1


test['title'] = test['Title'].map({1:0})
test['title'][test['title'] != 0] = 1

#checking to see where there are null values
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
#create a column that reflects the length of the name; my theory is this would reflect a survivor's info
train["Name_Length"]=train["Name"].str.len()
test["Name_Length"]=test["Name"].str.len()
#checking through another graph (see below), those with longer names tended to survive.
test['longname']=test['Name_Length']> 40
train['longname']=train['Name_Length']>40
#creat a binary column that tells if a person's name is longer than the average person's name. 
train['longername'] = train['longname'].astype(int)
test['longername']=test['longname'].astype(int)
#traindf2.head()
traindf2=train
testdf2=test
#curious to see how those with longer names fared
sns.countplot('longername', data=traindf2, hue='Survived')
traindf2.head()
traindf2['first'] = traindf2['Pclass']
traindf2['first'] = traindf2['first'].map({3:0})
traindf2['first'][traindf2['first'] != 0] = 1
traindf2['first'] = traindf2['first']

testdf2['first'] = testdf2['Pclass']
testdf2['first'] = testdf2['first'].map({3:0})
testdf2['first'][testdf2['first'] != 0] = 1
testdf2['first'] = testdf2['first']
#traindf.to_csv('titanic4.csv')
traindf2.head()
print(traindf2['Fare'].median())

#turn fare into a binary 
traindf2['highfare']=traindf2['Fare']>traindf2['Fare'].median()
testdf2['highfare']=testdf2['Fare']>testdf2['Fare'].median()
traindf2['high_fare'] = traindf2['highfare'].astype(int)
testdf2.tail()
#drop all unnessary columns
traindf2.drop('Parch',axis=1, inplace=True)
testdf2.drop('Parch',axis=1,inplace=True)
traindf2.drop(['Cabin'],axis=1,inplace=True)
testdf2.drop(['Cabin'],axis=1,inplace=True)
traindf2.drop(['Sex','Ticket','Embarked','Age'],axis=1,inplace=True)
testdf2.drop(['Sex','Ticket','Embarked','Age'],axis=1,inplace=True)
testdf2.drop("Name",axis=1, inplace=True)
traindf2.drop("Name",axis=1,inplace=True)
traindf2.drop(['longname'],axis=1, inplace=True)
testdf2.drop(['longname'],axis=1,inplace=True)
traindf2.drop(['Name_Length'],axis=1,inplace=True)
testdf2.drop(['Name_Length'],axis=1,inplace=True)
traindf2.drop('Fare', axis=1, inplace=True)
traindf2.drop('highfare', axis=1, inplace=True)
testdf2.drop('Fare', axis=1, inplace=True)
testdf2.drop('highfare', axis=1, inplace=True)
traindf2.drop('Pclass', axis=1, inplace=True)
testdf2.drop('Pclass', axis=1, inplace=True)
traindf2.drop('SibSp', axis=1, inplace=True)
testdf2.drop('SibSp', axis=1, inplace=True)
traindf2.drop('Title', axis=1, inplace=True)
testdf2.drop('Title', axis=1, inplace=True)
X = traindf2.drop("Survived", axis=1)
y = traindf2["Survived"]
X_test  = testdf2.drop("PassengerId", axis=1).copy()
traindf2.head()
X_train, X_test, y_train, y_test = train_test_split(traindf2.drop('Survived',axis=1), 
                                                    traindf2['Survived'], test_size=0.469, 
                                                    random_state=101)
logmodel = LogisticRegression()
#logmodel.fit(X_train,y_train)
logmodel.fit(X,y)
predictions = logmodel.predict(X_test)
print(classification_report(y_test,predictions))
submission = pd.DataFrame({
        "PassengerId": testdf2["PassengerId"],
        "Survived": predictions
    })
submission.to_csv('submission.csv', index=False)
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X,y)

Y_prediction = rfc.predict(X_test)

rfc.fit(X,y)
predrfc = rfc.predict(X_test)
print(classification_report(y_test,predictions))
testdf2.info()
dtree = DecisionTreeClassifier()
dtree.fit(X,y)
preddtree = dtree.predict(X_test)
print(classification_report(y_test,predictions))
testdf2.tail()
