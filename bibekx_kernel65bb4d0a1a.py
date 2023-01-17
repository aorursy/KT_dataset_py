import pandas as pd

import seaborn as sns

Data_test = pd.read_csv("../input/titanic/test.csv")

Data_train = pd.read_csv("../input/titanic/train.csv")
Data_train.head()

#Data Lookup reveals that there are 11 features and 1 target variable
Data_train.describe()

#Lets see if there are any missing values: Age has 177 missing values, what about " Cabin","Ticket","Embarked","Sex" and "Name"
Data_train.info()

#Turns out Cabin has 687 missing values and others do not
#Right now itself we know that "Cabin" and "Passenger Id " are useles, lets drop them

Data_train = Data_train.drop(['PassengerId'],axis=1)

Data_train = Data_train.drop(['Cabin'],axis=1)
#Lets look at other features and observe the classification power 

sns.barplot(x='Pclass', y='Survived', data=Data_train)
sns.barplot(x='SibSp', y='Survived', data=Data_train)
sns.barplot(x='Parch', y='Survived', data=Data_train)
sns.barplot(x='Embarked', y='Survived', data=Data_train)
sns.barplot(x='Sex', y='Survived', data=Data_train)
#Fare is a continuos value, lets convert it to a categorical

#We know from the quartiles that 25% = 7.91, 50% =14.45 , 75% =31 

#Lets create a new category out of it

#But first we have to fill the empty values



Data_train['Fare'].fillna(Data_train['Fare'].median(),inplace=True)

Data_train['Fare']=Data_train['Fare'].astype(int)

Data_train.loc[Data_train['Fare']<=7.91,'Fare']=0

Data_train.loc[(Data_train['Fare']>7.91) & (Data_train['Fare']<=14.45),'Fare']=1

Data_train.loc[(Data_train['Fare']>14.45) & (Data_train['Fare']<=31),'Fare']=2

Data_train.loc[Data_train['Fare']>31,'Fare']=4
sns.barplot(x='Fare', y='Survived', data=Data_train)

#Higher the fare, higher the chances of survival
#We filled the missing values in fare, lets fill out the missing values in age too, There are 177 of them missing

#Also embarked has 2 missing values (the most common embarked value is S)

Data_train["Age"].fillna(Data_train["Age"].median(),inplace=True)

Data_train['Embarked'] = Data_train['Embarked'].fillna('S')
#Now that out data is clean

y = Data_train["Survived"]


#But first we need to encode male and female as binary

#And embarkment port as ternary



#maybe not do the encoding yourselg=f

#genders = {"male": 0, "female": 1}

#Data_train['Sex'] = Data_train['Sex'].map(genders)



#Ports = {"S": 0, "C": 1,"Q":2}

#Data_train['Embarked'] = Data_train['Embarked'].map(Ports)

#So we are done here with data cleaning, lets get to feature engineering

Data_train["Family_size"] = Data_train['SibSp'] + Data_train['Parch']+1
#making age categorical: Quartiles are 20,28,38,80



Data_train['Age']=Data_train['Age'].astype(int)

Data_train.loc[Data_train['Age']<=20,'Age']=0

Data_train.loc[(Data_train['Age']>20) & (Data_train['Age']<=28),'Age']=1

Data_train.loc[(Data_train['Age']>28) & (Data_train['Age']<=38),'Age']=2

Data_train.loc[Data_train['Age']>38,'Age']=4
#Extracting title from name and adding it as one more feature using regular expressions

import re

def get_title(name):

    title_search=re.search('([A-Za-z]+)\.',name)

    #If the title exists extract it and return

    if title_search:

        return title_search.group(1)

    return ""

Data_train['title']=Data_train['Name'].apply(get_title)
#Lets look at the different types of titles

#We can see that no values are missing

Data_train['title'].value_counts()
# Some values like Mlle, Mme are mistyped they should be Miss and Mrs respectively also Ms and Miss has no real difference

#Others like Capt, Don, Countess, Jonkheer, Lady, Sir, Major, Col, Rev and Dr are just too rare in the data

Data_train['title'] = Data_train['title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer'], 'Rare')

Data_train['title']= Data_train['title'].replace('Mlle','Miss')

Data_train['title']= Data_train['title'].replace('Mme','Mr')

Data_train['title']= Data_train['title'].replace('Ms','Miss')
Data_train['title'].value_counts()

#Now this is something we can deal with, lets look at what titles vs survival
sns.barplot(x='title', y='Survived', data=Data_train)
#We do not still know what to do with names, so we ignore it as a feature. Also do not know what to make of ticket

#Lets drop them

Data_train = Data_train.drop(['Ticket'],axis=1)

Data_train = Data_train.drop(['Survived'],axis=1)
#So, 7 features in total

features=["Pclass","Sex","Age","Family_size","Fare","Embarked","title"]

X = pd.get_dummies(Data_train[features])

X.head()
#Lets look at out processed features, we can see that the features still have very different scales

X.describe()
#Lets build a model

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(criterion='entropy', n_estimators=200,max_depth=5,min_samples_split=10,min_samples_leaf=1,

max_features='auto',oob_score=True,random_state=42,n_jobs=-1)
model.fit(X,y)
#Do all the pre processing to test data in one go

#Except dropping the passenger ID



Data_test = Data_test.drop(['Ticket'],axis=1)



#genders = {"male": 0, "female": 1}

#Data_test['Sex'] = Data_test['Sex'].map(genders)



#Ports = {"S": 0, "C": 1,"Q":2}

#Data_test['Embarked'] = Data_test['Embarked'].map(Ports)



Data_test["Age"].fillna(Data_test["Age"].median(),inplace=True)

Data_test['Embarked'] = Data_test['Embarked'].fillna('S')



Data_test['Fare'].fillna(Data_test['Fare'].median(),inplace=True)

Data_test['Fare']=Data_test['Fare'].astype(int)

Data_test.loc[Data_test['Fare']<=7.91,'Fare']=0

Data_test.loc[(Data_test['Fare']>7.91) & (Data_test['Fare']<=14.45),'Fare']=1

Data_test.loc[(Data_test['Fare']>14.45) & (Data_test['Fare']<=31),'Fare']=2

Data_test.loc[Data_test['Fare']>31,'Fare']=4



Data_test['Age']=Data_test['Age'].astype(int)

Data_test.loc[Data_test['Age']<=20,'Age']=0

Data_test.loc[(Data_test['Age']>20) & (Data_test['Age']<=28),'Age']=1

Data_test.loc[(Data_test['Age']>28) & (Data_test['Age']<=38),'Age']=2

Data_test.loc[Data_test['Age']>38,'Age']=4



Data_test["Family_size"] = Data_test['SibSp'] + Data_test['Parch']+1

Data_test['title']=Data_test['Name'].apply(get_title)





#Test has one more title called Dona-- so we add Dona to the rare lise

Data_test['title'] = Data_test['title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer','Dona'], 'Rare')

Data_test['title']= Data_test['title'].replace('Mlle','Miss')

Data_test['title']= Data_test['title'].replace('Mme','Mr')

Data_test['title']= Data_test['title'].replace('Ms','Miss')



features=["Pclass","Sex","Age","Family_size","Fare","Embarked","title"]

X_test = pd.get_dummies(Data_test[features])

X_test.describe()
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': Data_test.PassengerId, 'Survived': predictions})

output.to_csv('my_submission_4.csv', index=False)