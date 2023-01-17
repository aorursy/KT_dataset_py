# EDA and Preparing Data libraries



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Visualization libraries

import seaborn as sns

import matplotlib.pyplot as plt



# Spliting data and creating model libraries

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from keras.models import Sequential #initialize neural network library

from keras.layers import Dense #build our layers library



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data_train = pd.read_csv("/kaggle/input/titanic/train.csv")

data_train.head()
data_train.info()
sns.countplot(data_train["Survived"])

plt.show()
sns.countplot(data_train["Pclass"])
data_train['Title'] = data_train.Name.str.extract(' ([A-Za-z]+)\.', expand=False)



data_train['Title'] = data_train['Title'].replace(['Lady', 'Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')

data_train['Title'] = data_train['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')

data_train['Title'] = data_train['Title'].replace('Mlle', 'Miss')

data_train['Title'] = data_train['Title'].replace('Ms', 'Miss')

data_train['Title'] = data_train['Title'].replace('Mme', 'Mrs')



plt.figure(figsize=(10,7))

sns.countplot(data_train.Title)

plt.title("data_train Passanger Name",color = 'blue',fontsize=15)

plt.show()
sns.countplot(data_train["Sex"])
g = sns.FacetGrid(data_train,row="Sex",col="Pclass")

g.map(sns.countplot,"Survived")

plt.show()
data_train["Age"].describe()
sns.distplot(data_train["Age"])
sns.countplot(data_train["SibSp"])
sns.countplot(data_train["Parch"])
data_train["Fare"].describe()
fare = ['above100$' if i>=100 else '32between100$' if (i<100 and i>=32) else 'Free' if i==0 else 'below32$' for i in data_train["Fare"]]

plt.figure(figsize=(10,7))

sns.countplot(fare)

plt.title("data_train Passanger Fare",color = 'blue',fontsize=15)

plt.show()
sns.countplot(data_train["Embarked"])
data_train.head()
#missing value

print(pd.isnull(data_train).sum())
# drop name and ticket

data_train = data_train.drop(["PassengerId","Name","Ticket"],axis=1)



# Sex

data_train["Sex"] = data_train["Sex"].replace("male",1)

data_train["Sex"] = data_train["Sex"].replace("female",2)



# Age

data_train["Age"] = data_train["Age"].replace(np.nan,data_train["Age"].median())



# Fare

data_train["Fare"] = data_train["Fare"].replace(np.nan,data_train["Fare"].median())



# Cabin

data_train.loc[data_train["Cabin"].str[0] == 'A', 'Cabin'] = 1

data_train.loc[data_train["Cabin"].str[0] == 'B', 'Cabin'] = 2

data_train.loc[data_train["Cabin"].str[0] == 'C', 'Cabin'] = 3

data_train.loc[data_train["Cabin"].str[0] == 'D', 'Cabin'] = 4

data_train.loc[data_train["Cabin"].str[0] == 'E', 'Cabin'] = 5

data_train.loc[data_train["Cabin"].str[0] == 'F', 'Cabin'] = 6

data_train.loc[data_train["Cabin"].str[0] == 'G', 'Cabin'] = 7

data_train.loc[data_train["Cabin"].str[0] == 'T', 'Cabin'] = 8

data_train["Cabin"] = data_train["Cabin"].fillna(data_train["Cabin"].mean())



# Embarked

data_train["Embarked"] = data_train["Embarked"].replace("S",1)

data_train["Embarked"] = data_train["Embarked"].replace("C",2)

data_train["Embarked"] = data_train["Embarked"].replace("Q",3)

data_train["Embarked"] = data_train["Embarked"].replace(np.nan,data_train["Embarked"].median())



# Title

data_train["Title"] = data_train["Title"].replace("Mr",1)

data_train["Title"] = data_train["Title"].replace("Mrs",2)

data_train["Title"] = data_train["Title"].replace("Miss",3)

data_train["Title"] = data_train["Title"].replace("Master",4)

data_train["Title"] = data_train["Title"].replace("Rare",5)

data_train["Title"] = data_train["Title"].replace("Royal",6)



#Family Size

data_train['FamilySize'] = data_train['SibSp'] + data_train['Parch']



data_train.drop(["SibSp","Parch"],axis=1,inplace=True)
data_test = pd.read_csv("/kaggle/input/titanic/test.csv")

data_test.head()
print(pd.isnull(data_test).sum())
#Title

data_test['Title'] = data_test.Name.str.extract(' ([A-Za-z]+)\.', expand=False)



data_test['Title'] = data_test['Title'].replace(['Lady', 'Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')

data_test['Title'] = data_test['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')

data_test['Title'] = data_test['Title'].replace('Mlle', 'Miss')

data_test['Title'] = data_test['Title'].replace('Ms', 'Miss')

data_test['Title'] = data_test['Title'].replace('Mme', 'Mrs')



# Sex

data_test["Sex"] = data_test["Sex"].replace("male",1)

data_test["Sex"] = data_test["Sex"].replace("female",2)



# Age

data_test["Age"] = data_test["Age"].replace(np.nan,data_test["Age"].median())



# Fare

data_test["Fare"] = data_test["Fare"].replace(np.nan,data_test["Fare"].median())



# Cabin

data_test.loc[data_test["Cabin"].str[0] == 'A', 'Cabin'] = 1

data_test.loc[data_test["Cabin"].str[0] == 'B', 'Cabin'] = 2

data_test.loc[data_test["Cabin"].str[0] == 'C', 'Cabin'] = 3

data_test.loc[data_test["Cabin"].str[0] == 'D', 'Cabin'] = 4

data_test.loc[data_test["Cabin"].str[0] == 'E', 'Cabin'] = 5

data_test.loc[data_test["Cabin"].str[0] == 'F', 'Cabin'] = 6

data_test.loc[data_test["Cabin"].str[0] == 'G', 'Cabin'] = 7

data_test.loc[data_test["Cabin"].astype(str).str[0] == 'T', 'Cabin'] = 8

data_test["Cabin"] = data_test["Cabin"].fillna(int(data_test["Cabin"].mean()))



# Embarked

data_test["Embarked"] = data_test["Embarked"].replace("S",1)

data_test["Embarked"] = data_test["Embarked"].replace("C",2)

data_test["Embarked"] = data_test["Embarked"].replace("Q",3)



# Title

data_test["Title"] = data_test["Title"].replace("Mr",1)

data_test["Title"] = data_test["Title"].replace("Mrs",2)

data_test["Title"] = data_test["Title"].replace("Miss",3)

data_test["Title"] = data_test["Title"].replace("Master",4)

data_test["Title"] = data_test["Title"].replace("Rare",5)

data_test["Title"] = data_test["Title"].replace("Royal",6)



#Family Size

data_test['FamilySize'] = data_test['SibSp'] + data_test['Parch']



# drop passenger id, name,ticket, sibsp and parch

data_test_x = data_test.drop(["PassengerId","Name","Ticket","SibSp","Parch"],axis=1)
X = data_train.drop(["Survived"],axis=1)

Y = data_train["Survived"]

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2)

print("x_train shape: ",x_train.shape)

print("y_train shape: ",y_train.shape)

print("x_test shape: ",x_test.shape)

print("y_test shape: ",y_test.shape)
classifier = Sequential() # initialize neural network

classifier.add(Dense(units = 128, activation = 'relu', input_dim = X.shape[1]))

classifier.add(Dense(units = 32, activation = 'relu'))

classifier.add(Dense(units = 16, activation = 'relu'))

classifier.add(Dense(units = 8, activation = 'relu'))

classifier.add(Dense(units = 4, activation = 'relu'))

classifier.add(Dense(units = 1, activation = 'sigmoid')) #output layer

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

model = classifier.fit(x_train,y_train,epochs=600)

mean = np.mean(model.history['accuracy'])

print("Accuracy mean: "+ str(mean))
y_predict = classifier.predict(x_test)

cm = confusion_matrix(y_test,np.argmax(y_predict, axis=1))



f, ax = plt.subplots(figsize=(5, 5))

sns.heatmap(cm, annot=True, fmt="d", linewidths=.5, ax=ax)
ids = data_test['PassengerId']

predict = classifier.predict(data_test_x)



#set the output as a dataframe and convert to csv file named submission.csv

output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': np.argmax(predict,axis=1)})

output.to_csv('submission.csv', index=False)