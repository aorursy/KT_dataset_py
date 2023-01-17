import pandas as pd

import numpy as np
train = pd.read_csv("/kaggle/input/titanic/train.csv",sep=",")
train.head()
train.info()
train.drop(['Cabin','Ticket'],inplace=True,axis=1)
train[train['Embarked'].isnull() == True]
train.iloc[61,9] = 'S'

train.iloc[829,9] = 'S'
#calculate the mean of age for each class and separated in genre

class1 = train[train['Pclass'] == 1].dropna()

male1 = [np.around(class1[class1['Sex'] == 'male'].Age.mean(),decimals=1)]

female1 = [np.around(class1[class1['Sex'] == 'female'].Age.mean(),decimals=1)]



class2 = train[train['Pclass'] == 2].dropna()

male2 = [np.around(class2[class2['Sex'] == 'male'].Age.mean(),decimals=1)]

female2 = [np.around(class2[class2['Sex'] == 'female'].Age.mean(),decimals=1)]



class3 = train[train['Pclass'] == 3].dropna()

male3 = [np.around(class3[class3['Sex'] == 'male'].Age.mean(),decimals=1)]

female3 = [np.around(class3[class3['Sex'] == 'female'].Age.mean(),decimals=1)]
#creat a data frame with just null values in age

nullValues = train[train['Age'].isnull() == True]

nullValues.head()
#creat a data frame separeting the class

nullClass1 = nullValues[nullValues['Pclass'] == 1]

nullClass2 = nullValues[nullValues['Pclass'] == 2]

nullClass3 = nullValues[nullValues['Pclass'] == 3]
#put the mean for male people of class 1

class1Male = nullClass1[nullClass1['Sex'] == 'male']

class1Male.Age = male1 * class1Male.Pclass.count()



#put the mean for female people of class 1

class1Female = nullClass1[nullClass1['Sex'] == 'female']

class1Female.Age = female1 * class1Female.Pclass.count()
#put the mean for male people of class 2

class2Male = nullClass2[nullClass2['Sex'] == 'male']

class2Male.Age = male2 * class2Male.Pclass.count()



#put the mean for female people of class 2

class2Female = nullClass2[nullClass2['Sex'] == 'female']

class2Female.Age = female2 * class2Female.Pclass.count()
#put the mean for male people of class 3

class3Male = nullClass3[nullClass3['Sex'] == 'male']

class3Male.Age = male3 * class3Male.Pclass.count()



#put the mean for female people of class 3

class3Female = nullClass3[nullClass3['Sex'] == 'female']

class3Female.Age = female3 * class3Female.Pclass.count()
#I eliminated the null values and replaced with the averages

train = train.dropna()

train = pd.concat([train,class1Male,class2Male,class3Male,class1Female,class2Female,class3Female])
#sort the data frame and drop de column PassengerId because we don't need this column to predict

train = train.sort_values('PassengerId')

train.head()
from sklearn.preprocessing import LabelEncoder
#1 -> male; 0 -> female

labelencoderSex = LabelEncoder()

train['Sex'] = labelencoderSex.fit_transform(train['Sex'])

train.head()
#0 -> C;1 -> Q;2 -> S

labelencoderEmbarked = LabelEncoder()

train['Embarked'] = labelencoderEmbarked.fit_transform(train['Embarked'])

train.head()
train.info()
train.describe()
%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns
genre = list(train.Sex.value_counts())

male = genre[0]

female = genre[1]
slices = [male,female]



fig, (ax1,ax2) = plt.subplots(1,2)

ax1.pie(slices, labels=['male','female'],colors=['b','r'],startangle = 90, shadow = True, explode = (0.1,0))

ax2.bar(['male'],male,label='k',color='b')

ax2.bar(['female'],female,color='r')
cityP = pd.DataFrame(train['Embarked'].value_counts().values,

                     index=['Southampton','Cherbourg','Queenstown'],

                    columns=['# people'])

cityP
plt.bar(['Southampton','Cherbourg','Queenstown'],cityP['# people'])

plt.ylabel('# of people')

plt.xlabel('City')
#which age show

np.sort(train['Age'].unique())
#how many passenger are in each age

ages = []

ages.append(list(train['Age'].value_counts().index))

ages.append(list(train['Age'].value_counts().values))

ages
plt.scatter(ages[0], ages[1], color='b', marker='x')

plt.ylabel("# of people")

plt.xlabel("Age")
survived = train['Survived'].value_counts()

survived
p = survived[0]/(survived[1]+survived[0]) * 100

p
plt.pie([p,100-p], labels=['Died','Alive'],colors=['black','gray'],startangle = 90, shadow = True, explode = (0.1,0))
pClass = train['Pclass'].value_counts()

pClass.index.name = 'Class'

pClass.name = 'Amount of people per class'

pClass
plt.bar(['3','1','2'],pClass)

plt.ylabel("# of people")

plt.xlabel("Class")
#separated the age of people per class

c3=list(train[train['Pclass'] == 3].Age.dropna())

c2=list(train[train['Pclass'] == 2].Age.dropna())

c1=list(train[train['Pclass'] == 1].Age.dropna())
plt.hist(c3, bins=[3,13,23,33,43,53,63,73,83], histtype = 'bar', 

         rwidth = 0.2, color='pink', label='Class 03',align='left')

plt.hist(c2, bins=[0,10,20,30,40,50,60,70,80], histtype = 'bar', 

         rwidth = 0.2, color='r', label='Class 02',align='mid')

plt.hist(c1, bins=[0,7,17,27,37,47,57,67,77], histtype = 'bar', 

         rwidth = 0.2, color='purple', label='Class 01',align='right')

plt.xlabel('Age')

plt.ylabel('# people')

plt.legend()
c1Fare = train[train['Pclass'] == 1].Fare.sum()

c2Fare = train[train['Pclass'] == 2].Fare.sum()

c3Fare = train[train['Pclass'] == 3].Fare.sum()

base = pd.Series([c1Fare,c2Fare,c3Fare])
plt.bar(['c1Fare','c2Fare','c3Fare'], base)
#evaluate the models

from sklearn.metrics import confusion_matrix, accuracy_score
train.drop(['Name','PassengerId'],axis=1,inplace=True)

train.head()
from sklearn.model_selection import train_test_split, StratifiedKFold
def modelSKF(model, dataset):

    acc,cof = 0,0

    skf = StratifiedKFold(n_splits=5)

    for train_index, test_index in skf.split(dataset.iloc[:,1:8],dataset.iloc[:,0]):

        Xtrain, Xtest = dataset.iloc[:,1:8].values[train_index], dataset.iloc[:,1:8].values[test_index]

        ytrain, ytest = dataset.iloc[:,0].values[train_index], dataset.iloc[:,0].values[test_index]

        model.fit(Xtrain,ytrain)

        prev = model.predict(Xtest)

        accuracy = accuracy_score(ytest,prev)

        if accuracy > acc:

            bestModel = model

            acc = accuracy

            conf = confusion_matrix(ytest,prev)

    return bestModel, acc, conf
def modelSplit(model, dataset):

    Xtrain, Xtest, ytrain, ytest = train_test_split(dataset.iloc[:,1:8],dataset.iloc[:,0],

                                                        test_size = 0.3, random_state = 0)

    model.fit(Xtrain,ytrain)

    prev = model.predict(Xtest)

    acc = accuracy_score(ytest,prev)

    conf = confusion_matrix(ytest,prev)

    return model, acc, conf
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(splitter='random',max_features='sqrt',max_depth=7, random_state=0) #create the model
tree,acc,conf = modelSKF(tree,train)

#tree,acc,conf = modelSplit(tree,train)
conf
acc
from sklearn.linear_model import LogisticRegression
logReg = LogisticRegression() #creat the model
logReg,acc,conf = modelSKF(logReg,train)

#logReg,acc,conf = modelSplit(logReg,train)
conf
acc
from sklearn.naive_bayes import GaussianNB
naive_bayes = GaussianNB() #create the model
naive_bayes,acc,conf = modelSKF(naive_bayes,train)

#naive_bayes,acc,conf = modelSplit(naive_bayes,train)
conf
acc
from sklearn.neighbors import KNeighborsClassifier
knc,acc,conf = modelSKF(KNeighborsClassifier(n_neighbors=3),train)

#knc,acc,conf = modelSplit(KNeighborsClassifier(n_neighbors=3),train)
conf
acc
test = pd.read_csv("/kaggle/input/titanic/test.csv",sep=",")

test['Sex'] = labelencoderSex.fit_transform(test['Sex'])

test['Embarked'] = labelencoderEmbarked.fit_transform(test['Embarked'])

test.drop(['Name','Ticket','Cabin'], axis=1, inplace=True)

test.info()
send = test.copy()

send.drop(['Pclass','Sex','Age','Fare','SibSp','Parch','Embarked'], axis=1, inplace=True)

send.head()
test.iloc[152,6] = 0
#calculate the mean of age for each class and separated in genre

class1 = test[test['Pclass'] == 1].dropna()

male1 = [np.around(class1[class1['Sex'] == 1].Age.mean(),decimals=1)]

female1 = [np.around(class1[class1['Sex'] == 0].Age.mean(),decimals=1)]



class2 = test[test['Pclass'] == 2].dropna()

male2 = [np.around(class2[class2['Sex'] == 1].Age.mean(),decimals=1)]

female2 = [np.around(class2[class2['Sex'] == 0].Age.mean(),decimals=1)]



class3 = test[test['Pclass'] == 3].dropna()

male3 = [np.around(class3[class3['Sex'] == 1].Age.mean(),decimals=1)]

female3 = [np.around(class3[class3['Sex'] == 0].Age.mean(),decimals=1)]
#creat a data frame with just null values in age

nullValues = test[test['Age'].isnull() == True]

nullValues.head()
#creat a data frame separeting the class

nullClass1 = nullValues[nullValues['Pclass'] == 1]

nullClass2 = nullValues[nullValues['Pclass'] == 2]

nullClass3 = nullValues[nullValues['Pclass'] == 3]
#put the mean for male people of class 1

class1Male = nullClass1[nullClass1['Sex'] == 1]

class1Male.Age = male1 * class1Male.Pclass.count()



#put the mean for female people of class 1

class1Female = nullClass1[nullClass1['Sex'] == 0]

class1Female.Age = female1 * class1Female.Pclass.count()
#put the mean for male people of class 2

class2Male = nullClass2[nullClass2['Sex'] == 1]

class2Male.Age = male2 * class2Male.Pclass.count()



#put the mean for female people of class 2

class2Female = nullClass2[nullClass2['Sex'] == 0]

class2Female.Age = female2 * class2Female.Pclass.count()
#put the mean for male people of class 3

class3Male = nullClass3[nullClass3['Sex'] == 1]

class3Male.Age = male3 * class3Male.Pclass.count()



#put the mean for female people of class 3

class3Female = nullClass3[nullClass3['Sex'] == 0]

class3Female.Age = female3 * class3Female.Pclass.count()
#I eliminated the null values and replaced with the averages

test = test.dropna()

test = pd.concat([test,class1Male,class2Male,class3Male,class1Female,class2Female,class3Female])
#sort the data frame and drop de column PassengerId because we don't need this column to predict

test = test.sort_values('PassengerId')

test.drop(['PassengerId'],axis=1,inplace=True)

test.head()
pred = pd.Series(naive_bayes.predict(test.iloc[:,0:7]))

pred.head()
send['Survived'] = pred

send['Survived'].astype(int)

send.head()