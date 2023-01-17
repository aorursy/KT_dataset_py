import numpy as np
import pandas as pd

import seaborn as sns

from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
%matplotlib inline
#loading the data from csv to the DataFrames
#data_train = pd.read_csv('/Users/deepanjal.gupta/Documents/Projects/Kaggle/1. Titanic Problem/train.csv')
#data_test = pd.read_csv('/Users/deepanjal.gupta/Documents/Projects/Kaggle/1. Titanic Problem/test.csv')

data_train = pd.read_csv('../input/train.csv')
data_test = pd.read_csv('../input/test.csv')

#assiging the number of the exapmles in the train/test set 
m_train = data_train.shape[0]
m_test  = data_test.shape[0]
#code to calculate the number of missing features in percentge
data_bar_train = pd.DataFrame({'Feature':data_train.isnull().sum().index.values,'Percent_Missing':data_train.isnull().sum().values/m_train*100})
data_bar_test = pd.DataFrame({'Feature':data_test.isnull().sum().index.values,'Percent_Missing':data_test.isnull().sum().values/m_train*100})

data_bar_train = data_bar_train.sort_values(by = ['Percent_Missing','Feature'])
data_bar_test = data_bar_test.sort_values(by = ['Percent_Missing','Feature'])
#plotting the percentages of the missing feature values in the dataset.
plt.subplots(figsize=(15, 4))
plt.title('Training Set')
sns.barplot(x='Percent_Missing', y='Feature',data = data_bar_train,label="Total", color='b')

plt.subplots(figsize=(15, 4))
plt.title('Test Set')
sns.barplot(x='Percent_Missing', y='Feature',data = data_bar_test,label="Total", color='b')
#data_train + data_test 
total_data = pd.concat([data_train,data_test])

#replacing the missing ages with the mean age of the dataset
mean_age = total_data.describe()['Age']['mean']
data_train.loc[data_train['Age'].isnull(),'Age'] = mean_age
data_test.loc[data_test['Age'].isnull(),'Age'] = mean_age

#replacing the missing fares with the fares age of the dataset
mean_fare = total_data.describe()['Fare']['mean']
data_train.loc[data_train['Fare'].isnull(),'Fare'] = mean_fare
data_test.loc[data_test['Fare'].isnull(),'Fare'] = mean_fare

#replacing the missing Embarked data with the most common location
Embarked_code = total_data['Embarked'].value_counts().argmax()
data_train.loc[data_train['Embarked'].isnull(),'Embarked'] = Embarked_code
data_test.loc[data_test['Embarked'].isnull(),'Embarked'] = Embarked_code
#notice that only the cabin data is missing, rest all data has been put in place.
print('__data_train nulls__')
print(data_train.isnull().sum())
print()
print('*********************')
print('__data_test nulls__')
print(data_test.isnull().sum())
a = data_train.loc[:,['Survived','Embarked']].groupby(['Embarked']).sum()
b = data_train.loc[:,['Survived','Embarked']].groupby(['Embarked']).count()
#plotting the count of passengers from three different locations
b.plot.bar(color='b')
plt.title('count of passengers from three different locations')

#plotting the percentage survived in the training set
(a/b*100).plot.bar(color='g')
plt.title('the percentage survived in the training set')

#we can clearly see that the C passengers ahd a higher chances of survival than S & Q.
#plotting the distribution of the fare of tickets
plt.figure(figsize = (10,10))
data_train['Fare'].plot.hist( 4)

data_train[data_train['Fare']<100]['Fare'].plot.hist( 4)
plt.title('Fare Distribution')
not_survived_fare = data_train[data_train['Survived'] == 0]['Fare']
survived_fare     = data_train[data_train['Survived'] == 1]['Fare']

plt.subplots(1,1,figsize=(12,4))
sns.boxplot(not_survived_fare)
plt.title('Fare Distribution for non - survivers')

plt.subplots(1,1,figsize=(12,4))
sns.boxplot(survived_fare)
plt.title('Fare Distribution for survivers')


#we can clearly see that higer the fare, higher the survival rate
#we can see mainly the young survied
not_survived_age = data_train[data_train['Survived'] == 0]['Age']
survived_age     = data_train[data_train['Survived'] == 1]['Age']

plt.subplots(figsize=(12,4))
not_survived_age.plot.hist(20,color = 'r')
plt.subplots(figsize=(12,4))
survived_age.plot.hist(20,color = 'g')
#family impact vs alone travellers
data_train['FamilyPresent'] = data_train['SibSp'] + data_train['Parch']
data_test['FamilyPresent'] = data_test['SibSp'] + data_test['Parch']

data_train.loc[data_train['FamilyPresent']==0,'FamilyPresent'] = 0
data_train.loc[data_train['FamilyPresent']>0,'FamilyPresent'] = 1

data_test.loc[data_test['FamilyPresent']==0,'FamilyPresent'] = 0
data_test.loc[data_test['FamilyPresent']>0,'FamilyPresent'] = 1

#plotting the effect of family presense on the rate of survival
a = data_train.loc[:,['Survived','FamilyPresent']].groupby(['FamilyPresent']).sum()
b = data_train.loc[:,['Survived','FamilyPresent']].groupby(['FamilyPresent']).count()


#plotting the count of family presense 
b.plot.bar(color='b')
plt.title('family present count')

#plotting the percentage survived in the training set
(a/b*100).plot.bar(color='g')
plt.title('the percentage survived in the training set')

#effect of class on the rate of survival

a = data_train.loc[:,['Survived','Pclass']].groupby(['Pclass']).sum()
b = data_train.loc[:,['Survived','Pclass']].groupby(['Pclass']).count()


#plotting the count of class
b.plot.bar(color='b')
plt.title('family present count')

#plotting the percentage survived in the training set
(a/b*100).plot.bar(color='g')
plt.title('the percentage survived in the training set')

# we can clearly see that the first class passengers had a higher chance of survival
#data_train.loc[data_train['Sex']=='male','Sex'] = 1 #male
#data_train.loc[data_train['Sex']=='female','Sex'] = 2 #female
data_train.loc[data_train['Age']<16,'Sex'] = 'child' #kid

#data_test.loc[data_test['Sex']=='male','Sex'] = 1
#data_test.loc[data_test['Sex']=='female','Sex'] = 2
data_test.loc[data_test['Age']<16,'Sex'] = 'child' #kid
#effect of gender on the rate of survival

a = data_train.loc[:,['Survived','Sex']].groupby(['Sex']).sum()
b = data_train.loc[:,['Survived','Sex']].groupby(['Sex']).count()


#plotting the count of gender
b.plot.bar(color='b')
plt.title('gender present count')

#plotting the percentage survived in the training set
(a/b*100).plot.bar(color='g')
plt.title('the percentage survived in the training set')

# we can clearly see that the women and children had a higher chance of survival
#defining the input features
data_train_input = data_train.copy()
data_test_input = data_test.copy()


#converting everyting to a number
data_train_input.loc[data_train['Sex']=='male','Sex'] = 1   #male
data_train_input.loc[data_train['Sex']=='female','Sex'] = 2 #female
data_train_input.loc[data_train['Sex']=='child','Sex'] = 3  #child

data_test_input.loc[data_test['Sex']=='male','Sex'] = 1   #male
data_test_input.loc[data_test['Sex']=='female','Sex'] = 2#female
data_test_input.loc[data_test['Sex']=='child','Sex'] = 3  #child 



data_train_input.loc[data_train['Embarked']=='S','Embarked'] = 1   #male
data_train_input.loc[data_train['Embarked']=='C','Embarked'] = 2 #female
data_train_input.loc[data_train['Embarked']=='Q','Embarked'] = 3  #child

data_test_input.loc[data_test['Embarked']=='S','Embarked'] = 1   #male
data_test_input.loc[data_test['Embarked']=='C','Embarked'] = 2#female
data_test_input.loc[data_test['Embarked']=='Q','Embarked'] = 3  #child 



#features of our model:
list(data_train_input.loc[:,['Pclass','Sex','Age','Fare','Embarked','FamilyPresent']].columns)
features = ['Pclass','Sex','Age','Fare','Embarked','FamilyPresent']
#creating our feature Matrix
X_train = np.array(data_train_input.loc[:,['Pclass','Sex','Age','Fare','Embarked','FamilyPresent']])
X_test = np.array(data_test_input.loc[:,['Pclass','Sex','Age','Fare','Embarked','FamilyPresent']])
#Creating output array for the training set
Y_train = np.array(data_train_input.loc[:,['Survived']]).reshape(-1)

print(X_train.shape)
print(Y_train.shape)


#Logistic Regression
logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

logreg.score(X_train, Y_train)

#printing the coef_
a = logreg.coef_
a = a.reshape(-1)

pd.DataFrame({'FeatureName':features,'Value':a})
FinalAnswer = pd.DataFrame({'PassengerId':data_test_input['PassengerId'],'Survived':Y_pred})
FinalAnswer.to_csv('titanic.csv', index=False)