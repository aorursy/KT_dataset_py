# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Read the training dataset

train_dataset = pd.read_csv("../input/train.csv")

test_dataset = pd.read_csv("../input/test.csv")
# Always good practise to print the top 5 records of the dataset

train_dataset.head()
# Print all the columns of the dataset

train_dataset.columns
print (train_dataset.info())

train_dataset.describe()
# If you notice, stats for only integer datatypes are displayed above, if you want to include the basic

# stats for all datatypes , run

train_dataset.describe(include='all')
#To Check the number of  missing values

train_dataset.isnull().sum()
# Lets group the dataset based on Gender

p_gender=train_dataset.groupby('Sex')

p_gender.groups
# Count of Female passengers who survived

p_gender.get_group('female')['Survived'].value_counts()
p_gender.get_group('male')['Survived'].value_counts()
#value counts will be give the count of female passengers based on the values in survived volumn

# Eg 1 233

#    0 81

# normalize will give the value of 233/314 , so multiply by 100 to get the perccentage

print("Percentage of female passengers who have survived:", 

      train_dataset["Survived"][train_dataset["Sex"] == 'female'].value_counts(normalize=True)[1]*100)

print("Percentage of Male passengers who have survived:", 

      train_dataset['Survived'][train_dataset['Sex']=='male'].value_counts(normalize=True)[1]*100)

sns.countplot(x="Sex",hue="Survived",data=train_dataset)
pclass= train_dataset.groupby('Pclass')
pclass.get_group(1)['Survived'].value_counts()
pclass.get_group(2)['Survived'].value_counts()
pclass.get_group(3)['Survived'].value_counts()
sns.countplot(x='Pclass',hue='Survived',data=train_dataset)
#Calcualte the percentage of passengers survived per class

print("Percentage of PClass=1 passengers who have survived",

      train_dataset['Survived'][train_dataset['Pclass']==1].value_counts(normalize=True)[1]*100)

print("Percentage of PClass=2 passengers who have survived",

      train_dataset['Survived'][train_dataset['Pclass']==2].value_counts(normalize=True)[1]*100)

print("Percentage of PClass=1 passengers who have survived",

      train_dataset['Survived'][train_dataset['Pclass']==3].value_counts(normalize=True)[1]*100)
sns.countplot(x='SibSp',hue='Survived',data=train_dataset)
# Lets check the percentage 

print("Percentage of passengers with SibSp=0 who survived",

     train_dataset['Survived'][train_dataset['SibSp']==0].value_counts(normalize=True)[1]*100)

print("Percentage of passengers with SibSp=1 who survived",

     train_dataset['Survived'][train_dataset['SibSp']==1].value_counts(normalize=True)[1]*100)

print("Percentage of passengers with SibSp=2 who survived",

     train_dataset['Survived'][train_dataset['SibSp']==2].value_counts(normalize=True)[1]*100)

print("Percentage of passengers with SibSp=3 who survived",

     train_dataset['Survived'][train_dataset['SibSp']==3].value_counts(normalize=True)[1]*100)

print("Percentage of passengers with SibSp=4 who survived",

     train_dataset['Survived'][train_dataset['SibSp']==4].value_counts(normalize=True)[1]*100)

sns.countplot(x='Parch',hue='Survived',data=train_dataset)
# Lets calculate the percentage 

print("Number of passengers with Parch=0 who survived",

     train_dataset['Survived'][train_dataset['Parch']==0].value_counts(normalize=True)[1]*100)

# Lets calculate the percentage 

print("Number of passengers with Parch=1 who survived",

     train_dataset['Survived'][train_dataset['Parch']==1].value_counts(normalize=True)[1]*100)

# Lets calculate the percentage 

print("Number of passengers with Parch=2 who survived",

     train_dataset['Survived'][train_dataset['Parch']==2].value_counts(normalize=True)[1]*100)

# Lets calculate the percentage 

print("Number of passengers with Parch=3 who survived",

     train_dataset['Survived'][train_dataset['Parch']==3].value_counts(normalize=True)[1]*100)
test_dataset.isnull().sum()
# As Embarked has only 2 missing values, lets just impute with the major class

train_dataset['Embarked'].value_counts()
# As majority of the passengers have embarked at 'S' 

# Lets replace the 2 missing values with this class only



train_dataset['Embarked'].fillna('S',inplace=True)
train_dataset['Embarked'].value_counts()
print ( train_dataset['Age'].isnull().sum() )

print ( test_dataset['Age'].isnull().sum() )
print ( train_dataset[['Name','Sex']][train_dataset['Age'].isnull()] )

print ( test_dataset[['Name','Sex']][test_dataset['Age'].isnull()] )

# We need to impute 177 missing values in train dataset and 86 in test dataset

# Lets create a list with both training and testing datasets

complete_dataset = [train_dataset,test_dataset]

# Lets capture the title from the name field

for dataset in complete_dataset:

    dataset['Title'] = dataset['Name'].str.extract('([A-Za-z)]+)\.')
# Lets print the unique title names along with the count 

print (train_dataset['Title'].value_counts())
# Looks like we have almost 17 various titles, lets try to print the gender

pd.crosstab(train_dataset['Title'],train_dataset['Sex'])
# Lets repalce these titles with more common names

for dataset in complete_dataset:

    dataset['Title'] = dataset['Title'].replace(['Capt','Col','Don','Dr','Jonkheer','Major','Rev','Dona'],'Rare')

    dataset['Title'] = dataset['Title'].replace(['Ms','Mlle'],'Miss')

    dataset['Title'] = dataset['Title'].replace(['Mme'],'Mrs')

    dataset['Title'] = dataset['Title'].replace(['Sir','Countess','Lady'],'Royal')

train_dataset['Title'].value_counts()
train_dataset.groupby('Title')['Age'].mean()
# Now lets impute the missing ages with the above values

train_dataset.loc[(train_dataset['Age'].isnull()) & (train_dataset['Title']=='Master'),'Age'] = 5

train_dataset.loc[(train_dataset['Age'].isnull()) & (train_dataset['Title']=='Miss'),'Age'] = 22

train_dataset.loc[(train_dataset['Age'].isnull()) & (train_dataset['Title']=='Mr'),'Age'] = 33

train_dataset.loc[(train_dataset['Age'].isnull()) & (train_dataset['Title']=='Mrs'),'Age'] = 36

train_dataset.loc[(train_dataset['Age'].isnull()) & (train_dataset['Title']=='Rare'),'Age'] = 46

train_dataset.loc[(train_dataset['Age'].isnull()) & (train_dataset['Title']=='Royal'),'Age'] = 44



test_dataset.groupby('Title')['Age'].mean()
test_dataset.loc[(test_dataset['Age'].isnull()) & (test_dataset['Title']=='Master'),'Age'] = 8

test_dataset.loc[(test_dataset['Age'].isnull()) & (test_dataset['Title']=='Miss'),'Age'] = 22

test_dataset.loc[(test_dataset['Age'].isnull()) & (test_dataset['Title']=='Mr'),'Age'] = 33

test_dataset.loc[(test_dataset['Age'].isnull()) & (test_dataset['Title']=='Mrs'),'Age'] = 39

test_dataset.loc[(test_dataset['Age'].isnull()) & (test_dataset['Title']=='Rare'),'Age'] = 44
test_dataset['Age'].isnull().sum()
train_dataset['Cabin'].isnull().sum()

# Since majority of the values are missing, lets just drop this column from feature list
print (train_dataset.dtypes)
for dataset in complete_dataset:

    dataset['Sex'].replace(['male','female'],[0,1],inplace=True)

    dataset['Embarked'].replace(['S','C','Q'],[0,1,2],inplace=True)
train_dataset.describe()
# As Age is a continous data ,it cannot be used in ML algorithms

# So we need to convert it .

print ( train_dataset['Age'].max())

print ( train_dataset['Age'].min())



print ( test_dataset['Age'].max())

print ( test_dataset['Age'].min())

# As the minimum age in the train and test dataset is 17 and maximum is 80,let us create bins using these values

# Lets consider 4 bins, so lets take age 20 as limit for each bin

for dataset in complete_dataset:

    dataset['Age_range']=0

    dataset.loc[dataset['Age']<=20,'Age_range']=0

    dataset.loc[(dataset['Age']>20)&(dataset['Age']<=40),'Age_range']=1

    dataset.loc[(dataset['Age']>40)&(dataset['Age']<=60),'Age_range']=2

    dataset.loc[(dataset['Age']>60)&(dataset['Age']<=80),'Age_range']=3



# Check the corelation of features

plt.figure(figsize=(18,10))

sns.heatmap(train_dataset.corr(),annot=True,cmap='RdYlGn',linecolor='b',linewidths='0.5')

plt.show()

#Lets run couple of models and based on the accuracy , lets decide which model to use

# Import all the libraries first

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix
# Split the training dataset into train and test datasets so that we can use test data

# to measure the models performance



X_features = train_dataset[['Age_range','Sex','Pclass','SibSp','Parch','Embarked']]

y_target = train_dataset['Survived']



X_train,X_test,y_train,y_test =train_test_split(X_features,y_target)



print (X_train.shape)

print (X_test.shape)



print(y_train.shape)

print(y_test.shape)
# Lets start with Logistic regression

logR = LogisticRegression()

logR.fit(X_train,y_train)

y_pred = logR.predict(X_test)



print ("Accuracy os Logistic regression is",accuracy_score(y_pred,y_test))
logR
# Parameter tuning Logistic Regression

logR = LogisticRegression(class_weight='balanced',C=1,max_iter=200)

logR.fit(X_train,y_train)

y_pred = logR.predict(X_test)



print ("Accuracy os Logistic regression with fine tuning is",accuracy_score(y_pred,y_test))



# As the accuracy has decreased , looks like the detafult parameters are best fit
svm = SVC()

svm.fit(X_train,y_train)

y_pred = svm.predict(X_test)



print ("Accuracy of SVM is",accuracy_score(y_pred,y_test))
# Parameter tuning

svm
svm = SVC(kernel='rbf',class_weight='balanced')

svm.fit(X_train,y_train)

y_pred = svm.predict(X_test)



print ("Accuracy of SVM after parameter tuning is ",accuracy_score(y_pred,y_test))
dt = DecisionTreeClassifier()

dt.fit(X_train,y_train)

y_pred = dt.predict(X_test)



print ("Accuracy of Decision tree is",accuracy_score(y_pred,y_test))

dt
rf = RandomForestClassifier()

rf.fit(X_train,y_train)

y_pred = rf.predict(X_test)



print ("Accuracy of Random forest is",accuracy_score(y_pred,y_test))



rf
knn = KNeighborsClassifier()

knn.fit(X_train,y_train)

y_pred = knn.predict(X_test)



print ("Accuracy of KNN is",accuracy_score(y_pred,y_test))



knn
nb = GaussianNB()

nb.fit(X_train,y_train)

y_pred = nb.predict(X_test)



print ("Accuracy of Gaussian NB is",accuracy_score(y_pred,y_test))

nb
gb = GradientBoostingClassifier()

gb.fit(X_train,y_train)

y_pred = gb.predict(X_test)



print ("Accuracy of Gradient boosting classifier is",accuracy_score(y_pred,y_test))



gb
# As SVM has 86% accuracy , am using svm for predicting test dataset

X_features = test_dataset[['Age_range','Sex','Pclass','SibSp','Parch','Embarked']]



pids = test_dataset['PassengerId']

predictions = svm.predict(X_features)



#Create a dataframe 

output = pd.DataFrame({ 'PassengerId' : pids, 'Survived': predictions })

output

# Output the dataframe to submission.csv

output.to_csv('submission.csv', index=False)