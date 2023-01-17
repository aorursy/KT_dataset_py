# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pandas
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

train=pd.read_csv('/kaggle/input/titanic/train.csv')
train.head()
train["Survived"].shape
test=pd.read_csv('/kaggle/input/titanic/test.csv')
test.head()
print(train.shape)

print(test.shape)
#Checking null values present on each column so that we can remove the unnecessary column

train.isnull().sum()
#Checking null values present on each column so that we can remove the unnecessary column

test.isnull().sum()
#Removing column "Cabin" since it has many null values

train.drop(['Cabin'], axis=1,inplace=True )

test.drop(['Cabin'], axis=1,inplace=True )
#Finding out missing values of "Embarked" and then filling them up by S

train['Embarked'].value_counts()
train['Embarked'].fillna('S', inplace=True)

#test['Embarked'].value_counts()
#test['Embarked'].fillna('S', inplace=True)
#Finding out missing values of "Fair" and fiiling them up by mean values

#train['Fare'].fillna(train['Fare'].mean(),inplace=True)

test['Fare'].fillna(test['Fare'].mean(),inplace=True)
#Finding out missing values for "Age" of train
#Creating a variable called train_age and storing the random values from  mean and std

train_age=np.random.randint(train['Age'].mean() - train['Age'].std() , train['Age'].mean() + train['Age'].std(), 177)

train_age
#Checking null values present in age of train
train['Age'][train['Age'].isnull()]
#Replacing these null values by train_age
train['Age'][train['Age'].isnull()]=train_age
train.isnull().sum()
#Finding out missing values for "Age" of test 
#Creating a variable called test_age and storing the random values from  mean and std

test_age=np.random.randint(train['Age'].mean() - train['Age'].std() , train['Age'].mean() + train['Age'].std(), 86)

test_age
#Checking null values present in age of test
test['Age'][test['Age'].isnull()]
#Replacing these null values by train_age
test['Age'][test['Age'].isnull()]=test_age
test.isnull().sum()
#for Pclass

train.groupby(['Pclass'])['Survived'].mean() #Therefore, Pclass can not be removed.
#For Sex

train.groupby(['Sex'])['Survived'].mean() #Therefore, Sex matters too
#For Embarked

train.groupby(['Embarked'])['Survived'].mean() #Therefore, Embarked matters too
#For Age since its numerical data so we are plotting the graph 

sns.distplot(train['Age'][train['Survived']==0])
sns.distplot(train['Age'][train['Survived']==1])

#Therefore, Age matters too
#For Fare

sns.distplot(train['Fare'][train['Survived']==0])
sns.distplot(train['Fare'][train['Survived']==1])

#Therefore, Fair matters too
#For Ticket, we have to remove it since it doesn't matter

train.drop(['Ticket'], axis=1, inplace=True)
test.drop(['Ticket'], axis=1, inplace=True)
test
#For "SibSp" and "Parch", we are going to add these two columns to a new column called "Family" for both train and test dataset

train['Family']=train['SibSp'] + train['Parch'] + 1
test['Family']=test['SibSp'] + test['Parch'] + 1

train['Family'].value_counts()
test['Family'].value_counts()
#For Family

train.groupby(['Family'])['Survived'].mean() #Therefore, Family matters too
#Creating a separate column for people travelling alone, with 2 or more than 2 & less than 4 and with more than 11

def cal1(number):
    if number==1:
        return"Alone"
    elif number>1 & number<5:
        return"Medium"
    else:
        return"Large"
    
train['Family_size']=train['Family'].apply(cal1)
train
test
#Creating a separate column for people travelling alone, with 2 or more than 2 & less than 4 and with more than 11

def cal2(number):
    if number==1:
        return"Alone"
    elif number>1 & number<5:
        return"Medium"
    else:
        return"Large"
    
test['Family_size']=test['Family'].apply(cal2)
test
#Removing columns like "SibSp", "Parch" and "Family"

train.drop(['SibSp','Parch', 'Family'], axis=1, inplace=True)
test.drop(['SibSp','Parch', 'Family'], axis=1, inplace=True)
#We need "PassengerId" for test so storing it in "passengerid"

passengerid=test['PassengerId'].values
#We don't need passenger id for training the dataset we only need it while testing so removing "PassengerId" and "Name"

train.drop(['PassengerId','Name'], axis=1, inplace=True)
test.drop(['PassengerId','Name'], axis=1, inplace=True)
train
test
#Converting categorical values into numerical values for train

train=pd.get_dummies(columns=['Pclass','Sex','Embarked','Family_size'], drop_first=True, data=train)
train
#Converting categorical values into numerical values for train

test=pd.get_dummies(columns=['Pclass','Sex','Embarked','Family_size'], drop_first=True, data=test)
test
train.shape
test.shape
X=train.iloc[:,1:].values
print("The shape of X:",X.shape)

Y=train.iloc[:,0].values
print("The shape of Y:",Y.shape)
#Splitting
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2)

print("The shape of X_train:",X_train.shape)
print("The shape of Y_train:",Y_train.shape)
print(Y_test.shape)
classifier1=DecisionTreeClassifier() 
classifier1.fit(X_train, Y_train)
#Predicting
Y_predict=classifier1.predict(X_test)
print(Y_predict.shape)

#Accuracy
AS1=accuracy_score(Y_test,Y_predict) 
print("The accuracy score using decision tree classifier:", AS1)
# Using Grid-Search-CV class and Training the model
#Creating Variable
param_dist={"criterion":["gini","entropy"],
            "max_depth":[1,2,3,4,5,6,7,8,None],
            "max_features":[1,2,3,4,5,6,7,None],
            "random_state":[0,1,2,3,4,5,6,7,8,9,None],
            "max_leaf_nodes":[0,1,2,3,4,5,6,7,8,9,None],
            "max_features" : ["auto","sqrt","log2",None],
            "min_samples_leaf" : [1,2,3,4,5,6,7,8,None],
            "min_samples_split" : [1,2,3,4,5,6,7,8,None]}

#Applying Grid-Search-CV
grid=GridSearchCV(classifier1, param_grid=param_dist, cv=10, n_jobs=-1)

#Training the model after applying Grid-Search-CV
grid.fit(X_train,Y_train)

OHV=grid.best_params_ 
print("The values of Optimal Hyperparameters are",OHV)
Acc=grid.best_score_
print("The Accuracy Score is",Acc)
print("Accuracy using DecisionTreeClassifier:", Acc*100,"%")
grid.best_estimator_
classifier2=DecisionTreeClassifier(criterion= 'gini', max_depth= 7, max_features= 'auto', max_leaf_nodes= None, min_samples_leaf= 8, min_samples_split= 2, random_state= 4)

classifier2.fit(X_train, Y_train)
# Predicting
Y_predict=classifier2.predict(X_test)
print(Y_predict.shape)

# Accuracy
AS2=accuracy_score(Y_test,Y_predict) 
print("The accuracy score using decision tree classifier:", AS2)
X_test=test.iloc[:,:].values
Y_test=classifier2.predict(X_test)
Y_test.shape
passengerid.shape
#Creating an empty dataframe since passenger_id and Y_test have same number of rows
Final = pd.DataFrame()
Final
#Adding these 2 columns "passengerid" and "survived" then passing Y_test value in survived column

Final['passengerid'] = passengerid
Final['survived'] = Y_test
Final
#Converting it into csv file

Final.to_csv('submission.csv', index=False)

