# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt
import seaborn as sns

#loading the data set
df = pd.read_csv('/kaggle/input/titanic/train.csv')
df.shape
df.head()
df.columns.values
df.info()
df.isnull().sum()
#dropping cabin column
df.drop(columns = ['Cabin'], inplace = True) 
#inplace = True changes the data inplace doesn't hae to return back 
#filing missing values for age
# strategy - mean, we can use different techniques for now we are filling it with mean
df['Age'].fillna(df['Age'].mean(), inplace = True)
df.info()
#inputting missing values for embarked
#we will fill the with most appeared value in embarked column
df['Embarked'].value_counts()
#people have boarded from S more. so we can assume others with no city can be S
df['Embarked'].fillna('S', inplace = True)

df['SibSp'].value_counts()
df['Parch'].value_counts()
df['Survived'] = df['Survived'].astype('category')
df['Pclass'] = df['Pclass'].astype('category')
df['Sex'] = df['Sex'].astype('category')
df['Age'] = df['Age'].astype('int')
df['Embarked'] = df['Embarked'].astype('category')
df.info()
df.describe()
#its good practice to check data by using describe
#univariate analysis
#we will start analysis with SUrvived column

sns.countplot(df['Survived'])
death_percent = round((df['Survived'].value_counts().values[0]/891)*100)
print('{} percent of people died'.format(death_percent))

#what about Pclass column
print((df['Pclass'].value_counts()/891)*100)

sns.countplot(df['Pclass'])

print((df['Sex'].value_counts()))
sns.countplot(df['Sex'])
print(df['SibSp'].value_counts())
sns.countplot(df['SibSp'])
print((df['Embarked'].value_counts()/891)*100)

sns.countplot(df['Embarked'])
#Age column

sns.distplot(df['Age']) 
print(df['Age'].skew())
print(df['Age'].kurt())
sns.boxplot(df['Age'])
print('people with age in between 60 and 70',df[(df['Age']>60) & (df['Age']<70)].shape[0])
print('people with age in between 70 and 75',df[(df['Age']>70) & (df['Age']<75)].shape[0])
print('people with age less than 75',df[(df['Age']<75)].shape[0])

print('-'*50)

print('People with age between 0 and 1', df[df['Age']<1].shape[0])
#Fare column
sns.distplot(df['Fare'])
print(df['Fare'].skew())
print(df['Fare'].kurt())
sns.boxplot(df['Fare'])
print('people with fare between 200$ to 300$', df[(df['Fare'] > 200) & (df['Fare'] < 300)].shape[0])
print('People with fare more than 300$', df[df['Fare'] > 300].shape[0])
#multivariate analysis
#Survival with Pclass


sns.countplot(df['Survived'], hue = df['Pclass'])

pd.crosstab(df['Pclass'], df['Survived']).apply( lambda r: round((r/r.sum())*100,1), axis =1)
#survival with sex

sns.countplot(df['Survived'], hue = df['Sex'])

pd.crosstab(df['Sex'],df['Survived']).apply(lambda r: round ((r/r.sum())*100,1),axis = 1)
#Survival on Embarked 

#its really not so good analysis but we ll see if we can see any information on this

sns.countplot(df['Survived'], hue = df['Embarked'])
pd.crosstab(df['Embarked'],df['Survived']).apply(lambda r: round((r/r.sum())*100, 1),axis = 1)
#survival with age 

plt.figure(figsize = (15,6))

sns.distplot(df[df['Survived']==0]['Age'])
sns.distplot(df[df['Survived']==1]['Age'])
#Survive with Fare

plt.figure(figsize = (15,6))
sns.distplot(df[df['Survived']==0]["Fare"])
sns.distplot(df[df['Survived']==1]['Fare'])
sns.pairplot(df)
sns.heatmap(df.corr())
#feature engineering

#we will create a new column by name family which will be the sum of SibSp and Parch cols

df['family_size'] = df['Parch'] + df['SibSp']
df.sample(5)
def family_type(number):
    if number == 0:
        return 'Alone'
    elif number > 0 and number <= 4:
        return 'Medium'
    else: 
        return 'Large'
df['family_type']=df['family_size'].apply(family_type)
df.sample(5)
#dropping SibSp, Parch and family_size

df.drop(columns = ['SibSp', 'Parch', 'family_size'], inplace = True)
pd.crosstab(df['family_type'],df['Survived']).apply(lambda r: round((r/r.sum())*100,1), axis = 1)

#handling outliers in age(Almost normal)

df = df[df['Age']<(df['Age'].mean() + 3 * df['Age'].std())]

#df.shape

df.shape
#handling outliers from fare column
#finding quartiles

Q1 = np.percentile(df['Fare'],25)
Q3 = np.percentile(df['Fare'],75)

outlier_low = Q1 - 1.5*(Q3 - Q1)
outlier_high = Q3 + 1.5 * (Q3 - Q1)

df = df[(df['Fare'] > outlier_low) & (df['Fare'] < outlier_high)]
#one hot encoding
#in categorical data , having values for categories will make our model give errors, because some where in the model the values of the categories will udergo some mathematical formula and give us wrong resutls 
#so one hot encoding will give us the binary values to the categorical columns


#columns to be transformed are Pclass, Sex, Embarked, family_type
#pd.get_dummies(data = df, columns = ['Pclass', 'Sex', 'Embarked', 'family_type'], drop_first = True)

df = pd.get_dummies(data = df, columns = ['Pclass', 'Sex', 'Embarked', 'family_type'], drop_first = True)
plt.figure(figsize = (15,6))
sns.heatmap(df.corr(), cmap = 'summer')
df.info()
#let us remove Name and PassengerID and Ticket columns which can't be used or not helpful for our model
df.drop(columns = ['PassengerId', 'Ticket', 'Name'], inplace = True)
X = df.loc[:,df.columns !='Survived']

Y = df.Survived
X.head()
from sklearn.model_selection import train_test_split



X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.20, random_state = 1)

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
y_pred_train = logreg.predict(X_train)
y_pred_test = logreg.predict(X_test)
from sklearn.metrics import accuracy_score

print('Accuracy score for test data is :', accuracy_score(Y_test,y_pred_test))
