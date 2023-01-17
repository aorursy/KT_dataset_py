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
import warnings

warnings.filterwarnings('ignore')
titanic = pd.read_csv('/kaggle/input/titanic/train.csv')

titanic.head()

## checking the head of our data set
titanic.info()

## checking info of all columns
titanic.shape

## checking shape of data set
titanic.describe()

## statistical information about numerical variable
round(100*(titanic.isnull().sum()/len(titanic)),2)

## checking missing value percentage in all columns
titanic.drop('Cabin',axis=1,inplace=True)

## cabin almost have 77% of missing values hence remove this column from data set
age_median = titanic['Age'].median(skipna=True)

titanic['Age'].fillna(age_median,inplace=True)

## as there is 19% of missing values in age column hence it is not a good idea to remove this row wise or column wise hence impute those missing values with the median of age 

titanic = titanic[titanic['Embarked'].isnull()!=True]

## as embarked has a very small amount of missing values hence remove those rows which have missing values in embarked column 

titanic.shape

## checking shape after removing null values
titanic_dub = titanic.copy()

## creating copy of the data frame to check duplicate values
titanic_dub.shape

## comparing shapes of two data frames
titanic.shape

## shape of original data frame
import seaborn as sns

import matplotlib.pyplot as plt

## importing libraries for data visualitation
plt.figure(figsize=(15,5), dpi=80)

plt.subplot(1,4,1)

sns.boxplot(y=titanic['Age'])

plt.title("Outliers in 'Age'")



plt.subplot(1,4,2)

ax = sns.boxplot(y=titanic['Fare'])

ax.set_yscale('log')

plt.title("Outliers in 'Fare'")



plt.subplot(1,4,3)

sns.boxplot(y=titanic['SibSp'])

plt.title("Outliers in 'SibSp'")





plt.subplot(1,4,4)

sns.boxplot(y=titanic['Parch'])

plt.title("Outliers in 'Parch'")

#ax.set_yscale('log')

plt.tight_layout()

plt.show()



## plotting all four variables to check for outliers

## it clearly shows that all four variables has some outliers




sns.catplot(x="SibSp", col = 'Survived', data=titanic, kind = 'count', palette='pastel')

sns.catplot(x="Parch", col = 'Survived', data=titanic, kind = 'count', palette='pastel')

plt.tight_layout()

plt.show()



## plotting of sibsp and parch in basis of survived and not survived
def alone(x):

    if (x['SibSp']+x['Parch']>0):

        return (1)

    else:

        return (0)

titanic['Alone'] = titanic.apply(alone,axis=1)

## creating a function to make one variable which tells us whether a person is single or accompanied by some on the ship
sns.catplot(x="Alone", col = 'Survived', data=titanic, kind = 'count', palette='pastel')

plt.show()
## drop parch and sibsp

titanic = titanic.drop(['Parch','SibSp'],axis=1)

titanic.head()

sns.distplot(titanic['Fare'])

plt.show()
titanic['Fare'] = titanic['Fare'].map(lambda x: np.log(x) if x>0 else 0)

## converting fare into a logarithmic scale
sns.distplot(titanic['Fare'])

plt.show()

## again check the distribution of fare 
sns.catplot(x="Sex", y="Survived", col="Pclass", data=titanic, saturation=.5, kind="bar", ci=None, aspect=0.8, palette='deep')

sns.catplot(x="Sex", y="Survived", col="Embarked", data=titanic, saturation=.5, kind="bar", ci=None, aspect=0.8, palette='deep')

plt.show()



## plotting of survive on basis of pclass
survived_0 = titanic[titanic['Survived']==0]

survived_1 = titanic[titanic['Survived']==1]

## divided our dataset into survived or not survived to check the distribution of age in both the cases 
survived_0.shape

## checking shape of the data set that contains the data of passengers who not survived
survived_1.shape

## checking shape of the data set that contains the data of passengers who survived
sns.distplot(survived_0['Age'])

plt.show()

## checking distribution of age in not survived data set
sns.distplot(survived_1['Age'])

plt.show()

## checking distribution of age in survived dataset
sns.boxplot(x='Survived',y='Fare',data=titanic)

plt.show()

## checking survival rate on basis of fare
Pclass_dummy = pd.get_dummies(titanic['Pclass'],prefix='Pclass',drop_first=True)

Pclass_dummy.head()

## creating dummy variables for pclass



## joing dummy variables

titanic = pd.concat([titanic,Pclass_dummy],axis=1)

titanic.head()
titanic.drop('Pclass',axis=1,inplace=True)

## as there is no use of pclass after joining the columns that contains dummy variables  for pclass
Embarked_dummy = pd.get_dummies(titanic['Embarked'],drop_first=True)

Embarked_dummy.head()

## creating dummy variables for embarked and dropping first column
titanic = pd.concat([titanic,Embarked_dummy],axis=1)

titanic.drop('Embarked',axis=1,inplace=True)

## joining dummy variables
titanic.head()

## checking head of the data set after joining dummy variables
def sex_map(x):

    if x == 'male':

        return (1)

    elif x == 'female':

        return (0)

titanic['Sex'] = titanic['Sex'].apply(lambda x:sex_map(x))



## creating function for convert sex into binary values
titanic = titanic[['Survived','Sex','Age','Fare','Alone','Pclass_2','Pclass_3','Q','S']]
## First, let's calculate the Inter Quantile Range for our dataset,

IQR = titanic.Age.quantile(0.75) - titanic.Age.quantile(0.25)

## Using the IQR, we calculate the upper boundary using the formulas mentioned above for age

upper_limit = titanic.Age.quantile(0.75) + (IQR * 1.5)

upper_limit_extreme = titanic.Age.quantile(0.75) + (IQR * 3)

upper_limit, upper_limit_extreme

## Now, let’s see the ratio of data points above the upper limit & extreme upper limit. ie, the outliers.

total = np.float(titanic.shape[0])

print('Total Passenger: {}'.format(titanic.Age.shape[0]/total))

print('Passenger those age > 54.5: {}'.format(titanic[titanic.Age>54.5].shape[0]/total))

print('passenger those age > 74.0: {}'.format(titanic[titanic.Age>74.0].shape[0]/total))
## hence replace more than 54.5 age values with age_median



titanic['Age'] = titanic['Age'].apply(lambda x:age_median if x>54.5 else x)
## let's define lower boundary of age



lower_limit = titanic.Age.quantile(0.25) - (IQR * 1.5)

lower_limit_extreme = titanic.Age.quantile(0.25) - (IQR * 3)

lower_limit, lower_limit_extreme
## Now, let’s see the ratio of data points above the upper limit & extreme upper limit. ie, the outliers.

total = np.float(titanic.shape[0])

print('Total Passenger: {}'.format(titanic.Age.shape[0]/total))

print('Passenger those age < 2.5: {}'.format(titanic[titanic.Age<2.5].shape[0]/total))

print('passenger those age < -17.0: {}'.format(titanic[titanic.Age<-17.0].shape[0]/total))
## hence replace less than 2.5 age values with age_median



titanic['Age'] = titanic['Age'].apply(lambda x:age_median if x<2.5 else x)
sns.boxplot(titanic['Age'])

plt.show()

## check distribution of Age after replacing outliers
## First, let's calculate the Inter Quantile Range for our dataset,

IQR_fare = titanic.Fare.quantile(0.75) - titanic.Fare.quantile(0.25)

## Using the IQR, we calculate the upper boundary using the formulas mentioned above for age

upper_limit_fare = titanic.Fare.quantile(0.75) + (IQR_fare * 1.5)

upper_limit_extreme_fare = titanic.Fare.quantile(0.75) + (IQR_fare * 3)

upper_limit_fare, upper_limit_extreme_fare

## Now, let’s see the ratio of data points above the upper limit & extreme upper limit. ie, the outliers.

total = np.float(titanic.shape[0])

print('Total Passenger: {}'.format(titanic.Fare.shape[0]/total))

print('Passenger those fare > 6.213966625918822: {}'.format(titanic[titanic.Fare>6.213966625918822].shape[0]/total))

print('passenger those fare > 8.700333209612108: {}'.format(titanic[titanic.Fare>8.700333209612108].shape[0]/total))
fare_median = titanic['Fare'].median()

titanic['Fare'] = titanic['Fare'].apply(lambda x:fare_median if x>6.213966625918822 else x)
## let's define lower boundary of Fare



lower_limit_fare = titanic.Fare.quantile(0.25) - (IQR_fare * 1.5)

lower_limit_extreme_fare = titanic.Fare.quantile(0.25) - (IQR_fare * 3)

lower_limit_fare, lower_limit_extreme_fare
## Now, let’s see the ratio of data points above the upper limit & extreme upper limit. ie, the outliers.

total = np.float(titanic.shape[0])

print('Total Passenger: {}'.format(titanic.Fare.shape[0]/total))

print('Passenger those fare < -0.4163442639299415: {}'.format(titanic[titanic.Fare<-0.4163442639299415].shape[0]/total))

print('passenger those fare < -2.9027108476232275: {}'.format(titanic[titanic.Fare<-2.9027108476232275].shape[0]/total))
sns.boxplot(titanic['Fare'])

plt.show()

## check distribution of fare after replacing outliers 
y_train = titanic.pop('Survived')

X_train = titanic
import sklearn

from sklearn.tree import DecisionTreeClassifier



dt = DecisionTreeClassifier(max_depth=2,random_state=10).fit(X_train,y_train)
## check accuracy score with default decision tree

from sklearn.metrics import accuracy_score

score_default = accuracy_score(y_train,dt.predict(X_train))

score_default
from sklearn.ensemble import AdaBoostClassifier

from sklearn.metrics import accuracy_score

estm = list(range(1,500,3))

acc_score = []

for i in estm:

    ada = AdaBoostClassifier(base_estimator=dt,n_estimators=i,random_state=50).fit(X_train,y_train)

    pred = ada.predict(X_train)

    scores = accuracy_score(y_train,pred)

    acc_score.append(scores)

    

    

## created different estimators of ensembles and produce different accuracy score for each ensemble .
plt.plot(estm,acc_score)

plt.xlabel('n_estimators')

plt.ylabel('accuracy')

plt.ylim([0.85, 1])

plt.show()

## plotted number of estimators and accuracy score 
## let's make one final model 



ada_final = AdaBoostClassifier(base_estimator=dt,n_estimators=50,random_state=100).fit(X_train,y_train)
from sklearn.metrics import classification_report

print(classification_report(y_train,ada_final.predict(X_train)))
titanic_test = pd.read_csv('/kaggle/input/titanic/test.csv')
titanic_test.isnull().sum() ## check for null values
## impute Age using age_median



titanic_test['Age'].fillna(age_median,inplace=True)



## log transformation of fare 



titanic_test['Fare'] = titanic_test['Fare'].map(lambda x: np.log(x) if x>0 else 0)
## convert sex into numerical value



titanic_test['Sex'] = titanic_test['Sex'].apply(lambda x:sex_map(x))



## outliers treatment 



titanic_test['Age'] = titanic_test['Age'].apply(lambda x:age_median if x>54.5 else x)
## outlier treatment



titanic_test['Age'] = titanic_test['Age'].apply(lambda x:age_median if x<2.5 else x)



titanic_test['Fare'] = titanic_test['Fare'].apply(lambda x:fare_median if x>6.213966625918822 else x)
## creating alone column 



titanic_test['Alone'] = titanic_test.apply(alone,axis=1)
Embarked_dummy_test = pd.get_dummies(titanic_test['Embarked'],drop_first=True) ## creating dummy for test

titanic_test = pd.concat([titanic_test,Embarked_dummy_test],axis=1) ## joining dummy set
## creating dummy for pclass



Pclass_dummy_test = pd.get_dummies(titanic_test['Pclass'],prefix='Pclass',drop_first=True)
titanic_test = pd.concat([titanic_test,Pclass_dummy_test],axis=1) ## joining dummy set
test_cols = list(X_train.columns) ## required columns for fitting model
X_test = titanic_test[test_cols]
titanic_test['Survived'] = ada_final.predict(X_test) ## predicted survival using our model on test data 
titanic_res = titanic_test[['PassengerId','Survived']]

titanic_res.to_csv("Submission_titanic.csv",index=False) ## final submission file