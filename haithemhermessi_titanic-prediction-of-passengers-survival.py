from sklearn.metrics import classification_report, confusion_matrix

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.model_selection import learning_curve

from sklearn.metrics import accuracy_score

from sklearn.model_selection import cross_val_predict

from sklearn.model_selection import KFold

from matplotlib import pyplot as plt

import xgboost as xgb

import seaborn as sns

import pandas as pd

import numpy as np

import shap

import math

%matplotlib inline



#data import 

train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')

train=pd.DataFrame(train)

test=pd.DataFrame(test)

data_test=pd.read_csv('../input/titanic/test.csv')



print("Shape of the train set is", train.shape, " and the shape of the test is ",test.shape)
train.head(10)
test.head(10)
train.info()

#correlation matrix traces how variables are related to each other

#This will give an idea how to empute missing variable without dropping col

sns.heatmap(train.corr(), annot=True, cmap="coolwarm")

plt.show()
#drop PassengerId ,Ticket columns

train.drop(['PassengerId', 'Ticket','Name'], axis=1, inplace=True)

test.drop(['PassengerId', 'Ticket', 'Name'], axis=1, inplace=True)

train.shape, test.shape
train.info()
train.describe()
#Missing columns

cols_with_missing_train = train.isnull().sum()

print("Training set columns with missing values are :\n", cols_with_missing_train[cols_with_missing_train>0])



cols_with_missing_test = test.isnull().sum()

print("\n\n Test set columns with missing values are :\n", cols_with_missing_test[cols_with_missing_test>0])



print("\n\n Mean of survived passengers:\n",train["Survived"].mean())
#Suvived by Age 

ax = sns.boxplot(x="Survived", y="Age", 

                data=train)

ax = sns.stripplot(x="Survived", y="Age",

                   data=train, jitter=True,

                   edgecolor="gray")

plt.title("Suvived by Age in training data")
sns.countplot('Pclass',hue='Survived',data=train)

plt.show()
#Corelation between Age and Parch

sns.boxplot(x='Parch',y='Age', data=train, palette='hls')

plt.title("Age % Parch in training data")
sns.boxplot(x='Parch',y='Age', data=test, palette='hls')

plt.title("Age % Parch in test data")

#Mean age % parch is due to the correlation between Age and Parch

print("Correlation between Age and Parch \n",train.corr()["Age"].sort_values(ascending = False))



mean_age_train=train.groupby(['Parch'])['Age'].mean()

mean_age_test= test.groupby(['Parch'])['Age'].mean()      
#Imputer function to fill age using mean  on Parch

def fill_age(data,mean_age):

    for i in data['Age'].index:

        if (math.isnan(data['Age'][i])):

            if (data['Parch'][i]==0): 

                data['Age'][i]=mean_age[0]

            if (data['Parch'][i]==1): 

                data['Age'][i]=mean_age[1]

            if (data['Parch'][i]==2): 

                data['Age'][i]=mean_age[2]

            if (data['Parch'][i]==3): 

                data['Age'][i]=mean_age[3]

            if (data['Parch'][i]==4): 

                data['Age'][i]=mean_age[4]

            if (data['Parch'][i]==5): 

                data['Age'][i]=mean_age[5]

            if (data['Parch'][i]==6): 

                data['Age'][i]=mean_age[6]

            data['Age'][i]=mean_age[6]

    return data
fill_age(train,mean_age_train)

fill_age(test,mean_age_test)
#Embarked has missing values lets observe how it moves?

train[train['Embarked'].isnull()]
sns.boxplot(x="Embarked", y="Fare", hue="Pclass", data=train)
#Fill Embarked with C

train["Embarked"] = train["Embarked"].fillna('C')
#Categorical to numerical Embarked in train/test

from sklearn.preprocessing import LabelEncoder,OneHotEncoder

Enc=LabelEncoder()

train["Embarked"]=Enc.fit_transform(train["Embarked"])

test["Embarked"]=Enc.fit_transform(test["Embarked"])

train.head()
test.head()
#Cabin feature are missing a lot of values in both training and test datasets

train["Cabin"].isnull().sum(),test["Cabin"].isnull().sum()

train["Cabin"].unique()
train['Part']=train['Cabin'].str[0]

test['Part']=test['Cabin'].str[0]

train.drop(['Cabin'],axis=1, inplace=True)

test.drop(['Cabin'],axis=1, inplace=True)

train.head()
sns.boxplot(x="Part", y="Fare",  data=train)
train.isnull().sum()
train['Part'].unique()

#Fill Part with random choices

train["Part"] = train['Part'].fillna((pd.Series(np.random.choice(['C', 'E', 'G', 'D', 'A', 'B', 'F', 'T'], size=len(train.index)))))

test['Part'].unique()
#Fill Part with random choices

test["Part"] = test['Part'].fillna((pd.Series(np.random.choice([ 'B', 'E', 'A', 'C', 'D', 'F', 'G'], size=len(test.index)))))

#Categorical to numerical Part

train["Part"]=Enc.fit_transform(train["Part"])

test["Part"]=Enc.fit_transform(test["Part"])
#convert Sex from categorical to numeric

train['Sex'].replace(['male','female'],[0,1],inplace=True)

test['Sex'].replace(['male','female'],[0,1],inplace=True)
#Test Fare missing value 

test[test['Fare'].isnull()]
#We can replace missing value in Fare by taking median of all fares of those passengers who share 3rd Passenger class

median_fare=test[(train['Pclass'] == 3)]['Fare'].median()

median_fare

test["Fare"] = test["Fare"].fillna(median_fare)
from sklearn import preprocessing





#convert Age from float to Int

train['Age'] = train['Age'].astype(int)

test['Age']    = test['Age'].astype(int)





std_scale = preprocessing.StandardScaler().fit(train[['Age', 'Fare']])

train[['Age', 'Fare']] = std_scale.transform(train[['Age', 'Fare']])





std_scale = preprocessing.StandardScaler().fit(test[['Age', 'Fare']])

test[['Age', 'Fare']] = std_scale.transform(test[['Age', 'Fare']])
#train.drop(['Part'], axis=1, inplace=True)

#test.drop(['Part'], axis=1, inplace=True)

train.head()



test.head()

y_train = train["Survived"]

X_train = train.drop("Survived",axis=1)

X_test=test
X_train.shape , y_train.shape, X_test.shape
X_train.head()
from sklearn.linear_model import LogisticRegression

Linaer_reg = LogisticRegression()

Linaer_reg.fit(X_train,y_train)

predictions = Linaer_reg.predict(X_test)

LR_score= Linaer_reg.score(X_train, y_train)

predictions

data_test


output = pd.DataFrame({'PassengerId': data_test['PassengerId'], 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)