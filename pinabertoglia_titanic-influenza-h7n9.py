# installing libraries

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy.stats import mode
import string

#loading data
train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')

print("train data describe")
print(train_data.describe())

print("test data describe")
print(test_data.describe())
#check variables names
print("train data variable names")
print(train_data.columns)

print("test data variable names")
print(test_data.columns)
#print 10 rows of train data
train_data.head(10)
#print 10 rows of test data
test_data.head(10)
#store the dependant variable: Survived
survived_data = train_data.Survived
# head command returns the top few lines of data.
print(survived_data.head())
#store ID variable
id_data_train = train_data.PassengerId
print(id_data_train.head())
id_data_test = test_data.PassengerId
print(id_data_test.head())
#check variables with missing values
print(train_data.isnull().sum())
print(test_data.isnull().sum())
#check type of variable
train_data.dtypes
#check type of variable
test_data.dtypes
#let's look at the Age of passengers

import matplotlib.pyplot as plt
%matplotlib inline
train_data.boxplot(column="Age",by="Survived")
#Other style

import seaborn as sns
sns.set_style("whitegrid")
ax = sns.boxplot(x="Survived", y="Age", data=train_data)
sns.set_style("whitegrid")
ax = sns.boxplot(x="Pclass", y="Fare", data=train_data)
#Describe variable Fare
train_data.describe()
#graph
sns.set_style("whitegrid")
ax = sns.boxplot(x="Pclass", y="Fare", data=test_data)
#Describe variable Fare
test_data.describe()
#Recoding Age into categorical data in train

#def binning(col, cut_points, labels=None):
  #Define min and max values:
  #minval = col.min()
 # maxval = col.max()

  #create list by adding min and max to cut_points
 # break_points = [minval] + cut_points + [maxval]

  #Binning using cut function of pandas
 # colBin = pd.cut(col,bins=break_points,labels=labels,include_lowest=True)
 # return colBin

#Binning age:
#cut_points = [15,40,60]
#labels = ["0-15","16-40","41-60","+61"]
#train_data["Age_cat"] = binning(train_data["Age"], cut_points, labels)
#print(pd.value_counts(train_data["Age_cat"], sort=False))
#Recoding Age into categorical data in test

#def binning(col, cut_points, labels=None):
  #Define min and max values:
 # minval = col.min()
  #maxval = col.max()

  #create list by adding min and max to cut_points
  #break_points = [minval] + cut_points + [maxval]

  #Binning using cut function of pandas
#  colBin = pd.cut(col,bins=break_points,labels=labels,include_lowest=True)
 # return colBin

#Binning age:
#cut_points = [15,40,60]
#labels = ["0-15","16-40","41-60","+61"]
#test_data["Age_cat"] = binning(test_data["Age"], cut_points, labels)
#print(pd.value_counts(test_data["Age_cat"], sort=False))
#def agrupador(df):
 #   df['Company'] = df['SibSp'] + df['Parch']
  #  return df

#train_data = agrupador(train_data)
#test_data = agrupador(test_data)

train_data.head()
test_data.head()
#Drop some predictors from train and test AND Survived from train

X_train = train_data.drop(['PassengerId','Name','Cabin','Ticket','Survived','Embarked','SibSp','Parch','Pclass','Age'], axis=1)
X_test = test_data.drop(['PassengerId','Name','Cabin','Ticket','Embarked','SibSp','Parch','Pclass','Age'], axis=1)
#Check
X_train.head()
#Check
X_test.head()
one_hot_encoded_train_data = pd.get_dummies(X_train)
one_hot_encoded_test_data = pd.get_dummies(X_test)

final_train, final_test = one_hot_encoded_train_data.align(one_hot_encoded_test_data,
                                                                    join='left', 
                                                                    axis=1)
#Check
print(final_train.isnull().sum())
print(final_test.isnull().sum())
#Impute missing data

from sklearn.impute import SimpleImputer
my_imputer = SimpleImputer()
data_with_imputed_values_train = my_imputer.fit_transform(final_train)
data_with_imputed_values_test = my_imputer.fit_transform(final_test)

#check
print(data_with_imputed_values_train)
#check
print(data_with_imputed_values_test)
pd.DataFrame(data_with_imputed_values_train).describe()
pd.DataFrame(survived_data).describe()
#Let's model with XGBoost

from xgboost import XGBRegressor

titanic_model = XGBRegressor()
# Add silent=True to avoid printing out updates with each cycle
titanic_model.fit(data_with_imputed_values_train, survived_data, verbose=False)

#Get predicted data
predicted_survived = titanic_model.predict(data_with_imputed_values_test)
print(predicted_survived)
predicted_survived = np.around(predicted_survived,0)
print(predicted_survived)
pina_submission = pd.DataFrame({'PassengerId': id_data_test, 'Survived': predicted_survived})

pina_submission['Survived'] = pina_submission['Survived'].astype(np.int64)
pina_submission.to_csv('submission.csv', index=False)
pina_submission.describe()
pina_submission.head(20)

