import pandas as pd

#Load Datasets:
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv') 

#For Train Data Find Missing Values :
#There are 81 variables then divide into numerical and categorical variables by data types:
NumericalVariable = list(train.loc[: , train.dtypes !=object].columns.values)
NumericalVariable
CategoricalVariable = list(train.loc[: , train.dtypes==object].columns.values)
CategoricalVariable

#Check Null Values for numerical data:
null_values = train[NumericalVariable].isnull().sum()
null_values = null_values[null_values>0] #we want to see missing value. you know, 0 means no null values.
print('Train Data Missing Values:\n')
print('Numerical Data:\n')
print(null_values.sort_values(ascending=False))
print('--------------------------------------------------')

#Check Null Values for categorical data:
null_values = train[CategoricalVariable].isnull().sum()
null_values = null_values[null_values>0] 
print('Categorical Data:\n')
print(null_values.sort_values(ascending=False))
print('\n')
#For Test Data Find Missing Values:
NumericalVariable= list(test.loc[: , test.dtypes!=object].columns.values)
NumericalVariable
CategoricalVariable= list(test.loc[: , test.dtypes==object].columns.values)
CategoricalVariable

#Check Null Values for numerical data:
null_values = test[NumericalVariable].isnull().sum()
null_values = null_values[null_values>0]
print('Test Data Missing Values:\n')
print('Numerical Data:\n')
print(null_values.sort_values(ascending=False))
print('--------------------------------------------------')

#Check Null Values for categorical data:
null_values = test[CategoricalVariable].isnull().sum()
null_values = null_values[null_values>0]
print('Categorical Data:\n')
print(null_values.sort_values(ascending=False))