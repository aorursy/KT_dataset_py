import pandas as pd

import numpy as np
# 1. From numpy array

series_1 = pd.Series(np.array([10, 20, 30, 40, 50]))

print(series_1)
# 2. From dictionary

series_2 = pd.Series({'name': 'Daniel James', 'age': 25, 'country': 'India'})

print(series_2)
# 1. From numpy array

dataFrame_1 = pd.DataFrame(np.array([10, 20, 30, 40, 50]))

print(dataFrame_1)
# 2. From list of dictionaries

dataFrame_2 = pd.DataFrame([{'name': 'Daniel James', 'age': '25', 'country': 'India'}, {'name': 'Joseph James', 'age': '22', 'country': 'India'}])

print(dataFrame_2)
# 3, From csv

dataFrame_3 = pd.read_csv("../input/titanic/test.csv")

print(dataFrame_3)
# 4, To csv

dataFrame_3.to_csv("new_csv.csv")
test = pd.read_csv("../input/titanic/test.csv")
# If no arguments, it will take 5 as a default value, else it will take the argument as n.

test.head()
test.head(7)
test.head().T
print(test["Pclass"])
test.shape
test.describe()
test.info()
test.isnull().sum()
test = test.drop('PassengerId', axis=1)
print(test.iloc[3])
print(test.iloc[3:5])
print(test[(test['Fare'] == 8.6625) | (test['Fare'] == 12.2875)])



# OR



print(test[test['Fare'].isin([8.6625, 12.2875])])
getAgeGroup = lambda row: 'old' if row['Age'] >= 30 else 'young'

test['age_group'] = test.apply(getAgeGroup, axis = 1)

test.head()
# 1. highest age

print(test.Age.max())

# 2. different age groups (aggregate functions)

print(test.age_group.unique())

# 3. mean of different ages

print(test.Age.mean())
test.groupby('age_group').Age.max().reset_index()
unpivot = test.groupby(['age_group', 'Sex']).Age.max().reset_index()

unpivot.pivot(columns = 'age_group', index = 'Sex', values = 'Age').reset_index()