import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
from scipy.stats import skew
import pandas as pd
import numpy as np
import seaborn as sns
%matplotlib inline
import warnings

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()
import os
print(os.listdir("../input"))
data = pd.read_csv('../input/train.csv')
data.head()
numerical_features = data.select_dtypes(include=[np.number])
numerical_features.columns
data.describe()
data.drop('PassengerId', axis = 1).hist(figsize=(30,20), layout=(4,3))
plt.plot()
skew_values = skew(data[numerical_features.columns], nan_policy = 'omit')
pd.concat([pd.DataFrame(list(numerical_features.columns), columns=['Features']), 
           pd.DataFrame(list(skew_values), columns=['Skewness degree'])], axis = 1,) 
missing_values = data.isnull().sum().sort_values(ascending = False)
percentage_missing_values = missing_values/len(data)
combine_data = pd.concat([missing_values, percentage_missing_values], axis = 1, keys= ['Missing values count', 'Percentage'])
pd.pivot_table(combine_data, index=combine_data.index,margins=True ) 
plt.figure(figsize=(20,10))
sns.heatmap(data.drop('PassengerId', axis = 1).corr(), square=True, annot=True, vmax= 1,robust=True, yticklabels=1)
plt.show()
plt.figure(figsize=(20,10))
sns.boxplot(data = data.drop('PassengerId', axis = 1))
plt.show()
# Let's see survival and Fare relation
var = 'Fare'
plt.scatter(x = data[var], y = data['Survived'])
plt.xlabel('Fare')
plt.ylabel('Survived')
plt.show()
plt.figure(figsize=(20,5))
sns.boxplot(x =data[var])
plt.show()
data.drop(data[data['Fare']> 100].index,  inplace= True)
data.drop(['Age', 'Cabin'], axis = 1, inplace=True)
data.dropna(inplace=True)
categorical_features = data.select_dtypes(include=[np.object])
categorical_features.columns
print('Sex has {} unique values: {}'.format(len(data.Sex.unique()),data.Sex.unique()))
print('Embarked has {} unique values: {}'.format(len(data.Embarked.unique()),data.Embarked.unique()))
data.drop(['Name', 'Ticket'], axis = 1, inplace=True)
data  = pd.get_dummies(data)
data.head()