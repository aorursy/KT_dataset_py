import pandas

import numpy



#names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

data = pandas.read_csv('../input/adult.csv')

print(data.shape)
# If Columns dont have names

# data.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income']
data.head(5)
# Replace '?' missing values with NaN (not a number) 

data = data.replace({'?': numpy.nan})



data.head(5)
# Drop the columns where all elements are missing values:

# data = data.dropna(axis=1, how='all')



# Drop the columns where any of the elements are missing values

#data = data.dropna(axis=1, how='any')



# Drop the rows where any of the elements are missing values

data = data.dropna(axis=0, how='any')



# Keep only the rows which contain 2 missing values maximum

# data = data.dropna(thresh=2)



# Drop the columns where any of the elements are missing values

# data = data.dropna(axis=1, how='any')



# Fill all missing values with the mean of the particular column

#data = data.fillna(data.mean())



# Fill any missing value in column 'workclass' with the column median

#data = data['workclass'].fillna(data['workclass'].median())



# Fill any missing value in column 'occupation' with the column mode

# data = data['occupation'].fillna(data['occupation'].mode())
data.shape
data.head()
data["education"] = data["education"].astype('category')
data.dtypes
data["education"] = data["education"].cat.codes
data.head()
pandas.get_dummies(data, columns=["sex"]).head()
data = data.values





X = data[:,0:14]

Y = data[:,14] 
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1

                                                    ,random_state = 0)



X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size = 0.11

                                                   ,random_state = 0)
X_train.shape
X_test.shape