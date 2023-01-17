import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import preprocessing
data = pd.read_csv('/kaggle/input/pima-diabetes-database/Pima Indians Diabetes Database.csv')

data.head()
data.info()
data.isnull().sum()
data.describe()
cols = ['Plasma glucose concentration a 2 hours in an oral glucose tolerance test', 'Diastolic blood pressure (mm Hg)', 'Triceps skin fold thickness (mm)', 

     '2-Hour serum insulin (mu U/ml)', 'Body mass index (weight in kg/(height in m)^2)', 'Diabetes pedigree function', 

     'Age (years)']

data[cols] = data[cols].replace(0,np.nan)

data.isnull().sum()
values = data.values

X = values[:,0:8]

y = values[:,8]
X.shape
y.shape
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

model = LinearDiscriminantAnalysis()

kfold = KFold(n_splits=3, random_state=7)

result = cross_val_score(model, X, y, cv=kfold, scoring = 'accuracy')

result.mean()
data_no_missing = data.copy()

data_no_missing.dropna(inplace=True)

data_no_missing.isnull().sum()
data_no_missing.shape
values = data_no_missing.values

X = values[:,0:8]

y = values[:,8]

model = LinearDiscriminantAnalysis()

kfold = KFold(n_splits=3, random_state=7)

result = cross_val_score(model, X, y, cv=kfold, scoring = 'accuracy')

result.mean()
data_imputed = data.copy()
## using pandas we can impute as 

## data_imputed.fillna(data_imputed.mean(),inplace=True)



## using sklearn

from sklearn.preprocessing import Imputer



values = data_imputed.values

X = values[:,0:8]

y = values[:,8]



imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis=0)

transformed_values = imputer.fit_transform(values)



## for strategy as mode we give 'most_frequent'



np.isnan(transformed_values).sum()



X = transformed_values[:,0:8]

y = transformed_values[:,8]



model = LinearDiscriminantAnalysis()

kfold = KFold(n_splits=3, random_state=7)

result = cross_val_score(model, X, y, cv=kfold, scoring = 'accuracy')

result.mean()
data_imputed = data.copy()



from sklearn.impute import SimpleImputer



values = data_imputed.values

X = values[:,0:8]

y = values[:,8]



imputer = SimpleImputer(missing_values = np.nan, strategy = 'constant', fill_value = 0)

transformed_values = imputer.fit_transform(values)



## for strategy as mode we give 'most_frequent'



np.isnan(transformed_values).sum()



X = transformed_values[:,0:8]

y = transformed_values[:,8]



model = LinearDiscriminantAnalysis()

kfold = KFold(n_splits=3, random_state=7)

result = cross_val_score(model, X, y, cv=kfold, scoring = 'accuracy')

result.mean()

data_imputed = data.copy()



values = data_imputed.values

X = values[:,0:8]

y = values[:,8]



imputer = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')

transformed_values = imputer.fit_transform(values)



np.isnan(transformed_values).sum()



X = transformed_values[:,0:8]

y = transformed_values[:,8]



model = LinearDiscriminantAnalysis()

kfold = KFold(n_splits=3, random_state=7)

result = cross_val_score(model, X, y, cv=kfold, scoring = 'accuracy')

result.mean()
data_imputed = data.copy()



values = data_imputed.values

X = values[:,0:8]

y = values[:,8]



imputer = SimpleImputer(missing_values = np.nan, strategy = 'median')

transformed_values = imputer.fit_transform(values)



np.isnan(transformed_values).sum()



X = transformed_values[:,0:8]

y = transformed_values[:,8]



model = LinearDiscriminantAnalysis()

kfold = KFold(n_splits=3, random_state=7)

result = cross_val_score(model, X, y, cv=kfold, scoring = 'accuracy')

result.mean()
assert pd.notnull(data).all().all()
cols = ['Plasma glucose concentration a 2 hours in an oral glucose tolerance test', 'Diastolic blood pressure (mm Hg)', 'Triceps skin fold thickness (mm)', 

     '2-Hour serum insulin (mu U/ml)', 'Body mass index (weight in kg/(height in m)^2)', 'Diabetes pedigree function', 

     'Age (years)']

data_with_nan = data.copy()

data_with_nan[cols] = data[cols].replace(0,np.nan)

data_with_nan.isnull().sum()