# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import missingno as msno

from sklearn.feature_selection import mutual_info_regression, SelectKBest

from sklearn.preprocessing import LabelEncoder



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/data.csv')

data.head()
print(f"Number of rows: {data.shape[0]}")

print(f"Number of cols: {data.shape[1]}")
def convert_currencies(x):

    if type(x) is float:

        return x

    

    multiply_by = 1000000 if 'M' in x else 1000

    x = x.replace('M', '').replace('K','').replace('€','')

    return float(x) * multiply_by
data['Wage'] = data['Wage'].apply(convert_currencies)

data['Value'] = data['Value'].apply(convert_currencies)

data['Release Clause'] = data['Release Clause'].apply(convert_currencies)

data.head()
# Count the NaNs

nan_count = data.isnull().sum()

cols_with_nans = nan_count[nan_count > 0]

cols_without_nans = nan_count[nan_count == 0]



print(f"Number of columns with missing values:  {cols_with_nans.size} of {data.shape[1]}")

print(f"Number of columns with complete values: {cols_without_nans.size} of {data.shape[1]}")
# Plot the completeness of every column

ax = msno.bar(data)
data.drop(columns='Loaned From', inplace=True)
data.dropna(axis=0, inplace=True)
ax = msno.bar(data)
# Just use some selected features

feature_cols = ['Age','International Reputation','Reactions','Balance','Weak Foot','Skill Moves','Work Rate','Body Type','Finishing','ShortPassing','Volleys','Dribbling','Curve','BallControl','LongShots']



# The target variable

label_col = 'Value'



# We need this as parameter for the mutual_info_regression method

categorical_cols = dt = [data.dtypes[t] == 'O' for t in data.dtypes.index if t in feature_cols]
# Create a label encoded data frame

data_enc = pd.DataFrame(columns=feature_cols)

data_enc



for col in feature_cols:

    enc = LabelEncoder()

    data_enc[col] = enc.fit_transform(data[col])

    

data_enc[label_col] = enc.fit_transform(data[label_col])
X=data_enc[feature_cols]

y=data_enc["Value"]



print(f'Shape of X: {X.shape}')

print(f'Shape of y: {y.shape}')
mutual_infos=mutual_info_regression(X.values,y.values, discrete_features=categorical_cols)



feature_importances = {}

for i,f in enumerate(feature_cols):

    feature_importances[f] = mutual_infos[i]    
sorted(feature_importances.items(), key=lambda kv: kv[1], reverse=True)