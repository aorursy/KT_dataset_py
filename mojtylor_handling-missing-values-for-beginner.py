import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('../input/titanic/test.csv')
df.shape
new_df = df.copy()
df.head(10)
df.isnull().sum()
print("Orginal shape before dropna()" ,new_df.shape)

drop = new_df.dropna()

print("Sshape after dropna()" ,drop.shape)
# Drop columns based on threshold limit

threshold = len(df) * 0.60

df_thresh=new_df.dropna(axis=1, thresh=threshold)

# View columns in the dataset

df_thresh.shape
# Mean of Age column

mean_age =df['Age'].mean()

median_age=df['Age'].median()

mode_age=df['Age'].mode()
# Alternate way to fill null values with mean

impute_df= df.copy()

impute_df['Age'] = impute_df['Age'].fillna(impute_df['Age'].mean())

impute_df['Age'] = impute_df['Age'].fillna(impute_df['Age'].median())

impute_df['Age'] = impute_df['Age'].fillna(impute_df['Age'].mode())
# Backward fill

df['Age'] = df['Age'].fillna(method='bfill')

# Forward fill

df['Age'] = df['Age'].fillna(method='ffill')
# Replacing the null values in the Age column with Mean

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

# Fit and transform to the parameters

new_df['Age'] = imputer.fit_transform(new_df[['Age']])

# Checking for any null values

new_df['Age'].isnull().sum()