# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# very inconvenient using csv model

import csv

train = open('../input/train.csv','r',newline='') # What's the difference between using newline='' and nothing?

train_data = csv.reader(train)    

# load the data in to construct a dataframe

# http://blog.enthought.com/enthought-canopy/with-and-without-the-canopy-data-import-tool-loading-data-theres-no-such-thing-as-a-simple-csv-file/#.Wa69lMiGM2w

df = pd.read_table('../input/train.csv', sep=',') # with header=None, no header

df
# Deal with NaNs

# Method 1: drop axis with NaN

# other methods:

# https://pandas.pydata.org/pandas-docs/stable/missing_data.html#dropping-axis-labels-with-missing-data-dropna

df1 = df.dropna(axis=1)

df1

# Other methods: interpolation, fillna, forward, backward
# change categorical to numerical

cat_columns = df1.select_dtypes(exclude=[np.number]).columns

cat_columns
# Method1: Use cat.codes: works only for type as 'category'

for col in cat_columns:

    df1[col] = df1[col].astype('category')

df1[cat_columns] = df1[cat_columns].apply(lambda x: x.cat.codes)

df1

# Another method: use label encoder

# https://www.analyticsvidhya.com/blog/2015/11/easy-methods-deal-categorical-variables-predictive-modeling/
## normalize data to remove the effect of data scale

# without using sklearn preprocessing:

# needs manipulation on the data type since the data

df2 = df1.copy()

for feature_name in df2.columns:

    max_value = df2[feature_name].max()

    min_value = df2[feature_name].min()

    df2[feature_name] = (df2[feature_name] - min_value) / (max_value - min_value)
df2
# Using sklearn pre-processing

from sklearn import preprocessing



X = df1.values #returns a numpy array

#min_max_scaler = preprocessing.MinMaxScaler()

#X_scaled = min_max_scaler.fit_transform(X)

X_normalized = preprocessing.normalize(X, norm='l2')

df3 = pd.DataFrame(X_normalized)
df4 = pd.DataFrame(data=df3.values, columns=df1.columns,index=df1.index)

df4
df5 = df2.drop(labels='SalePrice', axis=1)
from sklearn import linear_model

reg = linear_model.LinearRegression()

reg.fit(df5,df2['SalePrice'])

reg.coef_