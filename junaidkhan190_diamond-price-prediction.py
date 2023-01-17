# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
# Reading Data

data = pd.read_csv("../input/diamonds/diamonds.csv")
# Displaying data

data.head()
# Removing Unnamed: 0 column



data.drop(columns="Unnamed: 0", axis=1, inplace=True)
# Viewing data again after removing first column

data.head()
data.shape
# EDA



data.info()
# Checking for null data

data.isnull().any()
data.describe()
# Let's see how many zero values we are having in the data



data = data.replace(0.00, np.nan)
data.isnull().any()
data.isnull().sum()
data.dropna(inplace=True)
data.hist(bins=50, figsize=(8,8));
sns.pairplot(data, diag_kind='kde');
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
corr_mat = data.corr()

corr_mat['price'].sort_values(ascending=False).plot(kind='bar');
# Now let's work on the categorical feature.



input_cat_columns = data.select_dtypes(include = ['object']).columns.tolist()



input_cat_columns
for col in input_cat_columns:

    sns.catplot(x=col, y="price", kind='box', dodge=False, height=5, aspect=3, data=data)
data_one_hot_encoding = pd.get_dummies(data)

data_one_hot_encoding.head()
copy = data_one_hot_encoding.copy()
copy.drop(columns=['price','depth','x','y','z'], inplace=True)
copy.head()
X = copy.values

y = data_one_hot_encoding['price'].values
# importing libraries for our model



from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
reg_all = LinearRegression()
reg_all.fit(X_train, y_train)
y_pred = reg_all.predict(X_test)
# Let's find out the R^2 score

print('R^2 : {}'.format(reg_all.score(X_test, y_test)))