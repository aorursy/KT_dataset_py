# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
home_data = pd.read_csv('../input/train.csv')
home_data.head()
# See which columns have nan values
count_nan = home_data.isna().sum()
missing_nan = count_nan[count_nan > home_data.shape[0] * 0.15]

plt.figure()
plt.bar(missing_nan.index,missing_nan)
plt.show()

# Based on the bar charts, the four categories, "Alley", "PoolQC", "Fence", "MiscFeature", have more than 
# 15% of total values are nan values. Therefore, it is okay to remove them from the regression model.
drop_list = ["LotFrontage", "Alley", "FireplaceQu", "PoolQC", "Fence", "MiscFeature"]

copy_home_data = home_data.drop(drop_list, axis=1)
# The distribution of the sales price

plt.figure()
sns.distplot(home_data['SalePrice'])
plt.show()
#correlation matrix
corrmat = copy_home_data.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);
#saleprice correlation matrix
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(copy_home_data[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
# Scatterplot
sns.pairplot(copy_home_data[cols])
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# Impute missing data for the rest of the datas to see if data can be improved
missing_nan_index = count_nan[(count_nan > 0)][(count_nan < 700)]
missing_nan.index

target = copy_home_data.SalePrice
predictors = copy_home_data.drop(['SalePrice'], axis=1)
# include only numeric predictors
numeric_predictors = predictors.select_dtypes(exclude=['object'])

# split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(numeric_predictors, 
                                                    target,
                                                    train_size=0.7, 
                                                    test_size=0.3, 
                                                    random_state=0)

def score_dataset(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return mean_absolute_error(y_test, preds)

numeric_predictors
count_nan = numeric_predictors.isna().sum()
count_nan

# Without Imputing Numeric Predictors

# Imputing Numeric Predictors
