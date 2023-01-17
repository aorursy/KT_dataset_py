# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
mpg_df = pd.read_csv("../input/auto-mpg.csv")
mpg_df.head(50)
mpg_df = mpg_df.drop('car name', axis = 1)
mpg_df.head()
mpg_df['origin'] = mpg_df['origin'].replace({1: 'america', 2: 'europe', 3: 'asia'})
mpg_df.info()
mpg_df = pd.get_dummies(mpg_df, columns=['origin'])
mpg_df
mpg_df.describe().transpose()
mpg_df.dtypes
temp = pd.DataFrame(mpg_df.horsepower.str.isdigit())
temp[temp['horsepower'] == False]
mpg_df = mpg_df.replace('?', np.nan)
mpg_df[mpg_df.isnull().any(axis=1)]
mpg_df.median()
mpg_df = mpg_df.apply(lambda x: x.fillna(x.median()),axis=0)
mpg_df['horsepower'] = mpg_df['horsepower'].astype('float64')
mpg_df.describe()
mpg_df_attr = mpg_df.iloc[:, 0:12]
sns.pairplot(mpg_df_attr, diag_kind='kde')
%matplotlib inline
X = mpg_df.drop('mpg', axis=1)
X = X.drop({'origin_1', 'origin_2' ,'origin_3'}, axis=1)
X.head()
y = mpg_df[['mpg']]
y.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30 , random_state=1)
regression_model = LinearRegression()
regression_model.fit(X_train, y_train)
for idx, col_name in enumerate(X_train.columns):
    print("The coefficient for {} is {}".format(col_name, regression_model.coef_[0][idx]))
intercept = regression_model.intercept_[0]

print("The intercept for our model is {}".format(intercept))
regression_model.score(X_train, y_train)
regression_model.score(X_test, y_test)