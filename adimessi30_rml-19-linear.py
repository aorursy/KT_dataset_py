# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Reading the input file

df = pd.read_csv('/kaggle/input/housesalesprediction/kc_house_data.csv')

df.describe().abs()
df.info()
import seaborn as sns

import matplotlib.pyplot as plt
sns.set_style(style='ticks')

sns.pairplot(df)
# Filtering down only the required independent variables

subset_df = df[['sqft_living', 'price', 'bathrooms', 'grade',

       'sqft_above', 'sqft_basement', 'sqft_living15']]
plt.figure(figsize=(12, 10))

sns.heatmap(subset_df.corr().abs(), annot=True)
#Removing sqft_basement from the filtered dataframe

del subset_df['sqft_basement']

print(subset_df.columns)
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split
X_1 = subset_df[['sqft_above']]

y_1 = subset_df['sqft_living']
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X_1, y_1, test_size=0.2, random_state=123)
simple_linear_model = LinearRegression().fit(X_train_1, y_train_1)
x_test_value = X_test_1.iloc[1]

y_pred = simple_linear_model.predict([[1050]])

y_actual = y_test_1.iloc[1]

residual = y_pred - y_actual

print(residual)
print('Model coefficients', simple_linear_model.coef_)

print('Model intercept', simple_linear_model.intercept_)

r2_score_1 = simple_linear_model.score(X_test_1, y_test_1)

print('R square', r2_score_1)

print('Adjust R square', 1-(1-r2_score_1)*(len(X_train_1)-1)/(len(X_train_1)-1-1))
X_2 = subset_df[['price', 'bathrooms', 'grade',

       'sqft_above', 'sqft_living15']]

y_2 = subset_df['sqft_living']
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_2, y_2, test_size=0.2, random_state=123)
multi_linear_model_1 = LinearRegression().fit(X_train_2, y_train_2)
print('Model coefficients', multi_linear_model_1.coef_)

print('Model intercept', multi_linear_model_1.intercept_)

r2_score_2 = multi_linear_model_1.score(X_test_2, y_test_2)

print('R square', r2_score_2)

print('Adjust R square', 1-(1-r2_score_2)*(len(X_train_2)-1)/(len(X_train_2)-len(X_train_2.columns)-1))
X_3 = df[[x for x in df.columns if x not in ['id', 'date', 'sqft_living', 'yr_built', 'yr_renovated', 'zipcode',

       'lat', 'long']]]

y_3 = df['sqft_living']
X_train_3, X_test_3, y_train_3, y_test_3 = train_test_split(X_3, y_3, test_size=0.2, random_state=123)
multi_linear_model_2 = LinearRegression().fit(X_train_3, y_train_3)
print('Model coefficients', multi_linear_model_2.coef_)

print('Model intercept', multi_linear_model_2.intercept_)

r2_score_3 = multi_linear_model_2.score(X_test_3, y_test_3)

print('R square', r2_score_3)

print('Adjust R square', 1-(1-r2_score_3)*(len(X_train_3)-1)/(len(X_train_3)-len(X_train_3.columns)-1))