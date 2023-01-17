# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sklearn

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

sns.set()

pd.set_option('max_columns', 1000)

warnings.filterwarnings('ignore')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



df_train = pd.read_csv("../input/train.csv")



#Clean Missing data

num_missing = df_train.isnull().sum()

percent = num_missing / df_train.isnull().count()

df_missing = pd.concat([num_missing, percent], axis=1, keys=['MissingValues', 'Fraction'])

df_missing = df_missing.sort_values('Fraction', ascending=False)

df_missing[df_missing['MissingValues'] > 0]



#remove variable with a missing a colum



variables_to_keep = df_missing[df_missing['MissingValues'] == 0].index

df_train = df_train[variables_to_keep]





# Build the correlation matrix to see variables highly correlated with price

matrix = df_train.corr()

f, ax = plt.subplots(figsize=(16, 12))

sns.heatmap(matrix, vmax=0.7, square=True)



Variables_of_focus = matrix['SalePrice'].sort_values(ascending=False)

# Filter out the target variables (SalePrice) and variables with a low correlation score (v such that -0.6 <= v <= 0.6)

Variables_of_focus = Variables_of_focus[abs(Variables_of_focus) >= 0.6]

Variables_of_focus = Variables_of_focus[Variables_of_focus.index != 'SalePrice']

Variables_of_focus



#Overall quality seems highly correlated with sales price.Plot them together

data = pd.concat([df_train['SalePrice'], df_train['OverallQual']], axis=1)

data.plot.scatter(x='OverallQual', y='SalePrice')





#Analysing other variables of interest



cols = Variables_of_focus.index.values.tolist() + ['SalePrice']

sns.pairplot(df_train[cols], size=2.5)

plt.show()



#Draw correlationm matrix only with variables of interest

matrix = df_train[cols].corr()

f, ax = plt.subplots(figsize=(8, 6))

sns.heatmap(matrix, vmax=1.0, square=True)



#Build Random Forest Model



from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split



pred_vars = [v for v in Variables_of_focus.index.values if v != 'SalePrice']

target_var = 'SalePrice'



X = df_train[pred_vars]

y = df_train[target_var]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)



model = RandomForestClassifier(n_estimators=100, random_state=42)

model.fit(X_train, y_train)



y_pred = model.predict(X_test)

# Build a plot

plt.scatter(y_pred, y_test)

plt.xlabel('Prediction')

plt.ylabel('Real value')



# Now add the perfect prediction line

diagonal = np.linspace(0, np.max(y_test), 100)

plt.plot(diagonal, diagonal, '-r')

plt.show()



#Calculate Root mean square error

from sklearn.metrics import mean_squared_log_error, mean_absolute_error

print('MAE:\t$%.2f' % mean_absolute_error(y_test, y_pred))

print('MSLE:\t%.5f' % mean_squared_log_error(y_test, y_pred))





# Any results you write to the current directory are saved as output.