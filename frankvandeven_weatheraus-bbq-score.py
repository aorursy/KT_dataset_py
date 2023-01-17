# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn import linear_model, metrics

from sklearn.model_selection import train_test_split 
dataframe = pd.read_csv('/kaggle/input/weatheraus/weatherAUS.csv')

dataframe_clean = dataframe.dropna(how='any')

dataframe_clean['Date'] = pd.to_datetime(dataframe_clean['Date'])

dataframe_clean.info()
dataframe_clean[['Rainfall', 'Sunshine', 'WindSpeed3pm', 'Humidity3pm', 'Cloud3pm', 'Temp3pm']].describe()



dataframe_clean = dataframe_clean[dataframe_clean['Rainfall'] < dataframe_clean['Rainfall'].quantile(0.95)]



bbq_cijfer_determinants = ['Rainfall', 'Sunshine', 'WindSpeed3pm', 'Humidity3pm', 'Cloud3pm', 'Temp3pm']



dataframe_clean['rf_norm'] = (dataframe_clean['Rainfall'] - dataframe_clean['Rainfall'].min()) / (dataframe_clean['Rainfall'].max() - dataframe_clean['Rainfall'].min())

dataframe_clean['ss_norm'] = (dataframe_clean['Sunshine'] - dataframe_clean['Sunshine'].min()) / (dataframe_clean['Sunshine'].max() - dataframe_clean['Rainfall'].min())

dataframe_clean['ws_norm'] = (dataframe_clean['WindSpeed3pm'] - dataframe_clean['WindSpeed3pm'].min()) / (dataframe_clean['WindSpeed3pm'].max() - dataframe_clean['Rainfall'].min())

dataframe_clean['hm_norm'] = (dataframe_clean['Humidity3pm'] - dataframe_clean['Humidity3pm'].min()) / (dataframe_clean['Humidity3pm'].max() - dataframe_clean['Rainfall'].min())

dataframe_clean['cl_norm'] = (dataframe_clean['Cloud3pm'] - dataframe_clean['Cloud3pm'].min()) / (dataframe_clean['Cloud3pm'].max() - dataframe_clean['Rainfall'].min())

dataframe_clean['tp_norm'] = (dataframe_clean['Temp3pm'] - dataframe_clean['Temp3pm'].min()) / (dataframe_clean['Temp3pm'].max() - dataframe_clean['Rainfall'].min())



dataframe_clean.head()



dataframe_clean['bbq_score'] = (((-3 * dataframe_clean['rf_norm']  +

                                   2 * dataframe_clean['ss_norm']  +

                                  -2 * dataframe_clean['ws_norm']  +

                                  -2 * dataframe_clean['hm_norm']  +

                                  -2 * dataframe_clean['cl_norm']  +

                                   2 * dataframe_clean['tp_norm']) + 9) / 1.3)

dataframe_clean.head()



dataframe_clean['bbq_score'].describe()



dataframe_clean[dataframe_clean['bbq_score'] == dataframe_clean['bbq_score'].min()]



dataframe_adj = dataframe_clean.loc[:, 'Date':'bbq_score']

dataframe_adj.head()



dataframe_adj = dataframe_adj.drop('RISK_MM', axis=1)

dataframe_adj = dataframe_adj.drop('RainTomorrow', axis=1)

dataframe_adj.head()



dataframe_adj = dataframe_adj.set_index('Date')



dataframe_adj.head()



dataframe_adj['Location'] = dataframe_adj['Location'].astype('category')

dataframe_adj['WindDir3pm'] =  dataframe_adj['WindDir3pm'].astype('category')

dataframe_adj['RainToday'] =  dataframe_adj['RainToday'].astype('category')



plt.figure(figsize=(15, 5))

dataframe_adj.groupby('Location')['bbq_score'].max().plot(label='Max BBQ Score')

dataframe_adj.groupby('Location')['bbq_score'].mean().plot(label='Mean BBQ Score')

dataframe_adj.groupby('Location')['bbq_score'].min().plot(label='Min BBQ Score')

plt.yticks(range(0,11, 1))

plt.ylabel('BBQ score')

plt.legend(loc='center right')

plt.show()



plt.hist(dataframe_adj['bbq_score'], bins=50)

plt.show()



dataframe_adj.head()



with_dummies = pd.concat([dataframe_adj, pd.get_dummies(dataframe_adj[['Location', 'WindDir3pm', 'RainToday']])], axis=1)



with_dummies['bbq_score_tomorrow'] = with_dummies['bbq_score'].shift(-1)

with_dummies = with_dummies.drop(['bbq_score'], axis=1)

with_dummies



print(with_dummies.columns)



X = with_dummies.loc[:, ['rf_norm', 'ss_norm', 'ws_norm', 'hm_norm', 'cl_norm', 'tp_norm']]

y = with_dummies.loc[:, ['bbq_score_tomorrow']]

y = y.dropna(how='all')

X = X.reset_index()

X = X.drop(index=53573)

X = X.set_index('Date')



reg = linear_model.LinearRegression()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)



reg.fit(X_train, y_train)

print('Coefficients: \n', reg.coef_, reg.intercept_)



print('R-squared: {}'.format(reg.score(X_test, y_test))) 

  

# plot for residual error



## plotting residual errors in training data 

plt.figure(figsize=(15, 10))

plt.scatter(reg.predict(X_train), reg.predict(X_train) - y_train, 

            color = "green", s = .15, label = 'Train data') 

  

## plotting residual errors in test data 

plt.scatter(reg.predict(X_test), reg.predict(X_test) - y_test, 

            color = "blue", s = .15, label = 'Test data') 



## plotting legend 

plt.legend(loc = 'upper right') 

  

## plot title 

plt.title("Residual errors") 

  

## function to show plot 

plt.show() 