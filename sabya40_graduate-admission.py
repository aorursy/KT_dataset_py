# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import math

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
dataset = pd.read_csv('/kaggle/input/graduate-admissions/Admission_Predict.csv')
dataset.head(5)
data_prep = dataset.copy()
data_prep = data_prep.drop(['Serial No.'], axis = 1)
data_prep.head(5)
data_prep.isnull().sum()
data_prep.hist(rwidth = 0.9)
plt.tight_layout()
plt.subplot(3,3,1)
plt.title('CGPA Vs Chance Of Admit')
plt.scatter(data_prep['CGPA'], data_prep['Chance of Admit '], s= 2)

plt.subplot(3,3,2)
plt.title('GRE Score Vs Chance Of Admit')
plt.scatter(data_prep['GRE Score'], data_prep['Chance of Admit '], s= 2)

plt.subplot(3,3,3)
plt.title('LOR Vs Chance Of Admit')
plt.scatter(data_prep['LOR '], data_prep['Chance of Admit '], s= 2)

plt.subplot(3,3,4)
plt.title('Research Vs Chance Of Admit')
plt.scatter(data_prep['Research'], data_prep['Chance of Admit '], s= 2)

plt.subplot(3,3,5)
plt.title('SOP Vs Chance Of Admit')
plt.scatter(data_prep['SOP'], data_prep['Chance of Admit '], s= 2)

plt.subplot(3,3,6)
plt.title('TOEFL Vs Chance Of Admit')
plt.scatter(data_prep['TOEFL Score'], data_prep['Chance of Admit '], s= 2)

plt.subplot(3,3,7)
plt.title('University Rating Vs Chance Of Admit')
plt.scatter(data_prep['University Rating'], data_prep['Chance of Admit '], s= 2)
plt.tight_layout()
data_prep['Chance of Admit '].describe()
data_prep['Chance of Admit '].quantile([0.05, 0.1, 0.15, 0.9, 0.95, 0.99])
data_prep.columns
corr = data_prep[['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ',
                 'CGPA', 'Research', 'Chance of Admit ']].corr()

corr
data_prep = data_prep.drop(['TOEFL Score'], axis = 1)
data_prep.head(5)
df = pd.to_numeric(data_prep['Chance of Admit '], downcast = 'float')
plt.acorr(df, maxlags = 12)
df = data_prep['Chance of Admit ']
df2 = np.log(data_prep['Chance of Admit '])

plt.figure()
df.hist(rwidth = 0.9, bins = 20)

plt.figure()
df2.hist(rwidth = 0.9, bins = 20)
data_prep.head()
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data_prep['GRE Score']=scaler.fit_transform(data_prep[['GRE Score']])
data_prep.head()
Y = data_prep[['Chance of Admit ']]
X = data_prep.drop(['Chance of Admit '], axis = 1)

tr_size = int(0.7 * len(X))

X_train = X.values[0: tr_size]
X_test = X.values[tr_size : len(X)]

Y_train = Y.values[0: tr_size]
Y_test = Y.values[tr_size : len(Y)]
from sklearn.linear_model import LinearRegression

std_reg = LinearRegression()
std_reg.fit(X_train, Y_train)

r_sqr_train = std_reg.score(X_train, Y_train)
r_sqr_test = std_reg.score(X_test, Y_test)

Y_predict = std_reg.predict(X_test)

from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(Y_test, Y_predict))


log_sq_sum = 0.0
for i in range(0, len(Y_test)):
    log_a = math.log(Y_test[i] + 1)
    log_p = math.log(Y_predict[i] + 1)
    log_diff = (log_p - log_a)**2
    log_sq_sum = log_sq_sum + log_diff
    
rmsle = math.sqrt(log_sq_sum/len(Y_test))    
print('R Squared Value Train', r_sqr_train)
print('R Squared Value Test', r_sqr_test)
print('RMSE', rmse)
print("RMSLE", rmsle)
