# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/column_2C_weka.csv')
#First 5 row
df.head()
#Summary of data
df.info()
distrubution = df['class'].value_counts()
distrubution.plot(kind='pie',explode=(0, 0.1),autopct='%1.1f%%')
plt.show()
sns.pairplot(df, hue='class', markers=["o", "s"], diag_kind="kde")
sns.set_palette(sns.color_palette("Set1"))
sns.set(font_scale=1.3)
plt.show()
# Linear Regression Work
# Feature : pelvic_incidence, Target: sacral_slope

data_abnormal = df[df['class'] == 'Abnormal']
plt.scatter(data_abnormal.pelvic_incidence,data_abnormal.sacral_slope, color='m')

plt.title('Sacral Slope Changes with Pelvic Incidence')
plt.xlabel('Pelvic Incidence')
plt.ylabel('Sacral Slope')
plt.show()
from sklearn import linear_model

#linear regression model
linear_reg = linear_model.LinearRegression()

#Seperate data to train and test
#Assume first 100 data is train data and the other part is test data

x_train = data_abnormal.pelvic_incidence[:100].values.reshape(-1,1)
y_train = data_abnormal.sacral_slope[:100].values.reshape(-1,1)

linear_reg.fit(x_train, y_train)
#%% Prediction with last 100 data
y_head = linear_reg.predict(data_abnormal.pelvic_incidence[100:].values.reshape(-1,1))
y_test = data_abnormal.sacral_slope[100:].values.reshape(-1,1)

print('r2_score: ',r2_score(y_test, y_head))
x_test = data_abnormal.pelvic_incidence[100:].values.reshape(-1,1)

plt.plot(x_test, y_head, color='black', linewidth=3)
plt.scatter(x_train,y_train)
plt.xlabel('pelvic_incidence')
plt.ylabel('sacral_slope')
plt.show()
#linear regression model
linear_reg = linear_model.LinearRegression()

#This time we use all data for training and testing

x_train = data_abnormal.pelvic_incidence.values.reshape(-1,1)
y_train = data_abnormal.sacral_slope.values.reshape(-1,1)

linear_reg.fit(x_train, y_train)
#%% Prediction with all data
y_head = linear_reg.predict(data_abnormal.pelvic_incidence.values.reshape(-1,1))
y_test = data_abnormal.sacral_slope.values.reshape(-1,1)
print('r2_score: ',r2_score(y_test, y_head))
x_test = data_abnormal.pelvic_incidence.values.reshape(-1,1)

plt.plot(x_test, y_head, color='black', linewidth=3)
plt.scatter(x_train,y_train)
plt.xlabel('pelvic_incidence')
plt.ylabel('sacral_slope')
plt.show()