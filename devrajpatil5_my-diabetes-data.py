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
import pandas as pd
data = pd.read_csv("../input/Diab_pyth_data.csv")
data.head()
## import all libraries
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import math

import matplotlib as mpl

from mpl_toolkits.mplot3d import Axes3D 

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn import preprocessing

from sklearn.base import TransformerMixin

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

import statsmodels.api as sm

from sklearn.feature_selection import RFE

from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.metrics import r2_score

from sklearn.tree import DecisionTreeRegressor

## data Shape
data.shape
## Describe Data





data.describe
##Treat Missing Values



new_data= data.dropna(subset= ['Glucose Before fasting','Glucose Anytime','BMI', 'Blood Pressure', 'Age', 'Sex', 'Family member with Diabetes past or present', 'Pregnancies'],how = 'any')

new_data.isnull().sum(axis=0)
## New data Describe



new_data['Glucose Before fasting'].describe()
#Box Plot otgether



new_data.plot(kind= 'box')
#Correlation Matrix



plt.figure(figsize= (12,10))

sns.heatmap(new_data.corr(), annot = True);

plt.title("Correlation Heatmap")
#Histogram and Scatter matrix





pd.plotting.scatter_matrix(new_data,figsize= (12,10))
#Remove unwanted variables



new_data = new_data.drop('Sex', axis= 1)
new_data.head
X= new_data.iloc[:, :7]

Y = new_data.iloc[:, 7:8]
X.head
Y.head
#train test Split



X_train, X_test, Y_train, Y_test = train_test_split (X, Y, train_size = 0.7, test_size = 0.3)
#Build Linear Regression Model





from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train, Y_train)
import statsmodels.api as sm

model = sm.OLS(Y_train, X_train).fit()

predictions = model.predict(X_train)

model.summary()
# check Mullticolinearity



from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()

vif["VIF Factor"] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]



vif["features"] = X_train.columns

vif
## VIF should be less than 10, so we remove the variable above 10

X_train = X_train.drop('Glucose Before fasting', axis= 1)
X_train.head()
model = sm.OLS(Y_train, X_train).fit()

predictions = model.predict(X_train)

model.summary()
##check multicollinearity
from statsmodels.stats.outliers_influence import variance_inflation_factor





vif = pd.DataFrame()

vif["VIF Factor"] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]



vif["features"] = X_train.columns

vif
##model is perfectly correct and without mullticolinearity



### run model on test
X_test = X_test.drop('Glucose Before fasting', axis= 1)
model = sm.OLS(Y_test, X_test).fit()

predictions = model.predict(X_test)

model.summary()
from statsmodels.stats.outliers_influence import variance_inflation_factor





vif = pd.DataFrame()

vif["VIF Factor"] = [variance_inflation_factor(X_test.values, i) for i in range(X_test.shape[1])]



vif["features"] = X_test.columns

vif
## our model is perfectwithout multicollinearity