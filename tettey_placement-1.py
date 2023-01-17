# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LinearRegression
#import the dataset
data = pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')
data.head()
data.shape
# the predictor variables
data_features = ['ssc_p','hsc_p']
X = data[data_features]
#the response variable
y = data.mba_p
model = LinearRegression()
model.fit(X,y)
r2 = model.score(X,y)
#print the coefficients
print(f'alpha = {model.intercept_}')
print(f'Betas = {model.coef_}')
import statsmodels.api as sm
mod = sm.OLS(y,X)
fitmodel = mod.fit()
fitmodel.pvalues
data_features_1 = ['ssc_p','degree_p']
X1 = data[data_features_1]
data_features_2 = ['hsc_p','degree_p']
X2 = data[data_features_2]
#fitting the model
MR = model.fit(X1,y)
MR2 = model.fit(X2,y)
r2_MR = model.score(X1,y)
r2_MR2 = model.score(X2,y)
print(r2, r2_MR, r2_MR2)
data_features_3 = ['ssc_p','hsc_p','degree_p']
X3 = data[data_features_3]
train_X, test_X, train_y, test_y = train_test_split(X3, y, test_size=0.2, random_state=1)
model.fit(train_X, train_y)
print(f'alpha = {model.intercept_}')
print(f'betas = {model.coef_}')
mod2 = sm.OLS(train_y,train_X)
fitmodel2 = mod2.fit()
fitmodel2.pvalues
data_features_3 = ['hsc_p','degree_p']
X3 = data[data_features_3]
train_X, test_X, train_y, test_y = train_test_split(X3, y, test_size=0.2, random_state=1)
model.fit(train_X, train_y)
pred = model.predict(test_X)
result = pd.DataFrame({'Actual': test_y, 'Predicted': pred})
result.head()