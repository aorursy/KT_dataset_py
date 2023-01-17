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
import warnings
warnings.filterwarnings('ignore')
import os
import numpy as np 
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import pyplot as plt 
%matplotlib inline

import sklearn
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LinearRegression 

import tensorflow as tf
from tensorflow import keras 

from sklearn import preprocessing 
os.chdir('..')
data = pd.read_csv('/kaggle/input/climate_hour.csv')
#, parse_dates=['Date Time']
data.head()
data.shape
data.shape
data.isnull().any()
#data.isnull().all()
data.columns
data_num = data[list(data.dtypes[data.dtypes!='object'].index)]
#data_num = pd.DataFrame(data=data_num)
y_data = data_num.pop('T (degC)')
x_data = data_num
#train, test= np.split(data, [int(.8 *len(data))])
x_train, x_test = np.split(x_data, [int(.8 *len(x_data))])
y_train, y_test = np.split(y_data, [int(.8 *len(y_data))])
#x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=4)
x_train.head()
#print('Min date from train set: %s' % x_train['Date Time'].min().date())
#print('Max date from train set: %s' % x_train['Date Time'].max().date())
#print('Min date from test set: %s' % x_test['Date Time'].min().date())
#print('Max date from test set: %s' % x_test['Date Time'].max().date())
model = LinearRegression()
model.fit(x_train, y_train)
prediction = model.predict(x_test)
#error: 
np.mean((prediction-y_test)**2)
pd.DataFrame({'actual':y_test,
              'prediction':prediction,
              'diff':(y_test-prediction)})
import matplotlib.pyplot as plt 

plt.plot(y_test)
plt.show()
plt.plot(prediction)
plt.show()
from sklearn.preprocessing import PolynomialFeatures 

poly = PolynomialFeatures(degree=4)
x_poly = poly.fit_transform(x_train)

poly.fit(x_poly,y_train)
lin2 = LinearRegression()
lin2.fit(x_poly, y_train)
prediction2 = lin2.predict(poly.fit_transform(x_test))
np.mean((prediction2-y_test)**2)
pd.DataFrame({'actual':y_test,
              'prediction':prediction2,
              'diff':(y_test-prediction2)})