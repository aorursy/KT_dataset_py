# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, HuberRegressor, Ridge
import sklearn.metrics as metrics
from sklearn.model_selection import cross_val_predict

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
raw_data = pd.read_csv('../input/avocado.csv', encoding='utf-8', index_col='Date')
raw_data.head()
n_regions = raw_data.region.nunique()
regions = raw_data.region.unique()
print(n_regions, regions, sep="\n")
#x = regions[0]
place = 'Albany'
region = raw_data[raw_data.region == place]
region = region.sort_index()
price = region.AveragePrice

x= range(0,len(region.index))
y=price
plt.plot(x,y)
plt.show()
X = np.array(x)
X = X.reshape(-1,1)
X_train, y_train, X_test, y_test =  X[:150], y[:150], X[150:], y[150:] #division of training and test sets
models = [LinearRegression(), HuberRegressor(), Ridge()]
results = []
for model in models:
    predicted = cross_val_predict(model, X_train, y_train, cv= 5)
    testing = {'model':model,'cross_validation_method_1':metrics.mean_squared_error(y_pred=predicted, y_true=y_train)}
    model.fit(X_train, y_train)
    testing['singular_testing_method'] = metrics.mean_squared_error((model.predict(X_test)),y_test)
    testing['cross_validation_method_2'] = metrics.mean_squared_error(y_pred=cross_val_predict(model, X_test, y_test, cv= 5), y_true=y_test)
    results.append(testing)
    
for result in results:
    print("Model: {0},\n cross_validation_method_1: {1},\n singular_testing_method:{2},\n cross_validation_method_2: {3} \n".format(result['model'],result['cross_validation_method_1'],result['singular_testing_method'],result['cross_validation_method_2']))
    
