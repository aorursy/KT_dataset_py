# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from sklearn.datasets import load_boston
boston = load_boston()
boston.keys()
boston['data'].shape
print(boston['DESCR'])
bos = pd.DataFrame(boston['data'],columns = boston['feature_names'])
boston['target'].shape
bos['PRICE'] = boston['target']
bos.info()
bos.describe()
from sklearn.model_selection import train_test_split
X = bos.drop(['PRICE'],axis=1)
y = bos['PRICE']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)
predictions = lm.predict(X_test)
from scipy import stats
x = y_test
y = predictions
slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)

idx = ['slope', 'intercept', 'r_value', 'p_value', 'std_err']
data = np.array([slope, intercept, r_value, p_value, std_err])

print(pd.DataFrame(data= data, index= idx , columns =['values']))

regline = lambda S: 0.7466*x +6.0860359
S=np.array([x.min(),x.max()])
plt.figure(figsize=(12,10))
plt.scatter(y_test,predictions,color='Red',marker='*')
plt.plot(x,regline(S),lw=2.5, c="BLUE")
plt.xlabel("Prices",{'size':20})
plt.ylabel("Predictions",{'size':20})
plt.title("Predicted Prices vs Prices,{'size':20}")
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
