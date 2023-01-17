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
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston

import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LinearRegression
boston = load_boston()
print(boston.data)
bos = pd.DataFrame(boston.data)
bos.columns = boston.feature_names
bos['price'] = boston.target
X=bos.drop('price', axis=1)
model = LinearRegression()
model.fit(X,bos.price)
print('number of slopes',len(model.coef_))
print('slopes are ',model.coef_)
print('Intercept is ',model.intercept_)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(boston.data,boston.target,test_size=0.3 ,random_state = 5)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
model = LinearRegression()
model.fit(x_train,y_train)
pred = model.predict(x_test)
print('mse of the model is ',metrics.mean_squared_error(pred,y_test))
print('R2 of the model is ',metrics.r2_score(pred,y_test))
plt.scatter(bos.RM,bos.price)
plt.xlabel('average number of rooms per dwelling')
plt.ylabel('price of the house')
plt.title('Relationship bw price and no of rooms')
plt.show()