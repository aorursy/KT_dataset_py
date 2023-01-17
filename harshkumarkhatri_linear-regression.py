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
# This is code is the one which i have tested in my pc and added it here once the model worked correctly

from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
dataset=pd.read_csv("Linear Regression - Sheet1.csv")
print(dataset.head())
print(np.shape(dataset))
X=dataset['X'].values
y=dataset['Y'].values
print(np.shape(X))
print(np.shape(y))
X=X.reshape(-1,1)
y=y.reshape(-1,1)
print(np.shape(X))
print(np.shape(y))
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.33)
reg=linear_model.LinearRegression()
reg.fit(x_train,y_train)
plt.scatter(x_train,y_train,c='y')
preds=reg.predict(x_test)
plt.plot(x_test,preds,'r',linewidth=2)
plt.show()
print("Score for training \n")
print(reg.score(x_train,y_train))
print("Score ffor testing data")
print(reg.score(x_test,y_test))