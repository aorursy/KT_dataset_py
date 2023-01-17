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
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import matplotlib
matplotlib.rcParams.update({'font.size': 12})
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
data = pd.read_csv('../input/housesalesprediction/kc_house_data.csv')
data.head()
data.describe()
y=np.array(data['price'])
x=data.drop(['id','price','date'],axis=1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
alpha_ridge = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]
from sklearn.metrics import mean_squared_error as mse
score = []
for alpha in alpha_ridge:
    #alpha = alpha_ridge[i]
    rr = Ridge(alpha = alpha)
    rr.fit(x_train,y_train)
    y_pred = rr.predict(x_test)
    Ridge_train_score = rr.score(x_train,y_train)
    Ridge_test_score = rr.score(x_test, y_test)
    ridge_mse_error = mse(y_pred,y_test,squared=False)
    print("Alpha:%.15f"%alpha)
    print("Mean Square Error:",ridge_mse_error)
    print("Ridge regression train score:", Ridge_train_score)
    print("Ridge regression test score:", Ridge_test_score)
    score.append(ridge_mse_error)   
import matplotlib.pyplot as plt
plt.plot(alpha_ridge,score)
plt.xlabel('Alpha')
plt.ylabel('RMSE')
alpha_lasso = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]
from sklearn.linear_model import Lasso 
score=[]
for i in range(10):
    alpha = alpha_lasso[i]
    lassoreg = Lasso(alpha = alpha,normalize=True,max_iter=1e5)
    lassoreg.fit(x_train,y_train)
    y_pred = lassoreg.predict(x_test)
    Lasso_train_score = lassoreg.score(x_train,y_train)
    Lasso_test_score = lassoreg.score(x_test, y_test)
    Lasso_mse_error = mse(y_pred,y_test,squared=False)
    print("Alpha:%.15f"%alpha)
    print("Mean Square Error:",Lasso_mse_error)
    print("Ridge regression train score:", Lasso_train_score)
    print("Ridge regression test score:", Lasso_test_score)
    score.append(Lasso_mse_error)  
    
import matplotlib.pyplot as plt
plt.plot(alpha_lasso,score)
plt.xlabel('Alpha')
plt.ylabel('RMSE')
