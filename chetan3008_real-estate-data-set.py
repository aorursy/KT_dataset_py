# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
a=[]
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        a=os.path.join(dirname, filename)
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv(a,index_col='No')
data.describe()
data.boxplot()
import seaborn as sns
%matplotlib inline
sns.distplot(data['X3 distance to the nearest MRT station'])
#data = data[data['X3 distance to the nearest MRT station']<=5000]
sns.distplot(data['X3 distance to the nearest MRT station'])
data.info()
data.head()
#data_modified = data.drop(['X5 latitude','X6 longitude'],axis=1)
data_modified = data
data_modified.corr()['Y house price of unit area']
X = data_modified.drop('Y house price of unit area',axis=1)
X=pd.get_dummies(X)
y = data_modified['Y house price of unit area']
X
y
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
scalerX=MinMaxScaler(feature_range=(0,1))
X_train[X_train.columns]=scalerX.fit_transform(X_train[X_train.columns])
X_test[X_test.columns]=scalerX.transform(X_test[X_test.columns])
lm = LinearRegression()
lm.fit(X_train,y_train)
predict = lm.predict(X_test)
coeff_df = pd.DataFrame(lm.coef_,X_train.columns,columns=['Coefficient'])
coeff_df
import matplotlib.pyplot as plt
plt.scatter(y_test,predict)
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, predict))
print('MSE:', metrics.mean_squared_error(y_test, predict))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predict)))
lm.score(X_test,y_test)
from sklearn.tree import DecisionTreeRegressor
pz=DecisionTreeRegressor(max_depth=3)
pz.fit(X_train,y_train)
pz.score(X_test,y_test)

