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
data=pd.read_csv('/kaggle/input/melanoma-data/Train.csv')
test=pd.read_csv('/kaggle/input/melanoma-data/Test.csv')

data.head()
import seaborn as sns
import matplotlib.pyplot as plt

data_corr=data.corr(method='spearman')

plt.figure(figsize=(10,7))
sns.heatmap(data=data_corr , annot=True)
from scipy.stats import skew

data.skew()
(np.log1p(data['err_malign'])).skew()

X=data.drop(columns=['tumor_size'])
Y=data['tumor_size']
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.1,random_state=7)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import MinMaxScaler

pw=PowerTransformer()
ms=MinMaxScaler()

X_train=ms.fit_transform(X_train)
X_test=ms.transform(X_test)

Y_train=Y_train.values.reshape(-1,1)
Y_test=Y_test.values.reshape(-1,1)

Y_train=pw.fit_transform(Y_train)
Y_test=pw.transform(Y_test)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

l_model=LinearRegression()

l_model.fit(X_train,Y_train)

y_prediction=l_model.predict(X_test)

rmse=np.sqrt(mean_squared_error(y_prediction,Y_test))

print(rmse)
data['ratio']=data['damage_size']/data['exposed_area']

data['ratio_mean']=(data['ratio']+data['damage_ratio'])/2
import seaborn as sns
plt.figure(figsize=(10,7))
cor_mat= data.corr(method='spearman')
sns.heatmap(data=cor_mat,annot=True)
data.drop(columns=['damage_size','exposed_area','ratio_mean'],inplace=True)
import seaborn as sns
plt.figure(figsize=(10,7))
cor_mat= data.corr(method='spearman')
sns.heatmap(data=cor_mat>.7,annot=True)
X=data.drop(columns=['tumor_size'])
Y=data['tumor_size']
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.1,random_state=7)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

#from sklearn.preprocessing import PowerTransformer
#from sklearn.preprocessing import MinMaxScaler

#pw=PowerTransformer()
#ms=MinMaxScaler()

#X_train=ms.fit_transform(X_train)
#X_test=ms.transform(X_test)

#Y_train=Y_train.values.reshape(-1,1)
#Y_test=Y_test.values.reshape(-1,1)

#Y_train=pw.fit_transform(Y_train)
#Y_test=pw.transform(Y_test)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

l_model=LinearRegression()

l_model.fit(X_train,Y_train)

y_prediction=l_model.predict(X_test)

rmse=np.sqrt(mean_squared_error(y_prediction,Y_test))

print(rmse)
from sklearn.linear_model import Ridge
R_model=Ridge(alpha=0.3,random_state=2,solver='sag')
R_model.fit(X_train,Y_train)

y_test_predict=R_model.predict(X_test)

rmse=np.sqrt(mean_squared_error(Y_test,y_test_predict))


print('RMSE={}'.format(rmse))
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators =900)

model.fit(X_train, Y_train)
y_test_predict=model.predict(X_test)

rmse=np.sqrt(mean_squared_error(Y_test,y_test_predict))


print('RMSE={}'.format(rmse))
test=pd.read_csv('/kaggle/input/melanoma-data/Test.csv')
test.head()
test['ratio']=test['damage_size']/test['exposed_area']
test.drop(columns=['damage_size','exposed_area'],inplace=True)
test.head()
y_test=model.predict(test)
print(y_test)
sample=pd.read_csv('/kaggle/input/melanoma-data/sample_submission.csv')
sample
sample['tumor_size'] = y_test
sample.to_csv('y_test.csv',index=False)
rmse=np.sqrt(mean_squared_error(sample,y_test))
print(rmse)