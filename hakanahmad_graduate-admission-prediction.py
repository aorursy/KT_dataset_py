# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/graduate-admissions/Admission_Predict_Ver1.1.csv')

data.head()
data.drop('Serial No.',axis=1,inplace=True)
data.isna().sum()
sns.pairplot(data.iloc[:,[0,1,3,4,5,7]])

plt.show()
data_num = data.iloc[:,[0,1,3,4,5,7]]

num_corr = data_num.corr()

sns.heatmap(num_corr,annot=True)
data.drop(['GRE Score','TOEFL Score'],axis=1,inplace=True)
data.head()
sns.countplot(data['University Rating'])
sns.countplot(data['Research'])
X_train = data.iloc[:-100,:-1]

y_train = data.iloc[:-100,-1]

X_test = data.iloc[-100:,:-1]

y_test = data.iloc[-100:,-1]

X_train.shape,X_test.shape
from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error



lr = LinearRegression(fit_intercept=False)

lr.fit(X_train,y_train)

lr_pred = lr.predict(X_test)

print('The RMSE with Linear Regression: {}'.format(mean_squared_error(y_test,lr_pred)**(1/2)))
lr_coef = pd.DataFrame({'column':X_train.columns,'coef':lr.coef_})

lr_coef
from sklearn.model_selection import GridSearchCV

rf = RandomForestRegressor()

param_grid = {'n_estimators':np.arange(100,1100,100),

             'max_depth':np.arange(2,11),

             'criterion':['mse']}

clf = GridSearchCV(estimator=rf,param_grid=param_grid,verbose=1)

clf.fit(X_train,y_train)
print(clf.best_params_)

print(clf.best_score_)
rf_best = clf.best_estimator_

rf_best.fit(X_train,y_train)

rf_pred = rf_best.predict(X_test)

print('The RMSE with Random Forest: {}'.format(mean_squared_error(y_test,rf_pred)**(1/2)))
feature_importance = pd.DataFrame({'column':X_train.columns,'feature importance':rf_best.feature_importances_})

feature_importance
import tensorflow

from tensorflow.keras.models import Model

from tensorflow.keras.layers import Input, Dense



inputs = Input(shape=(5))

x = Dense(16,activation='linear')(inputs)

x = Dense(16,activation='linear')(x)

x = Dense(16,activation='linear')(x)

outputs = Dense(1,activation='sigmoid')(x)



model = Model(inputs,outputs)

model.summary()
model.compile(optimizer='adam',loss='mse')

model.fit(X_train,y_train,epochs=200,validation_split=0.3,verbose=0)
nn_pred = model.predict(X_test)

print('The RMSE with Neural Network: {}'.format(mean_squared_error(y_test,nn_pred)**(1/2)))
print('The RMSE with Random Forest: {}'.format(mean_squared_error(y_test,rf_pred)**(1/2)))