# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        



# Any results you write to the current directory are saved as output.
df=pd.read_csv('/kaggle/input/graduate-admissions/Admission_Predict_Ver1.1.csv').drop('Serial No.',axis=1)

df_train=df.iloc[0:399]

df_test=df.iloc[400:]

df_train.head()
import seaborn as sns

import matplotlib.pyplot as plt
df[df.isnull()].count()
sns.pairplot(df_train)
from sklearn.model_selection import train_test_split

import numpy as np



X=df_train.drop('Chance of Admit ',axis=1)

Y=df_train['Chance of Admit ']



fig,ax=plt.subplots(figsize=(10,6))

sns.heatmap(X.corr(),cmap='viridis')

X_train,X_val,Y_train,Y_val=train_test_split(X,Y,train_size=0.8,random_state=10)

X_test=df_test.drop('Chance of Admit ',axis=1)

Y_test=df_test['Chance of Admit ']
from sklearn.metrics import mean_squared_error

from sklearn.ensemble import RandomForestRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.svm import SVR

from sklearn.linear_model import LinearRegression,Lasso,Ridge




model_dict = {'Linear Regression ': LinearRegression(),

              'Lasso ': Lasso(normalize=True),

              'Ridge with alpha 0.005' : Ridge(alpha=0.005, normalize=True),

              'Ridge with alpha 0.05' : Ridge(alpha=0.05, normalize=True),

              'Random Forest ': RandomForestRegressor(n_estimators=50),

              'K-Neighbours ': KNeighborsRegressor(n_neighbors = 2),

              'SVM ': SVR()

             }

for key,val in model_dict.items():

    val.fit(X_train,Y_train)

    mse_train=mean_squared_error(Y_val,val.predict(X_val))

    print('{0} Root Mean squared error for validation set is {1:.4f} '.format(key,np.sqrt(mse_train)))

    mse_test=mean_squared_error(Y_test,val.predict(X_test))

    print('{0} Root Mean squared error for test set is {1:.4f} '.format(key,np.sqrt(mse_test)))

    print()

    

    