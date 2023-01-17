# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv("/kaggle/input/finance-accounting-courses-udemy-13k-course/udemy_output_All_Finance__Accounting_p1_p626.csv")

df.dropna(axis=0,inplace=True)

df.info()
df=df[['is_paid','is_wishlisted','num_subscribers','num_reviews','avg_rating','num_published_lectures','num_published_practice_tests','price_detail__amount','rating']]

X = df.iloc[:, :-1].values

y = df.iloc[:, -1].values
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder

ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[1])],remainder='passthrough')

X=np.array(ct.fit_transform(X))
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1,random_state=1)
from sklearn.linear_model import LinearRegression

regressor=LinearRegression()

regressor.fit(X_train,y_train)
y_pred=regressor.predict(X_test)

y_pred=y_pred.round(2)

y_pred
y_test
from sklearn.metrics import r2_score

cofficient_of_multiple_dermination=r2_score(y_pred,y_test)

print(cofficient_of_multiple_dermination)
import sklearn.metrics as metrics

def regression_results(y_true, y_pred):



    # Regression metrics

    explained_variance=metrics.explained_variance_score(y_true, y_pred)

    mean_absolute_error=metrics.mean_absolute_error(y_true, y_pred) 

    mse=metrics.mean_squared_error(y_true, y_pred) 

    median_absolute_error=metrics.median_absolute_error(y_true, y_pred)

    r2=metrics.r2_score(y_true, y_pred)

    print('explained_variance: ', round(explained_variance,4))    

    print('r2: ', round(r2,4))

    print('MAE: ', round(mean_absolute_error,4))

    print('MSE: ', round(mse,4))

    print('RMSE: ', round(np.sqrt(mse),4))

    

regression_results(y_test,y_pred)
n=12205

k=8

r2_adj=1-(((1-cofficient_of_multiple_dermination)*(n-1))/(n-k-1))

print(r2_adj)