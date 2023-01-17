import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston

boston = load_boston() # データセットの読み込み



boston_df = pd.DataFrame(boston.data, columns = boston.feature_names) 

boston_df['MEDV'] = boston.target



X = boston_df.drop('MEDV',axis=1)

y = boston_df['MEDV']



X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0)
from sklearn.linear_model import LinearRegression



lr = LinearRegression().fit(X_train,y_train)



print(lr.score(X_train,y_train))

print(lr.score(X_test,y_test))
py_train = lr.predict(X_train)



mean_y = np.mean(y_train)



#全変動を求める。

zen_hendo = 0

for num in y_train:

    zen_hendo += (num - mean_y)**2

    

#回帰変動を求める

kaiki_hendo = 0

for num in py_train:

    kaiki_hendo += (num - mean_y)**2

    

#決定係数を求める

train_R2 = kaiki_hendo/zen_hendo

print(train_R2)
py_test = lr.predict(X_test)



mean_y = np.mean(y_test)



#全変動を求める。

zen_hendo = 0

for num in y_test:

    zen_hendo += (num - mean_y)**2

    

#回帰変動を求める

kaiki_hendo = 0

for num_pred in py_test:

    kaiki_hendo += (num_pred - mean_y)**2

    

#決定係数を求める

test_R2 = kaiki_hendo/zen_hendo

print(test_R2)
#https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html



#The coefficient R^2 is defined as (1 - u/v), 

#where u is the residual sum of squares ((y_true - y_pred) ** 2).sum() 

#and v is the total sum of squares ((y_true - y_true.mean()) ** 2).sum(). 
py_test = lr.predict(X_test)



#uを求める。

u = 0

u = ((y_test - py_test) ** 2).sum()

    

#vを求める

v = 0

v = ((y_test - y_test.mean()) ** 2).sum()

    

#決定係数を求める

test_R2 = 1 - u/v

print(test_R2)