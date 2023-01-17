# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
Fish = pd.read_csv("../input/fish-market/Fish.csv",index_col=False)
Fish.info()

Fish.head()
y = Fish.Weight.values.reshape(-1,1)
X = Fish.Width.values.reshape(-1,1)

plt.scatter(X,y)
plt.ylabel("weight of fish in Gram")
plt.xlabel("diagonal width in cm")
plt.show()
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)
#************** linear regression cizimi *********
lr = LinearRegression()
lr.fit(X_train,y_train)

#****** predict *********
y_pred = lr.predict(X_test)

plt.plot(X_test,y_pred, color="red", label="linear_reg")
plt.scatter(X_test,y_test,label="point of actual")
plt.legend()
plt.show()
print("Predict weight of fish in 800 Gram: ", lr.predict([[5]]))

#********* Polynomial Regression *****y = b0 + b1*x1 + b2*x2 + b3*x3 +... ***********

polynomial_regression = PolynomialFeatures(degree = 15)    # 5.mertebeye kadar bakalim. Eger uygun degilse degistirmeliyiz
X_polynomial_1 = polynomial_regression.fit_transform(X_train)
X_polynomial_test = polynomial_regression.transform(X_test)


lr2 = LinearRegression()
lr2.fit(X_polynomial_1,y_train)

y_pred_1 = lr2.predict(X_polynomial_test)

plt.scatter(X_test,y_pred_1,color="green",label="poly_reg")
plt.scatter(X_test,y_test,label='point of actual')
plt.legend()
plt.show()
#X_polynomial_test
#********* Polynomial Regression *****y = b0 + b1*x1 + b2*x2 + b3*x3 +... ***********

polynomial_regression_2 = PolynomialFeatures(degree = 4) 
X_polynomial = polynomial_regression_2.fit_transform(X_train)
X_polynomial_test_2 = polynomial_regression_2.transform(X_test)


lr3 = LinearRegression()
lr3.fit(X_polynomial,y_train)

y_pred_2 = lr3.predict(X_polynomial_test_2)
if X_polynomial_test_2.shape[0]==y_pred_2.shape[0]:
    plt.scatter(X_test,y_pred_2,color="red",label="poly_reg")
    plt.scatter(X_test,y_test,label='point of actual')
    plt.legend()
    plt.show()
from sklearn.metrics import r2_score
r2_score_1 = r2_score(y_test,y_pred_2)
print(r2_score_1)
from sklearn.metrics import mean_squared_error as mse
mse_1 = mse(y_pred_2,y_test)/y_test.size
print(mse_1)
from sklearn.tree import DecisionTreeRegressor
reg = DecisionTreeRegressor()
reg.fit(X_train,y_train)
y_pred_3 = reg.predict(X_test)
plt.scatter(X_test,y_pred_3,color="yellow",label="Decision_tree",alpha=1,s=100)
plt.scatter(X_test,y_test,color="black",alpha=1,s=10,label='point of actual')
plt.legend()
plt.show()

reg.score(X_test,y_test)
from sklearn.metrics import r2_score
r2_score_2 = r2_score(y_test,y_pred_3)
print(r2_score_2)
from sklearn.metrics import mean_squared_error as mse
mse_2 = mse(y_pred_3,y_test)/y_test.size
print(mse_2)
from sklearn.ensemble import RandomForestRegressor
reg = RandomForestRegressor(n_estimators=200,max_depth=6)
reg.fit(X_train,y_train)
y_pred_4 = reg.predict(X_test)
plt.scatter(X_test,y_pred_4,color="yellow",label="poly",alpha=1,s=25)
plt.scatter(X_test,y_test,color="black",alpha=1,s=10)
plt.legend()
plt.show()
from sklearn.metrics import r2_score
r2_score_3 = r2_score(y_test,y_pred_4)
print(r2_score_3)
from sklearn.metrics import mean_squared_error as mse
mse_3 = mse(y_pred_4,y_test)/y_test.size
print(mse_3)