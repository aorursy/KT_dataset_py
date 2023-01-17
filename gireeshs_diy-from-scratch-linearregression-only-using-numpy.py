import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import warnings

warnings.filterwarnings('ignore')
X = np.matrix([[1, 1], 

               [1, 2],

              [1, 3],

              [1, 4]])

X
XT = np.matrix.transpose(X)

XT
y = np.matrix([[1], 

               [3],

              [3],

              [5]])

y
XT_X = np.matmul(XT, X)

XT_X
XT_y = np.matmul(XT, y)

XT_y
betas = np.matmul(np.linalg.inv(XT_X), XT_y)

betas
from sklearn.linear_model import LinearRegression



regressor = LinearRegression().fit(X = np.array([1, 2, 3, 4]).reshape(-1, 1), y = [1, 3, 3, 5])

print("The intercept is: ", str(regressor.intercept_), ". Which is almost 0.")

print("The coefficient is: ", str(regressor.coef_))
data_vw = pd.read_csv("/kaggle/input/used-car-dataset-ford-and-mercedes/vw.csv")

data_vw = data_vw[:300]

data_vw["Intercept"] = 1

data_vw = data_vw[["Intercept", "year", "mileage", "tax", "mpg", "engineSize", "price"]]

print(data_vw.shape)

data_vw.head()
np.set_printoptions(formatter={'float_kind':'{:f}'.format})

cross_tab = np.matmul(np.matrix.transpose(data_vw.values), data_vw.values)

cross_tab
X = data_vw[["Intercept", "year", "mileage", "tax", "mpg", "engineSize"]].values

y = data_vw[["price"]].values
XT = np.matrix.transpose(X)
XT_X = np.matmul(XT, X)

XT_X
XT_X_inv = np.linalg.inv(XT_X)

XT_X_inv
XT_y = np.matmul(XT, y)

XT_y
betas = np.matmul(XT_X_inv, XT_y)

betas
import statsmodels.api as sm



regressor = sm.OLS(y, X).fit()

print(regressor.summary())
yT_y = cross_tab[-1:, -1:]

n = cross_tab[:1, :1]

y_bar_square = np.square(cross_tab[:1, -1:])



SST = yT_y - (y_bar_square / n)

SST
n = cross_tab[:1, :1]

y_bar_square = np.square(cross_tab[:1, -1:])



SSR = np.sum(np.multiply(betas, XT_y)) - (y_bar_square / n)

SSR
r_square = SSR / SST

r_square