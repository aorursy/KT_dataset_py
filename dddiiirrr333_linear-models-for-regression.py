import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import SGDRegressor
model_LR = LinearRegression()
# 4, 9, 11, 14, 17 - X
# 6, 7, 11, 9, 13 - y
# y = w_1*x + w_0
model_LR.fit([[4], [9], [11], [14], [17]], [6, 7, 11, 9, 13])
# w_1
model_LR.coef_
# w_0
model_LR.intercept_
# y = 0.5*x + 3.7 
plt.figure(figsize=(14,10))
sns.set_style("whitegrid")
sns.regplot(x=[4, 9, 11, 14, 17], y=[6, 7, 11, 9, 13]);
