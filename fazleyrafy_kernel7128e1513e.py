import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.linear_model import LinearRegression
bd = pd.read_csv("../input/covid19-bangladesh-dataset/COVID-19-Bangladesh.csv")
bd.insert(loc=0, column = "Day", value = np.arange(len(bd)))
bd.tail()
bd.shape
bd.columns
bd.drop(['new_confirmed', 'new_deaths', 'active'], axis=1, inplace = True)
bd.drop(['total_quarantine', 'now_in_quarantine', 'released_from_quarantine'], axis=1, inplace = True)
bd.dtypes
bd.corr()
lm = LinearRegression()
lm
Y= bd["total_deaths"]
X = bd[["total_confirmed"]]
lm.fit(X,Y)
Yhat = lm.predict(X)
Yhat[0:5]
lm.intercept_
lm.coef_
# 𝑌ℎ𝑎𝑡=𝑎 + 𝑏𝑋, where 'a' refers to the intercept of the regression line0, in other words: the value of Y when X is 0;
# --> 'b' refers to the slope of the regression line, in other words: the value with which Y changes when X increases by 1 unit
