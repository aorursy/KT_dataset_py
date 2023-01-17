import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy.interpolate import *
data=pd.read_excel("https://www.ecdc.europa.eu/sites/default/files/documents/COVID-19-geographic-disbtribution-worldwide.xlsx", encoding="latin-1")
date=data["dateRep"][25330:25441].values.tolist()
case=data["cases"][25330:25441].values.tolist()
death=data["deaths"][25330:25441].values.tolist()
days=int(len(date))
x1=np.array(range(0,days))
x2=np.array(range(0,days))
y1=case
y2=death
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from math import sqrt

poly_reg = PolynomialFeatures(degree=19)
x1_poly = poly_reg.fit_transform(x1.reshape(-1, 1))

lin_reg = LinearRegression()
lin_reg.fit(x1_poly, y1)
y_pred_1 = lin_reg.predict(x1_poly)

rmse_1 = sqrt(mean_squared_error(y1, y_pred))
r2_1 = r2_score(y1, y_pred)

poly_reg = PolynomialFeatures(degree=9)
x2_poly = poly_reg.fit_transform(x2.reshape(-1, 1))

lin_reg = LinearRegression()
lin_reg.fit(x2_poly, y2)
y_pred_2 = lin_reg.predict(x2_poly)

rmse_2 = sqrt(mean_squared_error(y2, y_pred_2))
r2_2 = r2_score(y2, y_pred_2)
f, (ax0,ax1) = plt.subplots(1,2,figsize=(25,20))

ax0.text(0,4700,"Total number of cases\n as of 04.07.2020 is 203456")
ax0.scatter(x1,y1,label="Daily cases")
ax0.plot(x1,y_pred_1, label = 'Polynomial Fit: RMSE = ' + format(rmse_1,'.2f') + " R2 =" + format(r2_1,'.2f') ,color= 'red')
ax0.title.set_text("Daily covid-19 cases in Turkey")
ax0.set_xlabel('Days')
ax0.set_ylabel('Number of daily cases')
ax0.legend(loc=1)

ax1.scatter(x2,y2,label="Daily deaths")
ax1.plot(x2,y_pred_2, label = 'Polynomial Fit: RMSE = ' + format(rmse_2,'.2f') + " R2 =" + format(r2_2,'.2f') ,color= 'red')
ax1.title.set_text("Daily covid-19 deaths in Turkey")
ax1.set_xlabel('Days')
ax1.set_ylabel('Number of daily deaths')
ax1.legend(loc=1)