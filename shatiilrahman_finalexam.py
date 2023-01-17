import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os
print(os.listdir("../input"))

data = pd.read_csv("../input/Advertising.csv")
data.head()
data.shape
x = data['TV']
y = data['sales']
plt.scatter(x,y)
plt.show()
import statsmodels.api as sm
x1 = sm.add_constant(x)
model = sm.OLS(y,x1)
model_fit = model.fit()
print(model_fit.params)
beta_0 = model_fit.params[0] # to use for predicting data
beta_1 = model_fit.params[1] # N.A
model_fit.summary()
x_new = pd.DataFrame({'TV':[50]})
x_new.head()
predected_y = beta_0 + (beta_1 * x_new)
print(predected_y)

x_new = pd.DataFrame({'TV':[data.TV.min(),data.TV.max()]})
x_new.head()
predected_y = beta_0 + (beta_1 * x_new)
print(predected_y)
plt.scatter(x,y)
plt.plot(x_new,predected_y,c = 'red', linewidth=2)
model_fit.conf_int()
model_fit.pvalues
model_fit.rsquared
x = data[['TV','radio','newspaper']]
y = data['sales']
x1 = sm.add_constant(x)
model = sm.OLS(y,x1)
model_fit = model.fit()
print(model_fit.params)
#
model_fit.pvalues
model_fit.summary()