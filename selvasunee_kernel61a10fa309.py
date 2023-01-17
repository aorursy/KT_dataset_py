import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import statsmodels.api as sm #this package has mtcars dataset

from scipy.stats import linregress
data=pd.read_csv('../input/Advertising.csv')
data.head()
data.info()
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
data=data.drop("Unnamed: 0",axis=1)
y=data.Sales

#x=sm.add_constant(x) # adding a constant helps in calculating the intercept

x=data.drop("Sales",axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.30,random_state=123)

lm=LinearRegression()

model_adv=lm.fit(xtrain,ytrain)

lm_pred=lm.predict(xtest)
from sklearn.metrics import r2_score

r2_score(y,lm.predict(x))
from sklearn.metrics import mean_absolute_error, mean_squared_error

print(mean_absolute_error(ytest,lm_pred))

print(np.sqrt(mean_squared_error(ytest,lm_pred)))
model_adv=sm.OLS(ytrain,xtrain).fit() # fitted the model on data

insur_pred=model_adv.predict(xtrain) # making prediction on x values
fitted=model_adv.fittedvalues

residuals=model_adv.resid

sns.set(style="whitegrid")

ax=sns.residplot(fitted,residuals,color="red",lowess=True)

ax.set(xlabel='fitted values',ylabel="residuals",title="residual plot")

plt.show()
from statsmodels.stats.api import linear_rainbow

linear_rainbow(model_adv)
import scipy.stats as stats

import pylab

#find the standardised residual

st_residual=model_adv.get_influence().resid_studentized_internal

stats.probplot(st_residual,dist="norm",plot=pylab)

plt.show()
from scipy.stats import shapiro

test=shapiro(st_residual)

print(test)

if test[1]<0.05:

    print("reject null hypothysis")

else:

    print("fail to reject null hypothysis")
yaxis=np.sqrt(np.abs(st_residual))

ax=sns.residplot(fitted,yaxis,lowess=True,color="b")

ax.set(xlabel="Fitted values", ylabel="sqrt(standardized residuals)",title="Scale location Plot")

plt.show()
from statsmodels.stats.api import het_goldfeldquandt

het_goldfeldquandt(ytrain,xtrain)
lm=LinearRegression()

model=lm.fit(xtrain,ytrain)

lm_pred=lm.predict(xtest)
mean_absolute_error(ytest,lm_pred)
np.sqrt(mean_squared_error(ytest,lm_pred))
mean_absolute_error(ytrain,lm.predict(xtrain))
np.sqrt(mean_squared_error(ytrain,lm.predict(xtrain)))
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif=pd.DataFrame()

vif["VIF Values"]=[variance_inflation_factor(x.values,col) for col in range (0,x.shape[1])]
vif["vif index"]=['TV', 'Radio', 'Newspaper']
vif
from sklearn.preprocessing import PolynomialFeatures

pf=PolynomialFeatures()

# transformed the input variables for better model

x=pf.fit_transform(xtrain)

model_insur=sm.OLS(ytrain,x).fit() # fitted the model on data

insur_pred=model_insur.predict(x) # making prediction on x values
model_insur.rsquared
plt.scatter(xtrain.TV,ytrain,color="r")



#plt.plot(xtrain.TV,predict(xtrain),color="blue")
plt.scatter(xtrain.TV,ytrain,color="r")

plt.plot(xtrain.TV,insur_pred,color="blue")
plt.scatter(xtest.TV,ytest,color="r")

plt.plot(xtrain.TV,insur_pred,color="blue")
from sklearn.preprocessing import PolynomialFeatures

pf=PolynomialFeatures(degree=3)

# transformed the input variables for better model

x=pf.fit_transform(xtrain)

x=sm.add_constant(x)

model_insur=sm.OLS(ytrain,x).fit() # fitted the model on data

insur_pred=model_insur.predict(x) # making prediction on x values
shapiro(model_insur.resid)
model_insur.rsquared
plt.scatter(xtrain.TV,ytrain,color="r")

plt.plot(xtrain.TV,insur_pred,color="blue")
plt.scatter(xtest.TV,ytest,color="r")

plt.plot(xtrain.TV,insur_pred,color="blue")
from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor ,BaggingRegressor

rf=RandomForestRegressor()

x=pf.fit_transform(xtrain)

x=sm.add_constant(x)

model=rf.fit(x,ytrain)

pred_rf=rf.predict(x)

print(mean_absolute_error(ytrain,pred_rf))

print(np.sqrt(mean_squared_error(ytrain,pred_rf)))
plt.scatter(xtrain.TV,ytrain,color="r")

plt.plot(xtrain.TV,pred_rf,color="blue")
plt.scatter(xtest.TV,ytest,color="r")

plt.plot(xtrain.TV,pred_rf,color="blue")
print(mean_absolute_error(ytest,rf.predict(xtest)))

print(np.sqrt(mean_squared_error(ytest,rf.predict(ytest))))