import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import mean_squared_error
%matplotlib inline
data = pd.read_csv("../input/insurance.csv")
head = data.head(5)
print(head)
describe = data.describe()
print(describe)
sex = data.groupby(by = 'sex').size()
print(sex)
smoker = data.groupby(by = 'smoker').size()
print(smoker)
region = data.groupby(by = 'region').size()
print(region)
## check the distribution of charges
distPlot = sns.distplot(data['charges'])
plt.title("Distirbution of Charges")
plt.show(distPlot)
## check charges vs features
meanGender = data.groupby(by = "sex")["charges"].mean()
print(meanGender)
print(meanGender["male"] - meanGender["female"])
boxPlot1 = sns.violinplot(x = "sex", y = "charges", data = data)
meanSmoker = data.groupby(by = "smoker")["charges"].mean()
print(meanSmoker)
print(meanSmoker["yes"] - meanSmoker["no"])
boxPlot2 = sns.violinplot(x = "smoker", y = "charges", data = data)
meanRegion = data.groupby(by = "region")["charges"].mean()
print(meanRegion)
boxPlot3 = sns.violinplot(x = "region", y = "charges", data = data)
pairPlot = sns.pairplot(data)
sns.set(style = "ticks")
smokerPairs = sns.pairplot(data, hue = "smoker")
## Dummify sex, smoker and region
scaleMinMax = MinMaxScaler()
data[["age", "bmi", "children"]] = scaleMinMax.fit_transform(data[["age", "bmi", "children"]])
data = pd.get_dummies(data, prefix = ["sex", "smoker", "region"])
## retain sex = male, smoker = yes, and remove 1 region = northeast to avoid dummytrap
data = data.drop(data.columns[[4,6,11]], axis = 1)
head = data.head()
print(head)
dataX = data.drop(data.columns[[3]], axis = 1)
dataY = data.iloc[:, 3]
X_train, x_test, Y_train, y_test = train_test_split(dataX, dataY, random_state = 0)
import statsmodels.api as sm
from scipy import stats

X_train2 = sm.add_constant(X_train)
linearModel = sm.OLS(Y_train, X_train2)
linear = linearModel.fit()
print(linear.summary())
## try Linear Regression ##
from sklearn.linear_model import LinearRegression
linearModel = LinearRegression()
linear = linearModel.fit(X_train, Y_train)
linearPred = linear.predict(x_test)
mseLinear = metrics.mean_squared_error(y_test, linearPred)
rmseLinear = mseLinear**(1/2)
from sklearn.linear_model import Ridge
ridgeModel = Ridge()
ridge = ridgeModel.fit(X_train, Y_train)
ridgePred = ridge.predict(x_test)
mseRidge = metrics.mean_squared_error(y_test, ridgePred)
rmseRidge = mseRidge**(1/2)
from sklearn.linear_model import Lasso
lassoModel = Lasso()
lasso = lassoModel.fit(X_train, Y_train)
lassoPred = lasso.predict(x_test)
mseLasso = metrics.mean_squared_error(y_test, lassoPred)
rmseLasso = mseLasso**(1/2)
from sklearn.linear_model import ElasticNet
elasticNetModel = ElasticNet(alpha = 0.01, l1_ratio = 0.9, max_iter = 20)
ElasticNet = elasticNetModel.fit(X_train, Y_train)
ElasticNetPred = ElasticNet.predict(x_test)
mseElasticNet = metrics.mean_squared_error(y_test, ElasticNetPred)
rmseElasticNet = mseElasticNet**(1/2)
performanceData = pd.DataFrame({"model":["linear", "lasso", "ridge", "elasticnet"], "rmse":[rmseLinear, rmseLasso, rmseRidge, rmseElasticNet]})
print(performanceData)