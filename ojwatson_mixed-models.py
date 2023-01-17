# import modules needed

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns; sns.set()

import numpy as np

import pandas as pd

import sklearn as sk

from math import sqrt

import warnings

warnings.filterwarnings('ignore')
# read in our data

data = pd.read_csv('../input/data.csv')

data.head(30)
# plot the distribution of the bounce times - this creates the object that will be plotted when matplotlib.pyplot

sns.distplot(data.bounce_time)

plt.show()
# plot the distribution of the ages

sns.distplot(data.age, kde=False)

plt.show()
# lets use the scale function from the preprocess package within sklearn

from sklearn import preprocessing

data["age_scaled"] = preprocessing.scale(data.age.values)
# plot the distribution

sns.distplot(data.age_scaled, kde=False)

plt.show()
# let's use the lmplot function within seaborn

sns.lmplot(x = "age", y = "bounce_time", data = data)
from sklearn.linear_model import LinearRegression



# construct our linear regression model

model = LinearRegression(fit_intercept=True)

x = data.age_scaled

y = data.bounce_time



# fit our model to the data

model.fit(x[:, np.newaxis], y)



# and let's plot what this relationship looks like 

xfit = np.linspace(-3, 3, 1000)

yfit = model.predict(xfit[:, np.newaxis])

plt.scatter(x, y)

plt.plot(xfit, yfit);
print("Model slope:    ", model.coef_[0])

print("Model intercept:", model.intercept_)
# and let's store the rmse

y_predict = model.predict(x.values.reshape(-1,1))

RMSE = sqrt(((y-y_predict)**2).values.mean())



results = pd.DataFrame()

results["Method"] = ["Linear Regression"]

results["RMSE"] = RMSE

results
#!conda install -c districtdatalabs yellowbrick

import yellowbrick

from sklearn.linear_model import Ridge

from yellowbrick.regressor import ResidualsPlot



# Instantiate the linear model and visualizer

visualizer = ResidualsPlot(model = model)



visualizer.fit(x[:, np.newaxis], y)  # Fit the training data to the model

visualizer.poof()                    # Draw/show/poof the data
ax = sns.residplot(x = "age_scaled", y= "bounce_time", data = data, lowess = True)

ax.set(ylabel='Observed - Prediction')

plt.show()
sns.catplot(x="county", y="bounce_time", data=data, kind = "swarm")
# let's use the lmplot function within seaborn

grid = sns.lmplot(x = "age_scaled", y = "bounce_time", col = "county", sharex=False, col_wrap = 4, data = data, height=4)
sns.catplot(x="location", y="bounce_time", col="county", col_wrap=4, sharey=False, data=data, kind = "swarm")
# make a new data frame with one hot encoded columns for the counties

counties = data.county.unique()

data_new = pd.concat([data,pd.get_dummies(data.county)],axis=1)

data_new.head()
# construct our linear regression model

model = LinearRegression(fit_intercept=True)

x = data_new.loc[:,np.concatenate((["age_scaled"],counties))]

y = data.bounce_time



# fit our model to the data

model.fit(x, y)



# and let's plot what this relationship looks like 

visualizer = ResidualsPlot(model = model)

visualizer.fit(x, y)  # Fit the training data to the model

visualizer.poof()     
# and let's plot the predictions

performance = pd.DataFrame()

performance["residuals"] = model.predict(x) - data.bounce_time

performance["age_scaled"] = data.age_scaled

performance["predicted"] = model.predict(x)



ax = sns.residplot(x = "age_scaled", y = "residuals", data = performance, lowess=True)

ax.set(ylabel='Observed - Prediction')

plt.show()
data_new["y_predict"] = model.predict(x)

grid = sns.lmplot(x = "age_scaled", y = "y_predict", col = "county", sharey=False, col_wrap = 4, data = data_new, height=4)

grid.set(xlim=(-3,3))
# and let's store the rmse

y_predict = model.predict(x)

RMSE = sqrt(((y-y_predict)**2).values.mean())

results.loc[1] = ["Fixed", RMSE]

results
# coefficient for age and the counties

pd.DataFrame.from_records(list(zip(np.concatenate((["age_scaled"],counties)), model.coef_)))
#!conda install -c conda-forge statsmodels -y

import statsmodels.api as sm

import statsmodels.formula.api as smf



# construct our model, with our county now shown as a group

md = smf.mixedlm("bounce_time ~ age_scaled", data, groups=data["county"])

mdf = md.fit()

print(mdf.summary())
# and let's plot the predictions

performance = pd.DataFrame()

performance["residuals"] = mdf.resid.values

performance["age_scaled"] = data.age_scaled

performance["predicted"] = mdf.fittedvalues



sns.lmplot(x = "predicted", y = "residuals", data = performance)
ax = sns.residplot(x = "age_scaled", y = "residuals", data = performance, lowess=True)

ax.set(ylabel='Observed - Prediction')

plt.show()
# and let's store the rmse

y_predict = mdf.fittedvalues

RMSE = sqrt(((y-y_predict)**2).values.mean())

results.loc[2] = ["Mixed", RMSE]

results
# construct our model, but this time we will have a random interecept AND a random slope with respect to age

md = smf.mixedlm("bounce_time ~ age_scaled", data, groups=data["county"], re_formula="~age_scaled")

mdf = md.fit()

print(mdf.summary())
# and let's plot the predictions

performance = pd.DataFrame()

performance["residuals"] = mdf.resid.values

performance["age_scaled"] = data.age_scaled

performance["predicted"] = mdf.fittedvalues



sns.lmplot(x = "predicted", y = "residuals", data = performance)
ax = sns.residplot(x = "age_scaled", y = "residuals", data = performance, lowess=True)

ax.set(ylabel='Observed - Prediction')

plt.show()
# and let's store the rmse

y_predict = mdf.fittedvalues

RMSE = sqrt(((y-y_predict)**2).values.mean())

results.loc[3] = ["Mixed_Random_Slopes", RMSE]

results
from scipy import stats



# construct our linear regression model

lm = LinearRegression(fit_intercept=True)

x = data.age

y = data.bounce_time



# fit our model to the data

lm.fit(x[:, np.newaxis], y)



# let's get our fitted parameters for the intercept and coefficient and what our predictions are

params = np.append(lm.intercept_,lm.coef_)

predictions = lm.predict(x.values.reshape(-1, 1))



# and let's simulate some new data for the model and then compare what the error is for these 

newx = pd.DataFrame({"Constant":np.ones(len(x))}).join(pd.DataFrame(x))

MSE = (sum((y-predictions)**2))/(len(newx)-len(newx.columns))



# and whats the variance, standard deviation, t values and p-values

var_b = MSE*(np.linalg.inv(np.dot(newx.T,newx)).diagonal())

sd_b = np.sqrt(var_b)

ts_b = params/ sd_b

p_values =[2*(1-stats.t.cdf(np.abs(i),(len(newx)-1))) for i in ts_b]



# and let's group it together

names = ["intercept", "age"]

summary = pd.DataFrame()

summary["names"],summary["Coefficients"],summary["Standard Errors"] = [names,params,sd_b]

summary["t values"],summary["Probabilites"] = [ts_b,p_values]

print(summary)
from sklearn.metrics import mean_squared_error

from math import sqrt



model = LinearRegression(fit_intercept=True)

rms = np.empty(counties.size)

mse = np.empty(counties.size)



for i in range(counties.size):

    county = counties[i]

    x = data.age[data.county == county].values.reshape(-1,1)

    y = data.bounce_time[data.county == county]

    fit = model.fit(x, y)

    predict = model.predict(x)

    mse[i] = mean_squared_error(y, predict)

    rms[i] = sqrt(mse[i])



sqrt(mse.sum())

# construct our model, with our county now shown as a group

data["location_county"] = data["location"] + "_" + data["county"]

data.head()



md = smf.mixedlm("bounce_time ~ age_scaled", data, groups=data["location_county"], re_formula="~age_scaled")

mdf = md.fit()

print(mdf.summary())
# and let's store the rmse

y_predict = mdf.fittedvalues

RMSE = sqrt(((y-y_predict)**2).mean())

results.loc[3] = ["Nested_Mixed", RMSE]

results