# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)















# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
dftrain =  pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/train.csv")

dftrain["GrowthRate"] = dftrain["ConfirmedCases"] / dftrain.ConfirmedCases.shift(1)

print(dftrain.Date.unique())

dftest = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/test.csv")

print(dftest.columns.values)

dfsubmission  = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/submission.csv")
growth_per_country = {}

means_per_country = {}

last_per_country = {}

def train(country, region, growth):

    if growth < 2.0:

        print(growth)

        growth_list = growth_per_country.get((country,region), list())

        growth_list.append(growth)

        growth_per_country[(country,region)] = growth_list

        



def predict(country, region):

    if not (country,region) in means_per_country:

        means_per_country[(country,region)] = np.mean(growth_per_country[(country,region)])

    growth = means_per_country[(country,region)]

    return growth
## Training

for row in dftrain.itertuples():

    train(row.Country_Region, row.Province_State, row.GrowthRate)
## Current submission

for row in dftest.itertuples():

    if(row.ForecastId%100 == 0):

        print(row.ForecastId)

    if type(row.Province_State)!=str:

        dfnow = dftrain[dftrain.Country_Region == row.Country_Region]

    else:

        dfnow = dftrain[dftrain.Country_Region == row.Country_Region][dftrain.Province_State == row.Province_State]

    filterDate = dfnow["Date"].isin([row.Date])

    if len(dfnow[filterDate].values) == 0:

        growth = predict(row.Country_Region, row.Province_State)

        pred = pred * growth

        predfat = predfat * abs(growth - 0.1)

    else:

        pred = dfnow[filterDate]["ConfirmedCases"].values[0]

        predfat = dfnow[filterDate]["Fatalities"].values[0]

    dfsubmission.at[row.ForecastId-1, "ConfirmedCases"] = int(pred)

    dfsubmission.at[row.ForecastId-1, "Fatalities"] = int(predfat)

dfsubmission

dfsubmission.to_csv('submission.csv', index=False)  

# end, the rest is experimental code
print(dfsubmission.head(60))
dfsubmission["Date"] = dftest["Date"]

dfsub = dfsubmission[dftest.Country_Region == 'Ireland'][dftest.Province_State.isnull()]

dfsub[:20].plot("Date", "ConfirmedCases")
dssum
df = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/train.csv")

df.head(40)

print(df.columns.values)

df["GrowthRate"] = df["ConfirmedCases"] / df.ConfirmedCases.shift(1)

df["Growth"] = df["ConfirmedCases"] - df.ConfirmedCases.shift(1)

df["PredictedCasesByRate"] =  df["ConfirmedCases"].shift(1) * df.GrowthRate.shift(1)

df["PredictedCases"] = df["ConfirmedCases"].shift(1) + df.Growth.shift(1)

df["ErrorByRate"] =  (df.ConfirmedCases-df.PredictedCasesByRate)/df.ConfirmedCases

df["Error"] = (df.ConfirmedCases - df.PredictedCases)/df.ConfirmedCases 

df["FGrowth"] = df["Fatalities"]/df.Fatalities.shift(1)

print(df.head())



dff = df[df.Country_Region == 'Italy'][df.Province_State.isnull()][df.ConfirmedCases >= 100]

dff.plot("Date", ["ConfirmedCases","PredictedCases", "PredictedCasesByRate"])

dff.plot("Date", ["ErrorByRate", "Error"])

dff.plot("Date", "GrowthRate")

dff.head(15)
dftest = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/test.csv")

dftest.head(40)
dfg = df.groupby([df.Country_Region, df.Province_State])

dfg.head()
from functools import reduce

def test_sum(series):

    return reduce(lambda x, y: x + y, series)



dfg.agg(test_sum)
groups_dict =dfg.groups

for group, indexes in groups_dict.items():

    print(group)

    tempdf = df.loc[indexes[0]:indexes[-1]]

    print(tempdf.shape)

    if False:

        tempdf["Growth"] = tempdf.ConfirmedCases/tempdf.ConfirmedCases.shift(1)

        tempdf["FGrowth"] = tempdf.Fatalities/tempdf.Fatalities.shift(1)

        tempdf.plot("Date", ["Growth","FGrowth"])
dfa = df[df.Country_Region == 'Spain'][df.Province_State.isnull()][df.ConfirmedCases>10]

print(dftesta.shape)

print(dfa.shape)



from matplotlib import pyplot

from statsmodels.tsa.ar_model import AR

from sklearn.metrics import mean_squared_error



X = list(dfa.GrowthRate.values)



X = [x for x in X if not np.isnan(x) and not np.isinf(x)]

print(len(X))



train, test = X[:len(X)-6], X[len(X)-6:len(X)]

print(len(train))

# train autoregression

model = AR(train)

model_fit = model.fit()

window = model_fit.k_ar

coef = model_fit.params

# walk forward over time steps in test

history = train[len(train)-window:]

history = [history[i] for i in range(len(history))]

predictions = list()

for t in range(len(test)+31):

	length = len(history)

	lag = [history[i] for i in range(length-window,length)]

	yhat = coef[0]

	for d in range(window):

		yhat += coef[d+1] * lag[window-d-1]

	if t >= len(test):

		test.append(yhat)

	obs = test[t]

	predictions.append(yhat)

	history.append(obs)

	print('predicted=%f, expected=%f' % (yhat, obs))

error = mean_squared_error(test, predictions)

print('Test MSE: %.3f' % error)

# plot

pyplot.plot(train+test)

pyplot.plot(train+predictions, color='red')

pyplot.show()
dfa = df[df.Country_Region == 'Italy'][df.Province_State.isnull()][df.ConfirmedCases>10]

print(dftesta.shape)

print(dfa.shape)



from matplotlib import pyplot

from statsmodels.tsa.ar_model import AR

from sklearn.metrics import mean_squared_error



X = list(dfa.GrowthRate.values)



X = [x for x in X if not np.isnan(x) and not np.isinf(x)]

print(len(X))



train, test = X[10:len(X)-6], X[len(X)-6:len(X)]

print(len(train))

# train autoregression

model = AR(train)

model_fit = model.fit()

print('Lag: %s' % model_fit.k_ar)

print('Coefficients: %s' % model_fit.params)

# make predictions

predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)

for i in range(len(predictions)):

	print('predicted=%f, expected=%f' % (predictions[i], test[i]))

error = mean_squared_error(test, predictions)

print('Test MSE: %.3f' % error)

# plot results

pyplot.plot(train[:15]+test)

pyplot.plot(train[:15]+list(predictions), color='red')

pyplot.show()
#dfsubmission.to_csv('submission.csv', index=False)  