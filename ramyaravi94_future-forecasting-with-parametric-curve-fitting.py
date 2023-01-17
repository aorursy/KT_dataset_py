## For data

import pandas as pd

import numpy as np

## For plotting

import matplotlib.pyplot as plt

## For parametric fitting

from scipy import optimize
data = pd.read_csv("../input/ntt-data-global-ai-challenge-06-2020/COVID-19_and_Price_dataset.csv", sep=",")

data['Date'] = pd.to_datetime(data['Date'])

dtf = pd.DataFrame(data['Date'])

dtf['total'] = data['UnitedStates_total_cases']

dtf['new'] = data['UnitedStates_new_cases']

dtf.set_index('Date', inplace=True)

dtf.sort_index(inplace=True)

dtf.head(10)
fig, ax = plt.subplots(nrows = 2, ncols = 1, sharex = True, figsize = (13, 7))

ax[0].scatter(dtf.index, dtf['total'].values, color = 'black')

ax[0].set(title = 'total cases')

ax[1].bar(dtf.index, dtf['new'].values)

ax[1].set(title = 'new cases')

plt.show()
'''

Linear function: f(x) = a + b*x

'''

def f(x):

    

    return 10 + 25000*x



y_linear = f(x=np.arange(len(dtf)))

'''

Exponential function: f(x) = a + b^x

'''

def f(x):

    return 10 + 1.18**x



y_exponential = f(x=np.arange(len(dtf)))



'''

Logistic function: f(x) = a / (1 + e^(-b*(x-c)))

'''

def f(x): 

    return 2500000 / (1 + np.exp(-0.5*(x-20)))



y_logistic = f(x=np.arange(len(dtf)))
fig, ax =plt.subplots(figsize=(13,5))# plt.subplots(nrows = 3, ncols = 1, sharex = True, figsize = (13, 7)) 

ax.scatter(dtf["total"].index, dtf["total"].values, color="black")

ax.plot(dtf["total"].index, y_logistic, label="logistic", color="blue")

ax.plot(dtf["total"].index, y_linear, label="linear", color="red")

plt.legend()

plt.show()



fig, ax = plt.subplots(nrows = 2, ncols = 1, sharex = True, figsize = (13, 7)) 

ax[0].scatter(dtf["total"].index, dtf["total"].values, color="black")

ax[1].plot(dtf["total"].index, y_exponential, label="exponential", color="green")

plt.show()
'''

Logistic function: f(x) = capacity / (1 + e^-k*(x - midpoint) )

'''

def logistic_f(X, c, k, m):

    y = c / (1 + np.exp(-k*(X-m)))

    return y

## optimize from scipy

X_l = np.arange(len(dtf["total"]))

y_l = dtf["total"].values

p0_l = [np.max(dtf["total"]), 1, 1]

logistic_model, cov_l = optimize.curve_fit(logistic_f, X_l, y_l, p0_l)




'''

Plot parametric fitting.

'''

def utils_plot_parametric(dtf, zoom=30, figsize=(15,5)):

    ## interval

    dtf["residuals"] = dtf["ts"] - dtf["model"]

    dtf["conf_int_low"] = dtf["forecast"] - 1.96*dtf["residuals"].std()

    dtf["conf_int_up"] = dtf["forecast"] + 1.96*dtf["residuals"].std()

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)

    

    ## entire series

    dtf["ts"].plot(marker=".", linestyle='None', ax=ax[0], title="Parametric Fitting", color="black")

    dtf["model"].plot(ax=ax[0], color="green")

    dtf["forecast"].plot(ax=ax[0], grid=True, color="red")

    ax[0].fill_between(x=dtf.index, y1=dtf['conf_int_low'], y2=dtf['conf_int_up'], color='b', alpha=0.3)

   

    ## focus on last

    first_idx = dtf[pd.notnull(dtf["forecast"])].index[0]

    first_loc = dtf.index.tolist().index(first_idx)

    zoom_idx = dtf.index[first_loc-zoom]

    dtf.loc[zoom_idx:]["ts"].plot(marker=".", linestyle='None', ax=ax[1], color="black", 

                                  title="Zoom on the last "+str(zoom)+" observations")

    dtf.loc[zoom_idx:]["model"].plot(ax=ax[1], color="green")

    dtf.loc[zoom_idx:]["forecast"].plot(ax=ax[1], grid=True, color="red")

    ax[1].fill_between(x=dtf.loc[zoom_idx:].index, y1=dtf.loc[zoom_idx:]['conf_int_low'], 

                       y2=dtf.loc[zoom_idx:]['conf_int_up'], color='b', alpha=0.3)

    plt.show()

    return dtf[["ts","model","residuals","conf_int_low","forecast","conf_int_up"]]

'''

Forecast unknown future.

:parameter

    :param ts: pandas series

    :param f: function

    :param model: list of optim params

    :param pred_ahead: number of observations to forecast (ex. pred_ahead=30)

    :param freq: None or str - 'B' business day, 'D' daily, 'W' weekly, 'M' monthly, 'A' annual, 'Q' quarterly

    :param zoom: for plotting

'''

def forecast_curve(ts, f, model, start, pred_ahead=60, freq="D", zoom=30, figsize=(15,5)):

    ## fit

    X = np.arange(len(ts))

    fitted = f(X, model[0], model[1], model[2])

    dtf = ts.to_frame(name="ts")

    dtf["model"] = fitted

    

    ## index

    index = pd.date_range(start=start,periods=pred_ahead,freq=freq)

    index = index[1:]

    ## forecast

    Xnew = np.arange(len(ts)+1, len(ts)+1+len(index))

    preds = f(Xnew, model[0], model[1], model[2])

    dtf = dtf.append(pd.DataFrame(data=preds, index=index, columns=["forecast"]))

    dtf.reset_index(level = 0, inplace = True)

    ## plot

    utils_plot_parametric(dtf, zoom=zoom)

    return dtf
preds = forecast_curve(dtf["total"], logistic_f, logistic_model, start = '2020-06-30',

                       pred_ahead=81, freq="D", zoom=7)