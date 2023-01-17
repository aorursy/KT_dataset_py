import pandas as pd

import numpy as np

import warnings

import matplotlib.pyplot as plt



warnings.simplefilter('ignore')



# Common code for display result

def show_graph(df1,df2,title):

    data = pd.concat([df1, df2])

    data.reset_index(inplace=True, drop=True)

    for col in data.columns:

        if col.lower().startswith('pred'):

            data[col].plot(label=col,linestyle="dotted")

        else:

            data[col].plot(label=col)

    plt.title(title)

    plt.legend()

    plt.show()
from statsmodels.tsa.ar_model import AutoReg

from random import random



def AR_model(train,test):

    # fit model

    model = AutoReg(train['Act'], lags=1)

    model_fit = model.fit()

    # make prediction

    yhat=model_fit.predict(len(train), len(train) + len(test) - 1)

    res=pd.DataFrame({"Pred":yhat, "Act":test["Act"].values})

    return res

 

df_train = pd.DataFrame([x + random()*10 for x in range(1, 100)],

                     columns=['Act'])

df_test = pd.DataFrame([x + random()*10 for x in range(101, 200)],

                     columns=['Act'])

df_ret = AR_model(df_train, df_test)

show_graph(df_train, df_ret, "Autoregression (AR)")
from statsmodels.tsa.arima_model import ARMA

from random import random



def MA_model(train,test):

    # fit model

    model = ARMA(train['Act'], order=(0, 1))

    model_fit = model.fit(disp=False)

    # make prediction

    yhat = model_fit.predict(len(train), len(train) + len(test) - 1)

    res=pd.DataFrame({"Pred":yhat, "Act":test["Act"].values})

    return res

 

df_train = pd.DataFrame([x + random()*10 for x in range(0, 100)],

                     columns=['Act'])

df_test = pd.DataFrame([x + random()*10 for x in range(101, 201)],

                     columns=['Act'])

df_ret = MA_model(df_train, df_test)

show_graph(df_train, df_ret, "Moving Average (MA)")
from statsmodels.tsa.arima_model import ARMA

from random import random



def ARMA_model(train,test):

    # fit model

    model = ARMA(train['Act'], order=(1,2))

    model_fit = model.fit(disp=False)

    # make prediction

    yhat = model_fit.predict(len(train), len(train) + len(test) - 1)

    res=pd.DataFrame({"Pred":yhat, "Act":test["Act"].values})

    return res

 

df_train = pd.DataFrame([x + random()*10 for x in range(0, 100)],

                     columns=['Act'])

df_test = pd.DataFrame([x + random()*10 for x in range(101, 201)],

                     columns=['Act'])

df_ret = ARMA_model(df_train, df_test)

show_graph(df_train, df_ret, "Autoregressive Moving Average (ARMA)")
from statsmodels.tsa.arima_model import ARIMA

from random import random



def ARIMA_model(train,test):

    # fit model

    model = ARIMA(train['Act'], order=(1, 1, 1))

    model_fit = model.fit(disp=False)

    # make prediction

    yhat = model_fit.predict(len(train), len(train) + len(test) - 1, typ='levels')

    res=pd.DataFrame({"Pred":yhat, "Act":test["Act"].values})

    return res

 

df_train = pd.DataFrame([x + random()*10 for x in range(0, 100)],

                     columns=['Act'])

df_test = pd.DataFrame([x + random()*10 for x in range(101, 201)],

                     columns=['Act'])

df_ret = ARIMA_model(df_train, df_test)

show_graph(df_train, df_ret, "Autoregressive Integrated Moving Average (ARIMA)")
from statsmodels.tsa.statespace.sarimax import SARIMAX

from random import random



def SARIMA_model(train,test):

    # fit model

    model = SARIMAX(train['Act'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 2))

    model_fit = model.fit(disp=False)

    # make prediction

    yhat = model_fit.predict(len(train), len(train) + len(test) - 1)

    res=pd.DataFrame({"Pred":yhat, "Act":test["Act"].values})

    return res

 

df_train = pd.DataFrame([x + random()*10 for x in range(0, 100)],

                     columns=['Act'])

df_test = pd.DataFrame([x + random()*10 for x in range(101, 201)],

                     columns=['Act'])

df_ret = SARIMA_model(df_train, df_test)

show_graph(df_train, df_ret, "Seasonal Autoregressive Integrated Moving-Average (SARIMA)")
from statsmodels.tsa.statespace.sarimax import SARIMAX

from random import random



def SARIMAX_model(train,test):

    # fit model

    model = SARIMAX(train.drop('Exog', axis=1), exog=train['Exog'], order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))

    model_fit = model.fit(disp=False)

    # make prediction

    yhat = model_fit.predict(len(train), len(train) + len(test) - 1, exog=test["Exog"].values)

    res=pd.DataFrame({"Pred":yhat, "Act":test["Act"].values,"Exog":test["Exog"].values})

    return res



df_train = pd.DataFrame({'Act':[x + random()*10 for x in range(0, 100)],

                         'Exog':[x + random()*10 for x in range(101, 201)]})

df_test = pd.DataFrame({'Act':[x + random()*10 for x in range(101, 201)],

                         'Exog':[200 + random()*10 for x in range(201, 301)]})

df_ret = SARIMAX_model(df_train, df_test)

show_graph(df_train, df_ret, "Seasonal Autoregressive Integrated Moving-Average with Exogenous Regressors (SARIMAX)")
from statsmodels.tsa.vector_ar.var_model import VAR

from random import random



def VAR_model(train,test):

    # fit model

    model = VAR(train)

    model_fit = model.fit()

    # make prediction

    yhat = model_fit.forecast(model_fit.y, steps=len(test))

    res=pd.DataFrame({"Pred1":[x[0] for x in yhat], "Pred2":[x[1] for x in yhat], 

                      "Act1":test["Act1"].values, "Act2":test["Act2"].values})

    return res



df_train = pd.DataFrame({'Act1':[x + random()*10 for x in range(0, 100)],

                         'Act2':50+np.sin(np.linspace(0, 2*np.pi, 100))*50})

df_test = pd.DataFrame({'Act1':[x + random()*10 for x in range(101, 201)],

                         'Act2':50+np.sin(np.linspace(0, 2*np.pi, 100))*50})

df_ret = VAR_model(df_train, df_test)

show_graph(df_train, df_ret, "Vector Autoregression (VAR)")
from statsmodels.tsa.statespace.varmax import VARMAX

from random import random



def VARMA_model(train,test):

    # fit model

    model = VARMAX(train, order=(1, 2))

    model_fit = model.fit(disp=False)

    # make prediction

    yhat = model_fit.forecast(steps=len(test))

    res=pd.DataFrame({"Pred1":yhat['Act1'], "Pred2":yhat['Act2'], 

                      "Act1":test["Act1"].values, "Act2":test["Act2"].values})

    return res



df_train = pd.DataFrame({'Act1':[x + random()*10 for x in range(0, 100)],

                         'Act2':50+np.sin(np.linspace(0, 2*np.pi, 100))*50})

df_test = pd.DataFrame({'Act1':[x + random()*10 for x in range(101, 201)],

                         'Act2':50+np.sin(np.linspace(0, 2*np.pi, 100))*50})

df_ret = VARMA_model(df_train, df_test)

show_graph(df_train, df_ret, "Vector Autoregression Moving-Average (VARMA)")
from statsmodels.tsa.statespace.varmax import VARMAX

from random import random



def VARMAX_model(train,test):

    # fit model

    model = VARMAX(train.drop('Exog', axis=1), exog=train['Exog'], order=(1, 1))

    model_fit = model.fit(disp=False)

    # make prediction

    yhat = model_fit.forecast(steps=len(test),exog=test['Exog'])

    res=pd.DataFrame({"Pred1":yhat['Act1'], "Pred2":yhat['Act2'], 

            "Act1":test["Act1"].values, "Act2":test["Act2"].values, "Exog":test["Exog"].values})

    return res



df_train = pd.DataFrame({'Act1':[x + random()*10 for x in range(0, 100)],

                         'Act2':[x*3 + random()*10 for x in range(0, 100)],

                         'Exog':50+np.sin(np.linspace(0, 2*np.pi, 100))*50})

df_test = pd.DataFrame({'Act1':[x + random()*10 for x in range(101, 201)],

                         'Act2':[x*3 + random()*10 for x in range(101, 201)],

                         'Exog':50+np.sin(np.linspace(0, 2*np.pi, 100))*50})

df_ret = VARMAX_model(df_train, df_test)

show_graph(df_train, df_ret,"Vector Autoregression Moving-Average with Exogenous Regressors (VARMAX)")
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

from random import random



def SES_model(train,test):

    # fit model

    model = SimpleExpSmoothing(train['Act'])

    model_fit = model.fit()

    # make prediction

    yhat=model_fit.predict(len(train), len(train) + len(test) - 1)

    res=pd.DataFrame({"Pred":yhat, "Act":test["Act"].values})

    return res

 

df_train = pd.DataFrame([x + random()*10 for x in range(0, 100)],

                     columns=['Act'])

df_test = pd.DataFrame([x + random()*10 for x in range(101, 201)],

                     columns=['Act'])

df_ret = SES_model(df_train, df_test)

show_graph(df_train, df_ret,"Simple Exponential Smoothing (SES)")
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from random import random



def HWES_model(train,test):

    # fit model

    model = ExponentialSmoothing(train['Act'])

    model_fit = model.fit()

    # make prediction

    yhat=model_fit.predict(len(train), len(train) + len(test) - 1)

    res=pd.DataFrame({"Pred":yhat, "Act":test["Act"].values})

    return res

 

df_train = pd.DataFrame([x + random()*10 for x in range(0, 100)],

                     columns=['Act'])

df_test = pd.DataFrame([x + random()*10 for x in range(101, 201)],

                     columns=['Act'])

df_ret = HWES_model(df_train, df_test)

show_graph(df_train, df_ret, "Holt Winter’s Exponential Smoothing (HWES)")