#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'data:image/jpeg;base64,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',width=400,height=400)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings

import matplotlib.pyplot as plt

warnings.simplefilter('ignore')





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Common code for display result

def show_graph(df1,df2):

    data = pd.concat([df1, df2])

    data.reset_index(inplace=True, drop=True)

    for col in data.columns:

        if col.lower().startswith('pred'):

            data[col].plot(label=col,linestyle="dotted")

        else:

            data[col].plot(label=col)

    plt.title("Actual vs. Predicted")

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

show_graph(df_train, df_ret)
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

show_graph(df_train, df_ret)
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

show_graph(df_train, df_ret)
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

show_graph(df_train, df_ret)
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

show_graph(df_train, df_ret)
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

show_graph(df_train, df_ret)
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

show_graph(df_train, df_ret)
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

show_graph(df_train, df_ret)
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

show_graph(df_train, df_ret)
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

show_graph(df_train, df_ret)
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

show_graph(df_train, df_ret)