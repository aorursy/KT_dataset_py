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
import pandas as pd

import numpy as np

train = pd.read_csv("/kaggle/input/web-traffic-time-series-forecasting/train_1.csv").fillna(0) #handling missing values

train = train.replace(np.nan,0) #handling missing values
from matplotlib import rcParams

import numpy as np

from matplotlib import pyplot as plt

import matplotlib

rcParams['figure.figsize'] = 18,8

y = train.loc[1][1:]

plt.plot(y) #plot a random time series to get the idea of what our data looks like

plt.xlabel('Date-Time', fontsize=10)

plt.ylabel('Traffic', fontsize=10)

plt.title('Web Traffic- Original data')

plt.show()
#first order and second order differencing to enforce stationarity

first_order = y.diff()

second_order = first_order.diff()

plt.plot(y)

plt.xlabel('Date-Time', fontsize=10)

plt.ylabel('Traffic', fontsize=10)

plt.title('Web Traffic- Original data')

plt.show()

plt.plot(first_order)

plt.xlabel('Date-Time', fontsize=10)

plt.ylabel('Traffic', fontsize=10)

plt.title('Web Traffic- First Order difference')

plt.show()

plt.plot(second_order)

plt.xlabel('Date-Time', fontsize=10)

plt.ylabel('Traffic', fontsize=10)

plt.title('Web Traffic- Second Order difference')

plt.show()
import statsmodels.api as sm

series = y

cycle, trend = sm.tsa.filters.hpfilter(series, 50) #time series decomposition

fig, ax = plt.subplots(3,1)

ax[0].plot(series)

ax[0].set_title('Actual')

ax[1].plot(trend)

ax[1].set_title('Trend')

ax[2].plot(cycle)

ax[2].set_title('Cycle')

plt.show()
ind = pd.to_datetime(y.index)

arr = []

for i in range(len(y)):

    arr.append(y[i])

arr = pd.DataFrame(arr)

arr.index = ind

decomposition = sm.tsa.seasonal_decompose(arr) 

decomposition.plot();
#acf and pacf

from statsmodels.graphics.tsaplots import plot_acf,plot_pacf

plot_acf(arr);

plot_pacf(arr);

acf_values= sm.tsa.stattools.acf(y)

pacf_values= sm.tsa.stattools.pacf(y)
#Distributing data based on language

lang_dict = dict()

for i in range(len(train)):

    lang = train['Page'][i][train['Page'][i].find(".wikipedia")-2:train['Page'][i].find(".wikipedia")]

    temp=train.loc[i]

    if(lang not in lang_dict.keys()):

        lang_dict[lang] = [temp]

    else:

        lang_dict[lang].append(temp)
import numpy as np

from matplotlib import pyplot as plt

i=0

data_lang =[]

for lang in lang_dict.keys():

        data_lang.append([lang])

        for j in range(len(lang_dict[lang])):

            data_lang[i].append(sum(lang_dict[lang][j][1:]))

        i=i+1

stats_lang = []

for lang in data_lang:

    stats_lang.append([lang[0],sum(lang[1:]),len(lang[1:]),np.mean(lang[1:]),np.std(lang[1:])])

import pandas as pd

stats_lang = pd.DataFrame(stats_lang[:8])



index = np.arange(len(stats_lang))

plt.bar(index,stats_lang[:][3])

plt.xlabel('Language', fontsize=10)

plt.ylabel('Mean of web hits', fontsize=10)

plt.xticks(index, stats_lang[:][0], fontsize=10, rotation=30)

plt.title('Web Traffic mean based on language')

plt.show()
#Distibuting Data based on access type and type

type_dict = {"all-agent":list(),"spider":list()}

access_dict = {"access_dict":list(),"desktop":list(),"mobile-web":list()}

for i in range(len(train)):

    if("all-access" in train["Page"][i][train['Page'][i].find(".wikipedia"):]):

        access_dict["access_dict"].append(train.loc[i])

    if("desktop" in train["Page"][i][train['Page'][i].find(".wikipedia"):]):

        access_dict["desktop"].append(train.loc[i])

    if("mobile-web" in train["Page"][i][train['Page'][i].find(".wikipedia"):]):

        access_dict["mobile-web"].append(train.loc[i]) 

    if("all-agent" in train["Page"][i][train['Page'][i].find(".wikipedia"):]):

        type_dict["all-agent"].append(train.loc[i])

    if("spider" in train["Page"][i][train['Page'][i].find(".wikipedia"):]):

        type_dict["spider"].append(train.loc[i])
i=0

data_type =[]

for type_x in type_dict.keys():

        data_type.append([type_x])

        for j in range(len(type_dict[type_x])):

            data_type[i].append(sum(type_dict[type_x][j][1:]))

        i=i+1

stats_type = []

for type_x in data_type:

    stats_type.append([type_x[0],sum(type_x[1:]),len(type_x[1:]),np.mean(type_x[1:]),np.std(type_x[1:])])

import pandas as pd

stats_type = pd.DataFrame(stats_type[:8])



index = np.arange(len(stats_type))

plt.bar(index,stats_type[:][3])

plt.xlabel('Type', fontsize=10)

plt.ylabel('Mean of web hits', fontsize=10)

plt.xticks(index, stats_type[:][0], fontsize=10, rotation=30)

plt.title('Web Traffic mean based on Type')

plt.show()
i=0

data_access =[]

for access_x in access_dict.keys():

        data_access.append([access_x])

        for j in range(len(access_dict[access_x])):

            data_access[i].append(sum(access_dict[access_x][j][1:]))

        i=i+1

stats_access = []

for access_x in data_access:

    stats_access.append([access_x[0],sum(access_x[1:]),len(access_x[1:]),np.mean(access_x[1:]),np.std(access_x[1:])])

import pandas as pd

stats_access = pd.DataFrame(stats_access[:8])



index = np.arange(len(stats_access))

plt.bar(index,stats_access[:][3])

plt.xlabel('access', fontsize=10)

plt.ylabel('Mean of web hits', fontsize=10)

plt.xticks(index, stats_access[:][0], fontsize=10, rotation=30)

plt.title('Web Traffic mean based on access')

plt.show()
import pandas as pd

import numpy as np

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

%matplotlib inline

from matplotlib import pyplot as plt

from sklearn.metrics import mean_squared_error

from math import sqrt

import itertools

import statsmodels.api as sm

from scipy import stats

import warnings

warnings.filterwarnings('ignore')

from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor , RandomForestRegressor

from sklearn.tree import DecisionTreeRegressor

from matplotlib import rcParams

train = train.sample(10)
def split_sequence(sequence, n_steps):

    X, Y = list(), list()

    for i in range(len(sequence)):

        end_ix = i + n_steps

        if end_ix > len(sequence)-1:

            break

        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]

        X.append(seq_x)

        Y.append(seq_y)

    return np.array(X),np.array(Y)
def LSTM_MODEL(n,n_steps):

    n_features = 1

    model = Sequential()

    model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features),return_sequences = True))

    for layer in range(n):

        model.add(LSTM(50, activation='relu',return_sequences = True))

    model.add(LSTM(50, activation='relu'))

    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse')

    return model
rcParams['figure.figsize'] = 18,8

list_models = []

regr_1 = DecisionTreeRegressor(max_depth=4)

adaboostSVC = AdaBoostRegressor(n_estimators = 500, random_state = 42, learning_rate=0.01, base_estimator=regr_1)

est = GradientBoostingRegressor(n_estimators=500, learning_rate=0.1, max_depth=4, random_state=0, loss='ls')

regressor = RandomForestRegressor(max_depth=4 , random_state=0, n_estimators=500)

p,d,q = 1,0,2

for time_series in train.index:  

    

    #extracting the time series

    error_list = []

    list_of_model_pred = []

    print("For Time series:", time_series)

    print(train.loc[time_series][0])

    y = train.loc[time_series][1:]

    ind = pd.to_datetime(y.index)

    arr = []

    for i in range(len(y)):

        arr.append(y[i])

    arr = pd.DataFrame(arr)

    arr.index = ind

    #split the dataset into training and testing data

    test_X,test_Y = split_sequence(arr[0][-365:],30)

    train_X,train_Y = split_sequence(arr[0][0:-365],30)

    list_of_model_pred.append(y[-335:])

    #Adaboost Model

    model = adaboostSVC.fit(train_X, train_Y)

    pred_Y = model.predict(test_X)

    rmse = sqrt(mean_squared_error(test_Y,pred_Y))

    error_list.append(rmse)

    print("Adaboost Done with error: ",rmse)

    list_of_model_pred.append(pred_Y)



    #Gradient Boosting

    est.fit(train_X, train_Y)

    pred_Y = est.predict(test_X)

    rmse = sqrt(mean_squared_error(test_Y, pred_Y))

    error_list.append(rmse)

    print("Gradient Boost Done with error: ",rmse)

    list_of_model_pred.append(pred_Y)

    

    #Random Forrest

    regressor.fit(train_X,train_Y)

    pred_Y = regressor.predict(test_X)

    rmse = sqrt(mean_squared_error(test_Y,pred_Y))

    error_list.append(rmse)

    print("Random Forest Regressor Done with error: ",rmse)

    list_of_model_pred.append(pred_Y)

    

    # RNN and LSTM Model

    n_features = 1

    model = LSTM_MODEL(4,30)

    train_X1 = train_X.reshape((train_X.shape[0], train_X.shape[1], n_features))

    model.fit(train_X1, train_Y, epochs=200, verbose=0)

    test_X1 = test_X.reshape((test_X.shape[0], test_X.shape[1], n_features))

    pred_Y = model.predict(test_X1, verbose=0)

    rmse = sqrt(mean_squared_error(test_Y,pred_Y))

    error_list.append(rmse)

    print("RNN and LSTM Done with error: ",rmse)

    list_of_model_pred.append(pred_Y)

    

    #Sarimax Model

    y = train.loc[time_series][1:]

    mod = sm.tsa.statespace.SARIMAX(arr[:-365],

                                order=(p, d, q),

                                seasonal_order=(1,1,2, 12),

                                enforce_stationarity=True,

                                enforce_invertibility=False)

    results = mod.fit()

    pred = results.get_prediction(start=pd.to_datetime('2016-01-01'),end=pd.to_datetime('2016-12-31') )

    rmse = sqrt(mean_squared_error(arr['2016-01-01':'2017-01-02'], pred.predicted_mean))

    print("SARIMAX Done with error: ",rmse)

    list_of_model_pred.append(pred_Y)

    error_list.append(rmse)

    

    #Plot predicted vs Original for all Models

    label_list = ["Original Time Series","Adaboost","Gradient Boost","Random Forest","RNN and LSTM","SARIMAX"]

    plt.style.use('seaborn-darkgrid')

    palette = plt.get_cmap('Dark2')

    for i in range(len(list_of_model_pred)):

        plt.subplot(3,2, i+1)

        if(i!=0):

            plt.plot(list_of_model_pred[0], marker='', color='grey', linewidth=0.6, alpha=0.3)

            plt.xlabel('Date', fontsize=10)

            plt.ylabel('Number of web hits', fontsize=10)

        plt.title(label_list[i], loc='left', fontsize=12, fontweight=0, color=palette(i))

        plt.plot(list_of_model_pred[i], marker='', color=palette(i), linewidth=2.4, alpha=0.9, label=label_list[i])

        plt.xlabel('Date', fontsize=10)

        plt.ylabel('Number of web hits', fontsize=10)

    plt.suptitle(train.loc[time_series][0], fontsize=13, fontweight=0, color='black', style='italic', y=1.02)

    list_models.append(error_list)

    plt.show()
rcParams['figure.figsize'] = 18,8

list_models = pd.DataFrame(list_models)

index = ["time series: "+str(i) for i in train.index]

list_models.index = index

list_models.columns = [i for i in label_list[1:]]

list_models.head(10)

list_models.plot.line()

plt.title("ERROR")

plt.show()

list_models.boxplot()

plt.title("ERROR")

plt.show()