

# О времена, о нравы...

# Рассмотрим средства для анализа временных рядов





from scipy import stats

from theano import theano, tensor as tt

import pandas as pd

from sklearn import datasets

import numpy as np

from numpy import random

import pylab as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline

plt.mpl.style.use('ggplot')



from sklearn import datasets

from sklearn.preprocessing import scale

from sklearn.datasets import make_moons



from statsmodels.tsa.statespace.sarimax import SARIMAX

import plotly 

from itertools import product

from statsmodels.tsa import stattools,seasonal

from numpy.random import randn

import statsmodels as sm
# https://fred.stlouisfed.org/series/IPG2211A2N

df = pd.read_csv("/kaggle/input/timeseries/IPG2211A2N.csv")

df
df = df[300:].reset_index()

import plotly.express as px

fig = px.line(df, x="DATE", y="IPG2211A2N")

fig.show()

# Критерий стационарности - нам нужен стационарный ряд -> уменьшаем значени критерия Дикки-Фуллера

sm.tsa.stattools.adfuller(df.loc[12:,"IPG2211A2N"])[1]

# Уменьшаем дисперсию

df["IPG2211A2N"] = np.log(df["IPG2211A2N"])

df["log"] = np.log(df["IPG2211A2N"])

y = df["IPG2211A2N"].copy()

fig = px.line(df, x="DATE", y="log")

fig.show()
seasonal.seasonal_decompose(df["log"].to_numpy(),freq = 12).plot()
# Убираем сезонность 

df["log"] = df["log"] - df["log"].shift(12)

fig = px.line(df, x="DATE", y="log")

fig.show()

sm.tsa.stattools.adfuller(df.loc[12:,"log"])[1]
# Достаточно стационарно 

df["log"] = df["log"] - df["log"].shift(1)

fig = px.line(df, x="DATE", y="log")

fig.show()

sm.tsa.stattools.adfuller(df.loc[13:,"log"])[1]
seasonal.seasonal_decompose(df.loc[13:,"log"].to_numpy(),freq = 12).plot()
ps = range(0,6)

d=1

qs = range(0,4)

Ps = range(0,2)

D = 1

Qs = range(0,3)
parameters = product(ps,qs,Ps,Qs)

parameters_list = list(parameters)

len(parameters_list)
# Тут я подбирал параметры к Ариме

# result = []

# best_aic = float("inf")

# for p in parameters_list:

#     try:

#         model = SARIMAX(endog = df["IPG2211A2N"], order= (p[0],d,p[1]),

#                        seasonal_order=(p[2],D,p[3],12)).fit(disp = -1)

#     except:

#         continue

#     aic = model.aic

#     if aic < best_aic:

#         best_model = model

#         best_aic = aic

#         best_param = p

        
y = y[1:]
p =  (2, 1, 1, 2)

model = SARIMAX(endog = y, order= (p[0],1,p[1]),

                        seasonal_order=(p[2],1,p[3],12)).fit(disp = -1)
from sklearn.metrics import mean_squared_error as mse
y_arima = model.predict(0,670)
from numpy import pi

from sklearn.preprocessing import StandardScaler

# Создаем фичи для регрессии



# Добавляем сезонность с помощью ряда Фурье

# Эксперемент не удался

def app_Fourie(len_t, number_harmonic):

    list_cos = ['cos '+str(i)+'t' for i in range(1,1+(number_harmonic))]

    list_sin = ['sin '+str(i)+'t' for i in range(1,1+(number_harmonic))]



    answer = pd.DataFrame(columns = list_cos+list_sin)

    t = np.array([2*pi/(len_t)*k for k in range(len_t)])

    

    for i in range(number_harmonic):

        answer[list_cos[i]] = np.cos((i+1)*t)

        answer[list_sin[i]] = np.sin((i+1)*t)

    return answer



# Добавляем тренд полиномиальный

def app_Poly(len_t, degree, norm = False):

    poly = ['poly ' + str(i) for i in range(0,1+(degree))]

    t = np.arange(len_t)

    answer = pd.DataFrame(columns = poly)

    answer[poly[0]] = np.ones(len_t)

    for i in range(1,degree+1):

        answer[poly[i]] = answer[poly[i-1]]*t

    if norm == True:

        for i in range(1,degree+1):

            answer[poly[i]] = StandardScaler().fit_transform(answer[poly[i]].to_numpy().reshape(-1, 1))

    return answer



def add_Months(len_t):

    t = np.arange(len_t)//12

    answer = pd.DataFrame(columns = ['month'])

    answer['month'] = np.arange(len_t)%12

    return pd.get_dummies(answer['month'])



def add_Pct(ds,len_season):

    pct = ds.pct_change().to_numpy()

    answer = np.zeros((len(ds),len_season))

    for i in range(0,len(ds)-len_season):

        answer[i+len_season] = pct[i:i+len_season]

    return(pd.DataFrame(answer, columns = ["pct_"+ str(i) for i in range(len_season - 1, -1 ,-1)]))



y = df["IPG2211A2N"].copy().to_numpy()

X2 = app_Poly(len(y),3).join(add_Months(len(y)))

X_train2 = X2[:630].copy()

X_test2 = X2[630:].copy()

y_train = y[:630].copy()

y_test = y[630:].copy()
from tsfresh.utilities.dataframe_functions import roll_time_series

from tsfresh.utilities.dataframe_functions import make_forecasting_frame

from tsfresh import extract_features

from tsfresh.utilities.dataframe_functions import make_forecasting_frame

from tsfresh.utilities.dataframe_functions import impute
df_shift, y = make_forecasting_frame(y, kind="price", max_timeshift=12, rolling_direction=1)
# Создаем фичи

X = extract_features(df_shift, column_id="id", column_sort="time", column_value="value", impute_function=impute, 

                     show_warnings=False)
mse_test_res = model.predict(629,len(y)-1)

print('На тестовом множестве mse у ARIMA: ', mse(mse_test_res,y_test))
from sklearn.preprocessing import StandardScaler

X = StandardScaler().fit_transform(X)

X_all = X.copy()

X_train_all = X_all[:630].copy()

X_test_all = X_all[630:].copy()
# Уменьшаем размерность

from sklearn.decomposition import PCA

pca = PCA(n_components=0.99)



X = pca.fit_transform(X)

X_train = X[:630].copy()

y_train = y[:630].to_numpy().copy()

y_test = y[630:].to_numpy().copy()

X_test = X[630:].copy()

# Строим линейную модель

from sklearn.model_selection import KFold

from sklearn.linear_model import LinearRegression



# Model

def run_cv_model(train, test, target, model_fn, params={}, eval_fn=None, label='model'):

    kf = KFold(n_splits=7, random_state=42)

    fold_splits = kf.split(train, target)

    cv_scores = []

    pred_full_test = 0

    pred_train = np.zeros((train.shape[0]))

    i = 1

    for dev_index, val_index in fold_splits:

        print('Started ' + label + ' fold ' + str(i) + '/7')

        dev_X, val_X = train[dev_index], train[val_index]

        dev_y, val_y = target[dev_index], target[val_index]

        params2 = params.copy()

        pred_val_y, pred_test_y = model_fn(dev_X, dev_y, val_X, val_y, test, params2)

        pred_full_test = pred_full_test + pred_test_y

        pred_train[val_index] = pred_val_y

        if eval_fn is not None:

            cv_score = eval_fn(val_y, pred_val_y)

            cv_scores.append(cv_score)

            print(label + ' cv score {}: {}'.format(i, cv_score))

        i += 1

    pred_full_test = pred_full_test / 7

    results = {'label': label,

              'train': pred_train, 'test': pred_full_test,

              'cv': cv_scores}

    return results





def runLR(train_X, train_y, test_X, test_y, test_X2, params):

    print('Train LR')

    model = LinearRegression(**params)

    model.fit(train_X, train_y)

    print('Predict 1/2')

    pred_test_y = model.predict(test_X)

    print('Predict 2/2')

    pred_test_y2 = model.predict(test_X2)

    return pred_test_y, pred_test_y2





lr_params = {}

resultsLR = run_cv_model(X_train, X_test, y_train, runLR, lr_params, mse, 'lr')

y_lin = np.concatenate([resultsLR['train'][12:],resultsLR['test']])



print("На тестовом множестве mse у RL: ", mse(y_test,resultsLR['test']))
lr = LinearRegression().fit(X_train2,y_train)

print("На вручную сгенерированных фичах mse у RL: ", mse(y_test,lr.predict(X_test2)[0:-1]))
# Сравниваем линейную модель и Ариму, график можно приблизить в конце

import plotly.graph_objects as go

x = np.array([i for i in range(len(df["DATE"])+30)])

# Create traces

fig = go.Figure()

fig.add_trace(go.Scatter(x=x, y=np.exp(y)[610:], name = "answer"))

fig.add_trace(go.Scatter(x=x, y=np.exp(y_lin)[610:],name = "Linear"))

fig.add_trace(go.Scatter(x=x, y=np.exp(y_arima)[610:],name = "ARIMA"))
# Cмотрим на распределение

sns.distplot(y_train, color="b")
sns.distplot(np.exp(y_train), color="b")
def train_for_LSTM(data, time_steps):

    new_data = np.zeros((data.shape[0] - time_steps, time_steps, data.shape[1]))

    for i in range(data.shape[0] - time_steps):

        new_data[i,:,:] = data[i: i+time_steps]

    return(new_data)
X_for_LSTM = train_for_LSTM(X,12)

y_for_LSTM = y[12:].copy()
train_X = X_for_LSTM[0:618]

test_X = X_for_LSTM[618:]

train_y = y_for_LSTM[0:618]

test_y = y_for_LSTM[618:]
from keras.preprocessing import sequence

from keras.utils import np_utils

from keras.models import Sequential

from keras.layers.core import Dense, Dropout, Activation

from keras.layers.embeddings import Embedding

from keras.layers.recurrent import LSTM,LSTMCell,GRU,RNN,SimpleRNN

from keras.optimizers import Nadam,Adam

from keras.metrics import mae

from keras.activations import elu

from keras.layers import TimeDistributed

from keras.optimizers import Adam

from keras.layers import BatchNormalization

from keras.callbacks.callbacks import LearningRateScheduler,EarlyStopping

lr = LearningRateScheduler(lambda epoch: 0.02/((epoch+1)**0.5))

stop = EarlyStopping(restore_best_weights = True, patience = 30,verbose = 1)



model = Sequential()    

model.add(SimpleRNN(30, input_shape = (12, X.shape[1]),return_sequences=True))

model.add(GRU(20,return_sequences=True))

model.add(GRU(10,return_sequences=True))

model.add(LSTM(5))

model.add(Dense(1))



model.compile(loss = "mean_squared_error", optimizer = Adam())

LSTM_model = model.fit(train_X, 

              train_y, 

              epochs = 300, 

              batch_size = 30,

              validation_split=0.2,

              verbose = 2, callbacks = [lr,stop])
print("На тестовом множестве mse у LSTM: ", mse(y_test,model.predict(test_X)))
plt.plot(LSTM_model.history['loss'], label='loss')

plt.plot(LSTM_model.history['val_loss'], label = 'val_loss')

plt.xlabel('Epoch')

plt.ylabel('Accuracy')

plt.ylim([0, 0.05])

plt.legend(loc='lower right')
from keras.callbacks.callbacks import LearningRateScheduler,EarlyStopping

lr = LearningRateScheduler(lambda epoch: 0.02/((epoch+1)**0.5))

stop = EarlyStopping(restore_best_weights = True, patience = 160, verbose = 1)

model1 = Sequential()    



model1.add(Dense(50,activation="relu"))

model1.add(Dense(30,activation="relu"))

model1.add(Dense(10,activation="relu"))

model1.add(Dense(10,activation="relu"))

model1.add(Dense(1))



model1.compile(loss = "mse",

                  optimizer = Adam())

    

Perc_model = model1.fit(X_train, 

              y_train, 

              epochs = 500, 

              batch_size = 30, 

              verbose = 1,validation_split=0.2, callbacks = [lr,stop])

# Был разультат 0.002...

print("На тестовом множестве mse у Персептрона: ", mse(y_test,model1.predict(X_test)))
plt.plot(Perc_model.history['loss'], label='loss')

plt.plot(Perc_model.history['val_loss'], label = 'val_loss')

plt.xlabel('Epoch')

plt.ylabel('Accuracy')

plt.ylim([0, 0.05])

plt.legend(loc='lower right')
# Ради интереса попробуем SVR

from sklearn.svm import SVR

rf = SVR("linear",epsilon = 0.01).fit(X_train,y_train)

y_svr = np.concatenate([y[:630],rf.predict(X_test)])

print("На тестовом множестве mse у SVR: ", mse(y_test,rf.predict(X_test)))
# y_fun2 = np.concatenate([y_train,results.detach().numpy().reshape(1,-1)[0]])

y_lstm = model.predict(test_X)

y_pers = model1.predict(X_test[12:])

import plotly.graph_objects as go 

x = np.array([i for i in range(len(df["DATE"]))])

# Create traces

fig = go.Figure()

fig.add_trace(go.Scatter(x=x, y=np.exp(y_for_LSTM)[30:], name = "answer"))

fig.add_trace(go.Scatter(x=x, y=np.exp(y_lin)[30:],name = "Linear"))

fig.add_trace(go.Scatter(x=x, y=np.exp(y_pers)[30:],name = "Network"))

fig.add_trace(go.Scatter(x=x, y=np.exp(y_arima)[30:],name = "ARIMA"))

fig.add_trace(go.Scatter(x=x, y=np.exp(y_lstm)[30:],name = "LSTM"))



# fig.add_trace(go.Scatter(x=x, y=np.exp(y_svr),name = "SVM"))





fig.show()
y_lin.shape