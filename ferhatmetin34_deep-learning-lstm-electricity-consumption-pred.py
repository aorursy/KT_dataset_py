# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load





import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows',None)

pd.options.display.float_format = '{:.2f}'.format

import matplotlib,numpy,pandas,keras,sys

import matplotlib.pyplot as plt

import seaborn as sns

import plotly

import plotly.graph_objects as go

from plotly.subplots import make_subplots

import plotly.express as px

from statsmodels.tools.eval_measures import rmse

from keras.preprocessing.sequence import TimeseriesGenerator

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM,Dropout,Flatten

from sklearn.preprocessing import MinMaxScaler,StandardScaler

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split

from matplotlib import dates

import tensorflow as tf

from keras.optimizers import Adam

from keras.callbacks import EarlyStopping

import warnings

warnings.filterwarnings("ignore")
versions = ( ("matplotlib", matplotlib.__version__),

            ("numpy", np.__version__),

            ("pandas", pd.__version__),

            ("keras", keras.__version__),

           ("plotly",plotly.__version__))



print(sys.version, "\n")

print("library" + " " * 4 + "version")

print("-" * 18)



for tup1, tup2 in versions:

    print("{:11} {}".format(tup1, tup2))
data=pd.read_csv("/kaggle/input/electricity-consumption-20152020/GercekZamanliTuketim-01072015-30062020.csv",encoding= 'unicode_escape')

df=data.copy()

df.head()
df.shape
df.info()
df=df.rename({"Tüketim Miktarý (MWh)":"tuketim"},axis=1)

df.head()
df.Tarih=[i.replace(".","-") for i in df.Tarih]

    
df.head()
df.Tarih=df["Tarih"]+" "+df["Saat"]


df.Tarih=pd.to_datetime(df.Tarih,format="%d-%m-%Y %H:%M")
df.head()
df.tuketim=df.tuketim.str.replace(" ","")
df.tuketim=df.tuketim.str.replace(".","")

df.tuketim=df.tuketim.str.replace(",",".")

df.tuketim=df.tuketim.astype(float)
df.dtypes
df=df.set_index("Tarih")
df.head()
df.isnull().sum()
df.eq(0).sum()
df[df.tuketim==0]=np.nan

df.tuketim=df.tuketim.interpolate()
df.head()
df.index.min(),df.index.max()
df.describe()
df["tuketim"]["2015-12"].plot(figsize=(12,5));
fig = px.line(df, y='tuketim',title="Electricity Consumption")

fig.show()
fig = px.line(df["tuketim"]["2016"], y='tuketim',title="Electricity Consumption")

fig.show()
fig = px.line(df["tuketim"]["2017"], y='tuketim',title="Electricity Consumption")

fig.show()
fig = px.line(df["tuketim"]["2018"], y='tuketim',title="Electricity Consumption")

fig.show()
fig = px.line(df["tuketim"]["2019"], y='tuketim',title="Electricity Consumption")

fig.show()
fig = px.line(df["tuketim"]["2020"], y='tuketim',title="Electricity Consumption")

fig.show()
df.head()
df.groupby(df.index.hour).mean().plot.bar(figsize=(12,5),color="orangered",title="Hourly Consumption");
df.groupby(df.index.year).mean().plot.bar(figsize=(12,5),color="orangered",title="Yearly Consumption");
df.groupby(df.index.weekday).mean().plot.bar(figsize=(12,5),color="orangered",title="Consumption According to Weekdays");
df.groupby(df.index.day).mean().plot.bar(figsize=(12,5),color="orangered");
df.tuketim.plot(figsize=(12,5));

df.rolling(window=7).mean()["tuketim"].plot(figsize=(12,5));
df.tuketim.plot(figsize=(12,5));

df.rolling(window=30).mean()["tuketim"].plot(figsize=(12,5)); 
df.tuketim.expanding(30).mean().plot(figsize=(12,5));
df=df.drop("Saat",axis=1)
df.head()
df.resample("D").mean()[:10]
df.resample("D").sum()[:10]
df.resample("M").mean()
train=df.iloc[:-24]

test=df.tail(24)



train.shape,test.shape


scaler=MinMaxScaler()

scaler.fit(train)

scaled_train=scaler.transform(train)

scaled_test=scaler.transform(test)
n_input = 24

n_features = 1

generator = TimeseriesGenerator(scaled_train, scaled_train, length=n_input, batch_size=3)


model = Sequential()

model.add(LSTM(200, activation='relu',return_sequences=True, input_shape=(n_input, n_features)))

model.add(Dropout(0.01))

model.add(LSTM(50,activation="tanh",return_sequences=False))

#model.add(Dropout(0.1))

#model.add(LSTM(200,activation="relu",return_sequences=True))

#model.add(Dropout(0.1))

#model.add(LSTM(50,activation="tanh",return_sequences=False))

model.add(Dense(1))

model.compile(optimizer="Adam", loss='mse')
model.summary()




monitor_val_acc = EarlyStopping(monitor="mse", patience=2)

model.fit_generator(generator,epochs=7,callbacks= [monitor_val_acc])
model.history.history.keys()
loss_per_epoch = model.history.history['loss']

plt.plot(range(len(loss_per_epoch)),loss_per_epoch);
first_eval_batch = scaled_train[-24:]
first_eval_batch
first_eval_batch = first_eval_batch.reshape((1, n_input, n_features))
first_eval_batch.shape
model.predict(first_eval_batch)

scaled_test[0]
test_predictions = []



first_eval_batch = scaled_train[-n_input:]

current_batch = first_eval_batch.reshape((1, n_input, n_features))
current_batch
test_predictions = []



first_eval_batch = scaled_train[-n_input:]

current_batch = first_eval_batch.reshape((1, n_input, n_features))



for i in range(len(test)):

    

    # get prediction 1 time stamp ahead ([0] is for grabbing just the number instead of [array])

    current_pred = model.predict(current_batch)[0]

    

    # store prediction

    test_predictions.append(current_pred) 

    

    # update batch to now include prediction and drop first value

    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
current_pred
current_batch
test_predictions
#scaled_test
true_predictions = scaler.inverse_transform(test_predictions)
test['Predictions'] = true_predictions
test
test.plot(figsize=(12,8));
from statsmodels.tools.eval_measures import mse,rmse



mse(test["tuketim"],test["Predictions"])
rmse(test["tuketim"],test["Predictions"])