# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from dateutil.relativedelta import relativedelta

import keras
from keras.models import Model, Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Input, Lambda
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.losses import mse
from keras import backend as K

from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.neighbors import BallTree
from sklearn import metrics

from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

import matplotlib.pyplot as plt # plotting
import seaborn as sns
sns.set(style="darkgrid")

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import os
#print(os.listdir("../input"))
df = pd.read_csv('../input/Hack2018_ES_cleaned_temp.csv', parse_dates=[9])

# Clean data
# Parse GPS values
def parse_coord(value):
    if type(value) == str:
        return float(value.replace(',', '.'))
    return value

df.longitud_corregida = df.longitud_corregida.apply(parse_coord)
df.latitude_corregida = df.latitude_corregida.apply(parse_coord)
df.temp = df.temp.apply(parse_coord)

# Discarding ungeolocated values
lat_limit = 20
lon_limit = 0

discarded = df[(df.latitude_corregida <= lat_limit) | (df.longitud_corregida <= lon_limit)]
print("Discarding {} entries outside Catalonia".format(len(discarded)))
df = df[(df.latitude_corregida > lat_limit) & (df.longitud_corregida > lon_limit)]

def twoDigits(n):
    if n < 10:
        return '0' + str(n)
    return str(n)

def format_month(d):
    month = "{}-{}-01".format(d.year, twoDigits(d.month))
    return pd.Timestamp(month)

def format_date(d):
    month = "{}-{}-{}".format(d.year, twoDigits(d.month), twoDigits(d.day))
    return pd.Timestamp(month)

def format_month_day(d):
    month = "2000-{}-{}".format(twoDigits(d.month), twoDigits(d.day))
    return pd.Timestamp(month)

# Add month value
def format_n_month(d):
    return "{}".format(d.month if d.month > 9 else "0" + str(d.month))

# Filter by date
len_before = len(df)
df = df[(df.Fecha >= '2015-09-01 00:00:00') & (df.Fecha < '2018-09-01 00:00:00')]
print("Discarding {} because date out of range".format(len_before - len(df)))

df['month'] = df.Fecha.apply(format_month)
df['n_month'] = df.Fecha.apply(format_n_month)
df['date'] = df.Fecha.apply(format_date)

# Remove unfrequent disease
disease_counts = df.E_class.value_counts()
frequent_disease = set(disease_counts[disease_counts > 10].index.values)
df = df[df.E_class.apply(lambda x: x in frequent_disease)]

# Any results you write to the current directory are saved as output.
df.head()
vectors = df.pivot_table(values='nasistencias', index=['date','poblacion'], columns=['E_class'], aggfunc=np.sum).reset_index()
vectors = vectors.fillna(0)
vectors = pd.merge(df[['date', 'poblacion', 'temp']].drop_duplicates(['date', 'poblacion']), vectors)
vectors.head()
date_limit = '2017-09-01 00:00:00'

train = vectors[vectors.date < date_limit]
test = vectors[vectors.date >= date_limit]
shape = test.values[:,3:].shape

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def basic_predict(dt, town, temperature, prod=False):
    """Predict last year's value (ignoring temperature)"""
    data = train
    if prod:
        data = vectors
    source_date = dt - relativedelta(months=12)
    result = data[(data.date == format_date(source_date)) & (data.poblacion == town)]
    if len(result) > 0:
        return result.values[0][3:]
    return np.array([0.0]*(len(train.columns) - 3))

train_date_months = train.date.apply(format_month_day)
vectors_date_months = vectors.date.apply(format_month_day)
def basic_predict_all_years(dt, town, temperature, prod=False):
    """Predict as mean of all past dates (ignoring temperature)"""
    (data, date_months) = (train, train_date_months)
    if prod:
        (data, date_months) = (vectors, vectors_date_months)
    source_date = dt
    result = data[(date_months == format_month_day(source_date)) & (data.poblacion == town)]
    if len(result) > 0:
        return np.mean(result.values[:,3:], axis=0)
    return np.array([0.0]*(len(train.columns) - 3))

def basic_predict_all_years_and_temp(dt, town, temperature, prod=False):
    """Predict as mean of all past dates and picking closest temperature"""
    (data, date_months) = (train, train_date_months)
    if prod:
        (data, date_months) = (vectors, vectors_date_months)
    source_date = dt
    day_match = data[(date_months == format_month_day(source_date)) & (data.poblacion == town)]
    temp_match = data[(data.poblacion == town)]
    tmp_diff = np.abs(temp_match.temp - temperature)
    order = np.argsort(tmp_diff)
    result = np.vstack([day_match.values[:,3:], temp_match.values[order,3:][:3,:]])
    
    if len(result) > 0:
        return np.mean(result, axis=0)
    return np.array([0.0]*(len(train.columns) - 3))

def test_predictor(predictor):
    preds = test.apply(lambda row: predictor(row.date, row.poblacion, row.temp), axis=1)
    preds = np.vstack(preds.values)
    #print("Shape: {}".format(preds.shape))
    print("{} MAE: {}".format(predictor.__name__, metrics.mean_absolute_error(test.values[:,3:], preds)))
    #print("{} day level value MAPE: {}".format(predictor.__name__, mean_absolute_percentage_error(test.values[:,2:], preds)))


#row = test.iloc[60]
#print(row)
#basic_predict(row.date, row.poblacion, 0)
#basic_predict_all_years(row.date, row.poblacion, 0)
#basic_predict_all_years_and_temp(row.date, row.poblacion, row.temp)

test_predictor(basic_predict)
test_predictor(basic_predict_all_years)
test_predictor(basic_predict_all_years_and_temp)
from datetime import datetime

towns = sorted(df.poblacion.unique())
selected_date = datetime.now().date()

selected_town = 'Terrassa'
def town_updater(town):
    global selected_town
    selected_town = town
    return town

temperature = 25
def temperature_updated(x):
    global temperature
    temperature = x
    update()
    return x

town_selector = widgets.Select(
    options=towns,
    value=selected_town,
    description='Municipi:',
    disabled=False
)

datepicker = widgets.DatePicker(
    description='Pick a Date',
    disabled=False,
    value=selected_date
)

temp_selector = widgets.IntSlider(
    value=25,
    min=-20,
    max=40,
    step=1,
    description='Temperatura:',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='d'
)

def update(dt,town,temp):
    #print((datepicker.value, town_selector.value, temp_selector.value))
    prediction = basic_predict_all_years_and_temp(dt, town, temp, prod=True)
    plt.figure(figsize=(12,5))
    plt.bar(vectors.columns[3:], prediction)
    plt.title("Forecating {} on {} at {}ºC".format(town, dt, temp))
    plt.ylim((0, max(4, np.max(prediction))))
    plt.xlabel("Pathology")
    plt.ylabel("Count")
    plt.xticks(rotation=90)

interaction = interact(update,dt=datepicker, town=town_selector, temp=temp_selector)