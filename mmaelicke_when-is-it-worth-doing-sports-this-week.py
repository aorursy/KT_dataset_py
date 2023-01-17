import pandas as pd

import numpy as np

import warnings

from datetime import datetime as dt

from datetime import timedelta as td

from pandas.plotting import scatter_matrix

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.preprocessing import StandardScaler

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics

from sklearn.exceptions import UndefinedMetricWarning
df = pd.read_csv('../input/activities.csv')

df.head()
df['time'] = (df['End Timestamp (Raw Milliseconds)'] - df['Begin Timestamp (Raw Milliseconds)']) / 1000
scatter_matrix(df[['time', 'Elevation Gain (Raw)', 'Distance (Raw)']], figsize=(12,12))
df = df[~(df['Distance (Raw)'] > 1000)]



df = df[~(df['time'] > 100000)]
scatter_matrix(df[['time', 'Elevation Gain (Raw)', 'Distance (Raw)']], figsize=(12,12))
data = df[['Activity ID', 'Activity Name']].copy()

data.columns = ['id', 'name']



# date

data['date'] = df['Begin Timestamp (Raw Milliseconds)'].apply(lambda i: dt.fromtimestamp(i / 1000))



# distance, time, elevation gain

data['time'] = df['time']

data['distance'] = df['Distance (Raw)']

data['elevation_gain'] = df['Elevation Gain (Raw)']



# day of the week with Monday=0

data['dow'] = df['Begin Timestamp (Raw Milliseconds)'].apply(lambda i: dt.fromtimestamp(i / 1000).weekday())



# hour of day 

data['hod'] = df['Begin Timestamp (Raw Milliseconds)'].apply(lambda i: dt.fromtimestamp(i / 1000).hour)



# temperature, wind speed, wind direction, humidity, condition, rainfall

data['temperature'] = df['Temperature (Raw)']

data['wind_speed'] = df['Wind Speed (Raw)']

data['wind_direction'] = df['Wind Direction']

data['humidity'] = df['Humidity (Raw)']

data['sky'] = df['Condition']

data['rainfall'] = df['Rainfall'].map({'no': 0, 'yes':1})
print('Dataset length: %d:\n' % len(data), data.isnull().sum())

data.dropna(axis=0, how='any', inplace=True)

print('After Cleanup: %d\n' % len(data), data.isnull().sum())





data = data.copy()

data.head()
data['load_amount'], data['load_hours'], data['load_gain'] = np.nan, np.nan, np.nan

for i,row in data.iterrows():

    start = (row.date - td(5)).date()

    extract = data[np.logical_and(data.date < row.date,  data.date > start)]

    

    # was there a activity

    if extract.empty:

        data.loc[i, 'load_amount'] = 0

        data.loc[i, 'load_hours'] = 0

        data.loc[i, 'load_gain'] = 0

    else:

        data.loc[i, 'load_amount'] = len(extract)

        data.loc[i, 'load_hours'] = extract.time.sum() / 3600

        data.loc[i, 'load_gain'] = extract.elevation_gain.sum()

    

data.head()
x = data.distance.values

y = data.time.values





lm2 = LinearRegression()

lm2.fit(x.reshape(-1,1), y.reshape(-1,1))



X = np.linspace(np.min(x), np.max(x), 1000)

fig, ax = plt.subplots(1,1, figsize=(12,8))



Y = lm2.predict(X.reshape(1000, 1))

ax.plot(X, Y, '-g', lw=2)

ax.set_xlabel('distance [km]', fontsize=14)

ax.set_ylabel('activity time [s]', fontsize=14)

ax.plot(x, y, '.k')

ax.plot(X, Y * 1.05, '--g')

ax.plot(X, Y * .95, '--g')
Y_ = lm2.predict(x.reshape(-1,1))



# lower 0.95                     ==> good, 0

# lower 1.05 and not lower 0.95  ==> fair, 1

# higher 1.05                    ==> poor, 2

l95 = y < 0.95 * Y_.flatten()

l105 = y < 1.05 * Y_.flatten()

h105 = y > 1.05 * Y_.flatten()





label = y.copy()



label[l95] = 0

label[np.logical_and(l105, ~l95)] = 1

label[h105] = 2



print(label)
fig, ax = plt.subplots(1,1, figsize=(12,8))



ax.plot(X, Y, '-b', lw=2)

ax.set_xlabel('distance [km]', fontsize=14)

ax.set_ylabel('activity time [s]', fontsize=14)

ax.plot(X, Y * 1.05, '--b')

ax.plot(X, Y * .95, '--b')



ax.plot(x[np.where(label==0)], y[np.where(label==0)], '.g')

ax.plot(x[np.where(label==1)], y[np.where(label==1)], '.y')

ax.plot(x[np.where(label==2)], y[np.where(label==2)], '.r')
# adding the target

data['rating'] = label

data.head()
X_train, X_test, y_train, y_test = train_test_split(data.drop('rating', axis=1), 

                                                    data.rating,

                                                    test_size=0.2,

                                                    stratify=data.rating,

                                                    random_state=42

                                                   )



print('Using %d training and %d test sets' % (len(X_train), len(X_test)))
# using the DataFrameSelector I think orgiginally from Aurélien Géron's 

# 'Hands-On Machine Learning with Scikit-Learn & TensorFlow' (O'REILLY)



class DataFrameSelector(BaseEstimator, TransformerMixin):

    def __init__(self, attribute_names):

        self.attribute_names = attribute_names

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        return X[self.attribute_names].values



class Reshaper(BaseEstimator, TransformerMixin):

    def __init__(self):

        pass

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        return X.reshape(-1,1)
numerical_pipeline = Pipeline([

        ('selector', DataFrameSelector(['temperature', 'humidity', 'wind_speed'])),

        ('std_scaler', StandardScaler()),

    ])



categorical_pipeline = Pipeline([

        ('selector', DataFrameSelector('sky')),

        ('OneHot', CountVectorizer()),

    ])



preprocess_waether = FeatureUnion(transformer_list=[

        ('numerical', numerical_pipeline),

        ('categorical', categorical_pipeline),

#        ('rainfall', DataFrameSelector(['rainfall'])),

    ])
prepared_weather = preprocess_waether.fit_transform(X_train).todense()
warnings.filterwarnings('ignore', category=UndefinedMetricWarning)

randomForest = RandomForestClassifier(random_state=42)



param_grid = [

    dict(n_estimators=[3,5,10,30], max_depth=[2,3,5,7], criterion=['gini', 'entropy']),

    dict(n_estimators=[3,5,10,30], max_depth=[2,3,5,7], criterion=['gini', 'entropy'], bootstrap=[False])

]



searcher = GridSearchCV(randomForest, param_grid, refit=True, 

                        scoring=metrics.make_scorer(metrics.f1_score, average='weighted')

                       )

searcher.fit(prepared_weather, y_train)
print('Parameter set:\n', searcher.best_params_)
warnings.filterwarnings('ignore', category=UndefinedMetricWarning)

weatherModel = searcher.best_estimator_



# split my training data into a validation and training dataset

_X_train, _X_val, _y_train, _y_val = train_test_split(prepared_weather, y_train, test_size=0.2, stratify=y_train)



training_errors = list()

validation_errors = list()



for m in range(1, len(_X_train)):

    weatherModel.fit(_X_train[:m], _y_train[:m])

    training_predictions = weatherModel.predict(_X_train[:m])

    validation_prediction = weatherModel.predict(_X_val)

    training_errors.append(1 - metrics.f1_score(training_predictions, _y_train[:m], average='weighted'))

    validation_errors.append(1 - metrics.f1_score(validation_prediction, _y_val, average='weighted'))
fig, ax = plt.subplots(1,1, figsize=(10,6))



ax.plot(range(1, len(_X_train)), training_errors, '-+r', label='training error')

ax.plot(range(1, len(_X_train)), validation_errors, '-+b', label='validation error')

plt.legend(loc='upper right')

ax.set_xlabel('samples', fontsize=14)

ax.set_ylabel('1 - score',fontsize=14)
weatherModel.fit(prepared_weather, y_train)

metrics.confusion_matrix(y_train, weatherModel.predict(prepared_weather))
prepared_test = preprocess_waether.transform(X_test)



print('Score:', weatherModel.score(prepared_test, y_test))

metrics.confusion_matrix(y_test, weatherModel.predict(prepared_test))
print('sky:', data.sky.unique())

print('wind_speed:', np.percentile(data.wind_speed.values, 80))



data2 = data[['id', 'dow', 'hod', 'humidity', 'distance']].copy()



data2['sky'] = data.sky.replace({

        'mostly clear': 'fair',

        'partly cloudy': 'cloudy', 'mostly cloudy': 'cloudy', 'mist':'cloudy',

        'light rain': 'rain', 'showers': 'rain', 'thunderstorm':'rain'

    })



data2['strong_wind'] = data.wind_speed > np.percentile(data.wind_speed.values, 80)



def f(t):

    if t < 17:

        return 'cold'

    elif t > 21:

        return 'warm'

    else:

        return 'fair'

data2['temperature'] = data.temperature.apply(f)

data2['rainfall'] = data.rainfall.astype(bool)

data2['rating'] = data.rating



data2 = data2.copy(deep=True)

print(data2.head())
Xf_train, Xf_test, yf_train, yf_test = train_test_split(data2.drop('rating', axis=1), 

                                                    data2.rating,

                                                    test_size=0.2,

                                                    stratify=data2.rating,

                                                    random_state=42

                                                   )



print('Using %d training and %d test sets' % (len(Xf_train), len(Xf_test)))
sky_vectorizer = CountVectorizer()

temperature_vectorizer = CountVectorizer()



numerical_pipeline = Pipeline([

        ('selector', DataFrameSelector(['distance', 'hod', 'dow', 'humidity'])),

        ('std_scaler', StandardScaler()),

    ])



multi_categorical_pipeline = FeatureUnion(transformer_list=[

        ('sky', Pipeline([

                    ('sky_selector', DataFrameSelector('sky')),

                    ('sky_OneHot', temperature_vectorizer),

        ])),

        ('temperature', Pipeline([

                    ('sky_selector', DataFrameSelector('temperature')),

                    ('sky_OneHot', sky_vectorizer),

        ])),

    ])



binary_pipeline = Pipeline([

        ('selector', DataFrameSelector(['strong_wind', 'rainfall'])),

    ])





preprocess_full = FeatureUnion(transformer_list=[

        ('numerical', numerical_pipeline),

        ('categorical', multi_categorical_pipeline),

        ('binary', binary_pipeline),

    ])
prepared_full = preprocess_full.fit_transform(Xf_train).todense()

print('Working on {} shaped data'.format(prepared_full.shape))
warnings.filterwarnings('ignore', category=UndefinedMetricWarning)

randomForest = RandomForestClassifier(random_state=42)



param_grid = [

    dict(n_estimators=[3,5,10,30], max_depth=[2,3,5,7], criterion=['gini', 'entropy']),

    dict(n_estimators=[3,5,10,30], max_depth=[2,3,5,7], criterion=['gini', 'entropy'], bootstrap=[False])

]



searcher = GridSearchCV(randomForest, param_grid, refit=True, 

                        scoring=metrics.make_scorer(metrics.f1_score, average='weighted')

                       )

searcher.fit(prepared_full, yf_train)
print('Best parameter set:\n', searcher.best_params_)
warnings.filterwarnings('ignore', category=UndefinedMetricWarning)

fullModel = searcher.best_estimator_



# split my training data into a validation and training dataset

_X_train, _X_val, _y_train, _y_val = train_test_split(prepared_full, yf_train, test_size=0.2, stratify=yf_train, random_state=42)



training_errors = list()

validation_errors = list()



for m in range(1, len(_X_train)):

    fullModel.fit(_X_train[:m], _y_train[:m])

    training_predictions = fullModel.predict(_X_train[:m])

    validation_prediction = fullModel.predict(_X_val)

    training_errors.append(1 - metrics.f1_score(training_predictions, _y_train[:m], average='weighted'))

    validation_errors.append(1 - metrics.f1_score(validation_prediction, _y_val, average='weighted'))
fig, ax = plt.subplots(1,1, figsize=(10,6))



ax.plot(range(1, len(_X_train)), training_errors, '-+r', label='training error')

ax.plot(range(1, len(_X_train)), validation_errors, '-+b', label='validation error')

plt.legend(loc='upper right')

ax.set_xlabel('samples', fontsize=14)

ax.set_ylabel('1 - score', fontsize=14)
fullModel.fit(prepared_full, yf_train)

metrics.confusion_matrix(yf_train, fullModel.predict(prepared_full))
prepared_fulltest = preprocess_full.transform(Xf_test).todense()



print('Score:', fullModel.score(prepared_fulltest, yf_test))

metrics.confusion_matrix(yf_test, fullModel.predict(prepared_fulltest))
from sklearn.model_selection import cross_val_score



cross_val_score(RandomForestClassifier(random_state=42),

    preprocess_full.fit_transform(data2.drop('rating', axis=1)).todense(),

    data2.rating,

    cv=5, scoring='f1_weighted'

)
used_data = ['distance', 'hod', 'dow', 'humidity']

used_data.extend(sky_vectorizer.get_feature_names())

used_data.extend(temperature_vectorizer.get_feature_names())

used_data.extend(['strong_wind', 'rainfall'])

list(zip(used_data, fullModel.feature_importances_))
grps = data2.distance.groupby(data2.rating)

print(grps)



for rating, dist in grps:

    dist.hist(label='%.0f' % rating)

plt.legend(loc='upper right')
# CAUTION: the commented can't be run in kaggle.com

# you can download the kernel and use the commented part from a local machine.

# all you need is a OpenWeatherMap API Key and put it into the URL. Then this cell will use

# real time forecating data instead of the used downloaded response.





#import requests

#url = 'http://api.openweathermap.org/data/2.5/forecast?q=Freiburg,de&APPID=YOUR_OWM_APP_KEY'

#

#response = requests.get(url).json()



# if you use the part above, comment these linese

import json

with open('../input/response.json') as f:

    response = json.load(f)
def map_conditions(code):

    if code == 800 or code == 801:

        return 'fair'

    # drizzle, rain, thunderstorm, atmosphere (well, yes...), snow

    elif code <= 799 or code >= 900:

        return 'rain'

    elif code >= 802 and code <= 804:

        return 'cloudy'

    
print(response.keys())

weather = response['list']

print(weather[0].keys())

print(weather[0]['weather'])



temperature = [e['main']['temp'] - 273.15 for e in weather]

humidity = [e['main']['humidity'] for e in weather]

wind = [e['wind']['speed'] for e in weather]

sky = [e['weather'][0]['id'] for e in weather]

dates = [dt.strptime(e['dt_txt'], '%Y-%m-%d %H:%M:%S') for e in weather]

rain = ['rain' in e for e in weather]



forecast = pd.DataFrame({'date': dates, 'temperature':temperature, 'humidity':humidity, 

                         'strong_wind':wind, 'sky': sky, 'rainfall': rain})

forecast.head()
forecast['strong_wind'] = forecast.strong_wind > np.percentile(data.wind_speed.values, 80)

forecast['sky'] = forecast.sky.apply(map_conditions)



def f(t):

    if t < 17:

        return 'cold'

    elif t > 21:

        return 'warm'

    else:

        return 'fair'

forecast['temperature'] = forecast.temperature.apply(f)

forecast['hod'] = forecast.date.apply(lambda t: t.hour)

forecast['dow'] = forecast.date.apply(lambda t: t.weekday())



forecast.head()
from matplotlib.colors import ListedColormap

distances = [5., 6., 7., 8., 9., 10., 11., 12., 15.]

predictions = np.ones(shape=(len(forecast), 9)) * 99

for i, distance in enumerate(distances):

    fcXkm = forecast.copy()

    fcXkm['distance'] = distance

    predictions[:,i] = fullModel.predict(preprocess_full.transform(fcXkm).todense())

    

#print(predictions)

result = pd.DataFrame(index=forecast.date, data=predictions, columns=['%d km' % d for d in distances])

def styler(v):

    if v == 0.0:

        c = 'green'

    elif v == 1.0:

        c = 'yellow'

    else:

        c = 'red'

    return 'background-color: %s' % c

result.style.applymap(styler)
