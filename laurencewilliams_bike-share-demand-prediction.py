import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



plt.rc("font", size=18)

sns.set(style="white")

sns.set(style="whitegrid", color_codes=True)
train = pd.read_csv('../input/bike-sharing-demand/train.csv')

test = pd.read_csv('../input/bike-sharing-demand/test.csv')

train.head()
unique_values = {}

for i in range(1, len(train.columns)-3):

    unique_values[train.columns[i]] = train[train.columns[i]].unique()

unique_values
train.describe()
train.isnull().sum()
#Datetime:



datasets = [train, test]



for dataset in datasets:

    dataset['datetime'] = pd.to_datetime(dataset.datetime)

    dataset['hour'] = dataset['datetime'].apply(lambda x: x.hour)

    dataset['day'] = dataset['datetime'].apply(lambda x: x.day)

    dataset['weekday'] = dataset['datetime'].apply(lambda x: x.weekday())

    dataset['month'] = dataset['datetime'].apply(lambda x: x.month)

    dataset['year'] = dataset['datetime'].apply(lambda x: x.year)
train.head(2)
test.head(2)
#Names for categorical data:

train_c = train.copy()

train_c['weather'] = train_c['weather'].map({1: 'Good', 2: 'Medium', 3: 'Bad', 4: 'Very Bad'})

train_c['weekday'] = train_c['weekday'].map({0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thur', 4: 'Fri',

                                            5: 'Sat', 6: 'Sun'})

train_c['month'] = train_c['month'].map({1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 

                                        6: 'Jun', 7: 'July', 8: 'Aug', 9: 'Sept', 10: 'Oct',

                                         11: 'Nov', 12: 'Dec'})

train_c['season'] = train_c['season'].map({1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'})

train_c['workingday'] = train_c['workingday'].map({0: 'No', 1: 'Yes'})

train_c['holiday'] = train_c['holiday'].map({0: 'No', 1: 'Yes'})
from numpy import mean

fig, ax = plt.subplots(nrows=2, ncols=2, figsize = (12,8))

sns.barplot(x = 'season', y = 'count', data = train_c, ci=None, color='salmon',

            hue = 'year', estimator = mean, ax =ax[0,0])

ax[0,0].set_title('Mean Count by Season hue: Year')

sns.barplot(x = 'season', y = 'count', data = train_c, ci=None, 

            color = 'salmon', hue = 'weather', estimator = mean, ax = ax[0,1])

ax[0,1].set_title('Mean Count by Season hue: Weather')

sns.barplot(x = 'month', y = 'count', data = train_c, ci=None, 

            color = 'indigo', hue = 'year', estimator = mean, ax = ax[1,0])

ax[1,0].set_title('Mean Count by Month hue: Year')

sns.barplot(x = 'month', y = 'count', data = train_c, ci=None, 

            color = 'indigo', hue = 'weather', estimator = mean, ax = ax[1,1])

ax[1,1].set_title('Mean Count by Season hue: Weather')

plt.tight_layout()
fig, ax = plt.subplots(nrows=1, ncols=4, figsize = (16,5))

sns.distplot(train_c['windspeed'], ax=ax[0])

ax[0].set_title('Distplot windspeed')

sns.distplot(train_c['temp'], ax=ax[1])

ax[1].set_title('Distplot temperature')

sns.distplot(train_c['atemp'], ax=ax[2])

ax[2].set_title('Distplot atemperature')

sns.distplot(train_c['humidity'], ax=ax[3])

ax[3].set_title('Distplot humidity')

plt.tight_layout()
fig, ax = plt.subplots(nrows=4, ncols=1, figsize = (12,12))

sns.boxplot(x='season',y='windspeed', hue= 'weather', data=train_c, palette='winter', ax = ax[0])

ax[0].set_title('Boxplot Wincdspeed by Season: Hue Weather')

sns.boxplot(x='season',y='temp', hue= 'weather', data=train_c, palette='winter', ax = ax[1])

ax[1].set_title('Boxplot Temperature by Season: Hue Weather')

sns.boxplot(x='season',y='atemp', hue= 'weather', data=train_c, palette='winter', ax = ax[2])

ax[2].set_title('Boxplot ATemperature by Season: Hue Weather')

sns.boxplot(x='season',y='humidity', hue= 'weather', data=train_c, palette='winter', ax = ax[3])

ax[3].set_title('Boxplot Humidity by Season: Hue Weather')

plt.tight_layout()
fig, ax = plt.subplots(1, figsize = (12,8))

grouped_hours = pd.DataFrame(train_c.groupby(['hour'], sort=True)['casual', 'registered', 'count'].mean())

grouped_hours.plot(ax=ax)

ax.set_xticks(grouped_hours.index.to_list())

ax.set_xticklabels(grouped_hours.index)

plt.xticks(rotation=45)

plt.title('Avg Count by Hour')
fig, ax = plt.subplots(1, figsize = (12,8))

sns.barplot(x = 'weekday', y = 'count', data = train_c, ci=None, 

            color = 'indigo', estimator = mean, ax = ax)

ax.set_title('Avg Count by Weekday')
fig, ax = plt.subplots(nrows=1, ncols=2, figsize = (15,8))



workingday = train_c.loc[train_c.workingday == 'Yes']

not_workingday = train_c.loc[train_c.workingday == 'No']

grouped_workingday = pd.DataFrame(workingday.groupby(['hour'], sort=True)['count'].mean())

grouped_notworkingday = pd.DataFrame(not_workingday.groupby(['hour'], sort=True)['count'].mean())



grouped_workingday.plot(ax=ax[0])

ax[0].set_xticks(grouped_workingday.index.to_list())

ax[0].set_xticklabels(grouped_workingday.index)

ax[0].tick_params(labelrotation=45)

ax[0].set_title('Avg Count by Hour - Working Day')



grouped_notworkingday.plot(ax=ax[1])

ax[1].set_xticks(grouped_notworkingday.index.to_list())

ax[1].set_xticklabels(grouped_notworkingday.index)

ax[1].tick_params(labelrotation=45)

ax[1].set_title('Avg Count by Hour - Not Working Day')
sns.set(style="ticks")
sns.pairplot(data=train_c,

                  y_vars=['count'],

                  x_vars=['temp', 'atemp', 'humidity', 'windspeed'])
sns.pairplot(data=train_c,

                  y_vars=['registered'],

                  x_vars=['temp', 'atemp', 'humidity', 'windspeed'])
sns.pairplot(data=train_c,

                  y_vars=['casual'],

                  x_vars=['temp', 'atemp', 'humidity', 'windspeed'])
Q1 = train.quantile(0.25)

Q3 = train.quantile(0.75)

IQR = Q3 - Q1
train = train.drop(['datetime'], axis = 1)
train_without_outliers =train[~((train < (Q1 - 1.5 * IQR)) |(train > (Q3 + 1.5 * IQR))).any(axis=1)]
print("train original shape", train.shape[0])

print("train_without_outliers observations", train_without_outliers.shape[0])
fig, ax = plt.subplots(nrows=4, ncols=2, figsize = (12,12))



sns.boxplot(x='season',y='windspeed', data=train, palette='winter', ax = ax[0,0])

ax[0,0].set_title('Boxplot Wincdspeed by Season WITH OUTLIER')

sns.boxplot(x='season',y='windspeed', data=train_without_outliers, palette='winter', ax = ax[0,1])

ax[0,1].set_title('Boxplot Wincdspeed by Season WITHOUT OUTLIER')



sns.boxplot(x='season',y='temp', data=train, palette='winter', ax = ax[1,0])

ax[1,0].set_title('Boxplot Temperature by Season WITH OUTLIERS')

sns.boxplot(x='season',y='temp', data=train_without_outliers, palette='winter', ax = ax[1,1])

ax[1,1].set_title('Boxplot Temperature by Season WITHOUT OUTLIERS')





sns.boxplot(x='season',y='atemp', data=train, palette='winter', ax = ax[2,0])

ax[2,0].set_title('Boxplot ATemperature WITH OUTLIERS')

sns.boxplot(x='season',y='atemp', data=train_without_outliers, palette='winter', ax = ax[2,1])

ax[2,1].set_title('Boxplot ATemperature by Season WITHOUT OUTLIERES')



sns.boxplot(x='season',y='humidity', data=train, palette='winter', ax = ax[3,0])

ax[3,0].set_title('Boxplot Humidity by Season WITH OUTLIERS')

sns.boxplot(x='season',y='humidity', data=train_without_outliers, palette='winter', ax = ax[3,1])

ax[3,1].set_title('Boxplot Humidity by Season WITHOUT OUTLIERS')



plt.tight_layout()
train.corr()

mask = np.array(train.corr())

mask[np.tril_indices_from(mask)] = False

fig,ax= plt.subplots()

fig.set_size_inches(30,15)

sns.heatmap(train.corr(), mask = mask, vmax = 0.8, square=True, annot=True, center = 0, 

            cmap="RdBu_r", linewidths=.5)
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split
X= train.drop(['count', 'casual', 'registered'], axis = 1)

y = train['count']



X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.3, random_state=5)



rf = RandomForestRegressor(n_estimators=100, random_state=2)
rf.fit(X_train, Y_train)
%matplotlib inline

import matplotlib as mp

plt.subplots(figsize=(15,10))

core_variables = pd.Series(rf.feature_importances_, index=X.columns)

core_variables = core_variables.nlargest(8)



# Colorize the graph based on likeability:

likeability_scores = np.array(core_variables)

 

data_normalizer = mp.colors.Normalize()

color_map = mp.colors.LinearSegmentedColormap(

    "my_map",

    {

        "red": [(0, 1.0, 1.0),

                (1.0, .5, .5)],

        "green": [(0, 0.5, 0.5),

                  (1.0, 0, 0)],

        "blue": [(0, 0.50, 0.5),

                 (1.0, 0, 0)]

    }

)



plt.title('Most Important Features')



#make the plot

core_variables.plot(kind='barh', color=color_map(data_normalizer(likeability_scores)))
from sklearn import preprocessing

from sklearn.preprocessing import StandardScaler

continuous_features = ['temp','atemp', 'humidity', 'windspeed']

data = [train_without_outliers]

for dataset in data:

    for col in continuous_features:

        transf = dataset[col].values.reshape(-1,1)

        scaler = preprocessing.StandardScaler().fit(transf)

        dataset[col] = scaler.transform(transf)

train_without_outliers.reset_index()
X= train_without_outliers.drop(['count', 'casual', 'registered'], axis = 1)

y = train_without_outliers['count']



X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.3, random_state=5)



rf_without_outliers = RandomForestRegressor(n_estimators=100, random_state=2)
rf_without_outliers.fit(X_train, Y_train)
plt.subplots(figsize=(15,10))

core_variables_without_outliers = pd.Series(rf_without_outliers.feature_importances_, index=X.columns)

core_variables_without_outliers = core_variables_without_outliers.nlargest(8)



# Colorize the graph based on likeability:

likeability_scores = np.array(core_variables)

 

data_normalizer = mp.colors.Normalize()

color_map = mp.colors.LinearSegmentedColormap(

    "my_map",

    {

        "red": [(0, 1.0, 1.0),

                (1.0, .5, .5)],

        "green": [(0, 0.5, 0.5),

                  (1.0, 0, 0)],

        "blue": [(0, 0.50, 0.5),

                 (1.0, 0, 0)]

    }

)



#make the plot

core_variables_without_outliers.plot(kind='barh', color=color_map(data_normalizer(likeability_scores)))
fig, ax = plt.subplots(nrows=1, ncols=2, figsize = (20,10))

core_variables.plot(kind='barh', color=color_map(data_normalizer(likeability_scores)), ax=ax[0])

ax[0].set_title('With outliers significance plot')

core_variables_without_outliers.plot(kind='barh', color=color_map(data_normalizer(likeability_scores)), ax=ax[1])

ax[1].set_title('Without outliers significance plot')
from sklearn.preprocessing import RobustScaler
X = train.drop(['count', 'casual', 'registered'], axis =1)

y = train['count']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=2)
transformer = RobustScaler().fit(X_train)

rescaled_X_train = transformer.transform(X_train)



transformer = RobustScaler().fit(X_test)

rescaled_X_test = transformer.transform(X_test)



y_train= y_train.values.reshape(-1,1)

y_test= y_test.values.reshape(-1,1)



transformer = RobustScaler().fit(y_train)

rescaled_y_train = transformer.transform(y_train)



transformer = RobustScaler().fit(y_test)

rescaled_y_test = transformer.transform(y_test)
rf = RandomForestRegressor(n_estimators=100)

rf.fit(rescaled_X_train, rescaled_y_train)
from sklearn.metrics import mean_squared_error

from sklearn import metrics

rf_prediction = rf.predict(rescaled_X_test)

print('MSE:', metrics.mean_squared_error(rescaled_y_test, rf_prediction))
plt.scatter(rescaled_y_test,rf_prediction)
from sklearn.preprocessing import MinMaxScaler
X = train.drop(['count', 'casual', 'registered'], axis =1)

y = train['count']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=2)
y_train= y_train.values.reshape(-1,1)

y_test= y_test.values.reshape(-1,1)



sc_X = MinMaxScaler()

sc_y = MinMaxScaler()

X_train = sc_X.fit_transform(X_train)

X_test = sc_X.fit_transform(X_test)

y_train = sc_y.fit_transform(y_train)

y_test = sc_y.fit_transform(y_test)
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=100)

rf.fit(X_train, y_train)
rf_prediction = rf.predict(X_test)

print('MSE:', metrics.mean_squared_error(y_test, rf_prediction))
plt.scatter(y_test,rf_prediction)
from sklearn.preprocessing import StandardScaler
train.head(2)
continuous_features= ['temp','atemp', 'humidity', 'windspeed', 'count']

train_copy = train.copy()

for col in continuous_features:

    transf = train_copy[col].values.reshape(-1,1)

    scaler = preprocessing.StandardScaler().fit(transf)

    train_copy[col] = scaler.transform(transf)
X = train_copy.drop(['count', 'casual', 'registered'], axis =1)

y = train_copy['count']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=2)
transformer = StandardScaler().fit(X_train)

standard_X_train = transformer.transform(X_train)



transformer = StandardScaler().fit(X_test)

standard_X_test = transformer.transform(X_test)



y_train= y_train.values.reshape(-1,1)

y_test= y_test.values.reshape(-1,1)



transformer = StandardScaler().fit(y_train)

standard_y_train = transformer.transform(y_train)



transformer = StandardScaler().fit(y_test)

standard_y_test = transformer.transform(y_test)
rf = RandomForestRegressor(n_estimators=100)

rf.fit(X_train, y_train)
rf_prediction = rf.predict(standard_X_test)

print('MSE:', metrics.mean_squared_error(standard_y_test, rf_prediction))
plt.scatter(standard_y_test,rf_prediction)
X = train_without_outliers[['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp','humidity', 'year', 

                            'month', 'day', 'hour', 'weekday','windspeed']]

y = train_without_outliers['count']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.1)
y_train= y_train.values.reshape(-1,1)

y_test= y_test.values.reshape(-1,1)



sc_X = MinMaxScaler()

sc_y = MinMaxScaler()

X_train = sc_X.fit_transform(X_train)

X_test = sc_X.fit_transform(X_test)

y_train = sc_X.fit_transform(y_train)

y_test = sc_y.fit_transform(y_test)
rf = RandomForestRegressor(n_estimators=100)

rf.fit(X_train, y_train)
rf_prediction = rf.predict(X_test)
print('MSE:', metrics.mean_squared_error(y_test, rf_prediction))
test[['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp','humidity', 'year', 'month', 'day', 'hour',

     'weekday','windspeed']] = sc_X.fit_transform(test[['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp','humidity', 

                                                          'year', 'month', 'day', 'hour', 'weekday','windspeed']])
test_pred= rf.predict(test[['season', 'holiday', 'workingday', 

                            'weather', 'temp', 'atemp','humidity', 'year', 'month', 'day', 

                            'hour', 'weekday','windspeed']])
test_pred=test_pred.reshape(-1,1)

test_pred.shape
test_pred
test_pred = sc_y.inverse_transform(test_pred)
test_pred
test_pred = pd.DataFrame(test_pred, columns=['count'])
submission1 = pd.concat([test['datetime'], test_pred],axis=1)
submission1.head()
submission1.dtypes
submission1['count'] = submission1['count'].astype('int')
submission1.to_csv('submission1.csv', index=False)
X = train_without_outliers[['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp','humidity', 'year', 

                            'month', 'day', 'hour', 'weekday','windspeed']]

y = train_without_outliers['count']



X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.1)
rescaled_X_train = RobustScaler().fit_transform(X_train)



rescaled_X_test = RobustScaler().fit_transform(X_test)



y_train= y_train.values.reshape(-1,1)

y_test= y_test.values.reshape(-1,1)



rescaled_y_train = RobustScaler().fit_transform(y_train)



rescaled_y_test = RobustScaler().fit_transform(y_test)
rf = RandomForestRegressor(n_estimators=100)

rf.fit(rescaled_X_train, rescaled_y_train)
rf_prediction = rf.predict(rescaled_X_test)
test[['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp','humidity', 'year', 'month', 'day', 'hour',

     'weekday','windspeed']] = sc_X.fit_transform(test[['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp','humidity', 

                                                          'year', 'month', 'day', 'hour', 'weekday','windspeed']])



test_pred= rf.predict(test[['season', 'holiday', 'workingday', 

                            'weather', 'temp', 'atemp','humidity', 'year', 'month', 'day', 

                            'hour', 'weekday','windspeed']])
test_pred=test_pred.reshape(-1,1)

test_pred.shape
test_pred = transformer.inverse_transform(test_pred)
test_pred