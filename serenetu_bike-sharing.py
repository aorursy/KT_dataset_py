import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from patsy import dmatrices
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
raw_data = pd.read_csv('../input/train.csv', sep = ',')
raw_test = pd.read_csv('../input/test.csv', sep = ',')
print(raw_data.head())
print(raw_test.head())
print(raw_data.info())
print(raw_test.info())

# convert datetime
raw_data['datetime'] = pd.to_datetime(raw_data['datetime'])
raw_test['datetime'] = pd.to_datetime(raw_test['datetime'])

print(raw_data.info())
print(raw_test.info())
# check nan
print(raw_data.isnull().any())
print(raw_test.isnull().any())
# add date column, time column and dayofweek column
def add_date_time_info(data_):
    data = data_.copy(deep = True)
    data['date'] = data['datetime'].dt.date
    data['time'] = data['datetime'].dt.hour
    data['dayofweek'] = data['datetime'].dt.dayofweek
    data['month'] = data['datetime'].dt.month
    data['day'] = data['datetime'].dt.day
    return data
    
raw_data = add_date_time_info(raw_data)
raw_test = add_date_time_info(raw_test)

print(raw_data.head())
print(raw_test.head())
def show_related(data, feature, bins = None, show_info = False):
    if bins is None:
        val_count = data['count'].groupby(data[feature]).count()
        val_sum = data['count'].groupby(data[feature]).sum()
    else:
        val_count = data['count'].groupby(pd.cut(data[feature], bins)).count()
        val_sum = data['count'].groupby(pd.cut(data[feature], bins)).sum()
    if show_info:
        print (val_count)
        print (val_sum)
    val_sum.divide(val_count).plot(kind = 'bar')
    plt.ylabel('counts per hour')
    plt.show()
    return
# Dayofweek Related, Time Related
show_related(raw_data, 'dayofweek')
show_related(raw_data, 'time')
show_related(raw_data, 'month')
show_related(raw_data, 'day')

# From the result, the features of 'time' and 'month' influence the 'counts'
# Season Related
show_related(raw_data, 'season')

# This feature has overlap with 'month', but the 'month' provide more details
# Holiday Related
show_related(raw_data, 'holiday')

# not related
# Workingday Related
show_related(raw_data, 'workingday')

# not related
# Weather Related
show_related(raw_data, 'weather')
# Temp Related
show_related(raw_data, 'temp', bins = 10)
# Atemp Related
show_related(raw_data, 'atemp', bins = 10)
# Humidity Related
show_related(raw_data, 'humidity', bins = 10)
# Windspeed Related
show_related(raw_data, 'windspeed', bins = 10)
'''
label, feature = dmatrices('count~C(time)+C(month)+C(weather)+atemp+humidity+windspeed', raw_data, return_type='dataframe')
label = np.ravel(label)
print (feature.columns)

junck, test = dmatrices('holiday~C(time)+C(month)+C(weather)+atemp+humidity+windspeed', raw_test, return_type='dataframe')
print (test.columns)

def norm(data_, feature_list):
    data = data_.copy(deep = True)
    for feature in feature_list:
        print (feature+':', data[feature].max(), data[feature].min())
        data[feature] = (data[feature] - data[feature].min()) / (data[feature].max() - data[feature].min())
    return data

feature = norm(feature, ['atemp', 'humidity', 'windspeed'])
test = norm(test, ['atemp', 'humidity', 'windspeed'])
print(feature.shape)
print(test.shape)
print(feature[['atemp', 'humidity', 'windspeed']].head())
print(test[['atemp', 'humidity', 'windspeed']].head())
'''

def to_dummies(data_):
    data = data.copy(deep = True)
    data
    return

label = raw_data['count']
feature = raw_data[['time', 'month', 'weather', 'atemp', 'humidity', 'windspeed']]
test = raw_test[['time', 'month', 'weather', 'atemp', 'humidity', 'windspeed']]

print(feature.head())
enc = OneHotEncoder(categorical_features = [0, 1, 2])
feature = enc.fit_transform(feature).toarray()
test = enc.transform(test).toarray()
print(feature.shape)
print(test.shape)

label_train, label_test, feature_train, feature_test = train_test_split(label, feature, test_size = 0.2, random_state = 1)
print (label_train.shape, label_test.shape, feature_train.shape, feature_test.shape)
# Try model 'Ridge', 'Lasso', 'ElasticNet'
model = linear_model.Ridge(alpha=0.1)
model.fit(feature_train, label_train)
print ('Ridge')
print(model.score(feature_train,label_train))
predict = model.predict(feature_test)
for i in range(len(predict)):
    if predict[i] < 0:
        predict[i] = 0
print(mean_squared_log_error(label_test, predict))

model = linear_model.Lasso(alpha=1.1)
model.fit(feature_train, label_train)
print('Lasso')
print(model.score(feature_train,label_train))
predict = model.predict(feature_test)
for i in range(len(predict)):
    if predict[i] < 0:
        predict[i] = 0
print(mean_squared_log_error(label_test, predict))

model = linear_model.ElasticNet()
model.fit(feature_train, label_train)
print ('ElasticNet')
print(model.score(feature_train,label_train))
predict = model.predict(feature_test)
for i in range(len(predict)):
    if predict[i] < 0:
        predict[i] = 0
print(mean_squared_log_error(label_test, predict))
# Choose 'Lasso', Train All
model = linear_model.Lasso(alpha=1.1)
model.fit(feature, label)
print('Lasso Train All')
predict = model.predict(test)
for i in range(len(predict)):
    if predict[i] < 0:
        predict[i] = 0
res = pd.DataFrame({'datetime': raw_test['datetime'], 'count': predict})
print(res.head())
print(res.shape)
res.to_csv('submission.csv', header = True, index = False)
