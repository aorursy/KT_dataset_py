!pip install folium
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import folium                                 # Visualização de mapas

import catboost



from matplotlib import pyplot as plt

%matplotlib inline

from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

from sklearn.compose import ColumnTransformer

from folium.plugins import HeatMap

from folium.plugins import FastMarkerCluster

from sklearn.metrics import log_loss



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('../input/sf-crime/train.csv')

df_test = pd.read_csv('../input/sf-crime/test.csv')
df_train.head()
df_test.head()
# def RdateTime(datetime):

#     dates = []

#     times = []

    

#     for date, time in datetime:

#         dates.append(date)

#         times.append(time)

        

#     dates_ = []

#     for date in dates:

#         dates_.append(date.split('-'))

        

#     dates_ = np.array(dates_)

    

#     times_ = []

#     for date in times:

#         times_.append(date.split(':'))

    

#     times_ = np.array(times_)

    

#     return dates_, times_
df_train.describe()
df_train.info()
print(df_train.duplicated().sum())

df_train.drop_duplicates(inplace=True)

print(df_train.duplicated().sum())
y = df_train['Category']

df_train_description = df_train['Descript']

df_train_resolution = df_train['Resolution']
df_train.head()
test_ID = df_test["Id"]

df_test.drop("Id", axis=1, inplace=True)
df_test.head()
y.value_counts()
le = LabelEncoder()

y = le.fit_transform(y)

print(le.classes_)
def create_update_time_of_day_features(df_, dic_labels):

    """

        Create a feature time of day or period from datatime

            Examples: 

                datetime1 = 2019-04-28 02:00:56 -> period = early morning

                datetime2 = 2019-04-28 08:00:56 -> period = morning

                datetime3 = 2019-04-28 14:00:56 -> period = afternoon

                datetime4 = 2019-04-28 20:00:56 -> period = evening

    """

    try:

        print('\nCreating or updating period feature\n...early morning from 0H to 6H\n...morning from 6H to 12H\n...afternoon from 12H to 18H\n...evening from 18H to 24H')

        conditions =   [(df_[dic_labels['datetime']].dt.hour >= 0) & (df_[dic_labels['datetime']].dt.hour < 6), 

                        (df_[dic_labels['datetime']].dt.hour >= 6) & (df_[dic_labels['datetime']].dt.hour < 12),

                        (df_[dic_labels['datetime']].dt.hour >= 12) & (df_[dic_labels['datetime']].dt.hour < 18),  

                        (df_[dic_labels['datetime']].dt.hour >= 18) & (df_[dic_labels['datetime']].dt.hour < 24)]

        choices = ['early morning', 'morning', 'afternoon', 'evening']

        df_['PeriodOfDay'] = np.select(conditions, choices, 'undefined')      

        print('...the period of day feature was created')

    except Exception as e:

        raise e
df_train['Dates'] = pd.to_datetime(df_train['Dates'])

df_test['Dates'] = pd.to_datetime(df_test['Dates'])

create_update_time_of_day_features(df_train,dic_labels={'datetime' : 'Dates'})

create_update_time_of_day_features(df_test,dic_labels={'datetime' : 'Dates'})
# date = pd.to_datetime(df_train['Dates'])

df_train['year'] = df_train['Dates'].dt.year

df_train['month'] = df_train['Dates'].dt.month

df_train['day'] = df_train['Dates'].dt.day

df_train['hour'] = df_train['Dates'].dt.hour

df_train['minute'] = df_train['Dates'].dt.minute

df_train["n_days"] = (df_train['Dates'] - df_train['Dates'].min()).apply(lambda x: x.days)

df_train.drop("Dates", axis=1, inplace=True)
# date_test = pd.to_datetime(df_test['Dates'])

df_test['year'] = df_test['Dates'].dt.year

df_test['month'] = df_test['Dates'].dt.month

df_test['day'] = df_test['Dates'].dt.day

df_test['hour'] = df_test['Dates'].dt.hour

df_test['minute'] = df_test['Dates'].dt.minute

df_test['special_time'] = df_train['minute'].isin([0, 30]).astype(int)

df_test["n_days"] = (df_test['Dates'] - df_test['Dates'].min()).apply(lambda x: x.days)

df_test.drop("Dates", axis=1, inplace=True)
df_train.head()
df_train["DayOfWeek"].value_counts()
df_train["DayOfWeek"].value_counts()
import folium



saoFrancisco = folium.Map(

    location=[37.762657, -122.435792],

    zoom_start=12

)

for _, ponto in df_train.head(1000).iterrows():

    folium.Marker(

       location=[ponto['Y'], ponto['X']]

    ).add_to(saoFrancisco)



saoFrancisco
df_train['Category'].value_counts().plot(kind='bar', figsize=(10,6));
df_train['PdDistrict'].value_counts().plot(kind='bar', figsize=(10,6));
df_train['DayOfWeek'].value_counts().plot(kind='bar', figsize=(10,6)); 
df_train['PeriodOfDay'].value_counts().plot(kind='bar', figsize=(10,6));
df_train[df_train['DayOfWeek'] == 'Friday']['PeriodOfDay'].value_counts().plot(kind='bar', figsize=(10,6));
df_train[df_train['PdDistrict'] == 'SOUTHERN']['DayOfWeek'].value_counts().plot(kind='bar', figsize=(10,6));
df_train[df_train['PdDistrict'] == 'SOUTHERN']['PeriodOfDay'].value_counts().plot(kind='bar', figsize=(10,6));
SOUTHERN = df_train[df_train['PdDistrict'] == 'SOUTHERN']

SOUTHERN[SOUTHERN['DayOfWeek'] == 'Friday']['PeriodOfDay'].value_counts().plot(kind='bar', figsize=(10,6));
df_train.drop(["Category", "Descript", "Resolution", "Address"], axis=1, inplace=True)

# df_train.drop("Address", axis=1, inplace=True)

df_test.drop("Address", axis=1, inplace=True)
df_train.head()
df_test.head()
categorical_features = ["DayOfWeek", "PdDistrict", "PeriodOfDay"]

ct = ColumnTransformer(transformers=[("categorical_features", OrdinalEncoder(), categorical_features)],

                       remainder="passthrough")

df_train = ct.fit_transform(df_train)
categorical_features = ["DayOfWeek", "PdDistrict","PeriodOfDay"]

ct = ColumnTransformer(transformers=[("categorical_features", OrdinalEncoder(), categorical_features)],

                       remainder="passthrough")

df_test = ct.fit_transform(df_test)
df_train
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_train, y, test_size=0.25, random_state=42,stratify=y)
from sklearn.preprocessing import StandardScaler
std = StandardScaler()

X_train_std = std.fit_transform(X_train)

X_test_std = std.transform(X_test)

X_predict_std = StandardScaler().fit_transform(df_test)
cbc = catboost.CatBoostClassifier(n_estimators=5000, learning_rate=0.05,

                                  random_seed=0, task_type="GPU", verbose=50)
cbc.fit(X_train_std, y_train)

prob = cbc.predict_proba(X_test_std)
from sklearn.linear_model import SGDClassifier
SGDC = SGDClassifier(max_iter=100,tol=0.01)
SGDC.fit(X_train_std,y_train)

y_pred_SGDC = SGDC.predict(X_test_std)
from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_score

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import mean_squared_error
# print(' Accuracy|',accuracy_score(y_test,y_pred_SGDC))

# print('Precision|',precision_score(y_test,y_pred_SGDC,average='macro'))

print('      MAE|',mean_absolute_error(y_test,y_pred_SGDC))

print('      MSE|',mean_squared_error(y_test,y_pred_SGDC))

print('     RMSE|',np.sqrt(mean_squared_error(y_test,y_pred_SGDC)))
y_train = np.float64(y_train)
from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(n_estimators=50,criterion='entropy',max_depth=10)

RF.fit(X_train_std,y_train)
y_pred_RF = RF.predict(X_test_std)
print(' Accuracy|',accuracy_score(y_test,y_pred_RF))

print('Precision|',precision_score(y_test,y_pred_RF,average='macro'))

print('      MAE|',mean_absolute_error(y_test,y_pred_RF))

print('      MSE|',mean_squared_error(y_test,y_pred_RF))

print('     RMSE|',np.sqrt(mean_squared_error(y_test,y_pred_RF))) 

# print(' LOG_LOSS|',log_loss(y_test,y_pred_RF,labels=y_test))
RF2 = RandomForestClassifier(n_estimators=50,criterion='entropy',max_depth=10)

RF2.fit(X_train_std,y_train)
from keras.models import Sequential

from keras.layers import SimpleRNN, Embedding, Dense
model = Sequential()
model.add(Dense(26,input_dim=26, activation='relu'))

model.add(Dense(13,activation='relu'))

model.add(Dense(6,activation='relu'))

model.add(Dense(1,activation='relu'))
model.summary()
model.compile(optimizer='sgd',loss='mse',metrics=['mae','acc'])
from keras.callbacks import EarlyStopping
callback_early = EarlyStopping(min_delta=0.01,patience=20)
history = model.fit(x=X_train_std,y=y_train, batch_size=10, epochs=50,validation_split=0.1,callbacks=[callback_early])
loss, mae, acc = model.evaluate(X_test_std,y_test)

loss, mae, acc
ypred_RN = model.predict(X_test_std)
print('      MAE|',mean_absolute_error(y_test,ypred_RN))

print('      MSE|',mean_squared_error(y_test,ypred_RN))

print('     RMSE|',np.sqrt(mean_squared_error(y_test,ypred_RN)))
X2 = df_train.drop('Category', axis=1).values

y2 = df_train['Category'].values
X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size=0.15, random_state=42,stratify=y)
df_test['PeriodOfDay_undefined']
test = df_test.drop('Id',axis=1)

# test = test.drop('PeriodOfDay_undefined',axis=1)
test.info()
test = test.values
std = StandardScaler()

X_train = std.fit_transform(X_train)

X_test = std.transform(X_test)
test = StandardScaler().fit_transform(test);
RF = RandomForestClassifier(n_estimators=50,criterion='entropy',max_depth=10)
RF.fit(X_train,y_train)
y_pred_RF = RF.predict(X_predict_std)
Y = df_train['Category'].astype('category')
np.count_nonzero(Y.cat.categories.unique()), np.count_nonzero(np.unique(y_pred_RF))
submit = pd.DataFrame({'Id': df_test.Id.tolist()})

for category in Y.cat.categories:

    submit[category] = np.where(y_pred_RF == category, 1, 0)
submit.to_csv('San_Francisco_Crime_RF.csv', index = False)