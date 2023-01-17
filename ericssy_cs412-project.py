# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from sklearn import preprocessing


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 50)

train = pd.read_csv("/kaggle/input/bigquery-geotab-intersection-congestion/train.csv")
test = pd.read_csv("/kaggle/input/bigquery-geotab-intersection-congestion/test.csv")
train.head()

test.head()
# address_encoding = {"Street": "Street",
#  "St": "Street",
#  "Avenue": "Avenue",
#  "Ave": "Avenue",
#  "Boulevard": "Boulevard",
#  "Road": "Road",
#  "Drive": "Drive",
#  "Lane": "Lane",
#  "Tunnel":"Tunnel",
#  "Highway": "Highway",
#  "Way":"Way",
#  "Parkway":"Parkway",
#  "Parking": "Parking",
#  "Oval": "Oval",
#  "Square": "Square",
#  "Place": "Place",
#  "Bridge": "Bridge"} 

address_encoding = {
    "Street": 0,
     "St": 0,
     "Avenue": 1,
     "Ave": 1,
     "Boulevard": 2,
     "Road": 3,
     "Drive": 4,
     "Lane": 5,
     "Tunnel": 6,
     "Highway": 7,
     "Way": 8,
     "Parkway": 9,
     "Parking": 10,
     "Oval": 11,
     "Square": 12,
     "Place": 13,
     "Bridge": 14
}
def encode(x):
    if pd.isna(x):
        return None
    for road in address_encoding:
        if (road in x):
            return address_encoding[road]
    return None
train["EntryAdressEncoded"] = train['EntryStreetName'].apply(encode)
train['ExitAddressEncoded'] = train['ExitStreetName'].apply(encode)
test["EntryAdressEncoded"] = test['EntryStreetName'].apply(encode)
test['ExitAddressEncoded'] = test['ExitStreetName'].apply(encode)
pd.set_option('display.max_rows', 200)
train

def check_if_entry_equals_exit_address(entry_address, exit_address):
    if (entry_address == exit_address):
        return True
    return False

train['IfEntryExitSame'] = train.apply(lambda x: check_if_entry_equals_exit_address(x.EntryStreetName, x.ExitStreetName), axis=1)
test['IfEntryExitSame'] = test.apply(lambda x: check_if_entry_equals_exit_address(x.EntryStreetName, x.ExitStreetName), axis=1)
directions = {
    'N': 0,
    'NE': 1/4,
    'E': 1/2,
    'SE': 3/4,
    'S': 1,
    'SW': 5/4,
    'W': 3/2,
    'NW': 7/4
}
train['EntryHeading'] = train['EntryHeading'].map(directions)
train['ExitHeading'] = train['ExitHeading'].map(directions)

test['EntryHeading'] = test['EntryHeading'].map(directions)
test['ExitHeading'] = test['ExitHeading'].map(directions)
le = preprocessing.LabelEncoder()
train["Intersection"] = train["IntersectionId"].astype(str) + train["City"]
test["Intersection"] = test["IntersectionId"].astype(str) + test["City"]

print(train["Intersection"].sample(6).values)
pd.concat([train["Intersection"],test["Intersection"]],axis=0).drop_duplicates().values

le.fit(pd.concat([train["Intersection"],test["Intersection"]]).drop_duplicates().values)
train["Intersection"] = le.transform(train["Intersection"])
test["Intersection"] = le.transform(test["Intersection"])
one_hot = pd.get_dummies(train['City'])
train = train.drop('City',axis = 1)
train = train.join(one_hot)

one_hot = pd.get_dummies(test['City'])
test = test.drop('City',axis = 1)
test = test.join(one_hot)


def isLeft(entry, exit):
    if entry == exit:
        return False
    left_dir = []
    current_dir = entry
    for i in range(3):
        current_dir -= 1/4
        if current_dir < 0:
            current_dir += 2.0
        left_dir.append(current_dir)
    return exit in left_dir
train['TurnLeft'] = train.apply(lambda x: isLeft(x.EntryHeading, x.ExitHeading), axis = 1)
test['TurnLeft'] = test.apply(lambda x: isLeft(x.EntryHeading, x.ExitHeading), axis = 1)

train = train.dropna()
train
test
train.columns

Y = train[["TotalTimeStopped_p20", "TotalTimeStopped_p50", "TotalTimeStopped_p80", "DistanceToFirstStop_p20","DistanceToFirstStop_p50", "DistanceToFirstStop_p80" ]]
X = train[[ 'IntersectionId', "Intersection", 'Latitude', 'Longitude',
         'EntryHeading', 'ExitHeading', "TurnLeft", 'Hour', 'Weekend',
       'Month', 'EntryAdressEncoded', 'ExitAddressEncoded', 'IfEntryExitSame',
       'Atlanta', 'Boston', 'Chicago', 'Philadelphia']]

# X = train 
# X = X.drop(['TotalTimeStopped_p20', 'TotalTimeStopped_p40',
#        'TotalTimeStopped_p50', 'TotalTimeStopped_p60', 'TotalTimeStopped_p80',
#        'TimeFromFirstStop_p20', 'TimeFromFirstStop_p40',
#        'TimeFromFirstStop_p50', 'TimeFromFirstStop_p60',
#        'TimeFromFirstStop_p80', 'DistanceToFirstStop_p20',
#        'DistanceToFirstStop_p40', 'DistanceToFirstStop_p50',
#        'DistanceToFirstStop_p60', 'DistanceToFirstStop_p80' ], axis = 1)
Y
X
X_train, X_validate, y_train, y_validate = train_test_split(X, Y, test_size = 0.2, random_state = 1)

print(X_train.shape)
print(X_validate.shape)
print(y_train.shape)
print(y_validate.shape)
model = LinearRegression()
model.fit(X_train, y_train)
#reg.score(X_validate, y_validate)
y_pred = model.predict(X_validate)
y_pred
y_validate

mean_squared_error(y_validate, y_pred)
model.score(X_validate, y_validate)
from sklearn.cross_decomposition import PLSRegression
pls2 = PLSRegression(n_components=2)
pls2.fit(X_train, y_train)
y_pred = pls2.predict(X_validate)
y_pred
mean_squared_error(y_validate, y_pred)
model.score(X_validate, y_validate)

from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=3, random_state = 0)
model.fit(X_train, y_train)
y_pred = model.predict(X_validate)
mean_squared_error(y_validate, y_pred)
model.score(X_validate, y_validate)
n_list = []
mse_list = []
r_score_list = []
for n in range(1, 10):
    model = RandomForestRegressor(n_estimators=n, random_state = 0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_validate)
    mse_list.append(mean_squared_error(y_validate, y_pred))
    r_score_list.append(model.score(X_validate, y_validate))
    n_list.append(n)


    

print(mse_list)
print(r_score_list)

import matplotlib.pyplot as plt

fig = plt.figure()
ax = plt.axes()
ax.plot(n_list, mse_list)
plt.title("Number of trees vs. MSE")
plt.xlabel("Number of Trees")
plt.ylabel("Mean Squared Error");
# for i, v in enumerate(mse_list):
#     ax.annotate(str(round(v, 1)), xy=(i,v), xytext=(-7,7), textcoords='offset points')
fig = plt.figure()
ax = plt.axes()
ax.plot(n_list, r_score_list)
plt.title("Number of trees vs. R-squared Coefficient")
plt.xlabel("Number of Trees")
plt.ylabel("R-squared Coefficient");
# for i, v in enumerate(r_score_list):
#     ax.annotate(str(round(v, 3)), xy=(i,v), xytext=(0,0), textcoords='offset points')