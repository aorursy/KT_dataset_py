import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib

from sklearn import preprocessing 

import matplotlib.pyplot as plt

from scipy.stats import skew

from scipy.stats.stats import pearsonr





%config InlineBackend.figure_format = 'retina' #set 'png' here when working on notebook

%matplotlib inline



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test_id.csv')



print(train.shape)

print(test.shape)
train['UPDATE_TIME'] = train['UPDATE_TIME'].astype('datetime64[ns]')

test['UPDATE_TIME'] = test['UPDATE_TIME'].astype('datetime64[ns]')



train.head()
train.describe()
train.shape
date_of_train = train.groupby('UPDATE_TIME').mean().reset_index()

date_of_test = test.groupby('UPDATE_TIME').mean().reset_index()
date_of_train.head()
date_of_train['date'] = date_of_train['HOUR_ID']

date_of_test['date'] = date_of_test['HOUR_ID']

for i in range(date_of_train.shape[0]):

    date_of_train.at[i,'date'] = i

    

for i in range(date_of_test.shape[0]):

    date_of_test.at[i,'date'] = i + date_of_train.shape[0]
train['date'] = train['HOUR_ID']

for i in range(train.shape[0]):

    train.at[i, 'date'] = date_of_train.loc[date_of_train['UPDATE_TIME'] == train.at[i,'UPDATE_TIME']].date
test['date'] = test['HOUR_ID']

for i in range(test.shape[0]):

    test.at[i, 'date'] = date_of_test.loc[date_of_test['UPDATE_TIME'] == test.at[i,'UPDATE_TIME']].date
train.head()
zone_1 = train.loc[train['ZONE_CODE'] == 'ZONE01']

zone_2 = train.loc[train['ZONE_CODE'] == 'ZONE02']

zone_3 = train.loc[train['ZONE_CODE'] == 'ZONE03']
zone_1.head()
print(zone_1.shape)

print(zone_2.shape)

print(zone_3.shape)

print(train.shape)
def reject_outliers(data):

    u = np.mean(data["BANDWIDTH_TOTAL"])

    s = np.std(data["BANDWIDTH_TOTAL"])

    data_filtered = data[(data["BANDWIDTH_TOTAL"]>(u-2*s)) & (data["BANDWIDTH_TOTAL"]<(u+2*s))]

    return data_filtered



# train = reject_outliers(train)
zone_1 = reject_outliers(zone_1)

zone_2 = reject_outliers(zone_2)

zone_3 = reject_outliers(zone_3)
zone_3['MAX_USER'].hist()
matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)

prices = pd.DataFrame({"price":zone_3["MAX_USER"], "log(price + 1)":np.log1p(zone_3["MAX_USER"])})

prices.hist()
matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)

prices = pd.DataFrame({"price":zone_3["BANDWIDTH_TOTAL"], "log(price + 1)":np.log1p(zone_3["BANDWIDTH_TOTAL"])})

prices.hist()
zone_3_train = zone_3.groupby(['date']).mean().reset_index()
y_user = zone_3_train['MAX_USER']

y_ban = zone_3_train['BANDWIDTH_TOTAL']
from sklearn.model_selection import train_test_split

x_train, x_valid, y_train, y_valid = train_test_split(zone_3_train['date'], y_user, shuffle = False, test_size=0.2, random_state=42)
from sklearn.linear_model import LinearRegression

reg_user_3 = LinearRegression().fit(np.array(x_train).reshape(-1,1), y_train)
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')



plt.scatter(x_train, y_train, color='black')

plt.plot(x_train, reg_user_3.predict(np.array(x_train).reshape(-1,1)), color = 'blue')

plt.gca().set_title("Gradient Descent Linear Regressor")
def SMAPE(y_true, y_pred):

    error = np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))

    error.replace([np.inf, -np.inf], np.nan, inplace=True)

    error.dropna(inplace=True)

    return np.mean(error)*100





print('SMAPE user : ', SMAPE(y_valid, reg_user_3.predict(np.array(x_valid).reshape(-1,1))))
from sklearn.model_selection import train_test_split

x_train, x_valid, y_train, y_valid = train_test_split(zone_3_train['date'], y_ban, shuffle = False, test_size=0.2, random_state=42)
from sklearn.linear_model import LinearRegression

reg_ban_3 = LinearRegression().fit(np.array(x_train).reshape(-1,1), y_train)
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')



plt.scatter(x_valid, y_valid, color='black')

plt.plot(x_valid, reg_ban_3.predict(np.array(x_valid).reshape(-1,1)), color = 'blue')

plt.gca().set_title("Gradient Descent Linear Regressor")
def SMAPE(y_true, y_pred):

    error = np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))

    error.replace([np.inf, -np.inf], np.nan, inplace=True)

    error.dropna(inplace=True)

    return np.mean(error)*100





print('SMAPE user : ', SMAPE(y_valid, reg_ban_3.predict(np.array(x_valid).reshape(-1,1))))
zone_2['MAX_USER'].hist()
matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)

prices = pd.DataFrame({"price":zone_2["MAX_USER"], "log(price + 1)":np.log1p(zone_2["MAX_USER"])})

prices.hist()
zone_2_train = zone_2.groupby(['date']).mean().reset_index()
y_user = zone_2_train['MAX_USER']

y_ban = zone_2_train['BANDWIDTH_TOTAL']
from sklearn.model_selection import train_test_split

x_train, x_valid, y_train, y_valid = train_test_split(zone_2_train['date'], y_user, shuffle = False, test_size=0.2, random_state=42)
from sklearn.linear_model import LinearRegression

reg_user_2 = LinearRegression().fit(np.array(x_train).reshape(-1,1), y_train)
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')



plt.scatter(x_valid, y_valid, color='black')

plt.plot(x_valid, reg_user_2.predict(np.array(x_valid).reshape(-1,1)), color = 'blue')

plt.gca().set_title("Gradient Descent Linear Regressor")
print('SMAPE user : ', SMAPE(y_valid, reg_user_2.predict(np.array(x_valid).reshape(-1,1))))
from sklearn.model_selection import train_test_split

x_train, x_valid, y_train, y_valid = train_test_split(zone_2_train['date'], y_ban, shuffle = False, test_size=0.2, random_state=42)
from sklearn.linear_model import LinearRegression

reg_ban_2 = LinearRegression().fit(np.array(x_train).reshape(-1,1), y_train)
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')



plt.scatter(x_train, y_train, color='black')

plt.plot(x_train, reg_ban_2.predict(np.array(x_train).reshape(-1,1)), color = 'blue')

plt.gca().set_title("Gradient Descent Linear Regressor")
print('SMAPE user : ', SMAPE(y_valid, reg_ban_2.predict(np.array(x_valid).reshape(-1,1))))
zone_1['BANDWIDTH_TOTAL'].hist()
matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)

prices = pd.DataFrame({"price":zone_1["MAX_USER"], "log(price + 1)":np.log1p(zone_1["MAX_USER"])})

prices.hist()
matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)

prices = pd.DataFrame({"price":zone_1["BANDWIDTH_TOTAL"], "log(price + 1)":np.log1p(zone_1["BANDWIDTH_TOTAL"])})

prices.hist()
zone_1_train = zone_1.groupby(['date']).mean().reset_index()
y_user = zone_1_train['MAX_USER']

y_ban = zone_1_train['BANDWIDTH_TOTAL']
from sklearn.model_selection import train_test_split

x_train, x_valid, y_train, y_valid = train_test_split(zone_1_train['date'], y_user, shuffle = False, test_size=0.2, random_state=42)
from sklearn.linear_model import LinearRegression

reg_user_1 = LinearRegression().fit(np.array(zone_1_train['date']).reshape(-1,1), y_user)
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')



plt.scatter(x_train, y_train, color='black')

plt.plot(x_train, reg_user_1.predict(np.array(x_train).reshape(-1,1)), color = 'blue')

plt.gca().set_title("Gradient Descent Linear Regressor")
print('SMAPE user : ', SMAPE(y_valid, reg_user_1.predict(np.array(x_valid).reshape(-1,1))))
reg_user_1 = LinearRegression().fit(np.array(zone_1_train['date']).reshape(-1,1), zone_1_train['MAX_USER'])

reg_user_2 = LinearRegression().fit(np.array(zone_2_train['date']).reshape(-1,1), zone_2_train['MAX_USER'])

reg_user_3 = LinearRegression().fit(np.array(zone_3_train['date']).reshape(-1,1), zone_3_train['MAX_USER'])

reg_ban_1 = LinearRegression().fit(np.array(zone_1_train['date']).reshape(-1,1), zone_1_train['BANDWIDTH_TOTAL'])

reg_ban_2 = LinearRegression().fit(np.array(zone_2_train['date']).reshape(-1,1), zone_2_train['BANDWIDTH_TOTAL'])

reg_ban_3 = LinearRegression().fit(np.array(zone_3_train['date']).reshape(-1,1), zone_3_train['BANDWIDTH_TOTAL'])
train.tail()
test.head()
pred_user = []

pred_ban = []

for i in range(test.shape[0]):

    if (test.iloc[i].ZONE_CODE == 'ZONE01'):

        pred_user.append(reg_user_1.predict(np.array(test.iloc[i].date).reshape(1,-1)))

        pred_ban.append(reg_ban_1.predict(np.array(test.iloc[i].date).reshape(1,-1)))

    elif (test.iloc[i].ZONE_CODE == 'ZONE02'):

        pred_user.append(reg_user_2.predict(np.array(test.iloc[i].date).reshape(1,-1)))

        pred_ban.append(reg_ban_2.predict(np.array(test.iloc[i].date).reshape(1,-1)))

    else:

        pred_user.append(reg_user_3.predict(np.array(test.iloc[i].date).reshape(1,-1)))

        pred_ban.append(reg_ban_3.predict(np.array(test.iloc[i].date).reshape(1,-1)))
d = {'MAX_USER': pred_user}

pred_u = pd.DataFrame(d) 

e = {'BANWIDTH': pred_ban}

pred_b = pd.DataFrame(e) 
pred_b.head()

test['MAX_USER'] = pred_u['MAX_USER'].astype(int).astype(str)

test['BANDWIDTH_TOTAL'] = pred_b['BANWIDTH'].astype(float).round(2).astype(str)

test['label'] = test['BANDWIDTH_TOTAL'].str.cat(test['MAX_USER'],sep=" ")



test[['id','label']].to_csv('sub_aivn.csv', index=False)

test.head()
