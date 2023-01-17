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
#Load csv files into dataframe.

df_train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/train.csv')

df_test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/test.csv')

df_submission = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/submission.csv')

df_countryinfo = pd.read_csv('/kaggle/input/countryinfo/covid19countryinfo.csv')

df_lockdown = pd.read_csv('/kaggle/input/covid19-lockdown-dates-by-country/countryLockdowndates.csv')

df_testdates = pd.read_csv('/kaggle/input/covid19-tests-conducted-by-country/TestsConducted_AllDates_11May2020.csv')
#convert to standard datatypes



df_train['Date'] = pd.to_datetime(df_train['Date'], format='%Y-%m-%d', errors='coerce')

df_test['Date'] = pd.to_datetime(df_test['Date'], format='%Y-%m-%d', errors='coerce')

df_lockdown['Date'] = pd.to_datetime(df_lockdown['Date'], format='%d/%m/%Y', errors='coerce')



monthReplace = {'Feb' : '02', 'Mar' : '03', 'Apr' : '04', 'May' : '05'}

df_testdates['Date'] = df_testdates['Date'].replace(' ', '-', regex=True)

df_testdates['Date'] = df_testdates['Date'].replace(monthReplace, regex=True)

df_testdates['Date'] = df_testdates['Date'].astype(str) + '-2020'



df_testdates['Date'] = pd.to_datetime(df_testdates['Date'], format='%d-%m-%Y', errors='coerce')
#Remove columns

df_lockdown = df_lockdown.drop(columns=['Reference'])

df_testdates = df_testdates.drop(columns=['Source_1', 'Source_2', 'FileDate', 'Units', 'Tested', 'Positive'])



#Select country population info for further merging

df_selectedinfo = pd.DataFrame(df_countryinfo, columns=['region', 'country', 'pop', 'density', 'medianage', 'urbanpop'])

df_selectedinfo['region'].fillna('', inplace=True)

df_selectedinfo = df_selectedinfo.dropna()

df_selectedinfo['geo'] = ['_'.join(x) for x in zip(df_selectedinfo['country'], df_selectedinfo['region'])]
#Concat primary key

df_train['Province_State'].fillna('', inplace=True)

df_train['geo'] = ['_'.join(x) for x in zip(df_train['Country_Region'], df_train['Province_State'])]

df_train = df_train.drop(columns=['Province_State'])



df_lockdown['Province'].fillna('', inplace=True)

df_lockdown['geo'] = ['_'.join(x) for x in zip(df_lockdown['Country/Region'], df_lockdown['Province'])]

df_lockdown = df_lockdown.drop(columns=['Province'])



df_test['Province_State'].fillna('', inplace=True)

df_test['geo'] = ['_'.join(x) for x in zip(df_test['Country_Region'], df_test['Province_State'])]

df_test = df_test.drop(columns=['Province_State'])
#Merge train data with lockdown data



df_merged = df_train.merge(df_lockdown,on=["geo"],how="left")

df_merged['Lockdown_length'] = df_merged['Date_x'] - df_merged['Date_y']



df_merged = df_merged.drop(columns=['Country/Region'])

df_merged = df_merged.rename(columns={'Date_x': 'Date', 'Date_y': 'Lockdown_Date'})



#From testing, 210 instances' country are missing in lockdown data



df_merged_test = df_test.merge(df_lockdown,on=["geo"],how="left")

df_merged_test['Lockdown_length'] = df_merged_test['Date_x'] - df_merged_test['Date_y']



df_merged_test = df_merged_test.drop(columns=['Country/Region'])

df_merged_test = df_merged_test.rename(columns={'Date_x': 'Date', 'Date_y': 'Lockdown_Date'})
#Merge pop info for each country

df_merged1 = df_merged.merge(df_selectedinfo,on=["geo"],how="left")

df_merged1 = df_merged1.drop(columns=['country', 'region'])

df_merged1 = df_merged1.rename(columns={'Country_Region': 'country'})



#Merge pop info for test set

df_merged_test1 = df_merged_test.merge(df_selectedinfo,on=["geo"],how="left")

df_merged_test1 = df_merged_test1.drop(columns=['country', 'region'])

df_merged_test1 = df_merged_test1.rename(columns={'Country_Region': 'country'})
#Merge countries general info for those doesn't provide province info

#df_merged2 = df_merged1.merge(df_selectedinfo, how='left', left_on=['Country_Region'], right_on=['country'])



df_merged2 = df_merged1.merge(df_selectedinfo.drop_duplicates('country'),how='left',on='country')



#Merge country info for test set

df_merged_test2 = df_merged_test1.merge(df_selectedinfo.drop_duplicates('country'),how='left',on='country')
def custom_info(geo_x,geo_y, info_x, info_y):

    if geo_x == geo_y:

        return info_x

    else:

        return info_y
def retype_lockdown(lock_type, lock_date, current_date):

  if lock_date > current_date:

    return "Before"

  else:

    return lock_type
df_merged2['pop'] = df_merged2.apply(lambda x: custom_info(x['geo_x'],x['geo_y'], x['pop_x'], x['pop_y']),axis=1)

df_merged2['density'] = df_merged2.apply(lambda x: custom_info(x['geo_x'],x['geo_y'], x['density_x'], x['density_y']),axis=1)

df_merged2['medianage'] = df_merged2.apply(lambda x: custom_info(x['geo_x'],x['geo_y'], x['medianage_x'], x['medianage_y']),axis=1)

df_merged2['urbanpop'] = df_merged2.apply(lambda x: custom_info(x['geo_x'],x['geo_y'], x['urbanpop_x'], x['urbanpop_y']),axis=1)



df_merged2 = df_merged2.drop(columns=['pop_x', 'pop_y', 'density_x', 'density_y', 'medianage_x', 'medianage_y', 'urbanpop_x', 'urbanpop_y', 'geo_y', 'region'])

df_merged2 = df_merged2.rename(columns={'geo_x': 'geo'})



df_merged2['Type'] = df_merged2.apply(lambda x: retype_lockdown(x['Type'], x['Lockdown_Date'], x['Date']),axis=1)





#Reformat test data

df_merged_test2['pop'] = df_merged_test2.apply(lambda x: custom_info(x['geo_x'],x['geo_y'], x['pop_x'], x['pop_y']),axis=1)

df_merged_test2['density'] = df_merged_test2.apply(lambda x: custom_info(x['geo_x'],x['geo_y'], x['density_x'], x['density_y']),axis=1)

df_merged_test2['medianage'] = df_merged_test2.apply(lambda x: custom_info(x['geo_x'],x['geo_y'], x['medianage_x'], x['medianage_y']),axis=1)

df_merged_test2['urbanpop'] = df_merged_test2.apply(lambda x: custom_info(x['geo_x'],x['geo_y'], x['urbanpop_x'], x['urbanpop_y']),axis=1)



df_merged_test2 = df_merged_test2.drop(columns=['pop_x', 'pop_y', 'density_x', 'density_y', 'medianage_x', 'medianage_y', 'urbanpop_x', 'urbanpop_y', 'geo_y', 'region'])

df_merged_test2 = df_merged_test2.rename(columns={'geo_x': 'geo'})



df_merged_test2['Type'] = df_merged_test2.apply(lambda x: retype_lockdown(x['Type'], x['Lockdown_Date'], x['Date']),axis=1)
df_x = df_merged2

df_testx = df_merged_test2



#Reformat population data

df_x['pop'] = df_x['pop'].str.replace(',', '').astype(float)

df_x['Lockdown_length'] = df_x['Lockdown_length'].astype('timedelta64[D]')



start = df_x['Date'].min()

df_x['Date_length'] = df_x['Date'] - start

df_x['Date_length'] = df_x['Date_length'].astype('timedelta64[D]')

df_x = df_x.drop(columns=['Lockdown_Date', 'Date'])





#Reformat for test set

df_testx['pop'] = df_testx['pop'].str.replace(',', '').astype(float)

df_testx['Lockdown_length'] = df_testx['Lockdown_length'].astype('timedelta64[D]')



df_testx['Date_length'] = df_testx['Date'] - start

df_testx['Date_length'] = df_testx['Date_length'].astype('timedelta64[D]')

df_testx = df_testx.drop(columns=['Lockdown_Date', 'Date'])
#OneHotEncoding

df_x = pd.get_dummies(df_x, columns=['Type'])

df_testx = pd.get_dummies(df_testx, columns=['Type'])
#Factorize country and geo

testdf = pd.concat([df_x, df_testx])

testdf['country'] = pd.factorize(testdf['country'])[0]

testdf['geo'] = pd.factorize(testdf['geo'])[0]



df1 = testdf[0:20580]

df2 = testdf[20580:]



df1 = df1.drop(columns=['ForecastId'])

df2 = df2.drop(columns=['Id', 'ConfirmedCases', 'Fatalities'])



train_id = df1['Id']

df_y = pd.DataFrame(df1, columns=['ConfirmedCases', 'Fatalities'])

df_x = df1.drop(columns=['Id', 'ConfirmedCases', 'Fatalities'])
df_x['missing'] = df_x.apply(lambda x: x.isna().sum(), axis=1)



df_x['Lockdown_length'].fillna(0, inplace=True)

df_x['pop'].fillna(-1, inplace=True)

df_x['pop'].fillna(-1, inplace=True)

df_x['density'].fillna(-1, inplace=True)

df_x['medianage'].fillna(-1, inplace=True)

df_x['urbanpop'].fillna(-1, inplace=True)



# for test set

df2['missing'] = df2.apply(lambda x: x.isna().sum(), axis=1)

df2['Lockdown_length'].fillna(0, inplace=True)

df2['pop'].fillna(-1, inplace=True)

df2['pop'].fillna(-1, inplace=True)

df2['density'].fillna(-1, inplace=True)

df2['medianage'].fillna(-1, inplace=True)

df2['urbanpop'].fillna(-1, inplace=True)
df2.isna().sum()
def RMSLE(pred,actual):

    return np.sqrt(np.mean(np.power((np.log(pred+1)-np.log(actual+1)),2)))
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.33, random_state=1)
# Regressors

from sklearn.neighbors import KNeighborsRegressor

from sklearn.linear_model import Lasso

from sklearn.linear_model import ElasticNet

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import RidgeCV



from sklearn.ensemble import ExtraTreesRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import AdaBoostRegressor

from skimage.io import imshow



from sklearn.tree import DecisionTreeRegressor



from sklearn.multioutput import MultiOutputRegressor



ESTIMATORS = {

    "K-nn": KNeighborsRegressor(),

    "Ridge": RidgeCV(),

    "Lasso": Lasso(),

    "ElasticNet": ElasticNet(random_state=1),

    "RandomForestRegressor": RandomForestRegressor(),

    "Decision Tree Regressor":DecisionTreeRegressor(max_depth=5),

    "MultiO/P GBR" :MultiOutputRegressor(GradientBoostingRegressor(n_estimators=5)),

    "MultiO/P AdaB" :MultiOutputRegressor(AdaBoostRegressor(n_estimators=5))

}

  



y_test_predict = dict()

y_mse = dict()



for name, estimator in ESTIMATORS.items():     

    estimator.fit(X_train, y_train)                    # fit() with instantiated object

    y_test_predict[name] = estimator.predict(X_test)   # Make predictions and save it in dict under key: name

    y_mse[name] = RMSLE(estimator.predict(X_test), y_test)

    print('RMSE for ',name,' is ',y_mse[name])
from sklearn.multioutput import RegressorChain

# define model

model = KNeighborsRegressor()

wrapper = RegressorChain(model)

# fit model

wrapper.fit(X_train, y_train)

# make a prediction

yhat = wrapper.predict(X_test)

# evaluate prediction

RMSLE(yhat, y_test)
import xgboost as xgb



model = MultiOutputRegressor(xgb.XGBRegressor()).fit(X_train, y_train)

y_test_predict1 = model.predict(X_test)

y_mse1 = RMSLE(y_test_predict1, y_test)

print('RMSE for is ',y_mse1)
from keras.models import Sequential

from keras.layers import Dense



in_dim = X_train.shape[1]

out_dim = y_train.shape[1]



#add layers

model = Sequential()

model.add(Dense(100, input_dim=in_dim, activation="relu"))

model.add(Dense(32, activation="relu"))

model.add(Dense(out_dim))

model.compile(loss="mse", optimizer="adam")
model.fit(X_train, y_train, epochs=100, batch_size=12, verbose=0)
ypred = model.predict(X_test)

mse = RMSLE(ypred, y_test)

print("RMSLE for MLP:", mse)
test_id = df2['ForecastId']

x_mytest = df2.drop(columns=['ForecastId'])
model = RandomForestRegressor()

# fit model

model.fit(X_train, y_train)

# make a prediction

yhat = model.predict(x_mytest)
submission = df_submission.copy()
df_yhat = pd.DataFrame(yhat, columns=['ConfirmedCases', 'Fatalities'])

submission['ConfirmedCases'] = df_yhat['ConfirmedCases']

submission['Fatalities'] = df_yhat['Fatalities']
submission.to_csv('submission.csv', index=False)