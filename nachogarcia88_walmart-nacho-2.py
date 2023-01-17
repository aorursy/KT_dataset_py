import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import operator

from sklearn.preprocessing import StandardScaler, MinMaxScaler, Imputer, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
import sklearn.metrics as metrics
from sklearn.feature_selection import SelectFromModel

import xgboost as xgb
stores = pd.read_csv('../input/stores.csv')
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
features = pd.read_csv('../input/features.csv')
#Training
df_all = pd.merge(train, features, how='left',on = ['Date','Store','IsHoliday'])
df_all = pd.merge(df_all, stores, how='left',on = 'Store')
df_all.IsHoliday = df_all.IsHoliday.astype('category')
df_all.Type = df_all.Type.astype('category')
df_all['Date'] = pd.to_datetime(df_all['Date'])
df_all['WeekNumber'] = df_all['Date'].dt.week
df_all['Year'] = df_all['Date'].dt.year
df_all['Day'] = df_all['Date'].dt.dayofweek
df_all['Month'] = df_all['Date'].dt.month
df_all['tmp_Weekly_Sales'] = df_all.Weekly_Sales.copy()
df_all = df_all[['Dept','Store','Type','Size', 
                 'Date','Day','WeekNumber','Month','Year','IsHoliday',
                 'Temperature','Fuel_Price', 'CPI', 'Unemployment',
                 'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4','MarkDown5',
                 'Weekly_Sales','tmp_Weekly_Sales']]
# Cleaning data
counter = 0
for i in np.nditer(df_all.Dept.unique()):
    for j in np.nditer(df_all.Store.unique()):
        if (len(df_all[df_all.Date.isin(['2010-12-24','2010-12-31'])][df_all.Dept == i][df_all.Store == j]) == 1):
            df_all.drop(df_all[df_all.Date == '2010-12-24'][df_all.Dept == i][df_all.Store == j].index, inplace=True)
            df_all.drop(df_all[df_all.Date == '2010-12-31'][df_all.Dept == i][df_all.Store == j].index, inplace=True)
            counter+=1
        elif (len(df_all[df_all.Date.isin(['2011-12-23','2011-12-30'])][df_all.Dept == i][df_all.Store == j]) == 1):
            df_all.drop(df_all[df_all.Date == '2011-12-23'][df_all.Dept == i][df_all.Store == j].index, inplace=True)
            df_all.drop(df_all[df_all.Date == '2011-12-30'][df_all.Dept == i][df_all.Store == j].index, inplace=True)
            counter+=1
# shift (6/7) sales on week 51 to 52 year 2010 (1 day shopping pre-Christmas -> 1/7)
df_modified = df_all.copy()
df_modified['tmp_Weekly_Sales'] = df_modified.Weekly_Sales.copy()

iD = pd.Index(df_modified.Date)

tmp24 = df_modified.iloc[iD.get_loc('2010-12-24'), 19].copy()
tmp31 = df_modified.iloc[iD.get_loc('2010-12-31'),19].copy()

new24 = pd.Series(tmp24.values/7 + tmp31.values*6/7, index = tmp24.index)
new31 = pd.Series(tmp31.values/7 + tmp24.values*6/7, index = tmp31.index)

df_modified.iloc[iD.get_loc('2010-12-24'), 20], df_modified.iloc[iD.get_loc('2010-12-31'),20] = new24, new31
data_train = df_modified.copy()
# shift (5/7) sales on week 51 to 52 year 2011 (1 day shopping pre-Christmas -> 2/7)
tmp24 = df_modified.iloc[iD.get_loc('2011-12-23'), 19].copy()
tmp31 = df_modified.iloc[iD.get_loc('2011-12-30'),19].copy()

new24 = pd.Series(tmp24.values*2/7 + tmp31.values*5/7, index = tmp24.index)
new31 = pd.Series(tmp31.values*2/7 + tmp24.values*5/7, index = tmp31.index)

df_modified.iloc[iD.get_loc('2011-12-23'), 20], df_modified.iloc[iD.get_loc('2011-12-30'),20] = new24, new31
data_train = df_modified.copy()
#Test
data_test = pd.merge(test, features, how='left',on = ['Date','Store','IsHoliday'])
data_test = pd.merge(data_test, stores, how='left',on = 'Store')
data_test.IsHoliday = data_test.IsHoliday.astype('category')
data_test.Type = data_test.Type.astype('category')
data_test['Date'] = pd.to_datetime(data_test['Date'])
data_test['WeekNumber'] = data_test['Date'].dt.week
data_test['Month'] = data_test['Date'].dt.month
data_test['Id'] = data_test.Store.astype('str') + '_' + data_test.Dept.astype('str') + '_' + data_test.Date.astype('str')
data_test = data_test[['Id','Dept','Store','Type','Size', 
                 'Date','WeekNumber','Month','IsHoliday',
                 'Temperature','Fuel_Price', 'CPI', 'Unemployment',
                 'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4','MarkDown5']]
# TAKING CARE OF MISSING DATA
markdowns = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4','MarkDown5']
imputer = Imputer(missing_values = 'NaN', strategy = 'median', axis = 0)
imputer1 = imputer.fit(data_train[markdowns])
imputer2 = imputer.fit(data_test[markdowns])
data_train[markdowns] = imputer1.transform(data_train[markdowns])
data_test[markdowns] = imputer2.transform(data_test[markdowns])
# ENCODING CATEGORICAL DATA
labelencoder_X = LabelEncoder()
data_train['Type'] = labelencoder_X.fit_transform(data_train['Type'])
data_train['IsHoliday'] = labelencoder_X.fit_transform(data_train['IsHoliday'])
data_test['Type'] = labelencoder_X.fit_transform(data_test['Type'])
data_test['IsHoliday'] = labelencoder_X.fit_transform(data_test['IsHoliday'])
# TARGET VARIABLE AND FEATURES DEFINITION
filters = ['Unnamed: 0','Date','Day','Year','Weekly_Sales','tmp_Weekly_Sales']
target = 'Weekly_Sales'
#features = list(f for f in data.columns if f not in filters+markdowns)
features = list(f for f in data_train.columns if f not in filters+markdowns)
features_Id = features+['Id']
X_train = data_train.loc[:,features]
y_train = data_train.loc[:,target]

X_test = data_test.loc[:,features]
dtrain = xgb.DMatrix(data = X_train.values, label = y_train.values, feature_names = features)
dtest = xgb.DMatrix(data = X_test.values, feature_names = features)
num_round = 500
params ={'objective': 'reg:linear',
         'max_depth' : 10,
         'learning_rate' : 0.2, 
         'n_estimators' : 300, 
         'subsample' : 0.9, 
         'colsample_bytree' : 0.5,
         'silent': 1}
# TRAINING XGBOOST MODEL
model = xgb.train(params,
                  dtrain,
                  num_round)
pred_test = model.predict(dtest)
result = pd.DataFrame({'Id': data_test.Id})
result['Weekly_Sales'] = pred_test
result.to_csv('submission_nacho.csv', index=False, sep=',')
