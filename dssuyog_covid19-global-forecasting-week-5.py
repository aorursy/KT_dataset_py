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
train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-5/train.csv')
test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-5/test.csv')
samplesub = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-5/submission.csv')
# train.head()   # 2020-01-23 - 2020-05-08
# test.head()  # 2020-04-27 - 2020-06-10
# test.shape
train.dtypes
# from pandas_profiling import ProfileReport
# ProfileReport(train)

train.isna().sum() / train.shape[0] * 100
train.Province_State.fillna(train.Country_Region, inplace=True)
train.County.fillna(train.Province_State, inplace=True)
train.isna().sum()
cols_list = ['Id', 'County', 'Province_State', 'Country_Region', 'Target']
train[cols_list] = train[cols_list].apply(lambda x: x.astype('category'))

train['Date']= pd.to_datetime(train['Date'])
# Or, train.Date = train.Date.apply(pd.to_datetime)
train.dtypes
train_date_min = train['Date'].min()
train['days_since'] = train['Date'].apply(lambda x: (x - pd.to_datetime(train_date_min)).days )
train[train['days_since']==1].head()

train['Date']=train['Date'].dt.strftime("%Y%m%d").astype(int)
# # Adding feature continent and sub_region

# coucon = pd.read_csv('/kaggle/input/country-to-continent/countryContinent.csv', encoding='latin-1')

# # ss = train.merge(coucon[['country', 'continent', 'sub_region']],how='left', left_on='Country_Region', right_on='country').fillna(np.nan)
# # train.drop(['country'],axis=1,inplace=True)
# # ss[ss['continent'].isna()]['Country_Region'].unique()

# def get_continent(x):
#     if coucon['country'].str.contains(x).any():
#         return coucon.loc[coucon['country'].str.contains(x), 'continent'].iloc[0] 
#     else:
#         np.nan

# train['continent'] = train['Country_Region'].apply(get_continent)

# train.loc[train['Country_Region']== 'Burma', ['continent']] = 'Asia'
# train.loc[train['Country_Region'].isin(['Congo (Brazzaville)', 'Congo (Kinshasa)']), ['continent']] = 'Africa'
# train.loc[train['Country_Region']== "Cote d'Ivoire", ['continent']] = 'Africa'
# train.loc[train['Country_Region']== "Czechia", ['continent']] = 'Europe'
# train.loc[train['Country_Region']== "Diamond Princess", ['continent']] = 'Asia'
# train.loc[train['Country_Region']== "Eswatini", ['continent']] = 'Africa'
# train.loc[train['Country_Region']== "India", ['continent']] = 'Asia'
# train.loc[train['Country_Region']== "Korea, South", ['continent']] = 'Asia'
# train.loc[train['Country_Region']== "Kosovo", ['continent']] = 'Europe'
# train.loc[train['Country_Region']== "Laos", ['continent']] = 'Asia'
# train.loc[train['Country_Region']== "MS Zaandam", ['continent']] = 'Americas'
# train.loc[train['Country_Region']== "North Macedonia", ['continent']] = 'Europe'
# train.loc[train['Country_Region']== "US", ['continent']] = 'Americas'
# train.loc[train['Country_Region']== "Vietnam", ['continent']] = 'Asia'
# train.loc[train['Country_Region']== "West Bank and Gaza", ['continent']] = 'Asia'

# train.continent = train.continent.astype('category')

# train['continent'].isna().sum()
# y = train['TargetValue']
# # features = ['Country_Region', 'Population', 'Weight', 'days_since', 'Target']
# features = ['Country_Region', 'Population', 'Weight', 'Date', 'Target'] #'days_since'
# X = train[features]
# X.columns
# # split dataset in train and test
# from sklearn.model_selection import train_test_split
# # 80/20 split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
# Merge less frequent levels. OHE on limited levels won't generate large no. of features.

# Define which columns should be encoded vs scaled
# numeric_ftrs  = X_train.select_dtypes(include=['float64','int64'])
# categorical_ftrs = X_train.select_dtypes(include=['category'])

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X = train.loc[:, 'Country_Region'].values
train.loc[:, 'Country_Region'] = le.fit_transform(X.astype(str))

X = train.loc[:,'Target'].values
train.loc[:,'Target'] = le.fit_transform(X)

print("ran")

# from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder()
# X = test.loc[:, 'Country_Region'].values
# test.loc[:, 'Country_Region'] = le.fit_transform(X.astype(str))

# X = test.loc[:,'Target'].values
# test.loc[:,'Target'] = le.fit_transform(X)
# print("ran")
y = train['TargetValue']
# features = ['Country_Region', 'Population', 'Weight', 'days_since', 'Target']
features = ['Country_Region', 'Population', 'Weight', 'Date', 'Target'] #'days_since'
X = train[features]
# split dataset in train and test
from sklearn.model_selection import train_test_split
# 80/20 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0)
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

model = RandomForestRegressor(n_jobs=-1)
estimators = 200
model.set_params(n_estimators=estimators)
pip = Pipeline([('scaler2' , StandardScaler()), ('RandomForestRegressor: ', model)])

# pip = Pipeline([('scaler2' , StandardScaler()), ('RandomForestRegressor: ', RandomForestRegressor())])
pip.fit(X_train , y_train)
predictions = pip.predict(X_test)

acc=pip.score(X_test,y_test)
acc
# from sklearn.ensemble import RandomForestRegressor

# rf = RandomForestRegressor(n_jobs=-1, n_estimators=100, oob_score=True, max_features='sqrt')
# # rf = RandomForestRegressor(n_jobs=-1,n_estimators=50,oob_score=True,max_features=0.8,min_samples_leaf=5)
# # rf = RandomForestRegressor(n_jobs=-1,n_estimators=50,min_samples_leaf=7,max_features=0.5,oob_score=True,max_depth=40)

# rf.fit(X_train, y_train)

# predictions = rf.predict(X_test)

# print('test Score: ', rf.score(X_test, y_test))

from sklearn.metrics import mean_absolute_error
val_mae = mean_absolute_error(predictions,y_test)
print(val_mae) # was 2.9

# break
test.Province_State.fillna(test.Country_Region, inplace=True)
test.County.fillna(test.Province_State, inplace=True)

cols_list = ['ForecastId', 'County', 'Province_State', 'Country_Region', 'Target']
test[cols_list] = test[cols_list].apply(lambda x: x.astype('category'))

test['Date']= pd.to_datetime(test['Date'])

test_date_min = test['Date'].min()
test['days_since'] = test['Date'].apply(lambda x: (x - pd.to_datetime(test_date_min)).days )
test[test['days_since']==1].head()

test['Date']=test['Date'].dt.strftime("%Y%m%d").astype(int)
# # Adding feature continent and sub_region

# coucon = pd.read_csv('/kaggle/input/country-to-continent/countryContinent.csv', encoding='latin-1')

# # ss = test.merge(coucon[['country', 'continent', 'sub_region']],how='left', left_on='Country_Region', right_on='country').fillna(np.nan)
# # test.drop(['country'],axis=1,inplace=True)
# # ss[ss['continent'].isna()]['Country_Region'].unique()

# def get_continent(x):
#     if coucon['country'].str.contains(x).any():
#         return coucon.loc[coucon['country'].str.contains(x), 'continent'].iloc[0] 
#     else:
#         np.nan

# test['continent'] = test['Country_Region'].apply(get_continent)

# test.loc[test['Country_Region']== 'Burma', ['continent']] = 'Asia'
# test.loc[test['Country_Region'].isin(['Congo (Brazzaville)', 'Congo (Kinshasa)']), ['continent']] = 'Africa'
# test.loc[test['Country_Region']== "Cote d'Ivoire", ['continent']] = 'Africa'
# test.loc[test['Country_Region']== "Czechia", ['continent']] = 'Europe'
# test.loc[test['Country_Region']== "Diamond Princess", ['continent']] = 'Asia'
# test.loc[test['Country_Region']== "Eswatini", ['continent']] = 'Africa'
# test.loc[test['Country_Region']== "India", ['continent']] = 'Asia'
# test.loc[test['Country_Region']== "Korea, South", ['continent']] = 'Asia'
# test.loc[test['Country_Region']== "Kosovo", ['continent']] = 'Europe'
# test.loc[test['Country_Region']== "Laos", ['continent']] = 'Asia'
# test.loc[test['Country_Region']== "MS Zaandam", ['continent']] = 'Americas'
# test.loc[test['Country_Region']== "North Macedonia", ['continent']] = 'Europe'
# test.loc[test['Country_Region']== "US", ['continent']] = 'Americas'
# test.loc[test['Country_Region']== "Vietnam", ['continent']] = 'Asia'
# test.loc[test['Country_Region']== "West Bank and Gaza", ['continent']] = 'Asia'

# test.continent = test.continent.astype('category')

# test['continent'].isna().sum()

test_X = test[features]

from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder()
X = test_X.loc[:, 'Country_Region'].values
test_X.loc[:, 'Country_Region'] = le.fit_transform(X.astype(str))

X = test_X.loc[:,'Target'].values
test_X.loc[:,'Target'] = le.fit_transform(X)

print("ran")
# test_preds = rf.predict(test_X)
test_preds = pip.predict(test_X)
pred_list =[int(x) for x in test_preds]
output = pd.DataFrame({'Id': test.index, 'TargetValue': pred_list})
print(output)
a=output.groupby(['Id'])['TargetValue'].quantile(q=0.05).reset_index()
b=output.groupby(['Id'])['TargetValue'].quantile(q=0.5).reset_index()
c=output.groupby(['Id'])['TargetValue'].quantile(q=0.95).reset_index()
a.columns=['Id','q0.05']
b.columns=['Id','q0.5']
c.columns=['Id','q0.95']
a=pd.concat([a,b['q0.5'],c['q0.95']],1)
a['q0.05']=a['q0.05']
a['q0.5']=a['q0.5']
a['q0.95']=a['q0.95']
a
a['Id'] =a['Id']+ 1
# a.columns

sub=pd.melt(a, id_vars=['Id'], value_vars=['q0.05','q0.5','q0.95'])
sub['variable']=sub['variable'].str.replace("q","", regex=False)
sub['ForecastId_Quantile']=sub['Id'].astype(str)+'_'+sub['variable']
sub['TargetValue']=sub['value']
sub=sub[['ForecastId_Quantile','TargetValue']]
sub.reset_index(drop=True,inplace=True)
sub.to_csv("submission.csv",index=False)
sub.head()

