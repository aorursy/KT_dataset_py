import numpy as np 

import pandas as pd 

import seaborn as sns



from sklearn.model_selection import train_test_split



from xgboost import XGBRegressor

from sklearn.multioutput import MultiOutputRegressor

from sklearn.impute import SimpleImputer
train_data = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv')

test_data = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv')

country_data = pd.read_csv('/kaggle/input/country-data-population/country_data_population.csv')

submission_csv = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/submission.csv')
train_data = pd.merge(train_data,country_data, left_on=['Country_Region'],right_on=['Country'], how='left')

train_data.drop(['Country','Lending Category','UrbanRuralDesignation','GeoRegion','Continent'], axis=1, inplace=True)



test_data = pd.merge(test_data,country_data, left_on=['Country_Region'],right_on=['Country'], how='left')

test_data.drop(['Country','Lending Category','UrbanRuralDesignation','GeoRegion','Continent'], axis=1, inplace=True)



train_data['GeoSubregion'].fillna('', inplace=True)

test_data['GeoSubregion'].fillna('', inplace=True)



train_data['Income Group'].fillna('', inplace=True)

test_data['Income Group'].fillna('', inplace=True)



train_data['Province_State'].fillna('', inplace=True)

test_data['Province_State'].fillna('', inplace=True)

train_data['Date'] = pd.to_datetime(train_data['Date'], infer_datetime_format=True)

test_data['Date'] = pd.to_datetime(test_data['Date'], infer_datetime_format=True)
train_data.loc[:, 'Date'] = train_data.Date.dt.strftime('%y%m%d')

train_data.loc[:, 'Date'] = train_data['Date'].astype(int)



test_data.loc[:, 'Date'] = test_data.Date.dt.strftime('%y%m%d')

test_data.loc[:, 'Date'] = test_data['Date'].astype(int)
all_data = pd.merge(train_data, test_data ,how='outer', on=['Country_Region','Date','Province_State'])

pd.set_option('display.max_rows', None)





all_data.sort_values(['Country_Region','Province_State','Date'], ascending = True, inplace = True)

all_data = all_data.reset_index(drop=True)

all_data.head(500)
#Populate days_since_confirmedcases - not a very pythonic way of doing this 



all_data['days_since_confirmedcases'] = 0.0

all_data['ConfirmedCases'].fillna(0.0, inplace=True)



count = 0

save_province_state = ""

save_country_region = ""

save_confirmedcases = 0.0

first_confirmedcase = 0



for index, row in all_data.iterrows():



    if save_province_state != row['Province_State'] or save_country_region != row['Country_Region']:

        

        save_province_state = row['Province_State'] 

        save_country_region = row['Country_Region']  

        first_confirmedcase = 0 

        save_confirmedcases = 0



        

    if (save_confirmedcases == 0.0) and (row['ConfirmedCases'] > 0.0):

        save_confirmedcases = row['ConfirmedCases']

        first_confirmedcase = count 

        

    if first_confirmedcase > 0:

        all_data['days_since_confirmedcases'][count] = count - first_confirmedcase



    count += 1

    

all_data.head(200)
train_data = pd.merge(train_data,all_data[['Province_State','Country_Region','Date','days_since_confirmedcases']], left_on=['Province_State','Country_Region','Date'],right_on=['Province_State','Country_Region','Date'], how='left')

test_data = pd.merge(test_data,all_data[['Province_State','Country_Region','Date','days_since_confirmedcases']], left_on=['Province_State','Country_Region','Date'],right_on=['Province_State','Country_Region','Date'], how='left')



convert_dict = {'Province_State': str}

train_data = train_data.astype(convert_dict)

test_data = test_data.astype(convert_dict)
test_data.head()
train_data.head()
sns.countplot(y="Country_Region", data=train_data,order=train_data["Country_Region"].value_counts(ascending=False).iloc[:10].index)
sns.regplot(x=train_data["ConfirmedCases"], y=train_data["Fatalities"], fit_reg=True)
sns.jointplot(x=train_data["ConfirmedCases"], y=train_data["Fatalities"],kind='scatter')

#get list of categorical variables

s = (train_data.dtypes == 'object')

object_cols = list(s[s].index)
from sklearn.preprocessing import LabelEncoder
object_cols
label_encoder1 = LabelEncoder()

label_encoder2 = LabelEncoder()

label_encoder3 = LabelEncoder()

label_encoder4 = LabelEncoder()



train_data['Province_State'] = label_encoder1.fit_transform(train_data['Province_State'])

test_data['Province_State'] = label_encoder1.transform(test_data['Province_State'])



train_data['Country_Region'] = label_encoder2.fit_transform(train_data['Country_Region'])

test_data['Country_Region'] = label_encoder2.transform(test_data['Country_Region'])



train_data['GeoSubregion'] = label_encoder3.fit_transform(train_data['GeoSubregion'])

test_data['GeoSubregion'] = label_encoder3.transform(test_data['GeoSubregion'])



train_data['Income Group'] = label_encoder4.fit_transform(train_data['Income Group'])

test_data['Income Group'] = label_encoder4.transform(test_data['Income Group'])





    
train_data.head()
test_data.head()
Test_id = test_data.ForecastId
train_data.drop(['Id'], axis=1, inplace=True)

test_data.drop('ForecastId', axis=1, inplace=True)
missing_val_count_by_column = (train_data.isnull().sum())

print(missing_val_count_by_column[missing_val_count_by_column>0])
from xgboost import XGBRegressor
train_data.head()
X_train = train_data[['Province_State','Country_Region','GeoSubregion','Income Group','days_since_confirmedcases','Date']]

y_train = train_data[['ConfirmedCases', 'Fatalities']]

X_test  = test_data[['Province_State','Country_Region','GeoSubregion','Income Group','days_since_confirmedcases','Date']]

y_train_confirm = y_train.ConfirmedCases

y_train_fatality = y_train.Fatalities
model1 = XGBRegressor(n_estimators=40000)

model1.fit(X_train, y_train_confirm)

y_pred_confirm = model1.predict(X_test)
model2 = XGBRegressor(n_estimators=20000)

model2.fit(X_train,y_train_fatality )

y_pred_fat = model2.predict(X_test)
df_sub = pd.DataFrame()

df_sub['ForecastId'] = Test_id

df_sub['ConfirmedCases'] = y_pred_confirm

df_sub['Fatalities'] = y_pred_fat

df_sub.to_csv('submission.csv', index=False)
df_sub