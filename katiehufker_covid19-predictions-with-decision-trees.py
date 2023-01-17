import pandas as pd

import numpy as np
starting_data = pd.read_csv('../input/covid-clean/clean_train.csv')
starting_data.head()
region_metadata = pd.read_csv('../input/covid19-forecasting-metadata/region_metadata.csv')
region_metadata.head()
merged_data = pd.merge(starting_data, region_metadata, how='left', on=['Country_Region', 'Province_State'])
len(merged_data)

merged_data.head()
lockdown = pd.read_csv('../input/covid19-lockdown-dates-by-country/countryLockdowndatesJHUMatch.csv')
lockdown.rename(columns={'Country/Region': 'Country_Region'}, inplace=True)

lockdown.head()
merge_lockdown = pd.merge(merged_data, lockdown, how='left', on=['Country_Region'])
merge_lockdown.head()
merge_lockdown['Date_x'] = pd.to_datetime(merge_lockdown['Date_x'], errors='coerce')

merge_lockdown['Date_y'] = pd.to_datetime(merge_lockdown['Date_y'], errors='coerce')

merge_lockdown['days_since_lockdown_raw'] = merge_lockdown['Date_x'] - merge_lockdown['Date_y']

merge_lockdown['days_since_lockdown_raw'] = merge_lockdown['days_since_lockdown_raw'].apply(lambda x: x.days)

merge_lockdown['days_since_lockdown'] = merge_lockdown['days_since_lockdown_raw'].apply(lambda x: -1 if x < 0 else x)

merge_lockdown.head()
merge_lockdown['type_cleaned'] = merge_lockdown.apply(lambda x: x['Type'] if x['days_since_lockdown'] >= 0 else 'None', axis=1)
merge_lockdown.tail()
cases = merge_lockdown['ConfirmedCases']

deaths = merge_lockdown['Fatalities']
data = merge_lockdown.drop(['ConfirmedCases','Fatalities', 'Date_y', 'Type', 'Reference', 'days_since_lockdown_raw'], axis=1)
data.head()
data[['days_since_lockdown']] = data[['days_since_lockdown']].fillna(value=-1)
data.head()
data = pd.get_dummies(data, columns=['continent', 'type_cleaned'])
features = data[['day_from_jan_first','Lat','Long',

                   'medianage','urbanpop','hospibed','sexratio',

                   'lung','avgtemp','avghumidity','days_from_firstcase', 'population', 'area', 'density_y', 'continent_Africa', 'continent_Americas','continent_Asia','continent_Europe','continent_Oceania', 'days_since_lockdown', 'type_cleaned_Full','type_cleaned_None','type_cleaned_Partial']]

feature_list = list(features.columns)
from sklearn.model_selection import train_test_split

features_train,features_test,cases_train,cases_test=train_test_split(features,cases,test_size=0.2)
print(cases_test.head())

print(features_test.head())
from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor(random_state = 0)

regressor = regressor.fit(features_train, cases_train)
def merge(list1, list2): 

      

    merged_list = [(list1[i], list2[i]) for i in range(0, len(list1))] 

    return merged_list 
feature_weights = merge(features_train.columns, regressor.feature_importances_)

sorted_features = sorted(feature_weights, key=lambda x: x[-1])

print(sorted_features)
predictions = regressor.predict(features_test)
df = pd.DataFrame({'Actual': cases_test, 'Predicted': predictions})

df
from sklearn import metrics

from statistics import mean

print('Mean Absolute Error:', metrics.mean_absolute_error(cases_test, predictions))  

print('Mean Squared Error:', metrics.mean_squared_error(cases_test, predictions))  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(cases_test, predictions)))

print(mean(predictions))

score = regressor.score(features_test, cases_test)

print(score)
# Import the model we are using

from sklearn.ensemble import RandomForestRegressor

# Instantiate model with 1000 decision trees

rf = RandomForestRegressor(n_estimators = 100, random_state = 42)

# Train the model on training data

rf.fit(features_train, cases_train);
predictions = rf.predict(features_test)
feature_weights = merge(features_train.columns, rf.feature_importances_)

sorted_features = sorted(feature_weights, key=lambda x: x[-1])

print(sorted_features)
print('Mean Absolute Error:', metrics.mean_absolute_error(cases_test, predictions))  

print('Mean Squared Error:', metrics.mean_squared_error(cases_test, predictions))  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(cases_test, predictions)))

print(mean(predictions))

score = rf.score(features_test, cases_test)

print(score)