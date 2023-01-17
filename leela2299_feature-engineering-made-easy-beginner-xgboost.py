import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline



import seaborn as sns
train=pd.read_csv('../input/covid19-global-forecasting-week-3/train.csv')

test=pd.read_csv('../input/covid19-global-forecasting-week-3/test.csv')

train.info()
train.head()
#there are 180 unique countries 

train.Country_Region.unique()

print(len(train.Country_Region.unique()))
#there are not any null values in the countries column

train.Country_Region.isnull().sum()
# there are 176 countries for which there are not state province values 



train[train['Province_State'].isnull()]['Country_Region'].unique()

print(len(train[train['Province_State'].isnull()]['Country_Region'].unique()))
# fill nan values in state values with string "unkown"

#train.loc[train['Province_State'].isnull(),"Province_State"]="unkown"
#Inorder to merge train and test set: Created a new attribute to keep track  



train['is_test']=False

test['is_test']=True

matrix=pd.concat([train,test],ignore_index=True)
print(matrix.head())

print(matrix.shape)
# Rename columns 

matrix.rename(columns={'Province_State':'state','Country_Region':'country'},inplace=True)

matrix.head()
#replace null values in state with unknown

matrix.loc[matrix['state'].isnull(),'state']='unknown'



# introducing the new column for the state and country by combinning them

matrix['Country_State']=matrix['country']+'_'+matrix['state']
matrix.head()
# convert Date column to date time object 

matrix['Date'] = pd.to_datetime(matrix['Date'])
# Adding a new columns day that have the day of year 

matrix['Day']=matrix['Date'].dt.dayofyear

matrix.Day.min()
# SUBRACTING WITH 21 TO MAKE DAY START FROM ONE

matrix['Day']=matrix.Day-21

matrix.head(3)
#creating a new_column for the growth factor of both confirmed and fatalities for each country

#growth factor= no of cases till date/no of days



group1=matrix.groupby('country')['ConfirmedCases'].transform('max')

group2=matrix.groupby('country')['Day'].transform('max')

matrix['cases_growth_factor']=group1/group2



group1=matrix.groupby('country')['Fatalities'].transform('max')

group2=matrix.groupby('country')['Day'].transform('max')

matrix['fatalities_growth_factor']=group1/group2



print(matrix.head())



    
#creating a new_column for the growth factor of both confirmed and fatalities for each state

#growth factor= no of cases till date/no of days



group1=matrix.groupby('Country_State')['ConfirmedCases'].transform('max')

group2=matrix.groupby('Country_State')['Day'].transform('max')

matrix['cases_growth_factor_state']=group1/group2



group1=matrix.groupby('Country_State')['Fatalities'].transform('max')

group2=matrix.groupby('Country_State')['Day'].transform('max')

matrix['fatalities_growth_factor_state']=group1/group2



print(matrix.head())
# mean encoding of country featrue







group=matrix.groupby('country')['ConfirmedCases'].transform('mean')

matrix['conf_country_mean']=group



group1=matrix.groupby('country')['Fatalities'].transform('max')

matrix['fatalities_country_mean']=group1

# mean encoding of the Country_State feature



group=matrix.groupby('Country_State')['ConfirmedCases'].transform('mean')

matrix['conf_state_mean']=group



group1=matrix.groupby('Country_State')['Fatalities'].transform('max')

matrix['fatalities_state_mean']=group1

# since we are using tree base models Label Encoder is the good option 



from sklearn.preprocessing import LabelEncoder

matrix['country']=LabelEncoder().fit_transform(matrix['country'])

matrix['state']=LabelEncoder().fit_transform(matrix['state'])

matrix['Country_State']=LabelEncoder().fit_transform(matrix['Country_State'])



print(matrix.head())
# dropping the date column

matrix.drop('Date',axis=1,inplace=True)
#seperate the train and test date based on the is_true column

new_train=matrix[matrix['is_test']==False]

new_test=matrix[matrix['is_test']==True]
# dropping trivial attribrutes to predict the confirmed cases

x_train=new_train.drop(['Fatalities','fatalities_growth_factor','fatalities_growth_factor_state','is_test','ForecastId','Id','ConfirmedCases',

                        'fatalities_country_mean','fatalities_country_mean'],

                       axis=1)

x_train.info()
#x_train.drop(['ConfirmedCases'],axis=1,inplace=True)

x_test=new_test.drop(['Fatalities','fatalities_growth_factor','fatalities_growth_factor_state','is_test','ForecastId','Id','ConfirmedCases',

                     'fatalities_country_mean','fatalities_country_mean'],axis=1)
y_train=new_train['ConfirmedCases']
from xgboost import XGBRegressor
model1=XGBRegressor(nestimators=10000)

model1.fit(x_train,y_train,eval_metric='rmse')

ConfirmedCases=model1.predict(x_test)
# dropping the irrelevant features to predict fatalities 



x1_train=new_train.drop(['Fatalities','cases_growth_factor','cases_growth_factor_state','is_test','ForecastId','Id','ConfirmedCases',

                        'conf_state_mean','conf_country_mean']

                        ,axis=1)

x1_test=new_test.drop(['Fatalities','cases_growth_factor','cases_growth_factor_state','is_test','ForecastId','Id','ConfirmedCases',

                      'conf_state_mean','conf_country_mean'],axis=1)
y1_train=new_train['Fatalities']
model2=XGBRegressor(nestimators=10000)

model2.fit(x1_train,y1_train,eval_metric='rmse')

Fatalities=model2.predict(x1_test)
xout = pd.DataFrame({'ForecastId': [], 'ConfirmedCases': [], 'Fatalities': []})

xdata = pd.DataFrame({'ForecastId': test['ForecastId'],'ConfirmedCases': ConfirmedCases, 'Fatalities': Fatalities})
xout = pd.concat([xout, xdata], axis=0)

xout.ForecastId = xout.ForecastId.astype('int')

xout.tail()

xout.to_csv('submission.csv', index=False)
print(xout.head())