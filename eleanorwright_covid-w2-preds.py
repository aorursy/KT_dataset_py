import pandas as pd

import numpy as np
data_main = pd.read_csv("../input/corona/train (1).csv", header=0)
#First ensure the province/state column is consistently filled

data_main['Province_State'].fillna('', inplace=True)

#now change the date into datetime for ease of analysis

data_main['Date'] = pd.to_datetime(data_main['Date'])

#add new column DayOfYear stating which day of the year it is, this is as we do not want any data ater March 31st (the 91st day of the year)

data_main['DOY'] = data_main.Date.dt.dayofyear

#the submission must be given in terms of cumulative values of cases and deaths so the data set will be transformed to mirror this.

data_main[['ConfirmedCases','Fatalities']]=data_main.groupby(['Country_Region', 'Province_State'])[['ConfirmedCases', 'Fatalities']].transform('cummax')
data_info = pd.read_csv("../input/corona/covid19countryinfo.csv", header=0)
data_info['region'].fillna('', inplace=True)
data_lockdown = pd.read_csv("../input/corona/countryLockdowndates.csv", header=0)

data_lockdown['Date'] = pd.to_datetime(data_lockdown['Date'])
data_new1 = pd.merge(data_main, data_info, how='left', left_on=['Country_Region', 'Province_State'], right_on=['country', 'region'])
data_new2 = pd.merge(data_new1, data_lockdown, how='left', left_on=['Country_Region', 'Province_State', 'Date'], right_on=['Country/Region', 'Province', 'Date'])

# note in this new data set the column Type denotes the type of lockdown, this value changes on the date on which Lockdown was introduced for that specific country/region.
data_new2.info()
coldrop = ['region', 'country', 'alpha3code', 'alpha2code', 'active1', 'active2', 'active3', 'newcases1', 'newcases2', 'newcases3', 'newdeaths1', 'newdeaths2', 'newdeaths3', 'critical1', 'critical2', 'critical3', 'Country/Region', 'Province', 'Reference']

data_new2.drop(coldrop, inplace=True, axis=1)
data_new2.fillna(0, inplace=True)
data_new2['quarantine'] = pd.to_datetime(data_new2['quarantine'])

data_new2['schools'] = pd.to_datetime(data_new2['schools'])

data_new2['publicplace'] = pd.to_datetime(data_new2['publicplace'])

data_new2['gathering'] = pd.to_datetime(data_new2['gathering'])

data_new2['nonessential'] = pd.to_datetime(data_new2['nonessential'])

data_new2['firstcase'] = pd.to_datetime(data_new2['firstcase'])



# Now change each to day of year format for ease of modelling.



data_new2['quarantine'] = data_new2.quarantine.dt.dayofyear

data_new2['schools'] = data_new2.schools.dt.dayofyear

data_new2['publicplace'] = data_new2.publicplace.dt.dayofyear

data_new2['gathering'] = data_new2.gathering.dt.dayofyear

data_new2['nonessential'] = data_new2.nonessential.dt.dayofyear

data_new2['firstcase'] = data_new2.firstcase.dt.dayofyear

data_new2['Location'] = data_new2.Country_Region.astype(str) + ":" + data_new2.Province_State.astype(str)



# Now drop the two columns 'Country_Region' and 'Province_State'



data_new2.drop('Country_Region', inplace=True, axis=1)

data_new2.drop('Province_State', inplace=True, axis=1)
objcols = ['pop', 'tests', 'testpop', 'density', 'medianage', 'urbanpop', 'gatheringlimit', 'hospibed', 'smokers', 'sex0', 'sex14', 'sex25', 'sex54', 'sex64', 'sex65plus', 'sexratio', 'lung', 'femalelung', 'malelung', 'gdp2019', 'healthexp', 'healthperpop', 'fertility', 'avgtemp', 'avghumidity', 'totalcases', 'active30', 'active31', 'deaths', 'newdeaths30', 'newdeaths31', 'recovered', 'critical30', 'critical31', 'casediv1m', 'deathdiv1m', 'Type', 'Location', 'newcases30', 'newcases31']

for i in objcols:

  data_new2[i] = pd.to_numeric(data_new2[i], errors='coerce')

data_new2.fillna(0, inplace=True)
corr_matrix = data_new2.corr()

corr_matrix['ConfirmedCases'].sort_values(ascending=False)
corr_matrix['Fatalities'].sort_values(ascending=False)
corr_matrix.to_csv('corrr.csv', index=False)
CCfeatures = ['Location', 'DOY', 'Id', 'newdeaths31', 'newdeaths30', 'tests', 'quarantine', 'firstcase', 'deathdiv1m']

CCtarget = ['ConfirmedCases']

Ffeatures = ['Location', 'DOY', 'Id', 'newdeaths31', 'newdeaths30', 'tests', 'deathdiv1m', 'quarantine', 'nonessential']

Ftarget = ['Fatalities']



yCC = data_new2.loc[:,CCtarget]

yF = data_new2.loc[:,Ftarget]

xCC = data_new2.loc[:,CCfeatures]

xF = data_new2.loc[:,Ffeatures]
from sklearn.model_selection import train_test_split

xCC_train,xCC_test,yCC_train,yCC_test=train_test_split(xCC,yCC,test_size=0.2, random_state = 140001742)

xF_train,xF_test,yF_train,yF_test=train_test_split(xF,yF,test_size=0.2, random_state = 140001742)
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()

lin_reg.fit(xCC_train, yCC_train)



# Now to check the validity of the model



lin_predictions = lin_reg.predict(xCC_test)



from sklearn.metrics import mean_squared_error

lin_mse = mean_squared_error(lin_predictions, yCC_test)

lin_rmse = np.sqrt(lin_mse)

print("MSE: %d" % lin_mse, end="\n")

print("RMSE: %d" % lin_rmse)
from sklearn.ensemble import RandomForestRegressor

rfr_reg = RandomForestRegressor()

rfr_reg.fit(xCC_train, yCC_train)



# Check validity



rfr_predictions = rfr_reg.predict(xCC_test)



rfr_mse = mean_squared_error(yCC_test, rfr_predictions)

rfr_rmse = np.sqrt(rfr_mse)

print("MSE: %d" % rfr_mse, end="\n")

print("RMSE: %d" % rfr_rmse)
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()

tree_reg.fit(xCC_train, yCC_train)



# Now to check the validity of the model



dtr_predictions = tree_reg.predict(xCC_test)



dtr_mse = mean_squared_error(yCC_test, dtr_predictions)

dtr_rmse = np.sqrt(dtr_mse)

print("MSE: %d" % dtr_mse, end="\n")

print("RMSE: %d" % dtr_rmse)
import math as math
from sklearn.ensemble import RandomForestClassifier

ranforclas = RandomForestClassifier(random_state=140001742)

ranforclas.fit(xCC_train, yCC_train)



# Now to check the validity of the model



ranforclas_predictions = ranforclas.predict(xCC_test)



rfc_mse = mean_squared_error(yCC_test, ranforclas_predictions)

rfc_rmse = math.sqrt(rfc_mse)

print("MSE: %d" % rfc_mse, end="\n")

print("RMSE: %d" % rfc_rmse)
lin_reg1 = LinearRegression()

lin_reg1.fit(xF_train, yF_train)



# check validity



lin_predictions1 = lin_reg1.predict(xF_test)



lin_mse1 = mean_squared_error(lin_predictions1, yF_test)

lin_rmse1 = np.sqrt(lin_mse1)

print("MSE: %d" % lin_mse1, end="\n")

print("RMSE: %d" % lin_rmse1)
rfr_reg1 = RandomForestRegressor()

rfr_reg1.fit(xF_train, yF_train)



# check validity



rfr_predictions1 = rfr_reg1.predict(xF_test)



rfr_mse1 = mean_squared_error(yF_test, rfr_predictions1)

rfr_rmse1 = np.sqrt(rfr_mse1)

print("MSE: %d" % rfr_mse1, end="\n")

print("RMSE: %d" % rfr_rmse1)
tree_reg1 = DecisionTreeRegressor()

tree_reg1.fit(xF_train, yF_train)



# Check Validity



dtr_predictions1 = tree_reg1.predict(xF_test)



dtr_mse1 = mean_squared_error(yF_test, dtr_predictions1)

dtr_rmse1 = np.sqrt(dtr_mse1)

print("MSE: %d" % dtr_mse1, end="\n")

print("RMSE: %d" % dtr_rmse1)
ranforclas1 = RandomForestClassifier(random_state=140001742)

ranforclas1.fit(xF_train, yF_train)



# Check Validity



ranforclas_predictions1 = ranforclas1.predict(xF_test)



rfc_mse1 = mean_squared_error(yF_test, ranforclas_predictions1)

rfc_rmse1 = math.sqrt(rfr_mse1)

print("MSE: %d" % rfc_mse1, end="\n")

print("RMSE: %d" % rfc_rmse1)
data_test = pd.read_csv("../input/corona/test (1).csv", header=0)
data_test['Province_State'].fillna('', inplace=True)

data_test['Date'] = pd.to_datetime(data_test['Date'])

data_test['DOY'] = data_test.Date.dt.dayofyear

data_test1 = pd.merge(data_test, data_info, how='left', left_on=['Country_Region', 'Province_State'], right_on=['country', 'region'])

data_test1['Date'] = pd.to_datetime(data_test1['Date'])

data_test1.info()
data_lockdown['Date'] = pd.to_datetime(data_lockdown['Date'])

data_test2 = pd.merge(data_test1, data_lockdown, how='left', left_on=['Country_Region', 'Province_State', 'Date'], right_on=['Country/Region', 'Province', 'Date'])
data_test2.fillna(0, inplace=True)
data_test2['quarantine'] = pd.to_datetime(data_test2['quarantine'])

data_test2['quarantine'] = data_test2.quarantine.dt.dayofyear

data_test2['nonessential'] = pd.to_datetime(data_test2['nonessential'])

data_test2['firstcase'] = pd.to_datetime(data_test2['firstcase'])

data_test2['nonessential'] = data_test2.nonessential.dt.dayofyear

data_test2['firstcase'] = data_test2.firstcase.dt.dayofyear
data_test2['Location'] = data_test2.Country_Region.astype(str) + ":" + data_test2.Province_State.astype(str)
objcols = ['pop', 'tests', 'testpop', 'density', 'medianage', 'urbanpop', 'gatheringlimit', 'hospibed', 'smokers', 'sex0', 'sex14', 'sex25', 'sex54', 'sex64', 'sex65plus', 'sexratio', 'lung', 'femalelung', 'malelung', 'gdp2019', 'healthexp', 'healthperpop', 'fertility', 'avgtemp', 'avghumidity', 'totalcases', 'active30', 'active31', 'deaths', 'newdeaths30', 'newdeaths31', 'recovered', 'critical30', 'critical31', 'casediv1m', 'deathdiv1m', 'Type', 'Location', 'newcases30', 'newcases31']

for i in objcols:

  data_test2[i] = pd.to_numeric(data_new2[i], errors='coerce')
data_test2.fillna(0, inplace=True)
CCfeatures = ['Location', 'DOY', 'ForecastId', 'newdeaths31', 'newdeaths30', 'tests', 'quarantine', 'firstcase', 'deathdiv1m']

Ffeatures = ['Location', 'DOY', 'ForecastId', 'newdeaths31', 'newdeaths30', 'tests', 'deathdiv1m', 'quarantine', 'nonessential']

XCC = data_test2.loc[:,CCfeatures]

XF = data_test2.loc[:,Ffeatures]
final_predictions_CC = tree_reg.predict(XCC)

final_predictions_F = rfr_reg1.predict(XF)
My_Preds = pd.DataFrame(data_test['ForecastId'])

My_Preds['ConfirmedCases'] = final_predictions_CC

My_Preds['Fatalities'] = final_predictions_F



print(My_Preds)
My_Preds.to_csv('submission.csv', index=False)