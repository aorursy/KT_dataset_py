import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
data = pd.read_excel('/kaggle/input/obd-dataset/TestData.xlsx')
data.head()
data['TROUBLE_CODES'].unique() # check the DTC (diagnostic trouble codes) codes present in the data and verify if our required codes are present
# we could observe that null values are present in the data
data = data.dropna(subset = ['TROUBLE_CODES']).reset_index(drop=True) # getting rid of null values from trouble codes
data['TIMESTAMP'] = pd.to_datetime(data['TIMESTAMP'], unit='ms') # converting Timestamp to proper format.
data.loc[(data['TROUBLE_CODES'].str.contains('P007E')) | (data['TROUBLE_CODES'].str.contains('P007F'))].info()
# checking data of interest for null values and data sparsity
# we could see some of the columns like FUEL_LEVEL, BAROMETRIC_PRESSURE(KPA)..etc having no data. 
data.loc[(data['TROUBLE_CODES'].str.contains('P007E')) | (data['TROUBLE_CODES'].str.contains('P007F'))].nunique()
# checking data of interest for unique & constant values
# we could see some of the columns like MAKE, MODEL, DTC_NUMBER having only single value for all the timestamps.
req_data = data[['TIMESTAMP','TROUBLE_CODES','ENGINE_COOLANT_TEMP','ENGINE_LOAD','ENGINE_RPM','INTAKE_MANIFOLD_PRESSURE','AIR_INTAKE_TEMP','SPEED','SHORT TERM FUEL TRIM BANK 1','THROTTLE_POS','TIMING_ADVANCE']]
# from the data sanity checks we have selected the columns in the dataset which vary with time to play with. 
req_data.head()
req_data.iloc[:,1:].describe().round(2)
# checking the distributions of the sensor data - insights are given at the end of the script
req_data = req_data.interpolate()
# interpolating missing values in the sensor data
req_data.iloc[:,1:].describe().round(2)
# checking the distributions of the sensor data after interpolation - insights are given at the end of the script
plt.figure(figsize = (15,5))
sns.heatmap(req_data.iloc[:,1:].corr(),annot=True)
# checking the correlations between the various sensor data - currently pearson correlation alone is taken into picture
req_data_1 = req_data[req_data.TROUBLE_CODES.str.contains('P007E')].reset_index(drop=True)
req_data_2 = req_data[req_data.TROUBLE_CODES.str.contains('P007F')].reset_index(drop=True)
# data slicing based on Trouble codes
req_data_1.head()
req_data_1 = req_data_1.set_index(req_data_1['TIMESTAMP']).resample('D').mean().reset_index().fillna(0)
req_data_1['TROUBLE_CODES'] = 'P007E'
req_data_2 = req_data_2.set_index(req_data_2['TIMESTAMP']).resample('D').mean().reset_index().fillna(0)
req_data_2['TROUBLE_CODES'] = 'P007F'
# generalizing trouble code column to two groups - P007E & P007F
combined_data_resampled = pd.concat([req_data_1,req_data_2],ignore_index=True,sort=True)
# retaining original data for comparision with the resampled data
data1 = req_data[req_data.TROUBLE_CODES.str.contains('P007E')].reset_index(drop=True)
data2 = req_data[req_data.TROUBLE_CODES.str.contains('P007F')].reset_index(drop=True)
# data slicing based on Trouble codes
combined_data = pd.concat([data1,data2],ignore_index=True,sort=True)
X = combined_data.drop(columns={'TROUBLE_CODES','TIMESTAMP'})
y = combined_data['TROUBLE_CODES']
X_resampled = combined_data_resampled.drop(columns={'TROUBLE_CODES','TIMESTAMP'})
y_resampled = combined_data_resampled['TROUBLE_CODES']
plt.figure(figsize = (15,5))
plt.plot(data1['TIMESTAMP'],data1['ENGINE_COOLANT_TEMP'], marker = "s", label = 'P007E')
plt.plot(data2['TIMESTAMP'],data2['ENGINE_COOLANT_TEMP'], marker = "o", label = 'P007F')
plt.legend(loc='best')
plt.ylabel('ENGINE_COOLANT_TEMP')
plt.show()

#For P007F DTC code, coolant temperature is high as the engine heats up due to the failure of bank 2.
# Dataset couldn't be downsampled as the trend of the data would drastically change for the given trouble codes.
mdl1 = LogisticRegression(random_state=0,solver='lbfgs',multi_class='auto').fit(np.array(X['ENGINE_COOLANT_TEMP']).reshape(-1,1) , y)
mdl1.score(np.array(X['ENGINE_COOLANT_TEMP']).reshape(-1,1),y)

# checking the accuracy of the classifier to decide on the relationship between sensor data and trouble code's
# accuracy score is low to set up a relationship
mdl1.predict(np.array([114.1]).reshape(-1,1)),mdl1.predict(np.array([114.2]).reshape(-1,1))
# from the classifier model decision threshold of sensor data between the fault codes = 114.1
plt.figure(figsize = (15,5))
plt.plot(req_data_1['TIMESTAMP'],req_data_1['ENGINE_COOLANT_TEMP'].rolling(window=3).mean(), marker = "s", label = 'P007E')
plt.plot(req_data_2['TIMESTAMP'],req_data_2['ENGINE_COOLANT_TEMP'].rolling(window=3).mean(), marker = "o", label = 'P007F')
plt.legend(loc='best')
plt.ylabel('ENGINE_COOLANT_TEMP')
plt.show()
# after downsampling
# P007E - sensor value has a dip from sept 1st to sept 4th, and increases after september 9th. 
# P007F - sensor value decreases to 0 until sept 3rd, and follows a seasonal pattern afterwards.
plt.figure(figsize = (15,5))
plt.plot(data1['TIMESTAMP'],data1['ENGINE_LOAD'], marker = "s", label = 'P007E')
plt.plot(data2['TIMESTAMP'],data2['ENGINE_LOAD'], marker = "o", label = 'P007F')
plt.legend(loc='best')
plt.ylabel('ENGINE_LOAD')
plt.show()

# Engine Loads are high for P007E codes in the initial timestamps.
mdl2 = LogisticRegression(random_state=0,solver='lbfgs',multi_class='auto').fit(np.array(X['ENGINE_LOAD']).reshape(-1,1) , y)
mdl2.score(np.array(X['ENGINE_LOAD']).reshape(-1,1),y)

# checking the accuracy of the classifier to decide on the relationship between sensor data and trouble code's
# accuracy score is low to set up a relationship
plt.figure(figsize = (15,5))
plt.plot(req_data_1['TIMESTAMP'],req_data_1['ENGINE_LOAD'].rolling(window=3).mean(), marker = "s", label = 'P007E')
plt.plot(req_data_2['TIMESTAMP'],req_data_2['ENGINE_LOAD'].rolling(window=3).mean(), marker = "o", label = 'P007F')
plt.legend(loc='best')
plt.ylabel('ENGINE_LOAD')
plt.show()
# after downsampling
# P007E - sensor value has a dip from sept 1st to sept 4th, and increases after september 9th. 
# P007F - sensor value decreases to 0 until sept 3rd, and follows a seasonal pattern afterwards.
plt.figure(figsize = (15,5))
plt.plot(data1['TIMESTAMP'],data1['ENGINE_RPM'], marker = "s", label = 'P007E')
plt.plot(data2['TIMESTAMP'],data2['ENGINE_RPM'], marker = "o", label = 'P007F')
plt.legend(loc='best')
plt.ylabel('ENGINE_RPM')
plt.show()

# ENGINE_RPM doesn't follow a specific trend for both P007E and P007F codes.
mdl3 = LogisticRegression(random_state=0,solver='lbfgs',multi_class='auto').fit(np.array(X['ENGINE_RPM']).reshape(-1,1) , y)
mdl3.score(np.array(X['ENGINE_RPM']).reshape(-1,1),y)

# checking the accuracy of the classifier to decide on the relationship between sensor data and trouble code's
# accuracy score is low to set up a relationship
plt.figure(figsize = (15,5))
plt.plot(req_data_1['TIMESTAMP'],req_data_1['ENGINE_RPM'].rolling(window=3).mean(), marker = "s", label = 'P007E')
plt.plot(req_data_2['TIMESTAMP'],req_data_2['ENGINE_RPM'].rolling(window=3).mean(), marker = "o", label = 'P007F')
plt.legend(loc='best')
plt.ylabel('ENGINE_RPM')
plt.show()
# after downsampling
# P007E - sensor value has a dip from sept 1st to sept 4th, and increases after september 9th. 
# P007F - sensor value decreases to 0 until sept 3rd, and follows a seasonal pattern afterwards.
plt.figure(figsize = (15,5))
plt.plot(data1['TIMESTAMP'],data1['INTAKE_MANIFOLD_PRESSURE'], marker = "s", label = 'P007E')
plt.plot(data2['TIMESTAMP'],data2['INTAKE_MANIFOLD_PRESSURE'], marker = "o", label = 'P007F')
plt.legend(loc='best')
plt.ylabel('INTAKE_MANIFOLD_PRESSURE')
plt.show()

#Air pressure is high for both the codes in initial timestamps
mdl4 = LogisticRegression(random_state=0,solver='lbfgs',multi_class='auto',max_iter = 200).fit(np.array(X['INTAKE_MANIFOLD_PRESSURE']).reshape(-1,1) , y)
mdl4.score(np.array(X['INTAKE_MANIFOLD_PRESSURE']).reshape(-1,1),y)

# checking the accuracy of the classifier to decide on the relationship between sensor data and trouble code's
# accuracy score is low to set up a relationship
plt.figure(figsize = (15,5))
plt.plot(req_data_1['TIMESTAMP'],req_data_1['INTAKE_MANIFOLD_PRESSURE'].rolling(window=3).mean(), marker = "s", label = 'P007E')
plt.plot(req_data_2['TIMESTAMP'],req_data_2['INTAKE_MANIFOLD_PRESSURE'].rolling(window=3).mean(), marker = "o", label = 'P007F')
plt.legend(loc='best')
plt.ylabel('INTAKE_MANIFOLD_PRESSURE')
plt.show()

# after downsampling
# P007E - sensor value decreases until september 4 and then gradually increases
# P007F - sensor value decreases until september 4, but appears to follow a seasonal pattern afterwards.
plt.figure(figsize = (15,5))
plt.plot(data1['TIMESTAMP'],data1['AIR_INTAKE_TEMP'], marker = "s", label = "P007E")
plt.plot(data2['TIMESTAMP'],data2['AIR_INTAKE_TEMP'], marker = "o", label = "P007F")
# plt.scatter(time3,workingset['AIR_INTAKE_TEMP'], c = 'g', marker = "x", label = "other data")
plt.legend(loc='best')
plt.xlabel('TIMESTAMP')
plt.ylabel('AIR_INTAKE_TEMP')
plt.show()

#Intake temperatures are high for initial time stamps for P007E codes.
mdl5 = LogisticRegression(random_state=0,solver='lbfgs',multi_class='auto',max_iter = 200).fit(np.array(X['AIR_INTAKE_TEMP']).reshape(-1,1) , y)
mdl5.score(np.array(X['AIR_INTAKE_TEMP']).reshape(-1,1),y)
plt.figure(figsize = (15,5))
plt.plot(req_data_1['TIMESTAMP'],req_data_1['AIR_INTAKE_TEMP'].rolling(window=3).mean(), marker = "s", label = 'P007E')
plt.plot(req_data_2['TIMESTAMP'],req_data_2['AIR_INTAKE_TEMP'].rolling(window=3).mean(), marker = "o", label = 'P007F')
plt.legend(loc='best')
plt.ylabel('AIR_INTAKE_TEMP')
plt.show()
# after downsampling
# P007E - sensor value decreases until september 4, and is almost constant with a small dip in september 9th and then increases afterwards. 
# P007F - sensor value decreases until september 4, but appears to follow a seasonal pattern afterwards.
plt.figure(figsize = (15,5))
plt.plot(data1['TIMESTAMP'],data1['SPEED'], marker = "s", label = 'P007E')
plt.plot(data2['TIMESTAMP'],data2['SPEED'], marker = "o", label = 'P007F')
plt.legend(loc='best')
plt.ylabel('SPEED')
plt.show()
# no specific trends are observed.
mdl6 = LogisticRegression(random_state=0,solver='lbfgs',multi_class='auto',max_iter = 200).fit(np.array(X['SPEED']).reshape(-1,1) , y)
mdl6.score(np.array(X['SPEED']).reshape(-1,1),y)
# checking the accuracy of the classifier to decide on the relationship between sensor data and trouble code's
# accuracy score is low to set up a relationship
plt.figure(figsize = (15,5))
plt.plot(req_data_1['TIMESTAMP'],req_data_1['SPEED'].rolling(window=3).mean(), marker = "s", label = 'P007E')
plt.plot(req_data_2['TIMESTAMP'],req_data_2['SPEED'].rolling(window=3).mean(), marker = "o", label = 'P007F')
plt.legend(loc='best')
plt.ylabel('SPEED')
plt.show()
# after downsampling
# P007E - sensor value has a dip from sept 1st to sept 4th, and increases after september 9th. 
# P007F - sensor value decreases to 0 until sept 5th, and follows a seasonal pattern afterwards.
plt.figure(figsize = (15,5))
plt.plot(data1['TIMESTAMP'],data1['SHORT TERM FUEL TRIM BANK 1'], marker = "s", label = 'P007E')
plt.plot(data2['TIMESTAMP'],data2['SHORT TERM FUEL TRIM BANK 1'], marker = "o", label = 'P007F')
# plt.scatter(time3,workingset['SHORT TERM FUEL TRIM BANK 1'], c = 'g', marker = "x", label = "other data")
plt.legend(loc='best')
plt.ylabel('SHORT TERM FUEL TRIM BANK 1')
plt.show()

# Fuel Trim values for bank1 are in ideal operating ranges for both the codes.
mdl7 = LogisticRegression(random_state=0,solver='lbfgs',multi_class='auto',max_iter = 200).fit(np.array(X['SHORT TERM FUEL TRIM BANK 1']).reshape(-1,1) , y)
mdl7.score(np.array(X['SHORT TERM FUEL TRIM BANK 1']).reshape(-1,1),y)
plt.figure(figsize = (15,5))
plt.plot(req_data_1['TIMESTAMP'],req_data_1['SHORT TERM FUEL TRIM BANK 1'].rolling(window=3).mean(), marker = "s", label = 'P007E')
plt.plot(req_data_2['TIMESTAMP'],req_data_2['SHORT TERM FUEL TRIM BANK 1'].rolling(window=3).mean(), marker = "o", label = 'P007F')
plt.legend(loc='best')
plt.ylabel('SHORT TERM FUEL TRIM BANK 1')
plt.show()
# after downsampling
# P007E - sensor value has a dip from sept 3rd to sept 5th, and increases after september 7th. 
# P007F - sensor value increases until sept 3rd, and is constant until september 11th.
plt.figure(figsize = (15,5))
plt.plot(data1['TIMESTAMP'],data1['THROTTLE_POS'], marker = "s", label = 'P007E')
plt.plot(data2['TIMESTAMP'],data2['THROTTLE_POS'], marker = "o", label = 'P007F')
# plt.scatter(time3,workingset['THROTTLE_POS'], c = 'g', marker = "x", label = "other data")
plt.legend(loc='best')
plt.ylabel('THROTTLE_POS')
plt.show()
mdl8 = LogisticRegression(random_state=0,solver='lbfgs',multi_class='auto',max_iter = 200).fit(np.array(X['THROTTLE_POS']).reshape(-1,1) , y)
mdl8.score(np.array(X['THROTTLE_POS']).reshape(-1,1),y)
plt.figure(figsize = (15,5))
plt.plot(req_data_1['TIMESTAMP'],req_data_1['THROTTLE_POS'].rolling(window=3).mean(), marker = "s", label = 'P007E')
plt.plot(req_data_2['TIMESTAMP'],req_data_2['THROTTLE_POS'].rolling(window=3).mean(), marker = "o", label = 'P007F')
plt.legend(loc='best')
plt.ylabel('THROTTLE_POS')
plt.show()
# after downsampling - follows the exact trend of 'AIR_INTAKE_TEMP'
# P007E - sensor value decreases until september 4, and is almost constant with a small dip in september 9th and then increases afterwards. 
# P007F - sensor value decreases until september 4, but appears to follow a seasonal pattern afterwards.
plt.figure(figsize = (15,5))
plt.plot(data1['TIMESTAMP'],data1['TIMING_ADVANCE'], marker = "s", label = 'P007E')
plt.plot(data2['TIMESTAMP'],data2['TIMING_ADVANCE'], marker = "o", label = 'P007F')
plt.legend(loc='best')
plt.ylabel('TIMING_ADVANCE')
plt.show()

#Timing advances varies widely, especially are higher for P007E codes compared to P007F codes.
mdl9 = LogisticRegression(random_state=0,solver='lbfgs',multi_class='auto',max_iter = 200).fit(np.array(X['TIMING_ADVANCE']).reshape(-1,1) , y)
mdl9.score(np.array(X['TIMING_ADVANCE']).reshape(-1,1),y)
plt.figure(figsize = (15,5))
plt.plot(req_data_1['TIMESTAMP'],req_data_1['TIMING_ADVANCE'].rolling(window=3).mean(), marker = "s", label = 'P007E')
plt.plot(req_data_2['TIMESTAMP'],req_data_2['TIMING_ADVANCE'].rolling(window=3).mean(), marker = "o", label = 'P007F')
plt.legend(loc='best')
plt.ylabel('TIMING_ADVANCE')
plt.show()
# after downsampling - almost follows the trend of 'INTAKE_MANIFOLD_PRESSURE'
# P007E - sensor value decreases until september 4 and then gradually increases
# P007F - sensor value decreases until september 4, but appears to follow a seasonal pattern afterwards.
model1 = LogisticRegression(random_state=0,solver='lbfgs',multi_class='auto',max_iter=100000).fit(X, y)
model1.score(X,y)
# when data is used without re-sampling, the accuracy of the classifier is low and requires more iterations to converge to a solution,
# and so, it is not feasible to establish a relationship between sensor data and P007E/F Trouble Codes.
# when data is used without re-sampling, considering case where we have only two classes in our target variable.
data1['TROUBLE_CODES'] = 'P007E'
data2['TROUBLE_CODES'] = 'P007F'
y = list(data1['TROUBLE_CODES']) + list(data2['TROUBLE_CODES'])
model2 = LogisticRegression(random_state=0,solver='lbfgs',multi_class='auto',max_iter=100000).fit(X, y)
model2.score(X,y)
model3 = LogisticRegression(random_state=0,solver='lbfgs',multi_class='auto',max_iter = 200).fit(X_resampled , y_resampled)
model3.score(X_resampled, y_resampled)
# data when downsampled gives us more feasibility to establish relationship between sensor data and P007E/F Trouble Codes.
sns.pairplot(X,kind='reg')
# bivariate plots between sensor data before re-sampling
plt.figure(figsize = (15,5))
sns.heatmap(X.corr(),annot=True)
sns.pairplot(X_resampled,kind='reg')
# bi-variate plots between sensor data after downsampling
plt.figure(figsize = (15,5))
sns.heatmap(X_resampled.corr(),annot=True)
