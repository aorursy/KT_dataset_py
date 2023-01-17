#install libraries
!pip install regressors -U
plt.rcParams['figure.figsize'] = [10, 5]
#01 - Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
plt.style.use('ggplot')
from datetime import datetime
import csv
import datetime as dt
from datetime import date
from collections import Counter
from regressors import stats
#02 - Load Plant 1 Data
plant_1 = pd.read_csv("../input/solar-power-generation-data/Plant_1_Generation_Data.csv",\
                      quoting=csv.QUOTE_NONE, error_bad_lines=False)
#Load Weather data of plant 1
weather_1 = pd.read_csv('../input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv', \
                      quoting=csv.QUOTE_NONE, error_bad_lines=False)


#Change data type for date time    
plant_1['DATE_TIME'] = plant_1['DATE_TIME'].apply(lambda x: dt.date.strftime(dt.datetime.strptime(x, '%d-%m-%Y %H:%M')\
                                                                            , "%m/%d/%Y %H:%M"))
weather_1['DATE_TIME'] = weather_1['DATE_TIME'].apply\
(lambda x: dt.date.strftime(dt.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'), "%m/%d/%Y %H:%M"))


#Merge Plant and Weather data
plant_data = plant_1.merge(weather_1[['AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION', 'DATE_TIME']],\
                      how = 'left',
                      left_on=['DATE_TIME'],\
                     right_on = ['DATE_TIME'])
plant_data['DATE_TIME'] = plant_data['DATE_TIME'].apply(lambda x: dt.datetime.strptime(x, '%m/%d/%Y %H:%M'))
plant_data['TIME'] = plant_data['DATE_TIME'].apply(lambda x: str(x.hour)+str(x.minute))
#01 - Stats
plant_data.describe()
test_df = plant_data.copy()
#04 - Compare total yield to DC_power
test = test_df[(test_df['SOURCE_KEY']=='1IF53ai7Xc0U56Y')]
fig, ax = plt.subplots() # Create the figure and axes object

# Plot the first x and y axes:
test.plot(x = 'DATE_TIME', y = 'TOTAL_YIELD', ax = ax) 
test.plot(x = 'DATE_TIME', y = 'DC_POWER', ax = ax, secondary_y = True) 
#03 - Data Cleaning Functions
def CompressData(dataframe):
    dc_ind = False
    one_ind = False
    data = []
    for row in dataframe.itertuples():
        if int(row.DC_POWER) == 0:
            if dc_ind is False:
                plant_id = row.PLANT_ID
                source_key = row.SOURCE_KEY
                dc_pow = row.DC_POWER
                ac_pow = row.AC_POWER
                tot_yld = row.TOTAL_YIELD
                start_time = row.DATE_TIME
                amb_temp = row.AMBIENT_TEMPERATURE
                mod_temp = row.MODULE_TEMPERATURE
                irrad = row.IRRADIATION
                dc_ind = True
            else:
                end_time = row.DATE_TIME
                minutes_diff = (end_time - start_time).total_seconds() / 60.0
                tots = row.TOTAL_YIELD
                yld_diff = tots - tot_yld
                dc_pow = (row.DC_POWER + dc_pow)/2
                ac_pow = (row.AC_POWER + ac_pow)/2
                mod_temp = (row.MODULE_TEMPERATURE +mod_temp)/2
                amb_temp = (row.AMBIENT_TEMPERATURE +amb_temp)/2
                irrad = (row.IRRADIATION + irrad)/2
                one_ind = True
                
                pass
        else:
            if dc_ind is True:
                if one_ind is True:
                    try:
                        data.append([plant_id, source_key, start_time, end_time, minutes_diff, dc_pow, \
                                 ac_pow, tots, yld_diff, amb_temp, mod_temp, irrad])
                        del end_time, minutes_diff, tots, yld_diff
                        one_ind = False
                    except Exception as e:
                        print("one", e, row)
                else:
                    try:
                        data.append([plant_id, source_key, data[-1][3], start_time, \
                                 (start_time - data[-1][3]).total_seconds() / 60.0 ,\
                                 dc_pow, ac_pow, tot_yld, (tot_yld-data[-1][7]), amb_temp, \
                                 mod_temp, irrad])
                    except Exception as e:
                        print("two",e, row)
                    
                try:
                    data.append([row.PLANT_ID, row.SOURCE_KEY, data[-1][3], row.DATE_TIME,\
                         (row.DATE_TIME - data[-1][3]).total_seconds() / 60.0, row.DC_POWER, row.AC_POWER,\
                             row.TOTAL_YIELD, row.TOTAL_YIELD - data[-1][7], row.AMBIENT_TEMPERATURE, \
                             row.MODULE_TEMPERATURE, row.IRRADIATION])
                    dc_ind = False
                except Exception as e:
                    print("three",e, row)
            else:
                try:
                    start_time = data[-1][3]
                except Exception as e:
                    print("special", data, e, row)
                minutes_diff = (row.DATE_TIME - start_time).total_seconds() / 60.0
                tot_yld = data[-1][7]
                yld_diff = row.TOTAL_YIELD-tot_yld
                try:
                    data.append([row.PLANT_ID, row.SOURCE_KEY, start_time, row.DATE_TIME, minutes_diff, \
                             row.DC_POWER, row.AC_POWER, row.TOTAL_YIELD, yld_diff, row.AMBIENT_TEMPERATURE, \
                             row.MODULE_TEMPERATURE, row.IRRADIATION])
                except Exception as e:
                    print("four",e, row)
    return data
#01 - Call the Cleaning function for each Inverter
Final_df = pd.DataFrame(columns=['plant_id', 'source_key', 'start_time', 'end_time',\
                                   'minutes', 'dc_pow', 'ac_pow', 'total_yield', 'yield_generated'\
                                ,'amb_temp', 'mod_temp', 'irrad'])
for i in set(test_df.SOURCE_KEY):
    dfs = test_df[test_df['SOURCE_KEY']== i]
    dfs.reset_index(drop=True)
    temp_df = pd.DataFrame(data=CompressData(dfs), columns= Final_df.columns)
    Final_df = Final_df.append(temp_df)
#Check time when DC_POWER is zero
Final_df['timestamp'] = Final_df['end_time'].apply(lambda x: str(x.hour)+str(x.minute))
a = list(Final_df[Final_df['dc_pow']==0].timestamp)
letter_counts = Counter(a)
df = pd.DataFrame.from_dict(letter_counts, orient='index')
df1 = df.head(60)
df1.plot(kind='bar')
#Delete records with TIME as 5:45
no_power_df = Final_df[Final_df['timestamp']=='545']
#Remove them from the test_df
Final_df = pd.concat([Final_df, no_power_df, no_power_df]).drop_duplicates(keep=False)
Final_df = Final_df.reset_index(drop = True)
#Add a new dimension as Yield Per Minute
Final_df['ypm'] = (Final_df['yield_generated']*1.00)/Final_df['minutes']
Final_df['effic'] = ((Final_df['ac_pow']/Final_df['dc_pow'])*100.0).fillna(0)
Final_df['datestamp'] = Final_df['end_time'].apply(lambda x: dt.datetime.strftime(x, '%Y%m%d'))
Final_df[Final_df.isna().any(axis = 1)]
Final_df.irrad.fillna(method='ffill', inplace=True)
Final_df.mod_temp.fillna(method='ffill', inplace=True)
Final_df.amb_temp.fillna(method='ffill', inplace=True)
test = Final_df[(Final_df['source_key']=='zBIq5rxdHJRwDNY')]
fig, ax = plt.subplots() # Create the figure and axes object
test['dtemp'] = test['mod_temp']*10

# Plot the first x and y axes:
test.plot(x = 'datestamp', y = 'dc_pow', ax = ax) 
test.plot(x = 'datestamp', y = 'yield_generated', ax = ax, secondary_y = True) 
test.plot(x = 'datestamp', y = 'dtemp', ax = ax, secondary_y = True) 
#OneHotEncoding for inverters
dummies = pd.get_dummies(Final_df.source_key, drop_first=True)
Solar_plant_1 = pd.concat([Final_df, dummies], axis = 'columns')
Solar_plant_1
del Solar_plant_1['source_key']
Solar_plant_1 = Solar_plant_1.reset_index(drop=True)

#Prep data for building model
yld_gen = Solar_plant_1['ypm']
data = Solar_plant_1.copy()
del data['ypm']
del data['plant_id'], data['start_time'], data['end_time'], data['minutes'], data['total_yield'], data['timestamp'], data['datestamp'], data['yield_generated']

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
#Split data
X_train, X_test, Y_train, Y_test = train_test_split(data, yld_gen, test_size = 0.3)

#Build Model
model = linear_model.LinearRegression()
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
Y_pred.shape
from sklearn.metrics import mean_squared_error, r2_score
print("Coefficients:  ", model.coef_)
print("Intercept:  ", model.intercept_)
print("MSE:  %.2f" % mean_squared_error(Y_test, Y_pred))
print("Coefficients of determination:  %.2f" % r2_score(Y_test, Y_pred))

print("\n=========== SUMMARY ===========")
xlabels = X_test.columns
stats.summary(model, X_test, Y_test, xlabels)
fig = sns.regplot(x=Y_pred, y=Y_test, scatter_kws={"color": "black"}, line_kws={"color": "red"})
fig.set(xlabel='Predicted YPM', ylabel='Recorded YPM')
plt.show()
#Source Plant 2 data
plant_2 = pd.read_csv("../input/solar-power-generation-data/Plant_2_Generation_Data.csv",\
                      quoting=csv.QUOTE_NONE, error_bad_lines=False)
#Load Weather data of plant 2
weather_2 = pd.read_csv('../input/solar-power-generation-data/Plant_2_Weather_Sensor_Data.csv', \
                      quoting=csv.QUOTE_NONE, error_bad_lines=False)


#Change data type for date time    
plant_2['DATE_TIME'] = plant_2['DATE_TIME'].apply(lambda x: dt.date.strftime(dt.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')\
                                                                            , "%m/%d/%Y %H:%M"))
weather_2['DATE_TIME'] = weather_2['DATE_TIME'].apply\
(lambda x: dt.date.strftime(dt.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'), "%m/%d/%Y %H:%M"))


#Merge Plant and Weather data
plant_data2 = plant_2.merge(weather_2[['AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION', 'DATE_TIME']],\
                      how = 'left',
                      left_on=['DATE_TIME'],\
                     right_on = ['DATE_TIME'])
plant_data2['DATE_TIME'] = plant_data2['DATE_TIME'].apply(lambda x: dt.datetime.strptime(x, '%m/%d/%Y %H:%M'))
plant_data2['TIME'] = plant_data2['DATE_TIME'].apply(lambda x: str(x.hour)+str(x.minute))
plant_data2.describe()
test_df2 = plant_data2.copy()

test = test_df2[(test_df2['SOURCE_KEY']=='9kRcWv60rDACzjR')]
fig, ax = plt.subplots() # Create the figure and axes object

# Plot the first x and y axes:
test.plot(x = 'DATE_TIME', y = 'TOTAL_YIELD', ax = ax) 
test.plot(x = 'DATE_TIME', y = 'DC_POWER', ax = ax, secondary_y = True) 
test = test_df2[(test_df2['SOURCE_KEY']=='9kRcWv60rDACzjR')&(test_df2['DC_POWER']==0)]
fig, ax = plt.subplots() # Create the figure and axes object

# Plot the first x and y axes:
test.plot(x = 'DATE_TIME', y = 'IRRADIATION', ax = ax) 
test.plot(x = 'DATE_TIME', y = 'MODULE_TEMPERATURE', ax = ax)
test.plot(x = 'DATE_TIME', y = 'DC_POWER', ax = ax, secondary_y = True) 
data = []
dumps = []
d2 = []
for j in set(test_df2.SOURCE_KEY):
    dfs = test_df2[test_df2['SOURCE_KEY']== j]
    dfs.reset_index(drop=True)
    ind = 0
    
    for i in dfs.itertuples():
        if ind ==0:
            data.append([i.DATE_TIME, i.PLANT_ID, i.SOURCE_KEY, i.DC_POWER, i.AC_POWER, i.DAILY_YIELD,\
                     i.TOTAL_YIELD, i.AMBIENT_TEMPERATURE, i.MODULE_TEMPERATURE, i.IRRADIATION, i.TIME])
            ind +=1
        else:
            if i.TOTAL_YIELD>=data[-1][6]:
                data.append([i.DATE_TIME, i.PLANT_ID, i.SOURCE_KEY, i.DC_POWER, i.AC_POWER, i.DAILY_YIELD,\
                     i.TOTAL_YIELD, i.AMBIENT_TEMPERATURE, i.MODULE_TEMPERATURE, i.IRRADIATION, i.TIME])
            elif dumps:
                if i.TOTAL_YIELD <= dumps[-1][6]:
                    d2.append([i.DATE_TIME, i.PLANT_ID, i.SOURCE_KEY, i.DC_POWER, i.AC_POWER, i.DAILY_YIELD,\
                     i.TOTAL_YIELD, i.AMBIENT_TEMPERATURE, i.MODULE_TEMPERATURE, i.IRRADIATION, i.TIME])
            else:
                dumps.append([i.DATE_TIME, i.PLANT_ID, i.SOURCE_KEY, i.DC_POWER, i.AC_POWER, i.DAILY_YIELD,\
                     i.TOTAL_YIELD, i.AMBIENT_TEMPERATURE, i.MODULE_TEMPERATURE, i.IRRADIATION, i.TIME])
dump_df = pd.DataFrame(data=dumps, columns= test_df2.columns)
data_df = pd.DataFrame(data=data, columns= test_df2.columns)
d2_df = pd.DataFrame(data=d2, columns= test_df2.columns)
test = data_df[(data_df['SOURCE_KEY']=='9kRcWv60rDACzjR')]
fig, ax = plt.subplots() # Create the figure and axes object
test.plot(x = 'DC_POWER', y = 'MODULE_TEMPERATURE', ax = ax, secondary_y = True) 
no_dc2 = data_df[(data_df['MODULE_TEMPERATURE']>38) &(data_df['DC_POWER']==0)]
data_df = pd.concat([data_df, no_dc2, no_dc2]).drop_duplicates(keep = False)
data_df = data_df.reset_index(drop=True)
test = data_df[(data_df['SOURCE_KEY']=='9kRcWv60rDACzjR')]
fig, ax = plt.subplots() # Create the figure and axes object

# Plot the first x and y axes:
test.plot(x = 'DC_POWER', y = 'MODULE_TEMPERATURE', ax = ax, secondary_y = True) 
test = data_df[(data_df['SOURCE_KEY']=='9kRcWv60rDACzjR')]
fig, ax = plt.subplots() # Create the figure and axes object
# Plot the first x and y axes:
test.plot(x = 'DATE_TIME', y = 'DC_POWER', ax = ax) 
test.plot(x = 'DATE_TIME', y = 'TOTAL_YIELD', ax = ax, secondary_y = True) 
#04 - Call the Cleaning function for each Inverter
Final_df2 = pd.DataFrame(columns=['plant_id', 'source_key', 'start_time', 'end_time',\
                                   'minutes', 'dc_pow', 'ac_pow', 'total_yield', 'yield_generated'\
                                ,'amb_temp', 'mod_temp', 'irrad'])
for i in set(data_df.SOURCE_KEY):
    dfs = data_df[data_df['SOURCE_KEY']== i]
    dfs.reset_index(drop=True)
    temp_df = pd.DataFrame(data=CompressData(dfs), columns= Final_df2.columns)
    Final_df2 = Final_df2.append(temp_df)
#Add a new dimension as Yield Per Minute
Final_df2['ypm'] = (Final_df2['yield_generated']*1.00)/Final_df2['minutes']
Final_df2['effic'] = ((Final_df2['ac_pow']/Final_df2['dc_pow'])*100.0).fillna(0)
Final_df2['datestamp'] = Final_df2['end_time'].apply(lambda x: dt.datetime.strftime(x, '%Y%m%d'))
Final_df2[Final_df2.isna().any(axis = 1)]
test = Final_df2[(Final_df2['source_key']=='9kRcWv60rDACzjR')]
fig, ax = plt.subplots() # Create the figure and axes object

# Plot the first x and y axes:
test.plot(x = 'dc_pow', y = 'mod_temp', ax = ax, secondary_y = True) 
#Verify if there is any records with 0 dc_power
Final_df2['time'] = Final_df2['end_time'].apply(lambda x: str(x.hour)+str(x.minute))

a = list(Final_df2[Final_df2['dc_pow']==0].time)
letter_counts = Counter(a)
df = pd.DataFrame.from_dict(letter_counts, orient='index')
df1 = df.head(60)
df.plot(kind='bar')
#Delete records with TIME as 4:45
no_power_df2 = Final_df2[Final_df2['time']=='545']
#Remove them from the test_df
Final_df2 = pd.concat([Final_df2, no_power_df2, no_power_df2]).drop_duplicates(keep=False)
Final_df2 = Final_df2.reset_index(drop = True)
#OneHotEncoding for inverters
dummies2 = pd.get_dummies(Final_df2.source_key, drop_first=True)
Solar_plant_2 = pd.concat([Final_df2, dummies2], axis = 'columns')
Solar_plant_2
del Solar_plant_2['source_key']
Solar_plant_2 = Solar_plant_2.reset_index(drop=True)

#Prep data for building model
yld_gen2 = Solar_plant_2['ypm']
data2 = Solar_plant_2.copy()
del data2['ypm']
del data2['plant_id'], data2['start_time'], data2['end_time'], data2['minutes'], data2['total_yield'], data2['time'], data2['datestamp'], data2['yield_generated']

#Split data
X2_train, X2_test, Y2_train, Y2_test = train_test_split(data2, yld_gen2, test_size = 0.3)

#Build Model
model2 = linear_model.LinearRegression()
model2.fit(X2_train, Y2_train)
Y2_pred = model2.predict(X2_test)
Y2_pred.shape
from sklearn.metrics import mean_squared_error, r2_score
print("Coefficients:  ", model2.coef_)
print("Intercept:  ", model2.intercept_)
print("MSE:  %.2f" % mean_squared_error(Y2_test, Y2_pred))
print("Coefficients of determination:  %.2f" % r2_score(Y2_test, Y2_pred))

# to print summary table:
print("\n=========== SUMMARY ===========")
xlabels = X2_test.columns
stats.summary(model2, X2_test, Y2_test, xlabels)
fig = sns.regplot(x=Y2_pred, y=Y2_test, scatter_kws={"color": "black"}, line_kws={"color": "red"})
fig.set(xlabel='Predicted YPM', ylabel='Recorded YPM')
plt.show()