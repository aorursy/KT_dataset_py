# load the most important libraries for the data analysis and preprocessing

import numpy as np

import pandas as pd

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

from datetime import datetime, timedelta

import matplotlib.dates as mdates

from sklearn import preprocessing

import pickle

from dateutil import rrule

sns.set()

from sklearn.feature_selection import SelectFromModel



# Turn off warnings

import warnings

warnings.filterwarnings('ignore')
weight_wrz = pd.read_csv('../input/data-for-swarm-detector/gewicht 12.csv', sep = ';');

weight_wrz = weight_wrz.append(pd.read_csv('../input/data-for-swarm-detector/gewicht 13.csv', sep = ';'));

weight_wrz = weight_wrz.append(pd.read_csv('../input/data-for-swarm-detector/gewicht 14.csv', sep = ';'));

weight_wrz = weight_wrz.append(pd.read_csv('../input/data-for-swarm-detector/gewicht 15.csv', sep = ';'));

weight_wrz = weight_wrz.append(pd.read_csv('../input/data-for-swarm-detector/gewicht 16.csv', sep = ';'));

weight_wrz = weight_wrz.append(pd.read_csv('../input/data-for-swarm-detector/gewicht 17.csv', sep = ';'));

weight_wrz = weight_wrz.append(pd.read_csv('../input/data-for-swarm-detector/gewicht 18.csv', sep = ';'));

weight_wrz = weight_wrz.append(pd.read_csv('../input/data-for-swarm-detector/gewicht 19.csv', sep = ';'));
# create Pandas time series for the data

time_arr = pd.to_datetime(weight_wrz.Time.str[0:18])

ts_wrz = pd.Series(data=np.array(weight_wrz.Value), name="s_Gewicht", index=pd.DatetimeIndex(time_arr), dtype="float")



# clean measurement errors (weight below 10kg or above 100kg)

ts_wrz[ts_wrz < 10] = np.NaN

ts_wrz[ts_wrz > 100] = np.NaN



# manual removal of measurement errors

ts_wrz['2018-10-03':'2018-10-13 17:00:00'] = np.NaN

ts_wrz['2016-06-03':'2016-08-24 17:00:00'] = np.NaN

ts_wrz['2013-11-07 09:50:00':'2013-11-07 16:00:00'] = np.NaN

ts_wrz['2017-09-18 04:50:00':'2017-09-18 15:00:00'] = np.NaN

ts_wrz['2016-12-10 10:45:00':'2016-12-10 15:13:00'] = np.NaN

ts_wrz['2017-05-12 05:30:00':'2017-05-12 19:30:00'] = np.NaN

ts_wrz['2017-05-15 09:00:00':'2017-05-15 13:30:00'] = np.NaN



# resampling time series to hours (biggest time period in the data)

ts_wrz_hour = ts_wrz.resample("H").mean()

#ts_wrz_day = ts_wrz.resample("D").mean()



# plot data over all years

ax = plt.figure(figsize=(6,3), dpi=200).add_subplot(111)

ts_wrz_hour.plot(ax=ax, title="Time course of the hive's weight in Würzburg")
weight_sch = pd.read_csv('../input/data-for-swarm-detector/gewicht 1519.csv', sep = ';');

time_arr_sch = pd.to_datetime(weight_sch.Time.str[0:18])

ts_sch = pd.Series(data=np.array(weight_sch.Value), name="s_Gewicht", index=pd.DatetimeIndex(time_arr_sch), dtype="float")



# manual removal of measurement errors

ts_sch['2015-02-13 03:00:00':'2015-02-15 17:00:00'] = np.NaN



# convert weight from grams into kilograms:

ts_sch = ts_sch/1000



# resampling time series to hours (biggest time period in the data)

ts_sch_hour = ts_sch.resample("H").mean()

#ts_sch_day = ts_sch.resample("D").mean()



# plot the time serie over all years

ax = plt.figure(figsize=(6,3), dpi=200).add_subplot(111)

ts_sch_hour.plot(ax=ax, title="Time course of the hive's weight in Schwartau")
# copy the original time series for this preprocessing experiment

ts_wrz_std = ts_wrz_hour.copy(deep=True)



# get start values for the years 2012 to 2020

start_values = {}

for year in range(2012,2020):

    start_values[year] = ts_wrz_std[pd.Timestamp(str(year)+'-03-01')]



# subtract start values (check: values on Mar/01 should be 0)

for item in ts_wrz_std.index:

    ts_wrz_std[item] = ts_wrz_std[item] - start_values[item.year]



# data in 2016 is not useful and, thus, removed

ts_wrz_tmp = ts_wrz_std['2012':'2015']

ts_wrz_tmp = ts_wrz_tmp.append(ts_wrz_std['2017':'2019'])



# finally create a copy for data analysis/preprocessing

ts_wrz_h_tmp = ts_wrz_hour.copy(deep=True)

ts_wrz_h_tmp = ts_wrz_hour['2012':'2015']

ts_wrz_h_tmp = ts_wrz_h_tmp.append(ts_wrz_hour['2017':'2019'])
# repeat the procedure for Schwartau

ts_sch_std = ts_sch_hour.copy(deep=True)



start_values = {}

for year in range(2015,2020):

    start_values[year] = ts_sch_std[pd.Timestamp(str(year)+'-03-01')]

          

for item in ts_sch_std['2015':'2020'].index:

    ts_sch_std[item] = ts_sch_std[item] - start_values[item.year]



ts_sch_tmp = ts_sch_std['2015']

ts_sch_tmp = ts_sch_tmp.append(ts_sch_std['2019'])



ts_sch_h_tmp = ts_sch_hour.copy(deep=True)

ts_sch_h_tmp = ts_sch_hour['2015']

ts_sch_h_tmp = ts_sch_h_tmp.append(ts_sch_hour['2019'])
# create a dataframe for the original weight in Würzburg

df_gewicht = ts_wrz_tmp.to_frame()

df_gewicht['Ort'] = "Würzburg"



# adding data of Schwartau

df_sch_tmp = ts_sch_tmp.to_frame()

df_sch_tmp['Ort'] = 'Schwartau'

df_gewicht = df_gewicht.append(df_sch_tmp)



# identifier is the city (Ort) combined with the year; also convert date to a proper datetime value; remove "Ort"

df_gewicht['id'] = df_gewicht['Ort']  + ", " +  np.datetime_as_string(df_gewicht.index.values, unit='Y')

df_gewicht['Day'] = pd.to_datetime(df_gewicht.index.dayofyear-1, unit='D', origin=pd.Timestamp('2019-01-01'))

df_gewicht.drop(columns=["Ort"], inplace=True)

# df_gewicht contains the time series with the original weight data



# same procedure to create the 'standardized' weight

df_gewicht_full = ts_wrz_h_tmp.to_frame()

df_gewicht_full['Ort'] = "Würzburg"



# adding data of Schwartau

df_sch_full_tmp = ts_sch_h_tmp.to_frame()

df_sch_full_tmp['Ort'] = 'Schwartau'

df_gewicht_full = df_gewicht_full.append(df_sch_full_tmp)



# create identifer, correct datetime field and remove 'Ort'

df_gewicht_full['id'] = df_gewicht_full['Ort']  + ", " +  np.datetime_as_string(df_gewicht_full.index.values, unit='Y')

df_gewicht_full['Day'] = pd.to_datetime(df_gewicht_full.index.dayofyear-1, unit='D', origin=pd.Timestamp('2019-01-01'))

df_gewicht_full.drop(columns=["Ort"], inplace=True)

# df_gewicht_full is the standardized weight (0kg on March 01)
# original time series

ax = plt.figure(figsize=(6,3), dpi=200).add_subplot(111)

pd.Series(data=np.array(weight_wrz.Value), index=pd.DatetimeIndex(time_arr), dtype="float")

piv = pd.pivot_table(df_gewicht_full, index=['Day'], columns=['id'])

piv['2019-03-01':'2019-07-01'].plot(ax=ax, title="Development of weight from March to July")

ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1))

ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

ax.xaxis.set_minor_formatter(mdates.DateFormatter('%d'))

ax.tick_params(which='minor',labelsize=5, pad=1)

ax.legend(loc='upper left', prop={'size': 6}, labels=piv.columns.levels[1])
# standardized time series (shift to 0kg axis)

ax = plt.figure(figsize=(6,3), dpi=200).add_subplot(111)

pd.Series(data=np.array(weight_wrz.Value), index=pd.DatetimeIndex(time_arr), dtype="float")

piv = pd.pivot_table(df_gewicht, index=['Day'],columns=['id'])

piv['2019-03-01':'2019-07-01'].plot(ax=ax, title="Development of (standardized) weight from March to July")

ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1))

ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

ax.xaxis.set_minor_formatter(mdates.DateFormatter('%d'))

ax.tick_params(which='minor',labelsize=5, pad=1)

ax.legend(loc='upper left', prop={'size': 6}, labels=piv.columns.levels[1])
# copy standardized weight from March 01 to July 01

df_ausschnitt = df_gewicht_full[(df_gewicht_full.Day >= '2019-03-01') & (df_gewicht_full.Day < '2019-07-01')].copy(deep=True)

df_ausschnitt["cycle"] = (df_ausschnitt.index.dayofyear * 24) + df_ausschnitt.index.hour

df_ausschnitt.reset_index(inplace=True)



# (manually) remove sections without swarm events

df_ausschnitt.drop(df_ausschnitt.query('id in ["Würzburg, 2018", "Würzburg, 2017", "Würzburg, 2013", "Schwartau, 2017", "Schwartau, 2016"]').index, inplace=True)



# (manually) cut time series after the swarm events - Würzburg

df_ausschnitt.drop(df_ausschnitt[(df_ausschnitt.id == "Würzburg, 2019") & (df_ausschnitt.Time >= '2019-05-01')].index, inplace=True)

df_ausschnitt.drop(df_ausschnitt[(df_ausschnitt.id == "Würzburg, 2015") & (df_ausschnitt.Time >= '2015-05-15')].index, inplace=True)

df_ausschnitt.drop(df_ausschnitt[(df_ausschnitt.id == "Würzburg, 2014") & (df_ausschnitt.Time >= '2014-05-09')].index, inplace=True)

df_ausschnitt.drop(df_ausschnitt[(df_ausschnitt.id == "Würzburg, 2012") & (df_ausschnitt.Time >= '2012-05-17')].index, inplace=True)

# Schwartau

df_ausschnitt.drop(df_ausschnitt[(df_ausschnitt.id == "Schwartau, 2015") & (df_ausschnitt.Time >= '2015-05-12')].index, inplace=True)

df_ausschnitt.drop(df_ausschnitt[(df_ausschnitt.id == "Schwartau, 2019") & (df_ausschnitt.Time >= '2019-05-13')].index, inplace=True)
ax = plt.figure(figsize=(6,3), dpi=200).add_subplot(111)

piv = pd.pivot_table(df_ausschnitt[['Day', 'id', 's_Gewicht']], index=['Day'],columns=['id'])



piv['2019-03-01':'2019-07-01'].plot(ax=ax, title= "Development of weight until the swarm events")

ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1))

ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

ax.tick_params(which='minor',labelsize=5, pad=1)

ax.legend(loc='upper left', prop={'size': 6}, labels=piv.columns.levels[1])
# create overall dataframe

df_gesamt = df_ausschnitt



# read in and add number of outgoing bees for Schwartau

daten = pd.read_csv('../input/data-for-swarm-detector/ausflge 1519.csv', sep = ';');

time_arr = pd.to_datetime(daten.Time.str[0:18])

ts = pd.Series(data= np.array(daten.Value), name="s_Ausflüge", index=pd.DatetimeIndex(time_arr), dtype="float")

ts = ts.resample("H").mean()

df_temp = ts.to_frame()

df_temp['Ort'] = "Schwartau"



# read in and add number of outgoing bees for Würzburg

daten = pd.read_csv('../input/data-for-swarm-detector/Ausflge 12141519.csv', sep = ';');

time_arr = pd.to_datetime(daten.Time.str[0:18])

ts = pd.Series(data= np.array(daten.Value), name="s_Ausflüge", index=pd.DatetimeIndex(time_arr), dtype="float")

ts = ts.resample("H").mean()

df_wrz = ts.to_frame()

df_wrz['Ort'] = "Würzburg"

df_temp = df_temp.append(df_wrz)



# create identifier and drop 'Ort' attribute

df_temp['id'] = df_temp['Ort']  + ", " +  np.datetime_as_string(df_temp.index.values, unit='Y')

df_temp.drop(columns=["Ort"], inplace=True)

df_temp.reset_index(inplace=True)

df_gesamt = pd.merge(df_gesamt, df_temp, how = 'left', on = ['Time', 'id'])
# read in and add number of incoming bees for Schwartau

daten = pd.read_csv('../input/data-for-swarm-detector/einflge 1519.csv', sep = ';');

time_arr = pd.to_datetime(daten.Time.str[0:18])

ts = pd.Series(data= np.array(daten.Value), name="s_Einflüge", index=pd.DatetimeIndex(time_arr), dtype="float")

ts = ts.resample("H").mean()

df_temp = ts.to_frame()

df_temp['Ort'] = "Schwartau"



# read in and add number of incoming bees for Würzburg

daten = pd.read_csv('../input/data-for-swarm-detector/Einflge 12141519.csv', sep = ';');

time_arr = pd.to_datetime(daten.Time.str[0:18])

ts = pd.Series(data= np.array(daten.Value), name="s_Einflüge", index=pd.DatetimeIndex(time_arr), dtype="float")

ts = ts.resample("H").mean()

df_wrz = ts.to_frame()

df_wrz['Ort'] = "Würzburg"

df_temp = df_temp.append(df_wrz)



# create identifier and drop 'Ort' attribute

df_temp['id'] = df_temp['Ort']  + ", " +  np.datetime_as_string(df_temp.index.values, unit='Y')

df_temp.drop(columns=["Ort"], inplace=True)

df_temp.reset_index(inplace=True)

df_gesamt = pd.merge(df_gesamt, df_temp, how = 'left', on = ['Time', 'id'])
# read in and add humidity in the hives for Schwartau

daten = pd.read_csv('../input/data-for-swarm-detector/nestfeuchte 19.csv', sep = ';');

time_arr = pd.to_datetime(daten.Time.str[0:18])

ts = pd.Series(data= np.array(daten.Value), name="s_Nestfeuchte", index=pd.DatetimeIndex(time_arr), dtype="float")



# manual data cleaning (values have to be between 0% and 100%)

ts[ts  >= 100] = np.NaN

ts[ts <= 0] = np.NaN

# resampling to hourly values

ts = ts.resample("H").mean()

df_temp = ts.to_frame()

df_temp['Ort'] = "Schwartau"



# read in and add humidity in the hives for Würzburg

daten = pd.read_csv('../input/data-for-swarm-detector/nestfeuchte 12141519.csv', sep = ';');

time_arr = pd.to_datetime(daten.Time.str[0:18])

ts = pd.Series(data= np.array(daten.Value), name="s_Nestfeuchte", index=pd.DatetimeIndex(time_arr), dtype="float")



# manual data cleaning (values have to be between 0% and 100%)

ts[ts  >= 100] = np.NaN

ts[ts <= 0] = np.NaN

# resampling to hourly values

ts = ts.resample("H").mean()

df_wrz = ts.to_frame()

df_wrz['Ort'] = "Würzburg"

df_temp = df_temp.append(df_wrz)



# create identifier and drop 'Ort' attribute

df_temp['id'] = df_temp['Ort']  + ", " +  np.datetime_as_string(df_temp.index.values, unit='Y')

df_temp.drop(columns=["Ort"], inplace=True)

df_temp.reset_index(inplace=True)

df_gesamt = pd.merge(df_gesamt, df_temp, how = 'left', on = ['Time', 'id'])
# read in and add temparature for Schwartau

daten = pd.read_csv('../input/data-for-swarm-detector/temperatur 1519.csv', sep = ';');

time_arr = pd.to_datetime(daten.Time.str[0:18])

ts = pd.Series(data= np.array(daten.Value), name="s_Temperatur", index=pd.DatetimeIndex(time_arr), dtype="float")

ts = ts.resample("H").mean()

df_temp = ts.to_frame()

df_temp['Ort'] = "Schwartau"



# read in and add temparature for Würzburg

daten = pd.read_csv('../input/data-for-swarm-detector/temperatur v3 12141519.csv', sep = ';');

time_arr = pd.to_datetime(daten.Time.str[0:18])

ts = pd.Series(data= np.array(daten.Value), name="s_Temperatur", index=pd.DatetimeIndex(time_arr), dtype="float")

ts = ts.resample("H").mean()

df_wrz = ts.to_frame()

df_wrz['Ort'] = "Würzburg"

df_temp = df_temp.append(df_wrz)



# create identifier and drop 'Ort' attribute

df_temp['id'] = df_temp['Ort']  + ", " +  np.datetime_as_string(df_temp.index.values, unit='Y')

df_temp.drop(columns=["Ort"], inplace=True)

df_temp.reset_index(inplace=True)

df_gesamt = pd.merge(df_gesamt, df_temp, how = 'left', on = ['Time', 'id'])



ax = plt.figure(figsize=(6,3), dpi=200).add_subplot(111)

ts['2012-03-01':'2019-07-01'].plot(ax=ax, title="Temperature")
# copy and describe resulting dataframe

df = df_gesamt.copy()

df.describe()
# manual correction of measurment errors (use last value)    

df.loc[(df.Time == "2014-03-31 05:00:00") & (df.id=="Würzburg, 2014"), 's_Nestfeuchte'] = df.loc[(df.Time == pd.to_datetime("2014-03-31 05:00:00") - timedelta(hours=1)) & (df.id=="Würzburg, 2014"), 's_Nestfeuchte'].values[0]

df.loc[(df.Time == "2014-04-01 02:00:00") & (df.id=="Würzburg, 2014"), 's_Nestfeuchte'] = df.loc[(df.Time == pd.to_datetime("2014-04-01 02:00:00") - timedelta(hours=1)) & (df.id=="Würzburg, 2014"), 's_Nestfeuchte'].values[0]

df.loc[(df.Time == "2014-04-01 07:00:00") & (df.id=="Würzburg, 2014"), 's_Nestfeuchte'] = df.loc[(df.Time == pd.to_datetime("2014-04-01 07:00:00") - timedelta(hours=1)) & (df.id=="Würzburg, 2014"), 's_Nestfeuchte'].values[0]

df.loc[(df.Time == "2014-04-01 08:00:00") & (df.id=="Würzburg, 2014"), 's_Nestfeuchte'] = df.loc[(df.Time == pd.to_datetime("2014-04-01 08:00:00") - timedelta(hours=1)) & (df.id=="Würzburg, 2014"), 's_Nestfeuchte'].values[0]

df.loc[(df.Time == "2014-04-01 09:00:00") & (df.id=="Würzburg, 2014"), 's_Nestfeuchte'] = df.loc[(df.Time == pd.to_datetime("2014-04-01 09:00:00") - timedelta(hours=1)) & (df.id=="Würzburg, 2014"), 's_Nestfeuchte'].values[0]

df.loc[(df.Time == "2014-04-01 10:00:00") & (df.id=="Würzburg, 2014"), 's_Nestfeuchte'] = df.loc[(df.Time == pd.to_datetime("2014-04-01 10:00:00") - timedelta(hours=1)) & (df.id=="Würzburg, 2014"), 's_Nestfeuchte'].values[0]

df.loc[(df.Time == "2014-04-01 11:00:00") & (df.id=="Würzburg, 2014"), 's_Nestfeuchte'] = df.loc[(df.Time == pd.to_datetime("2014-04-01 11:00:00") - timedelta(hours=1)) & (df.id=="Würzburg, 2014"), 's_Nestfeuchte'].values[0]

df.loc[(df.Time == "2014-04-01 12:00:00") & (df.id=="Würzburg, 2014"), 's_Nestfeuchte'] = df.loc[(df.Time == pd.to_datetime("2014-04-01 12:00:00") - timedelta(hours=1)) & (df.id=="Würzburg, 2014"), 's_Nestfeuchte'].values[0]

df.loc[(df.Time == "2014-04-01 13:00:00") & (df.id=="Würzburg, 2014"), 's_Nestfeuchte'] = df.loc[(df.Time == pd.to_datetime("2014-04-01 13:00:00") - timedelta(hours=1)) & (df.id=="Würzburg, 2014"), 's_Nestfeuchte'].values[0]

df.loc[(df.Time == "2019-04-09 11:00:00") & (df.id=="Würzburg, 2019"), 's_Nestfeuchte'] = df.loc[(df.Time == pd.to_datetime("2019-04-09 11:00:00") - timedelta(hours=1)) & (df.id=="Würzburg, 2019"), 's_Nestfeuchte'].values[0]

df.loc[(df.Time == "2019-04-09 12:00:00") & (df.id=="Würzburg, 2019"), 's_Nestfeuchte'] = df.loc[(df.Time == pd.to_datetime("2019-04-09 12:00:00") - timedelta(hours=1)) & (df.id=="Würzburg, 2019"), 's_Nestfeuchte'].values[0]

df.loc[(df.Time == "2019-04-09 13:00:00") & (df.id=="Würzburg, 2019"), 's_Nestfeuchte'] = df.loc[(df.Time == pd.to_datetime("2019-04-09 13:00:00") - timedelta(hours=1)) & (df.id=="Würzburg, 2019"), 's_Nestfeuchte'].values[0]

df.loc[(df.Time == "2019-04-09 14:00:00") & (df.id=="Würzburg, 2019"), 's_Nestfeuchte'] = df.loc[(df.Time == pd.to_datetime("2019-04-09 14:00:00") - timedelta(hours=1)) & (df.id=="Würzburg, 2019"), 's_Nestfeuchte'].values[0]

   

# replace spaces durch to time change (summer time!)

for col in ['s_Gewicht', 's_Ausflüge', 's_Einflüge', 's_Nestfeuchte', 's_Temperatur']:

    df.loc[(df.Time == "2012-03-25 02:00:00") & (df.id=="Würzburg, 2012"), col] = df.loc[(df.Time == "2012-03-25 01:00:00") & (df.id=="Würzburg, 2012"), col].values[0]

    df.loc[(df.Time == "2014-03-30 02:00:00") & (df.id=="Würzburg, 2014"), col] = df.loc[(df.Time == "2014-03-30 01:00:00") & (df.id=="Würzburg, 2014"), col].values[0]

    df.loc[(df.Time == "2015-03-29 02:00:00") & (df.id=="Würzburg, 2015"), col] = df.loc[(df.Time == "2015-03-29 01:00:00") & (df.id=="Würzburg, 2015"), col].values[0]

    df.loc[(df.Time == "2015-03-29 02:00:00") & (df.id=="Schwartau, 2015"), col] = df.loc[(df.Time == "2015-03-29 01:00:00") & (df.id=="Schwartau, 2015"), col].values[0]

    df.loc[(df.Time == "2019-03-31 02:00:00") & (df.id=="Würzburg, 2019"), col] = df.loc[(df.Time == "2019-03-31 01:00:00") & (df.id=="Würzburg, 2019"), col].values[0]

    df.loc[(df.Time == "2019-03-31 02:00:00") & (df.id=="Schwartau, 2019"), col] = df.loc[(df.Time == "2019-03-31 01:00:00") & (df.id=="Schwartau, 2019"), col].values[0]



for i in range(0,25):

    for col in ['s_Gewicht', 's_Ausflüge', 's_Einflüge', 's_Nestfeuchte', 's_Temperatur']:

        df.loc[(df.Time == pd.to_datetime("2019-03-09 00:00:00") + timedelta(hours=i)) & (df.id=="Würzburg, 2019"), col] = df.loc[(df.Time == pd.to_datetime("2019-03-09 00:00:00") + timedelta(hours=i) - timedelta(days=1)) & (df.id=="Würzburg, 2019"), col].values[0]

        df.loc[(df.Time == pd.to_datetime("2019-03-09 00:00:00") + timedelta(hours=i)) & (df.id=="Schwartau, 2019"), col] = df.loc[(df.Time == pd.to_datetime("2019-03-09 00:00:00") + timedelta(hours=i) - timedelta(days=1)) & (df.id=="Schwartau, 2019"), col].values[0]

        

# clear selected columns (Time, Day, encode identifiers)

df.drop(columns=["Time", "Day"], inplace=True)

df.replace(["Würzburg, 2012", "Würzburg, 2014", "Würzburg, 2015", "Würzburg, 2019", "Schwartau, 2015", "Schwartau, 2019"], [1, 2, 3, 4, 5, 6], inplace=True)

snams = ["s_Gewicht", "s_Ausflüge", "s_Einflüge", "s_Nestfeuchte", "s_Temperatur"]
## window size for rolling means / standard deviation

ws =  7



MAX = (df.groupby(df.id, as_index = False).cycle.max(skipna = True).rename(index = str, columns = {"cycle" : "MAX"}))

df = pd.merge(df, MAX, how = 'inner', on = 'id')

df['RUL'] = df.MAX - df.cycle

df.drop('MAX', 1, inplace = True)
# Calculate rolling means and standard deviations, groubed by time series ID

rmeans = df.groupby('id')[snams].rolling(window = ws, min_periods = 1).mean()

rsds = df.groupby('id')[snams].rolling(window = ws, min_periods = 1).std().fillna(0)



anams = [i.replace('s_','a_') for i in snams]

rmeans.columns = anams



sdnams = [i.replace('s_','sd_') for i in snams]

rsds.columns = sdnams



# add to dataframe

df = pd.concat([df, 

                   rmeans.reset_index(drop = True), 

                   rsds.reset_index(drop = True)], 

                  axis = 1)

df.head()
df = df.loc[:,df.nunique() != 1]
df_viz = df.copy()
dontscale = ['RUL', 'id']

scale = df.columns.drop(dontscale)

scaler = preprocessing.StandardScaler().fit(df[scale])

df = pd.concat([pd.DataFrame(scaler.transform(df[scale]), columns = scale),

                   df.RUL,

                   df.id], 

                  axis = 1)

df.head()
#1: Würzburg, 2012

#2: Würzburg, 2014

#3: Würzburg, 2015

#4: Würzburg, 2019

#5: Schwartau, 2015

#6: Schwartau, 2019



trainIDs = [1,3,5,6]

testIDs = [2, 4]

train = df[df.id.isin(trainIDs)]

test = df[df.id.isin(testIDs)]
cols = ['id', 'cycle','s_Ausflüge', 's_Einflüge', 's_Nestfeuchte', 's_Temperatur', 's_Gewicht']

data = (df[cols]

        .melt(id_vars = ['id', 'cycle'], var_name = 'sensor')

        .query('id in [1,3]')

        )

grid = (sns.FacetGrid(data, 

                      col = 'sensor', col_wrap = 4, 

                      hue = 'id', 

                      sharex = True, sharey = False,

                      legend_out = True,

                      margin_titles = True

                      )

         .map(plt.plot, 'cycle', 'value')

         .add_legend()

        )

plt.subplots_adjust(top=0.9)

grid.fig.suptitle('Sensor Data für zwei Bienenstöcke', fontsize = 24)
from sklearn.metrics import confusion_matrix, accuracy_score
X_train = train.drop(['RUL', 'id'], axis = 1)
X_test = test.drop(['RUL', 'id'], axis = 1)



# create copies with NaN values (humidity in Schwartau is null)

X_train_complete = X_train.copy()

X_test_complete = X_test.copy()



# Important remark: at this point we do not use the humidity

X_train = X_train.drop(['s_Nestfeuchte', 'a_Nestfeuchte', 'sd_Nestfeuchte'], axis = 1)

X_test = X_test.drop(['s_Nestfeuchte', 'a_Nestfeuchte', 'sd_Nestfeuchte'], axis = 1)
from sklearn.metrics import mean_squared_error

from math import sqrt

Y_train = train['RUL']

Y_test = test['RUL']
from sklearn.linear_model import LinearRegression

model_r_linearModel = LinearRegression(fit_intercept=True)

model_r_linearModel.fit(X_train, Y_train)

pred = model_r_linearModel.predict(X_test)

stats_r_linearModel = sqrt(mean_squared_error(pred, Y_test))

stats_r_linearModel
from scipy import stats

import statsmodels.api as sm

model_r_poissonReg = sm.genmod.GLM(Y_train, X_train, family=sm.families.Poisson()).fit_regularized()

pred = model_r_poissonReg.predict(X_test)

stats_r_poissonReg = sqrt(mean_squared_error(pred, Y_test))

stats_r_poissonReg
from sklearn.ensemble import RandomForestRegressor

model_r_randomForest = RandomForestRegressor()

model_r_randomForest.fit(X_train, Y_train)

pred = model_r_randomForest.predict(X_test)

stats_r_randomForest = sqrt(mean_squared_error(pred, Y_test))

stats_r_randomForest
from sklearn.svm import SVR

model_r_svm = SVR()

model_r_svm.fit(X_train, Y_train)

pred = model_r_svm.predict(X_test)

stats_r_svm = sqrt(mean_squared_error(pred, Y_test))

stats_r_svm
from sklearn.neural_network import MLPRegressor

model_r_nnet = MLPRegressor(hidden_layer_sizes=100, max_iter=300)

model_r_nnet.fit(X_train, Y_train)

pred = model_r_nnet.predict(X_test)

stats_r_nnet = sqrt(mean_squared_error(pred, Y_test))

stats_r_nnet
res = pd.Series({'Linear Regression' : stats_r_linearModel,

       'Random Forest' : stats_r_randomForest,

       'Support Vector Machines' : stats_r_svm,

       'Neural Network' : stats_r_nnet,

       'Regularized Poisson Regression' : stats_r_poissonReg})

res.sort_values(inplace=True)

print(res)

print("\nBest performance by " + res.keys()[0] + ".")