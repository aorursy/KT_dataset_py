import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#load datasets
p1_whe = pd.read_csv('/kaggle/input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv')
p1_gen = pd.read_csv('/kaggle/input/solar-power-generation-data/Plant_1_Generation_Data.csv')
p2_whe = pd.read_csv('/kaggle/input/solar-power-generation-data/Plant_2_Weather_Sensor_Data.csv')
p2_gen = pd.read_csv('/kaggle/input/solar-power-generation-data/Plant_2_Generation_Data.csv')
p1_gen.isnull().sum() #null value cheaking in plant 1 generation data
p2_gen.isnull().sum() #null value cheaking in plant 2 generation data
p1_whe.isnull().sum() #null value cheaking in plant 1 wheather sensor data
p2_whe.isnull().sum() #null value cheaking in plant 2 wheather sensor data
p1_gen.info() #plant 1
p2_gen.info() #plant 2
#saperate date and time, store into new column date, time

#plant 1 power genration data 
p1_gen[['date','time']] = p1_gen['DATE_TIME'].apply(lambda x: pd.Series(str(x).split(" ")))

#plant 2 power genration data
p2_gen[['date','time']] = p2_gen['DATE_TIME'].apply(lambda x: pd.Series(str(x).split(" ")))

#plant 1 wheather sensor data
p1_whe[['Date','Time']] = p1_whe['DATE_TIME'].apply(lambda x: pd.Series(str(x).split(' ')))

#plant 2 wheather sensor data
p2_whe[['Date','Time']] = p2_whe['DATE_TIME'].apply(lambda x: pd.Series(str(x).split(' ')))
#drop unnecessary columns like plant id and date_time

#plant 1
p1_gen.drop(['DATE_TIME','PLANT_ID'],axis=1,inplace=True)

#plant 2
p2_gen.drop(['DATE_TIME','PLANT_ID'],axis=1,inplace=True)
#power generation data
#saparate day, month ,hour and minute

#plant 1
p1_gen['day'] = pd.to_datetime(p1_gen.date).dt.day 
p1_gen['month'] = pd.to_datetime(p1_gen.date).dt.month
p1_gen['hour'] = pd.to_datetime(p1_gen.time).dt.hour
p1_gen['minute'] = pd.to_datetime(p1_gen.time).dt.minute

#plant 2
p2_gen['day'] = pd.to_datetime(p2_gen.date).dt.day 
p2_gen['month'] = pd.to_datetime(p2_gen.date).dt.month
p2_gen['hour'] = pd.to_datetime(p2_gen.time).dt.hour
p2_gen['minute'] = pd.to_datetime(p2_gen.time).dt.minute

#plant 1
p1_gen.drop(['date','time'],axis=1,inplace=True)

#plant 2
p2_gen.drop(['date','time'],axis=1,inplace=True)
#wheather sensor data
#extract day, month, hour, minute and drop PLANT_ID, Date, Time, DATE_TIME columns

#plant 1
p1_whe['day'] = pd.to_datetime(p1_whe.Date).dt.day
p1_whe['month'] = pd.to_datetime(p1_whe.Date).dt.month
p1_whe['hour'] = pd.to_datetime(p1_whe.Time).dt.hour
p1_whe['minute'] = pd.to_datetime(p1_whe.Time).dt.minute
p1_whe.drop(['PLANT_ID','Date','Time','DATE_TIME'],axis=1,inplace=True)

#plant 2
p2_whe['day'] = pd.to_datetime(p2_whe.Date).dt.day
p2_whe['month'] = pd.to_datetime(p2_whe.Date).dt.month
p2_whe['hour'] = pd.to_datetime(p2_whe.Time).dt.hour
p2_whe['minute'] = pd.to_datetime(p2_whe.Time).dt.minute
p2_whe.drop(['PLANT_ID','Date','Time','DATE_TIME'],axis=1,inplace=True)
#get last five records

p1_gen.tail() #plant 1
p1_whe.head() #plant 1
p2_gen.tail() #plant 2
p2_whe.head() #plant 2
p1_gen.describe() #plant 1
p1_whe.describe() #plant 1
p2_gen.describe() #plant 2
p2_whe.describe() #plant 2
print('Mean value of daily yield (for plant 1) :',p1_gen.describe().loc['mean','DAILY_YIELD'])

print('Mean value of daily yield (for plant 2) :',p2_gen.describe().loc['mean','DAILY_YIELD'])
#extract all 5th and 6th month data points

#platn 1 
p1_gen = p1_gen.loc[(p1_gen.month == 5) | (p1_gen.month == 6)]

#platn 2 
p2_gen = p2_gen.loc[(p2_gen.month == 5) | (p2_gen.month == 6)]
#plot_by_hour return bar plot of given column w.r.t hour and month
#each bar represent hour wise total power ganeration. 

#plant 1
def plot_by_hour(data, val = None, agg = 'sum'):
    dd = data
    by_hour = dd.groupby(['hour','month'])[val].agg(agg).unstack()
    return by_hour.plot(kind='bar', figsize=(15,4),width=1,grid=True,title='Plant 1'+' '+val,legend=True)

plot_by_hour(p1_gen,'DC_POWER') 
plot_by_hour(p1_gen,'AC_POWER')
plot_by_hour(p1_gen,'DAILY_YIELD') 


#plant 2
def plot_by_hour(data, val = None, agg = 'sum'):
    dd = data
    by_hour = dd.groupby(['hour','month'])[val].agg(agg).unstack()
    return by_hour.plot(kind='bar', figsize=(15,4),width=1,grid=True,title='Plant 2'+' '+val,legend=True)

plot_by_hour(p2_gen,'DC_POWER') 
plot_by_hour(p2_gen,'AC_POWER')
plot_by_hour(p2_gen,'DAILY_YIELD') 

#DC power comparison 
def line_plot(data, col=None, agg = 'sum'):
    dd = data
    hours = dd.groupby(['month',"hour"])[col].agg(agg)
    ax = hours.plot(kind="line", figsize=(15,7),title=col, grid =True, legend=True) 
    ax.legend(["plant 1", "plant 2"]);
    return ("Comparison of DC power generation")

line_plot(p1_gen,'DC_POWER')
line_plot(p2_gen,'DC_POWER')
#AC power comparison 
def line_plot(data, col=None, agg = 'sum'):
    dd = data
    hours = dd.groupby(['month',"hour"])[col].agg(agg)
    ax = hours.plot(kind="line", figsize=(15,7),title=col, grid =True, legend=True) 
    ax.legend(["plant 1", "plant 2"]);
    return "Comparison of AC power generation"

line_plot(p1_gen,'AC_POWER')
line_plot(p2_gen,'AC_POWER')
#Daily yeild comparison between plant 1 and 2
def line_plot(data, col=None, agg = 'sum'):
    dd = data
    hours = dd.groupby(['month',"hour"])[col].agg(agg)
    ax = hours.plot(kind="line", figsize=(15,7),title=col, grid =True, legend=True) 
    ax.legend(["plant 1", "plant 2"]);
    return 'Daily yeild comparison'

line_plot(p1_gen,'DAILY_YIELD')
line_plot(p2_gen,'DAILY_YIELD') 
#hour wise 5th and 6th month comparison of ambient temparature and module temparature. 

#plant 1
def func(data, val = None, agg = 'mean'):
    dd=data
    by_hr = dd.groupby(['hour','month'])[val].agg(agg).unstack()
    return by_hr.plot(kind='bar',figsize=(15,5),width=0.9,grid=True,title='Plant 1'+" "+val)

func(p1_whe,'AMBIENT_TEMPERATURE')
func(p1_whe,'MODULE_TEMPERATURE')
func(p1_whe,'IRRADIATION')

#plant 2
def func(data, val = None, agg = 'mean'):
    dd=data
    by_hr = dd.groupby(['hour','month'])[val].agg(agg).unstack()
    return by_hr.plot(kind='bar',figsize=(15,5),width=0.9,grid=True,title='Plant 2'+" "+val)

func(p2_whe,'AMBIENT_TEMPERATURE')
func(p2_whe,'MODULE_TEMPERATURE')
func(p2_whe,'IRRADIATION')
#irradiation comparison between plant 1 and plant 2 

def line_plot(data, col=None, agg = 'mean'):
    dd = data
    hours = dd.groupby(['month',"hour"])[col].agg(agg)
    ax = hours.plot(kind="line", figsize=(15,7), grid =True, legend=True) 
    ax.legend(['plant 1','plant 2'])
    return 'Irradiation comparison'

line_plot(p1_whe,'IRRADIATION')
line_plot(p2_whe,'IRRADIATION')
#AMBIENT TEMPERATURE comparison between plant 1 and plant 2 

def line_plot(data, col=None, agg = 'mean'):
    dd = data
    hours = dd.groupby(['month',"hour"])[col].agg(agg)
    ax = hours.plot(kind="line", figsize=(15,7), grid =True, legend=True) 
    ax.legend(['plant 1','plant 2'])
    return 'Ambient temperature comparison'

line_plot(p1_whe,'AMBIENT_TEMPERATURE')
line_plot(p2_whe,'AMBIENT_TEMPERATURE')
#MODULE TEMPERATURE comparison between plant 1 and plant 2 

def line_plot(data, col=None, agg = 'mean'):
    dd = data
    hours = dd.groupby(['month',"hour"])[col].agg(agg)
    ax = hours.plot(kind="line", figsize=(15,7), grid =True, legend=True) 
    ax.legend(['plant 1','plant 2'])
    return 'Module temperature comparision'

line_plot(p1_whe,'MODULE_TEMPERATURE')
line_plot(p2_whe,'MODULE_TEMPERATURE')
#Total irradiation per day

#plant 1
irr_per_day_p1 = pd.DataFrame(p1_whe.groupby(['month',"day"])['IRRADIATION'].agg('sum'))
irr_per_day_p1
#plant 1
irr_per_day_p2 = pd.DataFrame(p2_whe.groupby(['month',"day"])['IRRADIATION'].agg('sum'))
irr_per_day_p2
#max ambient and module temperature

print(' Maximum ambient temparature (plant 1): ',max(p1_whe.AMBIENT_TEMPERATURE),'\n','Maximum module temparature (plant 1): ',max(p1_whe.MODULE_TEMPERATURE))
print(' Maximum ambient temparature (plant 2): ',max(p2_whe.AMBIENT_TEMPERATURE),'\n','Maximum module temparature (plant 2): ',max(p2_whe.MODULE_TEMPERATURE))
#number of inverters in each plant

print(' Inverter in plant 1:',len(p1_gen.SOURCE_KEY.unique()),'\n','Inverter in plant 2:',len(p2_gen.SOURCE_KEY.unique()))
#DC/AC ratio of plant 1
p1_gen['DC/AC'] = p1_gen.DC_POWER/p1_gen.AC_POWER
p1_gen.fillna(0,inplace=True)

#DC/AC ratio of plant 2
p2_gen['DC/AC'] = p2_gen.DC_POWER/p2_gen.AC_POWER
p2_gen.fillna(0,inplace=True)
#plant 1
p1_gen.sort_values(by='DC/AC',ascending=False).reset_index(drop=True).head()
#plant 2
p2_gen.sort_values(by='DC/AC',ascending=False).reset_index(drop=True).head()
#Plant 1 inverter rank(DC to AC conversion)
inverter_rank_p1 = pd.DataFrame(p1_gen.groupby('SOURCE_KEY')['DC/AC'].agg('max'))
inverter_rank_p1.sort_values(by='DC/AC',ascending=False)
#plant 2
inverter_rank_p2 = pd.DataFrame(p2_gen.groupby('SOURCE_KEY')['DC/AC'].agg('max'))
inverter_rank_p2.sort_values(by='DC/AC',ascending=False)
#rank the inverter based on DC power generation for plant 1
inv_rank_by_dc_p2 = pd.DataFrame(p1_gen.groupby('SOURCE_KEY')['DC_POWER'].agg('sum'))
inv_rank_by_dc_p2.sort_values(by='DC_POWER',ascending=False)
#rank the inverter based on AC power generation
inv_rank_by_ac_p1 = pd.DataFrame(p1_gen.groupby('SOURCE_KEY')['AC_POWER'].agg('sum'))
inv_rank_by_ac_p1.sort_values(by='AC_POWER',ascending=False)
#rank the inverter based on DC power generation for plant 2
inv_rank_by_dc_p2 = pd.DataFrame(p2_gen.groupby('SOURCE_KEY')['DC_POWER'].agg('sum'))
inv_rank_by_dc_p2.sort_values(by='DC_POWER',ascending=False)
#rank the inverter based on AC power generation
inv_rank_by_ac_p2 = pd.DataFrame(p2_gen.groupby('SOURCE_KEY')['AC_POWER'].agg('sum'))
inv_rank_by_ac_p2.sort_values(by='AC_POWER',ascending=False)