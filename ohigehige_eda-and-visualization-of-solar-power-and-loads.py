import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns
df2015 = pd.read_csv('../input/solar-generation-and-demand-italy-20152016/TimeSeries_TotalSolarGen_and_Load_IT_2015.csv')

df2016 = pd.read_csv('../input/solar-generation-and-demand-italy-20152016/TimeSeries_TotalSolarGen_and_Load_IT_2016.csv')
df2015.head()
print(df2015.dtypes) # confirm the data type of the datasets
df2015.isnull().sum() # confirm the missing data
df2016.tail()
print(df2016.dtypes)
df2016.isnull().sum()
df2015.dropna(subset=['IT_load_new'], axis=0, inplace=True)         # deleting missing data

df2015["utc_timestamp"] = pd.to_datetime(df2015["utc_timestamp"])   # converting the timestamp format

df2015.columns =  ['Timestamp', 'Load', 'Solar_Gen']                # renaming the columns

df2015['Month'] = df2015['Timestamp'].dt.month                      # adding 'Month' columns

df2015['Weekday'] =df2015['Timestamp'].dt.weekday                   #adding 'Weekday' columns

df2015['Solar_Rate'] = df2015['Solar_Gen']/df2015['Load']*100       # adding 'Solar_Rate' columns

    # which indicates how much solar-genarated power contributes for the tolal electricity load/demand.

df2015 = df2015[['Timestamp', 'Month', 'Weekday', 'Load', 'Solar_Gen', 'Solar_Rate']] 



df2015.head()
df2015.isnull().sum()
print(df2015.dtypes)
df2016.dropna(subset=['IT_load_new'], axis=0, inplace=True)         

df2016["utc_timestamp"] = pd.to_datetime(df2016["utc_timestamp"])   

df2016.columns =  ['Timestamp', 'Load', 'Solar_Gen']                

df2016['Month'] = df2016['Timestamp'].dt.month                      

df2016['Weekday'] =df2016['Timestamp'].dt.weekday                   

df2016['Solar_Rate'] = df2016['Solar_Gen']/df2016['Load']*100       

df2016 = df2016[['Timestamp', 'Month', 'Weekday', 'Load', 'Solar_Gen', 'Solar_Rate']] 
# Max./Ave. Electricity Load, Solar-generated power in 2015



ave_load_2015 = df2015.groupby(['Month'])['Load'].mean()

ave_solar_gen_2015 = df2015.groupby(['Month'])['Solar_Gen'].mean()

monthly_aveload_2015 = (ave_load_2015.values)

monthly_aveload_2015 = monthly_aveload_2015.reshape(-1, 1)

monthly_avegen_2015 = (ave_solar_gen_2015.values)

monthly_avegen_2015 = monthly_avegen_2015.reshape(-1, 1)



max_load_2015 = df2015.groupby(['Month'])['Load'].max()

max_solar_gen_2015 = df2015.groupby(['Month'])['Solar_Gen'].max()

monthly_maxload_2015 = (max_load_2015.values)

monthly_maxload_2015 = monthly_maxload_2015.reshape(-1, 1)

monthly_maxgen_2015 = (max_solar_gen_2015.values)

monthly_maxgen_2015 = monthly_maxgen_2015.reshape(-1, 1)



# Max./Ave. Electricity Load, Solar-generated power in 2016



ave_load_2016 = df2016.groupby(['Month'])['Load'].mean()

ave_solar_gen_2016 = df2016.groupby(['Month'])['Solar_Gen'].mean()

monthly_aveload_2016 = (ave_load_2016.values)

monthly_aveload_2016 = monthly_aveload_2016.reshape(-1, 1)

monthly_avegen_2016 = (ave_solar_gen_2016.values)

monthly_avegen_2016 = monthly_avegen_2016.reshape(-1, 1)



max_load_2016 = df2016.groupby(['Month'])['Load'].max()

max_solar_gen_2016 = df2016.groupby(['Month'])['Solar_Gen'].max()

monthly_maxload_2016 = (max_load_2016.values)

monthly_maxload_2016 = monthly_maxload_2016.reshape(-1, 1)

monthly_maxgen_2016 = (max_solar_gen_2016.values)

monthly_maxgen_2016 = monthly_maxgen_2016.reshape(-1, 1)



# Max./Ave. Electricity Load vs Solar-generated power ratio



ave_rate_2015 = df2015.groupby(['Month'])['Solar_Rate'].mean()

ave_rate_2016 = df2016.groupby(['Month'])['Solar_Rate'].mean()

monthly_averate_2015 = (ave_rate_2015.values)

monthly_averate_2015 = monthly_averate_2015.reshape(-1, 1)

monthly_averate_2016 = (ave_rate_2016.values)

monthly_averate_2016 = monthly_averate_2016.reshape(-1, 1)



max_rate_2015 = df2015.groupby(['Month'])['Solar_Rate'].max()

max_rate_2016 = df2016.groupby(['Month'])['Solar_Rate'].max()

monthly_maxrate_2015 = (max_rate_2015.values)

monthly_maxrate_2015 = monthly_maxrate_2015.reshape(-1, 1)

monthly_maxrate_2016 = (max_rate_2016.values)

monthly_maxrate_2016 = monthly_maxrate_2016.reshape(-1, 1)



# Draw the charts



sns.set()

sns.set_style=("darkgrid")

fig = plt.figure()

plt.figure(figsize=(14,12))



month = np.arange(1,13).reshape(-1,1)



plt.subplot(221)

plt.xlabel('Month', fontsize=16)

plt.ylabel('Monthly Electricity Load (MW/hr)', fontsize=16)

plt.xlim(0.5,13)

plt.ylim(21000,57000)

plt.plot(month, monthly_aveload_2015, label='2015 Average', color="mediumseagreen", linewidth='3')

plt.plot(month, monthly_aveload_2016, label='2016 Average', color="forestgreen", linestyle="dashdot", linewidth='3')

plt.plot(month, monthly_maxload_2015, label='2015 Max', color="lightsalmon", linewidth='3')

plt.plot(month, monthly_maxload_2016, label='2016 Max', color="indianred", linestyle="dashdot", linewidth='3')

plt.legend(loc='best')



plt.subplot(223)

plt.xlabel('Month', fontsize=16)

plt.ylabel('Monthly Solar Power Generation (MW/hr)', fontsize=16)

plt.xlim(0.5,13)

plt.ylim(0,17000)

plt.plot(month, monthly_avegen_2015, label='2015 Average', color="limegreen", linewidth='3')

plt.plot(month, monthly_avegen_2016, label='2016 Average', color="green", linestyle="dashdot", linewidth='3')

plt.plot(month, monthly_maxgen_2015, label='2015 Max', color="coral", linewidth='3')

plt.plot(month, monthly_maxgen_2016, label='2016 Max', color="goldenrod", linestyle="dashdot", linewidth='3')

plt.legend(loc='best')



plt.subplot(224)

plt.xlabel('Month', fontsize=16)

plt.ylabel('Monthly Load vs\n Solar-generated Electricity Rate (%)', fontsize=16)

plt.xlim(0.5,13)

plt.ylim(-5, 70)

plt.plot(month, monthly_averate_2015, label='2015 Average', color="steelblue", linewidth='3')

plt.plot(month, monthly_averate_2016, label='2016 Average', color="cornflowerblue", linestyle="dashdot", linewidth='3')

plt.plot(month, monthly_maxrate_2015, label='2015 Max', color="mediumslateblue", linewidth='3')

plt.plot(month, monthly_maxrate_2016, label='2016 Max', color="darkviolet", linestyle="dashdot", linewidth='3')

plt.legend(loc='upper left')



plt.show()
# classify days in weekdays and weekends



df15jul = df2015[df2015["Month"] == 7]

df15julwd = df15jul[df15jul["Weekday"] < 5]

df15julwe = df15jul[df15jul["Weekday"] >= 5]



df16may = df2016[df2016["Month"] == 5]

df16maywd = df16may[df16may["Weekday"] < 5]

df16maywe = df16may[df16may["Weekday"] >= 5]



sns.set()

sns.set_style=("darkgrid")

fig = plt.figure()

plt.figure(figsize=(14,7))



plt.subplot(211)

plt.ylim(15000,60000)

plt.ylabel("Electricity Load (MW/hr)", fontsize=14)

plt.scatter(df15julwd["Timestamp"], df15julwd["Load"], color="mediumseagreen", marker=",", label="2015Jul Load Weekday")

plt.scatter(df15julwe["Timestamp"], df15julwe["Load"], color="coral", label="2015Jul Load Weekend")

plt.legend(loc="best")



plt.subplot(212)

plt.ylim(15000,60000)

plt.ylabel("Electricity Load (MW/hr)", fontsize=14)

plt.scatter(df16maywd["Timestamp"], df16maywd["Load"], color="forestgreen", marker=",", label="2016May Load Weekday")

plt.scatter(df16maywe["Timestamp"], df16maywe["Load"], color="indianred", label="2016May Load Weekend")

plt.legend(loc="best")



plt.show()
# Visualize the solar power generation on daily-basis



df15dec = df2015[df2015["Month"] == 12]

df15jul = df2015[df2015["Month"] == 7]

df16dec = df2016[df2016["Month"] == 12]

df16jul = df2016[df2016["Month"] == 7]



sns.set()

sns.set_style=("darkgrid")

fig = plt.figure()

plt.figure(figsize=(14,6))



plt.subplot(211)

plt.plot(df15jul["Timestamp"], df15jul["Solar_Gen"], linewidth="2", color="coral", label="2015Jul Solar Generation")

plt.ylim(0,13000)

plt.ylabel("Solar Power\n Generation (MW/hr)", fontsize=14)

plt.legend(loc="best")



plt.subplot(212)

plt.ylim(0,13000)

plt.ylabel("Solar Power\n Generation (MW/hr)", fontsize=14)

plt.plot(df16dec["Timestamp"], df16dec["Solar_Gen"], linewidth="2", color="green", label="2016Dec Solar Generation")

plt.legend(loc="best")



plt.show()