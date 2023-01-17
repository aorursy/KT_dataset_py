#import packages

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

sns.set_style('dark')



# import statsmodels.api as sm

# from shutil import copyfile



# # copy our file into the working directory (make sure it has .py suffix)

# copyfile(src = "../input/demandanalysishelperfunctions/features_preprocessing.py", 

#          dst = "../working/features_preprocessing.py")



# #import features_preprocessing

# import features_preprocessing





#import the dataset

data = pd.read_csv('/kaggle/input/energy-consumption-generation-prices-and-weather/energy_dataset.csv',

                   index_col=0,

                   parse_dates=[0])



#set the index to datetime

data.index = pd.to_datetime(data.index, utc=True)

#analyze 2015 to 2018 data inclusive

data = data['2015':'2018']

print(data.columns)
load = data['total load actual']

forecast = data['total load forecast']

load.head(), forecast.head()
df_demand = pd.concat([load, forecast], axis=1)

df_demand.columns = ['load', 'forecast']

df_demand.describe()
fig, ax = plt.subplots(figsize=(10, 6))

sns.distplot(df_demand['load'].dropna(), ax=ax, kde=False).set_title('load', fontsize=16)

sns.distplot(df_demand['forecast'].dropna(), ax=ax, kde=False).set_title('forecast', fontsize=16)

plt.xlabel('MW Power', fontsize=15)

plt.legend(['load', 'forecast'])

ax.set_yticks(np.linspace(0, 1500, 9))

plt.title('Distribution of energy demanded and TSO 1 day forecasts')

plt.show()
#group data by year

groups = df_demand['load'].groupby(pd.Grouper(freq='A'))



#set figure and axis

fig, axs = plt.subplots(len(groups), 1, figsize=(15,15))





for ax, (name, group) in zip(axs, groups):

    

    #plot the data

    ax.plot(pd.Series(group.values))



    ax.set_xlabel('Hour of Year')

    ax.set_ylabel('Total Load')

    ax.set_title(name.year)

    plt.subplots_adjust(hspace=0.5)
group_hours = df_demand['load'].groupby(pd.Grouper(freq='D', how='mean'))



fig, axs = plt.subplots(1,1, figsize=(8,7))



year_demands = pd.DataFrame()

    

for name, group in group_hours:

    year_demands[name.year] = pd.Series(group.values)

    

year_demands.plot(ax=axs)

axs.set_xlabel('Hour of the day')

axs.set_ylabel('Energy Demanded MWh')

axs.set_title('Mean yearly energy demand by hour of the day ');
#group data by year

groups = df_demand['load'].groupby(pd.Grouper(freq='A'))



#set figure and axis

fig, axs = plt.subplots(1, 1, figsize=(8,5))





for name, group in groups:

    

    sorted_load_count = pd.Series(group.values).sort_values(ascending=False).reset_index()

    sorted_load_count.drop('index', axis=1, inplace=True)

    #plot the data

    axs.plot(sorted_load_count)

    axs.set_xlabel('Cumulative Hours')

    axs.set_ylabel('Total Load')

    axs.set_title('Load Duration Curve 2015-2018')

axs.legend(['2015', '2016', '2017', '2018'])
fig, axs = plt.subplots(1, 2, figsize=(15,5))



for ax, col in zip(axs, df_demand.columns):

    groups = df_demand[col].groupby(pd.Grouper(freq='A'))



    df = pd.DataFrame()



    for name, group in groups:

        df[name.year] = pd.Series(group.values)



    df.boxplot(ax=ax)

    ax.set_xlabel('Year')

    ax.set_ylabel('Total Load')

    ax.set_title(col)
fig, axs = plt.subplots(1, 1, figsize=(8,5))



df = pd.DataFrame()



groups = df_demand['load'].groupby(pd.Grouper(freq='A'))



for name, group in groups:

    df[name.year] = pd.Series(group.values)



df.boxplot(ax=axs)

axs.set_xlabel('Year')

axs.set_ylabel('Total Load')

axs.set_title('load')
#group data by year

groups = df_demand['load'].groupby(pd.Grouper(freq='M', how='mean'))



fig, axs = plt.subplots(12, 1, figsize=(20,30))



months=pd.DataFrame()



for ax, (name, group) in zip(axs, groups):

    months[name.month] = pd.Series(group.values)

    ax.set_title('Month: ' + str(name.month))

    

months.plot(ax=axs, subplots=True, legend=False)

plt.subplots_adjust(hspace=0.5)

plt.show()

#set figure and axis

fig, axs = plt.subplots(len(groups), 1, figsize=(15,100))





for ax, (name, group) in zip(axs, groups):

    

    #plot the data

    ax.plot(pd.Series(group.values))



    ax.set_xlabel('Hour of Year')

    ax.set_ylabel('Total Load')

    ax.set_title("Year: " + str(name.year) + " Month: " + str(name.month))

    plt.subplots_adjust(hspace=0.8)
group_hours = df_demand['load'].groupby(pd.Grouper(freq='D', how='mean'))



fig, axs = plt.subplots(1,1, figsize=(8,7))



df = pd.DataFrame()

    

months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']    



for name, group in group_hours:

    df[name.month] = pd.Series(group.values)



    

colors = ['gray', 'lightcoral', 'firebrick', 'chocolate', 'darkorange', 'gold', 'olive', 'palegreen', 'teal', 'skyblue', 'blueviolet', 'purple']

df.columns = months

df.plot(ax=axs, color=colors)

axs.set_xlabel('Hour of the day')

axs.set_ylabel('Energy Demanded MWh')

axs.set_title('Mean monthly energy demand by hour of the day');
#group data by year

groups = df_demand['load'].groupby(pd.Grouper(freq='M', how='mean'))



#set figure and axis

fig, axs = plt.subplots(1, 1, figsize=(12,8))



load_curve = pd.DataFrame()



for name, group in groups:    

    load_curve[name.month] = pd.Series(sorted(group.values, reverse=True))



#plot the data

axs.plot(load_curve)

axs.set_xlabel('Cumulative Hours')

axs.set_ylabel('Total Load')

axs.set_title('Average Load Duration Curve Per Month')

axs.legend(['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
fig, axs = plt.subplots(1, 1,  figsize=(8,5))



groups = df_demand['load'].groupby(pd.Grouper(freq='M'))



df = pd.DataFrame()



for name, group in groups:

    df[name.month] = pd.Series(group.values)



df.boxplot(ax=axs)

axs.set_xlabel('Month Year')

axs.set_ylabel('Energy Demanded MWh')

axs.set_title('Box plot month of year 2015-2018')

plt.subplots_adjust(hspace=0.5)





plt.show()
group_hours = df_demand['load'].groupby(pd.Grouper(freq='D', how='mean'))



day_names = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']



fig, axs = plt.subplots(1,1, figsize=(8,7))



df = pd.DataFrame()

    

for name, group in group_hours:

    df[name.dayofweek] = pd.Series(group.values)

df = df.reindex(sorted(df.columns), axis=1)

df.columns = day_names

df.plot(ax=axs)

axs.set_xlabel('Hour of the day')

axs.set_ylabel('Energy Demanded MWh')

axs.set_title('Mean day of the week energy demand profile by hour of the day');
fig, axs = plt.subplots(1, 1, figsize=(15,5))



groups = df_demand['load'].groupby(pd.Grouper(freq='D'))



df = pd.DataFrame()



for name, group in groups:

    df[name.dayofweek] = pd.Series(group.values)



df = df.reindex(sorted(df.columns), axis=1)

df.columns = ['mon', 'tue', 'wed', 'thu', 'fri','sat', 'sun']

print(df.head())

    

df.boxplot(ax=axs)

axs.set_xlabel('Day of the week')

axs.set_ylabel('Energy Demanded MWh')

axs.set_title('Box plot mean energy demand day of the week')

plt.subplots_adjust(hspace=0.5)



plt.show()