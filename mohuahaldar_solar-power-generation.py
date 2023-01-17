import numpy as np # linear algebra

import pandas as pd # data processing



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
plant1_gen=pd.read_csv('/kaggle/input/solar-power-generation-data/Plant_1_Generation_Data.csv')

plant1_gen.head()
plant2_gen=pd.read_csv('/kaggle/input/solar-power-generation-data/Plant_2_Generation_Data.csv')

plant2_gen.head()
plant1_gen.shape
plant1_gen.info()
plant1_gen.isnull().sum()
plant2_gen.isnull().sum()
def modify_time(df):

    df['DATE_TIME']=pd.to_datetime(df.DATE_TIME)

    df['TIME']=df['DATE_TIME'].dt.time

    df['DAY']=df['DATE_TIME'].dt.day

    df['MONTH']=df['DATE_TIME'].dt.month

    df['YEAR']=df['DATE_TIME'].dt.year

    df['DATE_TIME']=pd.to_datetime(df.DATE_TIME).dt.date

    return df
plant1_gen=modify_time(plant1_gen)

plant1_gen
plant2_gen=modify_time(plant2_gen)

plant2_gen
plt.figure(figsize=(12,10))

plant1_gen.groupby('DATE_TIME')['DAILY_YIELD'].sum().plot(legend=True)

plt.show()
plt.figure(figsize=(12,10))

plant1_gen.groupby('DATE_TIME')['DAILY_YIELD'].mean().plot()

plt.show()
plant1_sen=pd.read_csv('/kaggle/input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv')

plant2_sen=pd.read_csv('/kaggle/input/solar-power-generation-data/Plant_2_Weather_Sensor_Data.csv')
plant1_sen=modify_time(plant1_sen)

plant1_sen
plant2_sen=modify_time(plant2_sen)

plant2_sen
def calculate_total_irr(df):

       return df.groupby('DATE_TIME')['IRRADIATION'].sum()

def plot_figure(groupby_val, ylabel,title):

    plt.figure(figsize=(12,10))

    groupby_val.plot.bar()

    plt.ylabel(ylabel)

    plt.title(title)

    plt.show()
total_irr_p1=calculate_total_irr(plant1_sen)

plot_figure(total_irr_p1,'TOTAL IRRADIATION','Total irradiation per day plant1')
total_irr_p2=calculate_total_irr(plant2_sen)

plot_figure(total_irr_p2,'TOTAL IRRADIATION','Total irradiation per day plant2')
def calculate_cum_amp(df):

    return df.groupby('DATE_TIME')['AMBIENT_TEMPERATURE'].sum()

def calculate_cum_mod(df):

    return df.groupby('DATE_TIME')['MODULE_TEMPERATURE'].sum()



cum_amb_p1=calculate_cum_amp(plant1_sen)

print('Max Ambient Temparature:',cum_amb_p1.max())

cum_mod_p1=calculate_cum_mod(plant1_sen)

print('Max Module Temparature:',cum_mod_p1.max())
def calculate_inv_count(df1, df2):

    return df1.groupby('SOURCE_KEY')['SOURCE_KEY'].count(), df2.groupby('SOURCE_KEY')['SOURCE_KEY'].count()
print(calculate_inv_count(plant1_sen, plant2_sen))
def plot_min_max(df,column, ylabel,title):

    plt.figure(figsize=(12,10))

    df.groupby('DATE_TIME')[column].max().plot()

    df.groupby('DATE_TIME')[column].min().plot()

    plt.ylabel(ylabel)

    plt.title('title')

    plt.show()



        
plot_min_max(plant1_gen, 'DC_POWER','DC_POWER','Max-Min DC Power per Day For Plant1')
plot_min_max(plant1_gen, 'AC_POWER','AC Power','Max-Min AC Power per Day For Plant1')
plot_min_max(plant2_gen, 'DC_POWER','DC_POWER','Max-Min DC Power per Day For Plant2')
plot_min_max(plant2_gen, 'AC_POWER','AC_POWER','Max-Min AC Power per Day For Plant2')
def calculate_max_ac_dc(df,column):

        cum_sum=df.groupby('SOURCE_KEY')[column].sum()

        index=cum_sum.argmax()

        return cum_sum.index[index], cum_sum.max()

       

       

    
calculate_max_ac_dc(plant1_gen,'DC_POWER')
calculate_max_ac_dc(plant1_gen,'AC_POWER')
calculate_max_ac_dc(plant2_gen,'DC_POWER')
calculate_max_ac_dc(plant2_gen,'AC_POWER')
def rank_inverters(df, column):

    plt.figure(figsize=(12,10))

    df.groupby('SOURCE_KEY')[column].sum().sort_values(ascending=False).plot(kind='bar')

    plt.ylabel('Cumulative '+column)

    plt.title('Inverters based on '+column)

    plt.show()
rank_inverters(plant1_gen, 'AC_POWER')
rank_inverters(plant1_gen, 'DC_POWER')
rank_inverters(plant2_gen, 'DC_POWER')
rank_inverters(plant2_gen, 'AC_POWER')