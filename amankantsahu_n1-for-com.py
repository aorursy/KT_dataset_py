# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_pgen2 = pd.read_csv('../input/solar-power-generation-data/Plant_2_Generation_Data.csv')

df_pgen2
df_copy=df_pgen2.copy()        #creating a copy of original data set 
df_copy['DATE_TIME']= pd.to_datetime(df_copy['DATE_TIME'])
df_copy['DATE']= pd.to_datetime(df_copy['DATE_TIME']).dt.date

df_copy['HOUR'] = pd.to_datetime(df_copy['DATE_TIME']).dt.hour

df_copy['MINUTES'] = pd.to_datetime(df_copy['DATE_TIME']).dt.minute
print('number of days for which observation is avialable = ',len(df_copy['DATE'].unique()))

df_copy.isnull().values.any()
print('Number of invetors in plant2= ',len(df_pgen2['SOURCE_KEY'].unique()))

import matplotlib.pyplot as plt
temp_max = df_copy.groupby(['SOURCE_KEY','DATE']).agg(DAILY_YIELD = ('DAILY_YIELD',max))

print('MAX daily yield for each day')

temp_max.head(50)
temp_min = df_copy.groupby(['SOURCE_KEY','DATE']).agg(DAILY_YIELD = ('DAILY_YIELD',min))

print('MIN daily yield for each day')

temp_min.head(50)

keys = df_copy['SOURCE_KEY'].unique()

_, ax = plt.subplots(1,1,figsize=(22,20))

for key in keys:

    data = df_copy[df_copy['SOURCE_KEY'] == key]

    ax.plot(data.DATE,

            data.DAILY_YIELD,

            marker='^',

            linestyle='',

            alpha=.5,

            ms=10,

            label=key

           )

ax.grid()

ax.margins(0.05)

ax.legend()

plt.title('DATE vs DAILY YIELD for plant2')

plt.xlabel('DATE')

plt.ylabel('DAILY YIELD')

plt.show()

keys = df_copy['SOURCE_KEY'].unique()

_, ax = plt.subplots(1,1,figsize=(22,20))

for key in keys:

    data = df_copy[df_copy['SOURCE_KEY'] == key]

    ax.plot(data.DAILY_YIELD,

            data.HOUR,

            marker='^',

            linestyle='',

            alpha=.5,

            ms=10,

            label=key

           )

ax.grid()

ax.margins(0.05)

ax.legend()

plt.title('HOUR vs DAILY YIELD for plant2')

plt.xlabel('DAILY YIELD')

plt.ylabel('HOUR')

plt.show()

dates = df_copy['DATE'].unique()

keys = df_copy['SOURCE_KEY'].unique()

_, ax = plt.subplots(1,1,figsize=(22,20))

for key in keys :

    data1=df_copy[df_copy['SOURCE_KEY'] == key]

    for date in dates:

        data2 = data1[data1['DATE'] ==  date]

        ax.plot(data2.DAILY_YIELD,

                data2.HOUR,

                marker='^',

                linestyle='',

                alpha=.5,

                ms=10,

                label=key

               )

ax.grid()

ax.margins(0.05)

ax.legend()

plt.title('HOUR vs DAILY YIELD for plant2')

plt.xlabel('HOUR')

plt.ylabel('DAILY YIELD')

plt.show()
print('mean of daily daily yield = ',df_pgen2['DAILY_YIELD'].mean())
data= df_copy.groupby(df_copy['SOURCE_KEY'])['DAILY_YIELD'].mean()

data

fig = plt.figure(figsize =(10, 9)) 

df_copy.groupby(df_copy['SOURCE_KEY'])['DAILY_YIELD'].mean().plot.bar()

plt.grid()

plt.title('MEAN DAILY YIELD of each INVETOR')

plt.ylabel('MEAN DAILY YIELD')

plt.show()
dates = df_copy['DATE'].unique()

count = 0

for date in dates:

    data =  df_copy[df_copy['DATE'] == date]['DAILY_YIELD'].mean()

    count+=1

    print(data)

count
dates = df_copy['DATE'].unique() 

for date in dates:

    print('On',date,'mean DAILY YIELD is ',df_copy[df_copy['DATE']==date]['DAILY_YIELD'].mean())

dates = df_copy['DATE'].unique() 

for date in dates:

    fig = plt.figure(figsize =(10, 9)) 

    df_copy.groupby(df_copy['SOURCE_KEY'])['DAILY_YIELD'].mean().plot.bar()

    plt.grid()

    plt.title(date)

    plt.ylabel('DAILY YIELD')

    plt.show()

    print('On',date,'mean DAILY YIELD is ',df_copy[df_copy['DATE']==date]['DAILY_YIELD'].mean())

data = df_copy.groupby(df_copy['SOURCE_KEY'])['TOTAL_YIELD'].mean()

data
fig = plt.figure(figsize =(10, 5)) 

df_copy.groupby(df_copy['SOURCE_KEY'])['TOTAL_YIELD'].mean().plot.bar()

plt.grid()

plt.title('MEAN TOTAL YIELD of each INVETOR')

plt.ylabel('MEAN TOTAL YIELD')

plt.show()
temp_max = df_copy.groupby(['SOURCE_KEY','DATE']).agg(AC_POWER_MAX = ('AC_POWER',max))

print('MAX ac power for each day')

temp_max.head(50)
temp_max = df_copy.groupby(['SOURCE_KEY','DATE']).agg(DC_POWER_MAX = ('DC_POWER',max))

print('MAX dc power for each day')

temp_max.head(50)

keys = df_copy['SOURCE_KEY'].unique() 

for key in keys:

     print('Invetor',key,'produce mean ac power is',df_copy[df_copy['SOURCE_KEY']==key]['AC_POWER'].mean())

keys = df_copy['SOURCE_KEY'].unique() 

for key in keys:

    fig = plt.figure(figsize =(15, 9)) 

    df_copy.groupby(df_copy['DATE'])['AC_POWER'].mean().plot.bar()

    plt.grid()

    plt.title(key)

    plt.ylabel('AC POWER')

    plt.show()

    print('Invetor',key,'produce mean ac power is',df_copy[df_copy['SOURCE_KEY']==key]['AC_POWER'].mean())
keys = df_copy['SOURCE_KEY'].unique() 

for key in keys:

    data = df_copy[df_copy['SOURCE_KEY']==key]

    _,ax = plt.subplots(1,1,figsize =(20,6)) 

    ax.plot(data.AC_POWER,

            data.DC_POWER,

            marker='+',

            linestyle=''

            )

    ax.grid()

    ax.margins(0.05)

    ax.legend()

    plt.title(key)

    plt.xlabel('AC POWER')

    plt.ylabel('DC POWER')

    plt.show()
keys = df_copy['SOURCE_KEY'].unique() 

for key in keys:

    data = df_copy[df_copy['SOURCE_KEY']==key]

    _,ax = plt.subplots(1,1,figsize =(20,6)) 

    ax.plot(data.DATE,

            data.AC_POWER,

            marker='^',

            linestyle=''

            )

    ax.grid()

    ax.margins(0.05)

    ax.legend()

    plt.title(key)

    plt.ylabel('AC POWER')

    plt.xlabel('DATE')

    plt.show()
dates = df_copy['DATE'].unique()

keys = df_copy['SOURCE_KEY'].unique() 

for key in keys:

    data = df_copy[df_copy['SOURCE_KEY']==key]

    print('Analyise of ',key,'for ecah day on hours based')

    for date in dates:

        data2 = df_copy[df_copy['DATE']==date]

        _,ax = plt.subplots(1,1,figsize =(20,6)) 

        ax.plot(data2.HOUR,

                data2.AC_POWER,

                marker='+',

                linestyle=''

                )

        ax.grid()

        ax.margins(0.05)

        ax.legend()

        plt.title(date)

        plt.xlabel('HOUR')

        plt.ylabel('AC POWER')

        plt.show()
keys = df_copy['SOURCE_KEY'].unique() 

for key in keys:

    data = df_copy[df_copy['SOURCE_KEY']==key]

    _,ax = plt.subplots(1,1,figsize =(20,6)) 

    ax.plot(data.DATE,

            data.DC_POWER,

            marker='^',

            linestyle=''

            )

    ax.grid()

    ax.margins(0.05)

    ax.legend()

    plt.title(key)

    plt.ylabel('DC POWER')

    plt.xlabel('DATE')

    plt.show()
dates = df_copy['DATE'].unique()

keys = df_copy['SOURCE_KEY'].unique() 

for key in keys:

    data = df_copy[df_copy['SOURCE_KEY']==key]

    print('Analyise of ',key,'for ecah day on hours based')

    for date in dates:

        data2 = df_copy[df_copy['DATE']==date]

        _,ax = plt.subplots(1,1,figsize =(20,6))

        ax.plot(data2.HOUR,

                data2.DC_POWER,

                marker='+',

                linestyle=''

                )

        ax.grid()

        ax.margins(0.05)

        ax.legend()

        plt.title(date)

        plt.xlabel('HOUR')

        plt.ylabel('DC POWER')

        plt.show()
print('DC POWER VS DAILY YIELD')

dates = df_copy['DATE'].unique()

keys = df_copy['SOURCE_KEY'].unique() 

for key in keys:

    data = df_copy[df_copy['SOURCE_KEY']==key]

    print('Analyise of ',key)

    for date in dates:

        data2 = df_copy[df_copy['DATE']==date]

        _,ax = plt.subplots(1,1,figsize =(20,5)) 

        ax.plot(data2.DAILY_YIELD,

                data2.DC_POWER,

                marker='+',

                linestyle=''

                )

        ax.grid()

        ax.margins(0.05)

        ax.legend()

        plt.title(date)

        plt.xlabel('DAILY YIELD')

        plt.ylabel('DC POWER')

        plt.show()
