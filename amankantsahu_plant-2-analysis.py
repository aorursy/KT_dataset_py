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
len(dates)
dates = df_copy['DATE'].unique() 
for date in dates:
    fig = plt.figure(figsize =(10, 9)) 
    df_copy.groupby(df_copy['SOURCE_KEY'])['DAILY_YIELD'].mean().plot.bar()
    plt.grid()
    plt.title(date)
    plt.ylabel('DAILY YIELD')
    plt.show()
keys = df_copy['SOURCE_KEY'].unique() 
for key in keys:
    fig = plt.figure(figsize =(15, 9)) 
    df_copy.groupby(df_copy['DATE'])['DAILY_YIELD'].mean().plot.bar()
    plt.grid()
    plt.title(key)
    plt.ylabel('DAILY YIELD')
    plt.show()
data = df_copy.groupby(df_copy['SOURCE_KEY'])['TOTAL_YIELD'].mean()
data
fig = plt.figure(figsize =(10, 5)) 
df_copy.groupby(df_copy['SOURCE_KEY'])['TOTAL_YIELD'].mean().plot.bar()
plt.grid()
plt.title('MEAN TOTAL YIELD of each INVETOR')
plt.ylabel('MEAN TOTAL YIELD')
plt.show()
keys = df_copy['SOURCE_KEY'].unique() 
for key in keys:
    fig = plt.figure(figsize =(15, 9)) 
    df_copy.groupby(df_copy['DATE'])['AC_POWER'].mean().plot.bar()
    plt.grid()
    plt.title(key)
    plt.ylabel('DAILY YIELD')
    plt.show()
keys = df_copy['SOURCE_KEY'].unique() 
for key in keys:
    fig = plt.figure(figsize =(15, 9)) 
    df_copy.groupby(df_copy['DATE'])['AC_POWER'].mean().plot.bar()
    plt.grid()
    plt.title(key)
    plt.ylabel('AC_POWER')
    plt.show()
keys = df_copy['SOURCE_KEY'].unique() 
for key in keys:
    data = df_copy[df_copy['SOURCE_KEY']==key]
    _,ax = plt.subplots(1,1,figsize =(20,5)) 
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
    _,ax = plt.subplots(1,1,figsize =(20,9)) 
    ax.plot(data.DATE,
            data.AC_POWER,
            marker='o',
            linestyle=''
            )
    ax.grid()
    ax.margins(0.05)
    ax.legend()
    plt.title(key)
    plt.xlabel('AC POWER')
    plt.ylabel('DATE')
    plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df_wgen2=pd.read_csv('../input/solar-power-generation-data/Plant_2_Weather_Sensor_Data.csv')
df_wgen2
df_wgen2['DATE_TIME'] = pd.to_datetime(df_wgen2['DATE_TIME'],format = '%Y-%m-%d %H:%M')

df_wgen2['DATE'] = df_wgen2['DATE_TIME'].apply(lambda x:x.date())
df_wgen2['TIME'] = df_wgen2['DATE_TIME'].apply(lambda x:x.time())

df_wgen2['DATE'] = pd.to_datetime(df_wgen2['DATE'],format = '%Y-%m-%d').dt.date
df_wgen2['HOUR'] = pd.to_datetime(df_wgen2['TIME'],format='%H:%M:%S').dt.hour
df_wgen2['MINUTES'] = pd.to_datetime(df_wgen2['TIME'],format='%H:%M:%S').dt.minute

#The below graph shows the relation between the module temperature and how it is changing with respect to date and time.
#We can see that module temperature is higher on few partcular days like 5/18.

_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_wgen2['DATE_TIME'],
        df_wgen2['MODULE_TEMPERATURE'],
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        label='Module temperature',
       color='r')
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Date & Time vs. Module Tempreture')
plt.xlabel('Date & Time')
plt.ylabel('Module Tempreture')
plt.show()
_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_wgen2['DATE_TIME'],
        df_wgen2['AMBIENT_TEMPERATURE'],
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        label='Ambient temperature',
       color='r')
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Date & Time vs. Ambient Tempreture')
plt.xlabel('Date & Time')
plt.ylabel('Ambient Tempreture')
plt.show()
_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_wgen2['DATE_TIME'],
        df_wgen2['IRRADIATION'],
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        label='Irradiation',
       color='r')
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Date & Time vs. Irradiation')
plt.xlabel('Date & Time')
plt.ylabel('Irradiation')
plt.show()
_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_wgen2['DATE'],
        df_wgen2['MODULE_TEMPERATURE'],
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        label='Module temperature',
       color='g')
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Date vs. Module Tempreture')
plt.xlabel('Date')
plt.ylabel('Module Tempreture')
plt.show()
_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_wgen2['DATE'],
        df_wgen2['AMBIENT_TEMPERATURE'],
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        label='Ambient temperature',
       color='g')
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Date vs. Ambient Tempreture')
plt.xlabel('Date')
plt.ylabel('Ambient Tempreture')
plt.show()
# We can see that irradiation constantly increases and decreases without any sort of deviation.
_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_wgen2['DATE'],
        df_wgen2['IRRADIATION'],
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        label='Irradiation',
       color='g')
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Date vs. Irradiation')
plt.xlabel('Date')
plt.ylabel('Irradiation')
plt.show()
#After 10hrs of the functioning of the plant we can see gradual increase in the module temperature till it strikes evening. It then follows a gradual decrese.
_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_wgen2['HOUR'],
        df_wgen2['MODULE_TEMPERATURE'],
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        label='Module Temperature',
        color='m'
       )
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Time vs. Module Temperature')
plt.xlabel('Time')
plt.ylabel('Module Temperature')
plt.show()
#Maximum temperature can be recorded after 15 hours of functioning of the plant.
_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_wgen2['HOUR'],
        df_wgen2['AMBIENT_TEMPERATURE'],
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        label='Ambient Temperature',
        color='m'
       )
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Time vs. Ambient Temperature')
plt.xlabel('Time')
plt.ylabel('Ambient Temperature')
plt.show()
#Maximum irradiation is possible after 12hrs of working of the plant(noon).
_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_wgen2['HOUR'],
        df_wgen2['IRRADIATION'],
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        label='Irradiation',
        color='m'
       )
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Time vs. Irradiation')
plt.xlabel('Time')
plt.ylabel('Irradiation')
plt.show()
_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_wgen2['MINUTES'],
        df_wgen2['MODULE_TEMPERATURE'],
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        label='Module Temperature',
        color='y'
       )
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Time vs. Module Temperature')
plt.xlabel('Time')
plt.ylabel('Module Temperature')
plt.show()
_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_wgen2['MINUTES'],
        df_wgen2['AMBIENT_TEMPERATURE'],
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        label='Ambient Temperature',
        color='y'
       )
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Time vs. Ambient Temperature')
plt.xlabel('Time')
plt.ylabel('Ambient Temperature')
plt.show()
_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_wgen2['MINUTES'],
        df_wgen2['IRRADIATION'],
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        label='Irradiation',
        color='y'
       )
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Time vs. Irradiation')
plt.xlabel('Time')
plt.ylabel('Irradiation')
plt.show()
_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_wgen2['HOUR']+df_wgen2['MINUTES'],
        df_wgen2['MODULE_TEMPERATURE'],
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        label='Module Temperature',
        color='c'
       )
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Time vs. Module Temperature')
plt.xlabel('Time')
plt.ylabel('Module Temperature')
plt.show()
_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_wgen2['HOUR']+df_wgen2['MINUTES'],
        df_wgen2['AMBIENT_TEMPERATURE'],
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        label='Ambient Temperature',
        color='c'
       )
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Time vs. Ambient Temperature')
plt.xlabel('Time')
plt.ylabel('Ambient Temperature')
plt.show()
_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_wgen2['HOUR']+df_wgen2['MINUTES'],
        df_wgen2['IRRADIATION'],
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        label='Irradiation',
        color='c'
       )
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Time vs. Irradiation')
plt.xlabel('Time')
plt.ylabel('Irradiation')
plt.show()
#The plot shows a gradual increase in the temperature of the plant
_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_wgen2.AMBIENT_TEMPERATURE,
        df_wgen2.MODULE_TEMPERATURE,
        marker='o',
        linestyle='',
        alpha=.4,
        ms=10,
        label='Module Temperature (centigrade)',
       color='b')
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Ambient Tempreture vs. Module Tempreture')
plt.xlabel('Ambient Tempreture')
plt.ylabel('Module Tempreture')
plt.show()
#The plot is quite hard to read but we can make out the steady increase and decrase of temperature with irradiation. Hence temperature is proportional to irradiation.
_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_wgen2.AMBIENT_TEMPERATURE,
        df_wgen2.IRRADIATION,
        marker='o',
        linestyle='',
        alpha=.4,
        ms=10,
        label='Irradiation',
       color='b')
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Ambient Tempreture vs. Irradiation')
plt.xlabel('Ambient Tempreture')
plt.ylabel('Irradition')
plt.show()
#Increase in irradiation shows us increase in module temperature
_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_wgen2.MODULE_TEMPERATURE,
        df_wgen2.IRRADIATION,
        marker='o',
        linestyle='',
        alpha=.4,
        ms=10,
        label='Irradiation',
       color='b')
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Module Tempreture vs. Irradiation')
plt.xlabel('Module Tempreture')
plt.ylabel('Irradition')
plt.show()
#Increase in irradiation results in the increase of the ambient and module temperature.
_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_wgen2['IRRADIATION'],
        df_wgen2['MODULE_TEMPERATURE']+df_wgen2['AMBIENT_TEMPERATURE'],
        marker='o',
        linestyle='',
        alpha=.5, #transparency
        ms=10, #size of the dot
        label='temperature (Module + Ambient)',
       color='r')
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Irradiation vs. Tempreture ')
plt.xlabel('Irradiation')
plt.ylabel('Tempreture')
plt.show()

_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_wgen2['DATE_TIME'],
        df_wgen2['MODULE_TEMPERATURE']+df_wgen2['AMBIENT_TEMPERATURE'],
        marker='o',
        linestyle='',
        alpha=.5, #transparency
        ms=10, #size of the dot
        label='temperature (Module + Ambient)',
       color='r')
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Date & Time vs. Tempreture ')
plt.xlabel('Date & Time')
plt.ylabel('Tempreture')
plt.show()

dates = df_wgen2['DATE'].unique()

_, ax = plt.subplots(1, 1, figsize=(18, 9))

for date in dates:
    df_data = df_wgen2[df_wgen2['DATE']==date]

    ax.plot(df_data.AMBIENT_TEMPERATURE,
            df_data.MODULE_TEMPERATURE,
            marker='o',
            linestyle='',
            alpha=.5,
            ms=12,
            label=pd.to_datetime(date,format='%Y-%m-%d').date()
           )

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Module Tempreture vs. Ambient Tempreture')
plt.xlabel('Ambient Tempreture')
plt.ylabel('Module Tempreture')
plt.show()

dates = df_wgen2['DATE'].unique()

_, ax = plt.subplots(1, 1, figsize=(18, 9))

for date in dates:
    df_data = df_wgen2[df_wgen2['DATE']==date]

    ax.plot(df_data.IRRADIATION,
            marker='o',
            linestyle='',
            alpha=.5,
            ms=10,
            label=pd.to_datetime(date,format='%Y-%m-%d').date()
           )

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Irradition Per Day')
plt.xlabel('Irradition')
plt.ylabel('')
plt.show()

_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_wgen2.DATE_TIME,
        df_wgen2.AMBIENT_TEMPERATURE.rolling(window=20).mean(),
        label='Ambient',
        color='r'
       )

ax.plot(df_wgen2.DATE_TIME,
        df_wgen2.MODULE_TEMPERATURE.rolling(window=20).mean(),
        label='Module',
        color='c'
       )

ax.plot(df_wgen2.DATE_TIME,
        (df_wgen2.MODULE_TEMPERATURE-df_wgen2.AMBIENT_TEMPERATURE).rolling(window=20).mean(),
        label='Difference',
        color='m'
       )

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Ambient Tempreture and Module Tempreture over 34 Days')
plt.xlabel('Date and Time')
plt.ylabel('Tempreture')
plt.show()
i=df_wgen2['IRRADIATION']
from pylab import rcParams
import statsmodels.api as sm
rcParams['figure.figsize'] = 18, 8
decomposition = sm.tsa.seasonal_decompose(i, model='additive',period=34)
fig = decomposition.plot()
plt.show()
m=df_wgen2['MODULE_TEMPERATURE']
from pylab import rcParams
import statsmodels.api as sm
rcParams['figure.figsize'] = 18, 8
decomposition = sm.tsa.seasonal_decompose(m, model='additive',period=34)
fig = decomposition.plot()
plt.show()
a=df_wgen2['AMBIENT_TEMPERATURE']
from pylab import rcParams
import statsmodels.api as sm
rcParams['figure.figsize'] = 18, 8
decomposition = sm.tsa.seasonal_decompose(a, model='additive',period=34)
fig = decomposition.plot()
plt.show()
result_inner = pd.merge(df_pgen2,df_wgen2, on='DATE_TIME',how='inner') #left, right, outer, inner
result_left = pd.merge(df_pgen2,df_wgen2, on='DATE_TIME',how='left') #left, right, outer, inner
result_left['IRRADIATION'] = result_left['IRRADIATION'].fillna(0)
result_left['AMBIENT_TEMPERATURE'] = result_left['AMBIENT_TEMPERATURE'].fillna(0)
result_left['MODULE_TEMPERATURE'] = result_left['MODULE_TEMPERATURE'].fillna(0)

X = result_left.iloc[:, 15:16].values #Irradiation
y = result_left.iloc[:, 3].values #DC_Power
X,y
result_inner['IRRADIATION'] = result_inner['IRRADIATION'].fillna(0)
result_inner['AMBIENT_TEMPERATURE'] = result_inner['AMBIENT_TEMPERATURE'].fillna(0)
result_inner['MODULE_TEMPERATURE'] = result_inner['MODULE_TEMPERATURE'].fillna(0)

X = result_inner.iloc[:, 15:16].values #Irradiation
y = result_inner.iloc[:, 3].values #DC_Power
X,y
X = result_left[['IRRADIATION','MODULE_TEMPERATURE']]
y = result_left['DC_POWER']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
coeff_df = pd.DataFrame(lin_reg.coef_,X.columns,columns = ['Coefficients'])
coeff_df

y_pred = lin_reg.predict(X_test)

compare_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
compare_df.head(10)

print("Train Set Accuracy")
print('Mean Absolute Error:', metrics.mean_absolute_error(y_train, lin_reg.predict(X_train)))
print('Mean Squared Error:', metrics.mean_squared_error(y_train, lin_reg.predict(X_train)))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, lin_reg.predict(X_train))))
print('---------------------------')
print("Test Set Accuracy")
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

lin_reg.predict([[0.4,50]]) #0.4 irradiation and 50 degrees of module temperature
from fbprophet import Prophet
df_wgen2 = pd.read_csv('../input/solar-power-generation-data/Plant_2_Weather_Sensor_Data.csv')
df_wgen2.head()
df_wgen2=df_wgen2.rename(columns={'DATE_TIME':'ds', 'IRRADIATION':'y'})
df_wgen2
m = Prophet()
m.fit(df_wgen2)

future = m.make_future_dataframe(periods=34)
future

forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

fig1 = m.plot(forecast)

fig2 = m.plot_components(forecast)
                        
