import numpy as np 
import pandas as pd

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

Plant_1_Generation_Data = pd.read_csv('../input/solar-power-generation-data/Plant_1_Generation_Data.csv')
df = pd.read_csv('../input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv')
Plant_2_Weather_Sensor_Data = pd.read_csv('../input/solar-power-generation-data/Plant_2_Weather_Sensor_Data.csv')
Plant_2_Generation_Data = pd.read_csv('../input/solar-power-generation-data/Plant_2_Generation_Data.csv')
print(('-'*50)+'PLANT 1 INFORMATION'+('-'*50))
print('\n')
print(('-')*50+'Generation Data'+('-')*50)
print('\n')
print(Plant_1_Generation_Data.columns)
print('\n')
print(Plant_1_Generation_Data.head())
print('\n')
print(Plant_1_Generation_Data.describe())
print('\n')
print(Plant_1_Generation_Data.info())
print('\n\n')
print(('-')*50+'Weather Sensor Data'+('-')*50)
print('\n')
print(df.columns)
print('\n')
print(df.head())
print('\n')
print(df.describe())
print('\n')
print(df.info())
print(('-'*50)+'PLANT 2 INFORMATION'+('-'*50))
print('\n')
print(('-')*50+'Generation Data'+('-')*50)
print('\n')
print(Plant_2_Generation_Data.columns)
print('\n')
print(Plant_2_Generation_Data.head())
print('\n')
print(Plant_2_Generation_Data.describe())
print('\n')
print(Plant_2_Generation_Data.info())
print('\n\n')
print(('-')*50+'Weather Sensor Data'+('-')*50)
print('\n')
print(Plant_2_Weather_Sensor_Data.columns)
print('\n')
print(Plant_2_Weather_Sensor_Data.head())
print('\n')
print(Plant_2_Weather_Sensor_Data.describe())
print('\n')
print(Plant_2_Weather_Sensor_Data.info())
df_pgen1 = pd.read_csv('../input/solar-power-generation-data/Plant_1_Generation_Data.csv')
df_psense1 = pd.read_csv('../input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv')
df_pgen1['DATE_TIME'] = pd.to_datetime(df_pgen1['DATE_TIME'],format = '%d-%m-%Y %H:%M')
df_psense1['DATE_TIME'] = pd.to_datetime(df_psense1['DATE_TIME'],format = '%Y-%m-%d %H:%M')

df_pgen1['DATE'] = df_pgen1['DATE_TIME'].dt.date
df_pgen1['TIME'] = df_pgen1['DATE_TIME'].dt.time

df_psense1['DATE'] = df_psense1['DATE_TIME'].dt.date
df_psense1['TIME'] = df_psense1['DATE_TIME'].dt.time
df_pgen1['DATE'] = pd.to_datetime(df_pgen1['DATE'],format = '%Y-%m-%d')
df_psense1['DATE'] = pd.to_datetime(df_psense1['DATE'],format = '%Y-%m-%d')
df_pgen1['HOUR'] = pd.to_datetime(df_pgen1['TIME'],format='%H:%M:%S').dt.hour
df_pgen1['MINUTES'] = pd.to_datetime(df_pgen1['TIME'],format='%H:%M:%S').dt.minute

df_psense1['HOUR'] = pd.to_datetime(df_psense1['TIME'],format='%H:%M:%S').dt.hour
df_psense1['MINUTES'] = pd.to_datetime(df_psense1['TIME'],format='%H:%M:%S').dt.minute
df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'],format = '%Y-%m-%d %H:%M')
df['DATE'] = pd.to_datetime(df['DATE_TIME'],format = '%Y-%m-%d %H:%M').dt.date
df['DATE'] = pd.to_datetime(df['DATE'])
print(('-'*20),'Plant_1_Generation_Data',('-'*20))
print(Plant_1_Generation_Data['DATE_TIME'].head())
print('\n')
print(('-'*20)+'-'*len('Plant_1_Generation_Data')+('-'*22))
print(Plant_1_Generation_Data['DATE_TIME'].tail())
print('\n\n')
print(('-'*20),'Plant_2_Generation_Data',('-'*20))
print(Plant_2_Generation_Data['DATE_TIME'].head())
print('\n')
print(('-'*20)+'-'*len('Plant_2_Generation_Data')+('-'*22))
print(Plant_2_Generation_Data['DATE_TIME'].tail())
print('\n\n')
print(('-'*20),'df',('-'*20))
print(df['DATE_TIME'].head())
print('\n')
print(('-'*20)+'-'*len('df')+('-'*22))
print(df['DATE_TIME'].tail())
print('\n\n')
print(('-'*20),'Plant_2_Weather_Sensor_Data',('-'*20))
print(Plant_2_Weather_Sensor_Data['DATE_TIME'].head())
print('\n')
print(('-'*20)+'-'*len('Plant_2_Weather_Sensor_Data')+('-'*22))
print(Plant_2_Weather_Sensor_Data['DATE_TIME'].tail())
Plant_1_Generation_Data['DATE_TIME_PARSED'] = pd.to_datetime(Plant_1_Generation_Data['DATE_TIME'],format = "%d-%m-%Y %H:%M")
df['DATE_TIME_PARSED'] = pd.to_datetime(df['DATE_TIME'],format = "%Y-%m-%d %H:%M:%S")
Plant_2_Generation_Data['DATE_TIME_PARSED'] = pd.to_datetime(Plant_2_Generation_Data['DATE_TIME'],format = "%Y-%m-%d %H:%M:%S")
Plant_2_Weather_Sensor_Data['DATE_TIME_PARSED'] = pd.to_datetime(Plant_2_Weather_Sensor_Data['DATE_TIME'],format = "%Y-%m-%d %H:%M:%S")
print(('*')*30+'DATE_TIME_PARSED'+('*')*30)
print(('-'*20),'Plant_1_Generation_Data',('-'*20))
print(Plant_1_Generation_Data['DATE_TIME_PARSED'].head())
print('\n\n')
print(('-'*20),'Plant_1_Weather_Sensor_Data',('-'*20))
print(df['DATE_TIME_PARSED'].head())
print('\n\n')
print(('-'*20),'Plant_2_Generation_Data',('-'*20))
print(Plant_2_Generation_Data['DATE_TIME_PARSED'].head())
print('\n\n')
print(('-'*20),'Plant_2_Weather_Sensor_Data',('-'*20))
print(Plant_2_Weather_Sensor_Data['DATE_TIME_PARSED'].head())
plant_1_irradiation_per_day = df.groupby(df.DATE_TIME_PARSED.dt.date)['IRRADIATION'].sum()
plant_2_irradiation_per_day = Plant_2_Weather_Sensor_Data.groupby(Plant_2_Weather_Sensor_Data.DATE_TIME_PARSED.dt.date)['IRRADIATION'].sum()
print('PLANT 1 IRRADIATION PER DAY')
print('\n')
print(plant_1_irradiation_per_day)
print('\n\n')
print('PLANT 2 IRRADIATION PER DAY')
print('\n')
print(plant_2_irradiation_per_day)
print('PLANT 1 IRRADIATION PER DAY')
plt.figure(figsize= (13,13))
plant_1_irradiation_per_day.plot(kind = 'bar')
plt.show()
print('PLANT 2 IRRADIATION PER DAY')
plt.figure(figsize= (13,13))
plant_2_irradiation_per_day.plot(kind = 'bar')
plt.show()
import matplotlib.pyplot as plt
plt.figure(figsize=(20,10)) 
plt.plot(df['DATE_TIME'],df['IRRADIATION'],label = 'IRRADIATION')
plt.legend()
plt.grid()
plt.margins(0.05)
plt.show()
_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_psense1.DATE_TIME,
        df_psense1.AMBIENT_TEMPERATURE.rolling(window=20).mean(),
        label='Ambient'
       )

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Ambient Temperature')
plt.xlabel('Date and Time')
plt.ylabel('Temperature')
plt.show()
print ("minimum="+ str(df_psense1.AMBIENT_TEMPERATURE.min()) )
print ("maximum="+ str(df_psense1.AMBIENT_TEMPERATURE.max()) )
print ("mean="+ str(df_psense1.AMBIENT_TEMPERATURE.mean()) )
import matplotlib.pyplot as plt
plt.figure(figsize=(20,10)) # To increase the size of the graph
plt.plot(df['DATE_TIME'],df['MODULE_TEMPERATURE']-df['AMBIENT_TEMPERATURE'],label = 'Difference')
plt.legend()
plt.grid()
plt.margins(0.05)
plt.show()

_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_psense1.DATE_TIME,
        df_psense1.AMBIENT_TEMPERATURE.rolling(window=20).mean(),
        label='Ambient'
       )

ax.plot(df_psense1.DATE_TIME,
        df_psense1.MODULE_TEMPERATURE.rolling(window=20).mean(),
        label='Module'
       )

ax.plot(df_psense1.DATE_TIME,
        (df_psense1.MODULE_TEMPERATURE-df_psense1.AMBIENT_TEMPERATURE).rolling(window=20).mean(),
        label='Difference'
       )

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Ambient Temperature and Module Temperature over 34 Days')
plt.xlabel('Date and Time')
plt.ylabel('Temperature')
plt.show()

plt.figure(figsize=(20,10))
plt.scatter(df['AMBIENT_TEMPERATURE'],df['MODULE_TEMPERATURE'],s = 50,label='Temperature Graph',alpha = 0.5)
plt.plot()
plt.legend()
plt.grid()
plt.margins(0.05)
plt.show()
_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_pgen1.DATE_TIME,
        df_pgen1.AC_POWER.rolling(window=500).mean(),
        )

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('AC_POWER over 34 Days')
plt.xlabel('Date and Time')
plt.ylabel('POWER')
plt.show()

print ("minimum="+ str(df_pgen1.AC_POWER.min()) )
print ("maximum="+ str(df_pgen1.AC_POWER.max()) )
print ("mean="+ str(df_pgen1.AC_POWER.mean()) )
_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_pgen1.DATE_TIME,
        (df_pgen1.DC_POWER/10).rolling(window=500).mean(),
        )

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('DC_POWER over 34 Days')
plt.xlabel('Date and Time')
plt.ylabel('POWER')
plt.show()

print ("minimum="+ str((df_pgen1.DC_POWER/10).min()) )
print ("maximum="+ str((df_pgen1.DC_POWER/10).max()) )
print ("mean="+ str((df_pgen1.DC_POWER/10).mean()) )
df['DATE']=='2020-05-15'
dates = df['DATE'].unique()
df[df['DATE']==dates[0]] 
df[df['DATE']==dates[0]][df['IRRADIATION']>0]

data = df[df['DATE']==dates[0]][df['IRRADIATION']>0]
plt.plot(data['AMBIENT_TEMPERATURE'],data['MODULE_TEMPERATURE'],marker='o',linestyle='',
        label=pd.to_datetime(dates[0],format='%Y-%m-%d').date())
plt.legend()
plt.show()
data = df[df['IRRADIATION']>0.1]
plt.figure(figsize=(20,10))
plt.plot(data['IRRADIATION'],data['MODULE_TEMPERATURE']-data['AMBIENT_TEMPERATURE'],marker='o',linestyle='',alpha=0.5,label='Difference Temp')
plt.legend()
plt.ylabel('Temperature Difference')
plt.xlabel('Irradiation')
df_gen = pd.read_csv('../input/solar-power-generation-data/Plant_1_Generation_Data.csv')
df_gen['DATE_TIME'] = pd.to_datetime(df_gen['DATE_TIME'],format='%d-%m-%Y %H:%M')
df_gen['DATE'] = pd.to_datetime(df_gen['DATE_TIME'],format='%d-%m-%Y %H:%M').dt.date
df_gen['DATE'] = pd.to_datetime(df_gen['DATE'])
inv_lst = df_gen['SOURCE_KEY'].unique()
_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_pgen1.DATE_TIME,
        df_pgen1.AC_POWER.rolling(window=500).mean(),
        label='AC'
       )

ax.plot(df_pgen1.DATE_TIME,
       (df_pgen1.DC_POWER/10).rolling(window=500).mean(),
        label='DC'
       )

ax.plot(df_pgen1.DATE_TIME,
       ((df_pgen1.DC_POWER/10)-df_pgen1.AC_POWER).rolling(window=500).mean(),
        label='Difference'
       )

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('AC POWER and DC POWER over 34 Days')
plt.xlabel('Date and Time')
plt.ylabel('kW')
plt.show()

Inverters_performance=df_pgen1.groupby("SOURCE_KEY").agg(LIFETIME_YIELD=("TOTAL_YIELD",max),
                                           SOURCE_KEY=("SOURCE_KEY",max)
                                        )
Inverters_performance
sns.barplot(x=Inverters_performance["SOURCE_KEY"], y=Inverters_performance["LIFETIME_YIELD"])

print("maximum=" +str(Inverters_performance['LIFETIME_YIELD'].max()))
print("minimum=" +str(Inverters_performance['LIFETIME_YIELD'].min()))
print("mean=" +str( Inverters_performance['LIFETIME_YIELD'].mean()))

_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_pgen1.DC_POWER/10,
        df_pgen1.AC_POWER,
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        label='AC POWER'
       )
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Scatter Plot showing How AC Power varies with DC Power')
plt.xlabel('DC Power')
plt.ylabel('AC Power')
plt.show()
#dc power vs daily yield
comparision=df_pgen1.groupby("DATE").agg(DAILY_YIELD=("DAILY_YIELD",max),
                                         DC_POWER=("DC_POWER",sum),
                                         AC_POWER=("AC_POWER",sum),
                                         DATE=("DATE",max)
                                         )
comparision

_, ax = plt.subplots(1, 1, figsize=(18, 9))



ax.plot(comparision.DC_POWER,
        comparision.DAILY_YIELD,
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        )

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('DC Power vs Daily Yield')
plt.xlabel('DC power')
plt.ylabel('daily yield')
plt.show()
#ac power vs daily yield
_, ax = plt.subplots(1, 1, figsize=(18, 9))



ax.plot(comparision.AC_POWER,
        comparision.DAILY_YIELD,
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        )

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('AC Power vs Daily Yield')
plt.xlabel('AC power')
plt.ylabel('daily yield')
plt.show()
plt.show()
import pandas as pd
r_left = pd.merge(df_gen,df,on='DATE_TIME',how='left')
r_left
r_left.isnull().sum()
r_left['AMBIENT_TEMPERATURE'].isnull().value_counts()
null_data1 = r_left[r_left.isnull().any(axis=1)]
null_data1
plt.figure(figsize=(20,10))
plt.plot(r_left['IRRADIATION'],r_left['DC_POWER'],marker='o',linestyle='',alpha=0.5,label='DC Power')
plt.legend()

plt.xlabel('Irradiation')
plt.ylabel('DC Power')
dates = r_left['DATE_x'].unique()
r_left[r_left['DATE_x']==dates[0]][r_left['IRRADIATION']>0.1]
data = r_left[r_left['DATE_x']==dates[1]][r_left['IRRADIATION']>0.1]
plt.plot(data['MODULE_TEMPERATURE'],data['DC_POWER'],marker='o',linestyle='',
         label = pd.to_datetime(dates[1],format='%Y-%m-%d').date())
plt.legend()
print(dates[1])
r_left['IRRADIATION'] = r_left['IRRADIATION'].fillna(0)  
r_left.info()
x = r_left.iloc[:,12:13].values
x.ndim
y = r_left.iloc[:,3].values
y
import matplotlib.pyplot as plt
plt.scatter(x,y,s=1)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
x_train.shape
x_test.shape
55022+13756
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
y_pred
y_test
plt.scatter(x_train,y_train,s=1)
plt.scatter(x_train,model.predict(x_train),s=1)
plt.xlabel('IRRADIATION')
plt.ylabel('DC_power')
plt.show()
#Linear regression model for dc power vs irradiation
import pickle
pickle.dump(model, open('model.pkl','wb'))
!pip install streamlit
!pip install pyngrok
%%writefile app1.py
import streamlit as st
import pickle

st.title("Irradiation vs DCpower")
st.subheader('Prediction')
st.write('This is useful in predicting DC POWER for given irradiation')

model = pickle.load(open('model.pkl', 'rb'))
temp = st.number_input('Enter IRRADIATION')
op = model.predict([[temp]])
if st.button("Predict"):
  st.title(f'The DC Power is {op}')
from pyngrok import ngrok
public_url = ngrok.connect(port='8501')
print(public_url)
!streamlit run app1.py
df.info()
day_summary = df.groupby('DATE').agg(TOTAL_IRRADIANCE=('IRRADIATION',sum),DATE=('DATE',max))
day_summary
day_summary = day_summary.rename(columns={'DATE':'ds','TOTAL_IRRADIANCE':'y'})
day_summary.info()

import fbprophet
op = fbprophet.Prophet(changepoint_prior_scale=0.25)
op.fit(day_summary)
forecast = op.make_future_dataframe(periods = 30,freq='D')
forecast = op.predict(forecast)
op.plot(forecast,xlabel='Date',ylabel='Irradiation')
plt.title('Irradiation Prediction');
forecast['ds'].value_counts()
df_new = df[['DATE_TIME','IRRADIATION']]
df_new
df_new = df_new.rename(columns={'DATE_TIME':'ds','IRRADIATION':'y'})
df_new

import fbprophet
op = fbprophet.Prophet(changepoint_prior_scale=0.25)
op.fit(df_new)
forecast = op.make_future_dataframe(periods = 300,freq='H')
forecast = op.predict(forecast)
op.plot(forecast,xlabel='Date',ylabel='Irradiation')
plt.title('Irradiation Prediction');
op.plot_components(forecast);
forecast