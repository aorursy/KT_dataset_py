#TASKS

# 1. Load the data from csv

import pandas as pd

df1 = pd.read_csv('../input/solar-power-generation-data/Plant_1_Generation_Data.csv')

df2 = pd.read_csv('../input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv')

df1
df2
df1.head()
df2.head()
df1['DATE_TIME'] = pd.to_datetime(df1['DATE_TIME'],format = '%d-%m-%Y %H:%M')   #datetime 

df1['DATE'] = pd.to_datetime(df1['DATE_TIME'],format = '%d-%m-%Y %H:%M').dt.date   #split

df1['DATE'] = pd.to_datetime(df1['DATE'] )      #datetime series

df1.info()
df2['DATE_TIME'] = pd.to_datetime(df2['DATE_TIME'],format = '%Y-%m-%d %H:%M')   #datetime 

df2['DATE'] = pd.to_datetime(df2['DATE_TIME'],format = '%Y-%m-%d %H:%M').dt.date   #split

df2['DATE'] = pd.to_datetime(df2['DATE'] )      #datetime series

df2.info()
df1.columns
df2.columns
df1.nunique()
df2.nunique()
df1.describe()
df2.describe()
dy_mean = df1['DAILY_YIELD'].mean()

print(f'The mean value of Daily Yield is {dy_mean}')
df2['IRRADIATION'].sum()
df2.groupby('DATE')["IRRADIATION"].sum()
df2['AMBIENT_TEMPERATURE'].max()
df2['MODULE_TEMPERATURE'].max()
df1['SOURCE_KEY'].nunique()
df1['DC_POWER'].max()
df1['DC_POWER'].min()
df1['AC_POWER'].max()
df1['AC_POWER'].min()
a = df1.groupby('DATE')["DC_POWER"].min()

print(a)
b = df1.groupby('DATE')["DC_POWER"].max()

print(b)
c = df1.groupby('DATE')["AC_POWER"].max()

print(c)
d = df1.groupby('DATE')["AC_POWER"].min()

print(d)
df1[df1['DC_POWER'] == df1['DC_POWER'].max()]['SOURCE_KEY']
s= df1.groupby('SOURCE_KEY').sum()

s['AC_POWER'].sort_values()
q= df1.groupby('SOURCE_KEY').sum()

q['DC_POWER'].sort_values()
#Yes

22*4*24
df1
df1['DATE'].value_counts()
import pandas as pd



df2 = pd.read_csv('../input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv')

df2
df2.info()
df2['DATE_TIME'] = pd.to_datetime(df2['DATE_TIME'],format = '%Y-%m-%d %H:%M')   #datetime 

df2['DATE_TIME']
import matplotlib.pyplot as plt 
#module temp daily basis

plt.figure(figsize=(20,10))

plt.plot(df2['DATE_TIME'],df2['MODULE_TEMPERATURE'],label ='Module Temperature')

plt.legend()

plt.xlabel('DATE TIME')

plt.ylabel('MODULE TEMPERATURE')

plt.xticks(rotation = 90)

plt.grid()

plt.margins(0.05)

plt.title("Module VS Date_time")

plt.show()

#ambient temp daily basis

plt.figure(figsize=(20,10))

plt.plot(df2['DATE_TIME'],df2['AMBIENT_TEMPERATURE'],label ='Ambient Temperature')

plt.legend()

plt.xlabel('DATE TIME')

plt.ylabel('AMBIENT TEMPERATURE')

plt.xticks(rotation = 90)

plt.grid()

plt.margins(0.05)

plt.title("Ambient VS Date_time")

plt.show()
#Difference in Temp 

plt.figure(figsize=(20,10))

plt.plot(df2['DATE_TIME'],df2['MODULE_TEMPERATURE']-df2['AMBIENT_TEMPERATURE'], label ='Difference' , c ='r')

plt.legend()

plt.xlabel('DATE TIME')

plt.ylabel('MODULE - AMBIENT ')

plt.xticks(rotation = 90)

plt.grid()

plt.margins(0.05)

plt.title("Module - Ambient VS Date_time")

plt.show()
#RED for ambient   # Blue for module   # black for Difference 

plt.figure(figsize=(20,10))

plt.plot(df2['DATE_TIME'],df2['AMBIENT_TEMPERATURE'], label ='AMBIENT' , c ='r')

plt.plot(df2['DATE_TIME'],df2['MODULE_TEMPERATURE'], label ='MODULE' )

plt.plot(df2['DATE_TIME'],df2['MODULE_TEMPERATURE']-df2['AMBIENT_TEMPERATURE'], label ='Difference' , c ='k')

plt.legend()

plt.grid()

plt.show()
#RED for ambient   # Blue for module   # black for Difference 

plt.figure(figsize=(20,10))

plt.plot(df2['DATE_TIME'],df2['AMBIENT_TEMPERATURE'].rolling(window=20).mean(), label ='AMBIENT' , c ='r')

plt.plot(df2['DATE_TIME'],df2['MODULE_TEMPERATURE'].rolling(window=20).mean(), label ='MODULE' )

plt.plot(df2['DATE_TIME'],(df2['MODULE_TEMPERATURE']-df2['AMBIENT_TEMPERATURE']).rolling(window=20).mean(), label ='Difference' , c ='k')

plt.legend()

plt.grid()

plt.show()


import numpy as np

a = np.random.randint(1,500,250)

b = np.random.randint(500,1000,250)

plt.plot(a,b, marker ='o' ,linestyle ='')

plt.show()
plt.figure(figsize=(18,9))

plt.plot(df2['AMBIENT_TEMPERATURE'],df2['MODULE_TEMPERATURE'],marker = 'o' ,linestyle = '', alpha = 0.5, ms=10)
df2.info()
df2['DATE'] = pd.to_datetime(df2['DATE_TIME'],format = '%Y-%m-%d %H:%M').dt.date

df2['DATE'] = pd.to_datetime(df2['DATE'])
dates =df2['DATE'].unique()

dates
dates[0]
df2.info()
#plot irradiation > 0 

data = df2[df2['DATE']==dates[0]][df2['IRRADIATION']>0]

data
data = df2[df2['DATE']==dates[0]][df2['IRRADIATION']>0]

plt.plot(data['AMBIENT_TEMPERATURE'],data['MODULE_TEMPERATURE'], marker = 'o', linestyle ='', alpha = 0.5, label = pd.to_datetime(date,format = '%Y-%m-%d').date())



plt.legend()

plt.show()
plt.figure(figsize=(19,9))

for date in dates:

    data = df2[df2['DATE']==date][df2['IRRADIATION']>0]

    plt.plot(data['AMBIENT_TEMPERATURE'],data['MODULE_TEMPERATURE'], marker = 'o', linestyle ='', alpha = 0.5, ms= 6,label = pd.to_datetime(date,format = '%Y-%m-%d').date())



plt.legend()

plt.show()
df2.columns
df2
#listing the inverters

inv_lst = df1['SOURCE_KEY'].unique()

inv_lst
#using groupby, max total yield for each inverter
TOTAL = df1.groupby('SOURCE_KEY')['TOTAL_YIELD'].max()

TOTAL
#plot bar graph inverters vs total yield

plt.figure(figsize=(20,10))

plt.bar(inv_lst,TOTAL)

plt.xticks(rotation=90)

plt.show()

#annotate or text 
df1.info()
df2.info()
#2 Dataframes

#Left  df1 : 68778 rows, 8 columns

#Right  df2 : 3182 rows, 7 columns

#output r_left : 68778 rows × 15 columns 14(15-1)

#LEFT MERGE 
r_left = pd.merge(df1,df2, on ='DATE_TIME', how='left')

r_left
r_left.info()
#check the amount null values in merged dataframe

r_left.isnull().sum()

#ambient temp

r_left['AMBIENT_TEMPERATURE'].isnull().value_counts()

r_left['MODULE_TEMPERATURE'].isnull().value_counts()

r_left['AC_POWER'].isnull().value_counts()
null_data1 = r_left[r_left.isnull().any(axis=1)]

null_data1
plt.figure(figsize=(20,10))

plt.plot(r_left['IRRADIATION'],r_left['DC_POWER'], marker='o',linestyle='',alpha =0.5,label='DC POWER')

plt.legend()

plt.show()
plt.figure(figsize=(20,10))

plt.plot(r_left['MODULE_TEMPERATURE'],r_left['DC_POWER'], marker='o',linestyle='',alpha =0.5,label='DC POWER')

plt.legend()

plt.show()
dates = r_left['DATE_x'].unique()

dates
r_left[r_left['DATE_x']==dates[0]][r_left['IRRADIATION' ]>0.1]
r_left.info()
data = r_left[r_left['DATE_x']==dates[0]][r_left['IRRADIATION' ]>0.1]

plt.plot(data['MODULE_TEMPERATURE'],data['DC_POWER'], marker='o',linestyle='',label = pd.to_datetime(dates[0],format = '%Y-%m-%d').date())

plt.legend()

plt.show()
plt.figure(figsize=(20,10))

for date in dates:

    data = r_left[r_left['DATE_x']==date][r_left['IRRADIATION']>0.1]

    plt.plot(data['MODULE_TEMPERATURE'],data['DC_POWER'],marker='o',linestyle='',label = pd.to_datetime(date,format='%Y-%m-%d').date())

plt.legend()

plt.xlabel('Module Temperature')

plt.ylabel('DC Power')

plt.title('MODULE TEMPERATURE VS DC POWER')

plt.show()
data_summary = df1.groupby(['SOURCE_KEY','DATE']).agg(READINGS = ('TOTAL_YIELD','count'),  

                                                      INV = ('SOURCE_KEY',max), 

                                                      DATE = ('DATE',max))

data_summary
r_left.info()

#Linear Regression
r_left.info()

#Linear Regression

# x - inpput    independent variable [irradiation]     12th column  

# y - output     dependent variable [dc power]    3rd column
#Fill irradiation column with 0 for all null values (NaN and NaT)

r_left['IRRADIATION'] = r_left['IRRADIATION'].fillna(0)

r_left.info()
#x = r_left.iloc[:,[12]].values      #taking all the values of irradiation column

x = r_left.iloc[:,12:13].values

x.ndim
y = r_left.iloc[:,3].values      #taking all the values of DC_POWER column

y
import matplotlib.pyplot as plt

plt.scatter(x,y,s=1)
from sklearn.model_selection import train_test_split

x_train , x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state=0)
x_train.shape
x_test.shape
55022+13756
y_train.shape
y_test.shape
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)      #y_pred is from the ML model of Regression

y_pred  
y_test      #y_test is from the output which was given initially (actual)
#Visualization  of TRAINING DATA

plt.scatter(x_train,y_train,s=1)

plt.scatter(x_train,model.predict(x_train),s=1)

plt.show()
#testing data

plt.scatter(x_test,y_test,s=1)

plt.scatter(x_test,model.predict(x_test),s=1)

plt.show()
plt.scatter(y_pred,y_test,s=1)    #final outcome (actual vs predicted)
df3 = pd.DataFrame({'Actual':y_test, 'Predicted': y_pred,'Difference in %': ((y_pred-y_test)/(y_test)*100)})

df3
df3.plot()
#METHOD 1 using sklearn library 

from sklearn.metrics import r2_score,mean_squared_error

print(mean_squared_error(y_pred,y_test))
#METHOD 2 using numpy (manually finding the value)

import numpy as np 

MSE = np.square(np.subtract(y_pred,y_test)).mean()

MSE
r2_score(y_pred,y_test)    
#Forecasting 
day_summary = df2.groupby('DATE').agg(TOTAL_IRRADIANCE=('IRRADIATION',sum),DATE=('DATE',max))

day_summary
# datetime DATE ds   TOTAL_IRRADIANCE  y

day_summary = day_summary.rename(columns={'DATE':'ds', 'TOTAL_IRRADIANCE':'y'})

day_summary.info()
#importting fbprophet

#to avoid to overfitting or underfitting of the model we use changepoint_proir_scale

import fbprophet

op = fbprophet.Prophet(changepoint_prior_scale=0.25)

op.fit(day_summary)
#make a future possible prediction[DATAFRAME] for 1month(period) fred D-days

forecast = op.make_future_dataframe(periods = 30, freq ='D')

forecast = op.predict(forecast)
op.plot(forecast, xlabel ='Date', ylabel = 'IRRADIATION')

plt.title("Irradiation Prediction")
#new dataframe for hours

df_new = df2[['DATE_TIME','IRRADIATION']]

df_new
df_new = df_new.rename(columns = {'DATE_TIME':'ds','IRRADIATION':'y'})

df_new.info()
import fbprophet

op = fbprophet.Prophet(changepoint_prior_scale=0.25)

op.fit(df_new)
forecast = op.make_future_dataframe(periods=300, freq = 'H')

forecast = op.predict(forecast)
op.plot(forecast, xlabel='Date',ylabel='Irradiation',figsize=(20,10))

plt.title('Irradiation Prediction');
op.plot_components(forecast);
forecast