# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import fbprophet



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
## Cell for data loading and pre-processing.



# Step 1 - loading data sets

df_pgen1 = pd.read_csv('../input/solar-power-generation-data/Plant_1_Generation_Data.csv')

df_psense1 = pd.read_csv('../input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv')



df_pgen2 = pd.read_csv('../input/solar-power-generation-data/Plant_2_Generation_Data.csv')

df_psense2 = pd.read_csv('../input/solar-power-generation-data/Plant_2_Weather_Sensor_Data.csv')



# Step 2 - correcting date_time format

df_pgen1['DATE_TIME'] = pd.to_datetime(df_pgen1['DATE_TIME'],format = '%d-%m-%Y %H:%M')

df_psense1['DATE_TIME'] = pd.to_datetime(df_psense1['DATE_TIME'],format = '%Y-%m-%d %H:%M')



df_pgen2['DATE_TIME'] = pd.to_datetime(df_pgen2['DATE_TIME'],format = '%Y-%m-%d %H:%M:%S')

df_psense2['DATE_TIME'] = pd.to_datetime(df_psense2['DATE_TIME'],format = '%Y-%m-%d %H:%M:%S')



# Step 3 - splitting date and time

df_pgen1['DATE'] = df_pgen1['DATE_TIME'].apply(lambda x:x.date())

df_pgen1['TIME'] = df_pgen1['DATE_TIME'].apply(lambda x:x.time())



df_psense1['DATE'] = df_psense1['DATE_TIME'].apply(lambda x:x.date())

df_psense1['TIME'] = df_psense1['DATE_TIME'].apply(lambda x:x.time())



df_pgen2['DATE'] = df_pgen2['DATE_TIME'].apply(lambda x:x.date())

df_pgen2['TIME'] = df_pgen2['DATE_TIME'].apply(lambda x:x.time())



df_psense2['DATE'] = df_psense2['DATE_TIME'].apply(lambda x:x.date())

df_psense2['TIME'] = df_psense2['DATE_TIME'].apply(lambda x:x.time())





# Step 4 - correcting data_time format for the DATE column

df_pgen1['DATE'] = pd.to_datetime(df_pgen1['DATE'],format = '%Y-%m-%d')

df_psense1['DATE'] = pd.to_datetime(df_psense1['DATE'],format = '%Y-%m-%d')

df_pgen2['DATE'] = pd.to_datetime(df_pgen2['DATE'],format = '%Y-%m-%d')

df_psense2['DATE'] = pd.to_datetime(df_psense2['DATE'],format = '%Y-%m-%d')



# Step 5 - splitting hour and minutes

df_pgen1['HOUR'] = pd.to_datetime(df_pgen1['TIME'],format='%H:%M:%S').dt.hour

df_pgen1['MINUTES'] = pd.to_datetime(df_pgen1['TIME'],format='%H:%M:%S').dt.minute



df_psense1['HOUR'] = pd.to_datetime(df_psense1['TIME'],format='%H:%M:%S').dt.hour

df_psense1['MINUTES'] = pd.to_datetime(df_psense1['TIME'],format='%H:%M:%S').dt.minute



df_pgen2['HOUR'] = pd.to_datetime(df_pgen2['TIME'],format='%H:%M:%S').dt.hour

df_pgen2['MINUTES'] = pd.to_datetime(df_pgen2['TIME'],format='%H:%M:%S').dt.minute



df_psense2['HOUR'] = pd.to_datetime(df_psense2['TIME'],format='%H:%M:%S').dt.hour

df_psense2['MINUTES'] = pd.to_datetime(df_psense2['TIME'],format='%H:%M:%S').dt.minute





#Changing the incorrectly noted DC Power in the first file

for i in range(len(df_pgen1.index)) : 

    df_pgen1['DC_POWER'].iloc[i] = df_pgen1['DC_POWER'].loc[i]/10
df_pgen1.head(50)
df_psense1.head(50)
df_pgen2.head(50)
df_psense2.head(50)
df_pgen1.count()
df_pgen1['SOURCE_KEY'].unique()

len(df_pgen1['SOURCE_KEY'].unique())
df_pgen1.isnull()
# Count total NaN at each row in a DataFrame 

myint = 0

type(myint)

for i in range(len(df_pgen1.index)) : 

    myint = myint +df_pgen1.iloc[i].isnull().sum()

print("Count of NAN is :"+str(myint))
df_pgen1['SOURCE_KEY'].value_counts()
df_pgen1['DATE_TIME'].value_counts()
df_pgen1.info()
df_pgen1['DATE'].value_counts()
import seaborn as sns #visualisation

plt.figure(figsize=(20,10))

c= df_pgen1.corr()

sns.heatmap(c,cmap='BrBG',annot=True)

c
import seaborn as sns #visualisation

plt.figure(figsize=(20,10))

c= df_psense1.corr()

sns.heatmap(c,cmap='BrBG',annot=True)

c
result_left = pd.merge(df_pgen1,df_psense1, on='DATE_TIME',how='left') 



import seaborn as sns #visualisation

plt.figure(figsize=(20,10))

c= result_left.corr()

sns.heatmap(c,cmap='BrBG',annot=True)

c
_, ax = plt.subplots(1, 1, figsize=(18, 9))



ax.plot(df_psense1['MODULE_TEMPERATURE'],

        df_psense1['AMBIENT_TEMPERATURE'],

        marker='o',

        linestyle='',

        alpha=.5, #transparency

        ms=3, #size of the dot

        label='Correlation Between MODULE_TEMPERATURE & AMBIENT_TEMEPRATURE')

ax.grid()

ax.margins(0.05)

ax.legend()

plt.title('Correlation Between MODULE_TEMPRATURE & AMBIENT_TEMPRATURE')

plt.xlabel('MODULE_TEMPRATURE')

plt.ylabel('AMBIENT_TEMPRATURE')

plt.show()
_, ax = plt.subplots(1, 1, figsize=(18, 9))



ax.plot(df_pgen1['DC_POWER'],

        df_pgen1['AC_POWER'],

        marker='o',

        linestyle='',

        alpha=.5, #transparency

        ms=3, #size of the dot

        label='Correlation Between DC Power & AC Power')

ax.grid()

ax.margins(0.05)

ax.legend()

plt.title('Correlation Between DC Power & AC Power')

plt.xlabel('AC_POWER')

plt.ylabel('DC_POWER')

plt.show()
_, ax = plt.subplots(1, 1, figsize=(24, 10))



ax.plot(df_psense1['HOUR'],

        df_psense1['IRRADIATION'],

        marker='o',

        linestyle='',

        alpha=.5,

        ms=10,

        label='Irradiation With Time')

ax.grid()

ax.margins(0.05)

ax.legend()

plt.title('Irradiation vs. Time')

plt.xlabel('Hour')

plt.ylabel('Irradiation')

plt.show()



df_data = df_psense1[df_psense1['DATE']=='2020-05-23T00:00:00.000000000']



_, ax = plt.subplots(1, 1, figsize=(18, 9))



#df_data = df_psense1[df_psense1['DATE']==date]#[df_psense1['IRRADIATION']>0]



df_data_irr_1 = df_data[(df_data['IRRADIATION']>0) & (df_data['IRRADIATION']<=0.5)]

df_data_irr_2 = df_data[(df_data['IRRADIATION']>0.5) & (df_data['IRRADIATION']<=1)]

df_data_irr_3 = df_data[df_data['IRRADIATION']>1]

df_data_noirr = df_data[df_data['IRRADIATION']==0]



ax.plot(df_data_irr_1.AMBIENT_TEMPERATURE,

        df_data_irr_1.MODULE_TEMPERATURE,

        marker='o',

        linestyle='',

        alpha=.5,

        ms=10,

        label='Irradiation_1'

       )



ax.plot(df_data_irr_2.AMBIENT_TEMPERATURE,

        df_data_irr_2.MODULE_TEMPERATURE,

        marker='o',

        linestyle='',

        alpha=.5,

        ms=10,

        label='Irradiation_2'

       )



ax.plot(df_data_irr_3.AMBIENT_TEMPERATURE,

        df_data_irr_3.MODULE_TEMPERATURE,

        marker='o',

        linestyle='',

        alpha=.5,

        ms=10,

        label='Irradiation_3'

       )



ax.plot(df_data_noirr.AMBIENT_TEMPERATURE,

        df_data_noirr.MODULE_TEMPERATURE,

        marker='o',

        linestyle='',

        alpha=.5,

        ms=10,

        label='No Irradiation'

       )



ax.grid()

ax.margins(0.05)

ax.legend()

plt.title('Module Tempreture vs. Ambient Tempreture')

plt.xlabel('Ambient Tempreture')

plt.ylabel('Module Tempreture')

plt.show()
_, ax = plt.subplots(1, 1, figsize=(18, 9))



ax.plot(result_left.MODULE_TEMPERATURE,

        result_left.DC_POWER,

        marker='o',

        linestyle='',

        alpha=.5,

        ms=10,

        label='DC POWER')



ax.grid()

ax.margins(0.05)

ax.legend()

plt.title('DC Power vs. Module Temperature')

plt.xlabel('Module Temperature')

plt.ylabel('DC Power')

plt.show()


    

import matplotlib.pyplot as plt



#SS Plotting How DCPower and AC Power changing with time

_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_pgen1.DATE_TIME,

        df_pgen1.DC_POWER.rolling(window=20).mean(),

        label='DC Power'

       )



ax.plot(df_pgen1.DATE_TIME,

        df_pgen1.AC_POWER.rolling(window=20).mean(),

        label='AC Power'

       )





ax.grid()

ax.margins(0.05)

ax.legend()

plt.title('DC Power and AC Power')

plt.xlabel('Date and Time')

plt.ylabel('Power')

plt.show()    
#SS Plotting How DCPower and AC Power changing with time

_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_pgen1.HOUR,

        df_pgen1.DC_POWER.rolling(window=24).mean(),

        label='DC Power'

       )



ax.plot(df_pgen1.HOUR,

        df_pgen1.AC_POWER.rolling(window=20).mean(),

        label='AC Power'

       )





ax.grid()

ax.margins(0.05)

ax.legend()

plt.title('DC Power and AC Power')

plt.xlabel('Time of Day')

plt.ylabel('Power')

plt.show()
#SS Plotting How DCPower and AC Power changing with time

_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_pgen2.HOUR,

        df_pgen2.DC_POWER.rolling(window=24).mean(),

        label='DC Power'

       )



ax.plot(df_pgen2.HOUR,

        df_pgen2.AC_POWER.rolling(window=20).mean(),

        label='AC Power'

       )





ax.grid()

ax.margins(0.05)

ax.legend()

plt.title('DC Power and AC Power')

plt.xlabel('Time of Day')

plt.ylabel('Power')

plt.show()
#SS Irradiation With Time

_, ax = plt.subplots(1, 1, figsize=(24, 10))



ax.plot(df_psense1['HOUR'],

        df_psense1['IRRADIATION'],

        marker='o',

        linestyle='',

        alpha=.5,

        ms=10,

        label='Irradiation With Time')

ax.grid()

ax.margins(0.05)

ax.legend()

plt.title('Irradiation vs. Time')

plt.xlabel('Hour')

plt.ylabel('Irradiation')

plt.show()
#SS Irradiation With Time

_, ax = plt.subplots(1, 1, figsize=(24, 10))



ax.plot(df_psense1['HOUR'],

        df_psense1['IRRADIATION'],

        marker='o',

        linestyle='',

        alpha=.5,

        ms=10,

        label='Irradiation With Time')

ax.grid()

ax.margins(0.05)

ax.legend()

plt.title('Irradiation vs. Time')

plt.xlabel('Hour')

plt.ylabel('Irradiation')

plt.show()
day_summary = df_psense1.groupby('DATE').agg(TOTAL_IRRADIANCE = ('IRRADIATION', sum),

                                         DATE = ('DATE',max)

                                        )
day_summary
#SS Irradiation With Time

_, ax = plt.subplots(1, 1, figsize=(24, 10))



ax.plot(day_summary['DATE'],

        day_summary['TOTAL_IRRADIANCE'],

        marker='o',

        linestyle='',

        alpha=.5,

        ms=10,

        label='Total Irradiation For Date')

ax.grid()

ax.margins(0.05)

ax.legend()

plt.title('Irradiation vs. Date')

plt.xlabel('DATE')

plt.ylabel('TOTAL_IRRADIANCE')

plt.show()
_, ax = plt.subplots(1, 1, figsize=(18, 9))



ax.plot(result_left.IRRADIATION,

        result_left.DC_POWER,

        marker='o',

        linestyle='',

        alpha=.5,

        ms=10,

        label='DC POWER')



ax.grid()

ax.margins(0.05)

ax.legend()

plt.title('DC Power vs. Irradiation')

plt.xlabel('Irradiation')

plt.ylabel('DC Power')

plt.show()
_, ax = plt.subplots(1, 1, figsize=(18, 9))



date = dates[4]

inverters = result_left['SOURCE_KEY_x'].unique()



for inverter in inverters:



    data = result_left[(result_left['DATE_x']==date)&(result_left['SOURCE_KEY_x']==inverter)&(result_left['IRRADIATION']>0.1)]



    ax.plot(data.MODULE_TEMPERATURE,

                data.DC_POWER,

                marker='o',

                linestyle='',

                alpha=.5,

                ms=10,

                label=inverter

               )

ax.grid()

ax.margins(0.05)

ax.legend()

plt.title('DC Power vs. Module Temperature')

plt.xlabel('Module Temperature')

plt.ylabel('DC Power')

plt.show()
result_left = pd.merge(df_pgen1,df_psense1, on='DATE_TIME',how='left')

result_left['IRRADIATION'] = result_left['IRRADIATION'].fillna(0)

result_left['AMBIENT_TEMPERATURE'] = result_left['AMBIENT_TEMPERATURE'].fillna(0)

result_left['MODULE_TEMPERATURE'] = result_left['MODULE_TEMPERATURE'].fillna(0)
from sklearn.model_selection import train_test_split 

from sklearn.linear_model import LinearRegression



# Splitting the dataset into the Training set and Test set





X = result_left[['IRRADIATION','MODULE_TEMPERATURE']]

y = result_left['DC_POWER']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)



lin_reg = LinearRegression()

lin_reg.fit(X_train, y_train)



coeff_df = pd.DataFrame(lin_reg.coef_,X.columns,columns = ['Coefficients'])

coeff_df











# Predict!



y_pred = lin_reg.predict(X_test)

# Compare



compare_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

compare_df.head(10)

# Validate



from sklearn import metrics

import numpy as np



print("Train Set Accuracy")

print('Mean Absolute Error:', metrics.mean_absolute_error(y_train, lin_reg.predict(X_train)))

print('Mean Squared Error:', metrics.mean_squared_error(y_train, lin_reg.predict(X_train)))  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, lin_reg.predict(X_train))))

print('---------------------------')

print("Test Set Accuracy")

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  

print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
from sklearn.model_selection import train_test_split 

from sklearn.linear_model import LinearRegression



# Splitting the dataset into the Training set and Test set





X = result_left[['IRRADIATION','MODULE_TEMPERATURE','HOUR_x']]

y = result_left['DC_POWER']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)



lin_reg = LinearRegression()

lin_reg.fit(X_train, y_train)



coeff_df = pd.DataFrame(lin_reg.coef_,X.columns,columns = ['Coefficients'])

coeff_df



# Predict!



y_pred = lin_reg.predict(X_test)







# Compare



compare_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

compare_df.head(10)


