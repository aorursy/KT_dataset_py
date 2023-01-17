# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

from datetime import date, timedelta
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # Scatter plots, Histogram, Bar Charts
import seaborn as sns # Heatmap
from plotly.offline import init_notebook_mode, iplot#Contours
init_notebook_mode(connected=True) 
import plotly.graph_objs as go 




# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_pgen1 = pd.read_csv('../input/solar-power-generation-data/Plant_1_Generation_Data.csv')
df_pgen2 = pd.read_csv('../input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv')
df_pgen1.info()
df_pgen2.info()
df_pgen1['DATE_TIME'] = pd.to_datetime(df_pgen1['DATE_TIME'], format = '%d-%m-%Y %H:%M') # Converting Date_Time to Date format
df_pgen1['DATE'] = df_pgen1['DATE_TIME'].apply(lambda x:x.date())
df_pgen1['TIME'] = df_pgen1['DATE_TIME'].apply(lambda x:x.time())
df_pgen1['DATE'] = pd.to_datetime(df_pgen1['DATE'],format = '%Y-%m-%d')
df_pgen1['HOUR'] = pd.to_datetime(df_pgen1['TIME'],format='%H:%M:%S').dt.hour
df_pgen1['MINUTES'] = pd.to_datetime(df_pgen1['TIME'],format='%H:%M:%S').dt.minute

df_pgen2['DATE_TIME'] = pd.to_datetime(df_pgen2['DATE_TIME'], format = '%Y-%m-%d %H:%M:%S')
df_pgen2['DATE'] = df_pgen2['DATE_TIME'].apply(lambda x:x.date())
df_pgen2['TIME'] = df_pgen2['DATE_TIME'].apply(lambda x:x.time())
df_pgen2['DATE'] = pd.to_datetime(df_pgen2['DATE'],format = '%Y-%m-%d')
df_pgen2['HOUR'] = pd.to_datetime(df_pgen2['TIME'],format='%H:%M:%S').dt.hour
df_pgen2['MINUTES'] = pd.to_datetime(df_pgen2['TIME'],format='%H:%M:%S').dt.minute

merged= pd.merge(df_pgen2,df_pgen1,on= 'DATE_TIME',how='left')

print("Number of Invertors",len(df_pgen1['SOURCE_KEY'].unique()))
merged.head()

#Exploring Data 
print(df_pgen1['DC_POWER'].mean())
print(df_pgen1[df_pgen1['SOURCE_KEY'] == 'wCURE6d3bPkepu2']['DC_POWER'].mean())
df_pgen1.head()
df_pgen1.tail()
df_pgen1.value_counts()
df_pgen1['DATE_TIME'].value_counts()
df_pgen1.describe()



null_data = merged[merged.isnull().any(axis = 1)]
null_data
merged['PLANT_ID_y']= merged['PLANT_ID_y'].fillna(0)
merged['SOURCE_KEY_y']= merged['SOURCE_KEY_y'].fillna(0)
merged['DC_POWER']= merged['DC_POWER'].fillna(0)
merged['AC_POWER']= merged['AC_POWER'].fillna(0)
merged['DAILY_YIELD']= merged['DAILY_YIELD'].fillna(0)
merged['TOTAL_YIELD']= merged['TOTAL_YIELD'].fillna(0)
merged['DATE_y']= merged['DATE_y'].fillna(0)
merged['TIME_y']= merged['TIME_y'].fillna(0)
merged['HOUR_y']= merged['HOUR_y'].fillna(0)
merged['MINUTES_y']= merged['MINUTES_y'].fillna(0)
merged['IRRADIATION']= merged['IRRADIATION'].fillna(0)
merged['AMBIENT_TEMPERATURE']= merged['AMBIENT_TEMPERATURE'].fillna(0)
merged['MODULE_TEMPERATURE']= merged['MODULE_TEMPERATURE'].fillna(0)
merged.isnull().count()
_, ax = plt.subplots(1, 1, figsize=(16, 9))

ax.plot(df_pgen1.DATE_TIME,df_pgen1.DC_POWER.rolling(window=20).mean()/10,label='DC Power')
ax.plot(df_pgen1.DATE_TIME,df_pgen1.AC_POWER.rolling(window=20).mean(),label='AC Power')


ax.grid()
ax.margins(0.05)
ax.legend()


plt.title('AC_POWER vs DC Power')
plt.xlabel('Date and Time')
plt.ylabel('POWER')
plt.show()
import plotly.graph_objects as go

fig = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = ((df_pgen1['AC_POWER'].sum()/(df_pgen1['DC_POWER'].sum()/10))*100),
    domain = {'x': [0, 1], 'y': [0, 1]},
    title = {'text': "AC Power generated from DC Power %"},
    gauge = {'axis': {'range': [None, 100]},
            'steps' : [
                 {'range': [0, 50], 'color': "#DC143C"},
                 {'range': [50, 95], 'color': "#FF8C00"},
                {'range': [80, 95], 'color': "yellow"},
                 {'range': [95, 100], 'color': "lightgreen"},
            ]}
))
    

fig.show()
iplot([go.Histogram2dContour(x=df_pgen1.head(10000)['AC_POWER'], 
                             y=df_pgen1.head(10000)['DC_POWER'], 
                             contours=go.Contours(coloring='heatmap')),
       go.Scatter(x=df_pgen1.head(20000)['AC_POWER'], y=df_pgen1.head(20000)['DC_POWER'], mode='markers')])
_, ax = plt.subplots(1, 1, figsize=(20, 9))
ax.plot(
    df_pgen2["MODULE_TEMPERATURE"], df_pgen2["IRRADIATION"], "o--",
    linestyle='',
    alpha=0.75, label="Irradiation",
    
)



#plt.plot(1, 1, figsize=(18, 9))
#plt.figsize(1800, 900)
plt.ylabel('Irradiation')
plt.xlabel('Module Temperature')
plt.legend()
plt.show()
#ax = plt_init()
_, ax = plt.subplots(1, 1, figsize=(20, 9))
ax.plot(
    df_pgen2["DATE_TIME"], df_pgen2["IRRADIATION"], "o--",
    alpha=0.75, label="Irradiation",
    
)
#min_max_scaler = MinMaxScaler()
ax.plot(
    df_pgen2["DATE_TIME"], df_pgen2["MODULE_TEMPERATURE"]/60, 
    alpha=0.75, label="Relative Module Temperature"
    
)


#plt.plot(1, 1, figsize=(18, 9))
#plt.figsize(1800, 900)
plt.legend()
plt.show()
#Line Graph to visualize how Module Temperature varies with Irradiation for Plant 1 for 34 days


plt.plot(df_pgen2.IRRADIATION,
        df_pgen2.MODULE_TEMPERATURE.rolling(window=4).mean(),
         marker='o',
         linestyle='',
        label='MODULE TEMPERATURE'
       )


ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Line Graph showing How Module Temperature varies with Irradiation for 34 Days for Plant 1')
plt.xlabel('Irradiation')
plt.ylabel('Module Temperature')
plt.show()
_, ax = plt.subplots(1, 1, figsize=(20, 9))
ax.plot(
    merged["MODULE_TEMPERATURE"],merged.DC_POWER.rolling(window=20).mean()/(12000/60), "o--",
    linestyle='',
    alpha=0.75, label="",
    
)



#plt.plot(1, 1, figsize=(18, 9))
#plt.figsize(1800, 900)


plt.legend()
plt.show()

_, ax = plt.subplots(1, 1, figsize=(16, 9))

ax.plot(df_pgen1.DATE_TIME,df_pgen1.DC_POWER.rolling(window=20).mean()/(12000/60),label='Relative DC_POWER')
ax.plot(df_pgen2.DATE_TIME,df_pgen2.MODULE_TEMPERATURE.rolling(window=20).mean(),label='MODULE')



ax.grid()
ax.margins(0.05)
ax.legend()


plt.title('Relative DC Power vs Module Temperature over a period of 34 Days')
plt.xlabel('Date and Time')
plt.ylabel('Temperature')
plt.show()
iplot([go.Histogram2dContour(y=df_pgen2.head(10000)['MODULE_TEMPERATURE'], 
                             x=df_pgen2.head(10000)['DATE_TIME'], 
                             contours=go.Contours(coloring='heatmap')),
       go.Scatter(y=df_pgen2.head(20000)['MODULE_TEMPERATURE'], x=df_pgen2.head(20000)['DATE_TIME'], mode='markers')])
#Plot bar graph of sourcekey vs total yield for a particular inverter
plt.figure(figsize= (20,10))
inv_lst= df_pgen1['SOURCE_KEY'].unique()
plt.bar(inv_lst,df_pgen1.groupby('SOURCE_KEY')['TOTAL_YIELD'].max())


plt.xticks(rotation = 45)
plt.grid()
plt.show()

df_pgen1['AC_POWER'].argmax() 
print("Plant 1:")

print("Maximum Total Yield:", df_pgen1['SOURCE_KEY'].values[df_pgen1['TOTAL_YIELD'].argmax()])
print("Minimum Total Yield:", df_pgen1['SOURCE_KEY'].values[df_pgen1['TOTAL_YIELD'].argmin()])

plt.plot(merged['IRRADIATION'],merged['DC_POWER'],c='cyan',marker ='o',linestyle='',alpha = 0.07,label ='DC POWER')
plt.legend()
plt.xlabel('irradiation')
plt.ylabel('dc power')
plt.show()
dates = df_pgen2['DATE'].unique()

_, ax = plt.subplots(1, 1, figsize=(18, 9))

for date in dates:
    df_data = df_pgen2[df_pgen2['DATE']==date]

    ax.plot(df_data.AMBIENT_TEMPERATURE,
            df_data.MODULE_TEMPERATURE,
            marker='.',
            linestyle='',
            alpha=.5,
            ms=10,
            label=pd.to_datetime(date,format='%Y-%m-%d').date()
           )

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Scatter Plot for Module Temperature vs Ambient Temperature for 34 Days for Plant 1')
plt.xlabel('Ambient Temperature')
plt.ylabel('Module Temperature')
plt.show()
data = merged[merged['DATE_x']== dates[1]][merged['IRRADIATION']>0.1]
plt.plot(data['MODULE_TEMPERATURE'],data['DC_POWER'],marker ='o',linestyle='',label = pd.to_datetime(dates[1],format='%Y-%m-%d').date)
plt.legend()
merged.info()

X = merged.iloc[:,5:6].values   #Irradiation
y =merged.iloc[:,12].values        #DC POWER


plt.scatter(X,y)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 0)
y_train.shape
y_test.shape
from sklearn.linear_model import LinearRegression
lin_reg =LinearRegression()
lin_reg.fit(X_train,y_train) 
y_pred =lin_reg.predict(X_test)
y_pred
plt.scatter(X,y)
plt.scatter(X_test,y_test,color ='blue')
plt.scatter(X_test,y_pred,color ='red')
plt.ylabel("DC Power")
plt.xlabel("Irradiation")
plt.legend()
plt.show()
print("Slope:",lin_reg.coef_ )  #slope  m
print("Y Intercept:",lin_reg.intercept_)  #y intercept
#we are create on program wherein we should get one url for our website
#flask library
#flask is a web application framework written in Python


!pip install flask-ngrok
#-m pip install --upgrade pip
#ngrok will make local url tunnel to a global url
#Create a website
from flask_ngrok import run_with_ngrok
from flask import Flask

app = Flask(__name__)
run_with_ngrok(app)

@app.route('/')
def home():
    return "Hi welcome to ML Model,           Please input a value in the path of the URL"

@app.route('/<float:x>')
def ml(x):
    b = ("Input: "+ str(x) +"\n" )
    a = ("DC Power: "+ str(lin_reg.predict([[x]])))
    b = b+" "+a
    return (b)
app.run()