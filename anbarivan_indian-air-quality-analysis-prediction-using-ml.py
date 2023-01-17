



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline

plt.rcParams['figure.figsize'] = (10, 7)



# Warnings

import warnings

warnings.filterwarnings('ignore')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data=pd.read_csv('../input/india-air-quality-data/data.csv',encoding="ISO-8859-1")

data.fillna(0, inplace=True)

data.head()
#Function to calculate so2 individual pollutant index(si)

def calculate_si(so2):

    si=0

    if (so2<=40):

     si= so2*(50/40)

    if (so2>40 and so2<=80):

     si= 50+(so2-40)*(50/40)

    if (so2>80 and so2<=380):

     si= 100+(so2-80)*(100/300)

    if (so2>380 and so2<=800):

     si= 200+(so2-380)*(100/800)

    if (so2>800 and so2<=1600):

     si= 300+(so2-800)*(100/800)

    if (so2>1600):

     si= 400+(so2-1600)*(100/800)

    return si

data['si']=data['so2'].apply(calculate_si)

df= data[['so2','si']]

df.head()
#Function to calculate no2 individual pollutant index(ni)

def calculate_ni(no2):

    ni=0

    if(no2<=40):

     ni= no2*50/40

    elif(no2>40 and no2<=80):

     ni= 50+(no2-14)*(50/40)

    elif(no2>80 and no2<=180):

     ni= 100+(no2-80)*(100/100)

    elif(no2>180 and no2<=280):

     ni= 200+(no2-180)*(100/100)

    elif(no2>280 and no2<=400):

     ni= 300+(no2-280)*(100/120)

    else:

     ni= 400+(no2-400)*(100/120)

    return ni

data['ni']=data['no2'].apply(calculate_ni)

df= data[['no2','ni']]

df.head()
#Function to calculate no2 individual pollutant index(rpi)

def calculate_(rspm):

    rpi=0

    if(rpi<=30):

     rpi=rpi*50/30

    elif(rpi>30 and rpi<=60):

     rpi=50+(rpi-30)*50/30

    elif(rpi>60 and rpi<=90):

     rpi=100+(rpi-60)*100/30

    elif(rpi>90 and rpi<=120):

     rpi=200+(rpi-90)*100/30

    elif(rpi>120 and rpi<=250):

     rpi=300+(rpi-120)*(100/130)

    else:

     rpi=400+(rpi-250)*(100/130)

    return rpi

data['rpi']=data['rspm'].apply(calculate_si)

df= data[['rspm','rpi']]

df.tail()

#many data values of rspm values is unawailable since it was not measure before
#Function to calculate no2 individual pollutant index(spi)

def calculate_spi(spm):

    spi=0

    if(spm<=50):

     spi=spm

    if(spm<50 and spm<=100):

     spi=spm

    elif(spm>100 and spm<=250):

     spi= 100+(spm-100)*(100/150)

    elif(spm>250 and spm<=350):

     spi=200+(spm-250)

    elif(spm>350 and spm<=450):

     spi=300+(spm-350)*(100/80)

    else:

     spi=400+(spm-430)*(100/80)

    return spi

data['spi']=data['spm'].apply(calculate_spi)

df= data[['spm','spi']]

df.tail()

#many data values of rspm values is unawailable since it was not measure before
#function to calculate the air quality index (AQI) of every data value

#its is calculated as per indian govt standards

def calculate_aqi(si,ni,spi,rpi):

    aqi=0

    if(si>ni and si>spi and si>rpi):

     aqi=si

    if(spi>si and spi>ni and spi>rpi):

     aqi=spi

    if(ni>si and ni>spi and ni>rpi):

     aqi=ni

    if(rpi>si and rpi>ni and rpi>spi):

     aqi=rpi

    return aqi

data['AQI']=data.apply(lambda x:calculate_aqi(x['si'],x['ni'],x['spi'],x['rpi']),axis=1)

df= data[['sampling_date','state','si','ni','rpi','spi','AQI']]

df.head()
df.state.unique()
state=pd.read_csv("../input/indian-states-lat-lon/lat.csv")

state.head()

df.head()
dff=pd.merge(state.set_index("state"),df.set_index("state"), right_index=True, left_index=True).reset_index()

dff.head()
from mpl_toolkits.basemap import Basemap

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')

%config InlineBackend.figure_format = 'retina'



m = Basemap(projection='mill',llcrnrlat=5,urcrnrlat=40, llcrnrlon=60,urcrnrlon=110,lat_ts=20,resolution='c')


longitudes = dff["lon"].tolist()

latitudes = dff["lat"].tolist()

#m = Basemap(width=12000000,height=9000000,projection='lcc',

            #resolution=None,lat_1=80.,lat_2=55,lat_0=80,lon_0=-107.)

x,y = m(longitudes,latitudes)
fig = plt.figure(figsize=(12,10))

plt.title("All affected areas")

m.plot(x, y, "o", markersize = 3, color = 'blue')

m.drawcoastlines()

m.fillcontinents(color='coral',lake_color='aqua')

m.drawmapboundary()

m.drawcountries()

plt.show()
#Visualization of AQI across india 



data['date'] = pd.to_datetime(data['date'],format='%Y-%m-%d') # date parse

data['year'] = data['date'].dt.year # year

data['year'] = data['year'].fillna(0.0).astype(int)

data = data[(data['year']>0)]



df = data[['AQI','year','state']].groupby(["year"]).median().reset_index().sort_values(by='year',ascending=False)

f,ax=plt.subplots(figsize=(15,10))

sns.pointplot(x='year', y='AQI', data=df)
#setting up date parameter

import warnings

import itertools

import dateutil

import statsmodels.api as sm

import matplotlib.pyplot as plt

import matplotlib.dates as mdates

import seaborn as sns

%matplotlib inline

df=data[['AQI','date']]

df["date"] = pd.to_datetime(df['date'])

df.tail(20)
#Calculating the yearly mean for the data 

df=df.set_index('date').resample('M')["AQI"].mean()

df.head()
#preprocessing the data values

data=df.reset_index(level=0, inplace=False)

data = data[np.isfinite(data['AQI'])]

data=data[data.date != '1970-01-31']

data = data.reset_index(drop=True)

data.head()
#visualizing the processed data of AQI



df=data.set_index('date')

df.sort_values(by='date',ascending=False)

df.plot(figsize=(15, 6))

plt.show()

y=df.AQI

#exctracting knowledge about data



#spliting dataframes into test and train

n = df.shape[0]

train_size = 0.65



features_dataframe = df.sort_values('date')

train = df.iloc[:int(n * train_size)]

test = df.iloc[int(n * train_size):]
#plotting the yearly variations of AQI



train.AQI.plot(figsize=(15,8), title= 'YEARLY VARIATIONS', fontsize=14)

test.AQI.plot(figsize=(15,8), title= 'YEARLY VARIATIONS', fontsize=14)

plt.show()
#Naive Forecast Approach to find the variations(trend)



dd= np.asarray(train.AQI)

y_hat = test.copy()

y_hat['naive'] = dd[len(dd)-1]

plt.figure(figsize=(12,8))

plt.plot(train.index, train['AQI'], label='Train')

plt.plot(test.index,test['AQI'], label='Test')

plt.plot(y_hat.index,y_hat['naive'], label='Naive Forecast')

plt.legend(loc='best')

plt.title("Naive Forecast",fontsize=20)



plt.legend(["actual ","predicted"])

plt.xlabel("YEAR",fontsize=20)

plt.ylabel("AQI",fontsize=20)

plt.tick_params(labelsize=20)

plt.show()

#various statmodel to identity huge variations od data values

import statsmodels.api as sm

train.index=pd.DatetimeIndex(freq="w", start=0 ,periods=224)



sm.tsa.seasonal_decompose(train.AQI).plot()

result = sm.tsa.stattools.adfuller(train.AQI)

plt.show()
#resampling the data to predict monthly AQI of india



df=data[['AQI','date']]



df['date']=pd.to_datetime(df['date'])

date=df.groupby(pd.Grouper(key='date',freq='1MS'))["AQI"].mean()

df.count()

#splitting the sampling date into month and year accordingly

data['month'] = data['date'].dt.month

data['year'] = data['date'].dt.year

data=data[['AQI','date','month','year']]

data.head()
#predicting JANUARY-AQI across india

data=data[data['month']==1]

data.head()
#Appling BOXPLOT analysis

df = data[['AQI','year']].groupby(["year"]).mean().reset_index().sort_values(by='year',ascending=False)

df=df.dropna()

dd=df

df.describe()
import seaborn as sns

sns.boxplot(x=df['AQI'])
#removing Outliers

df = df[np.isfinite(df['AQI'])]

df=df[df.AQI >153]

df=df[df.AQI <221]

#visualizing the filttered data



year=df['year'].values

AQI=df['AQI'].values

df['AQI']=pd.to_numeric(df['AQI'],errors='coerce')

df['year']=pd.to_numeric(df['year'],errors='coerce')



import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (20.0, 10.0)

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()

ax = Axes3D(fig)

ax.scatter(year,AQI, color='red')

plt.show()
#scatter plot of data points

cols =['year']

y = df['AQI']

x=df[cols]



plt.scatter(x,y)

plt.show()


x = (x - x.mean()) / x.std()

x = np.c_[np.ones(x.shape[0]), x]

x
# Applying GRADIENT DESCENT 



alpha = 0.1 #Step size

iterations = 3000 #No. of iterations

m = y.size #No. of data points

np.random.seed(4) #Setting the seed

theta = np.random.rand(2) #Picking random values to start with



def gradient_descent(x, y, theta, iterations, alpha):

    past_costs = []

    past_thetas = [theta]

    for i in range(iterations):

        prediction = np.dot(x, theta)

        error = prediction - y

        cost = 1/(2*m) * np.dot(error.T, error)

        past_costs.append(cost)

        theta = theta - (alpha * (1/m) * np.dot(x.T, error))

        past_thetas.append(theta)

        

    return past_thetas, past_costs

past_thetas, past_costs = gradient_descent(x, y, theta, iterations, alpha)

theta = past_thetas[-1]



#Printing the results...

print("Gradient Descent: {:.2f}, {:.2f}".format(theta[0], theta[1]))
#Plotting the cost function...

plt.title('Cost Function J')

plt.xlabel('No. of iterations')

plt.ylabel('Cost')

plt.plot(past_costs)

plt.show()
#Predicted val

newB=[ 200.17, -1.54]

def rmse(y,y_pred):

    rmse=np.sqrt(sum(y-y_pred))

    return rmse



   

y_pred=x.dot(newB)



dt = pd.DataFrame({'Actual': y, 'Predicted': y_pred})  

x=pd.concat([df, dt], axis=1)

x

x
#calculating the root mean squared error for the predicted AQi values

from sklearn import metrics

print(np.sqrt(metrics.mean_squared_error(y,y_pred)))
x_axis=x.year

y_axis=x.Actual

y1_axis=x.Predicted

plt.plot(x_axis,y_axis)

plt.plot(x_axis,y1_axis)

plt.title("Actual vs Predicted",fontsize=20)

plt.legend(["actual ","predicted"])

plt.xlabel("YEAR",fontsize=20)

plt.ylabel("AQI",fontsize=20)

plt.tick_params(labelsize=20)

plt.show()
#improving the accuracy by splitting the data on heavy variations



df=dd[['year','AQI']]





#huge variations aqi accures on year 2009-2010 (by moving average graph)

df=df[df.year<2011]

df.describe()
#applying boxplot analysis

import seaborn as sns

sns.boxplot(x=df['AQI'])
#removing outliers

df = df[np.isfinite(df['AQI'])]

df=df[df.AQI >200]

df=df[df.AQI <226]

df
#plotting data points

cols =['year']

y = df['AQI']

x=df[cols]



plt.scatter(x,y)

plt.show()
x = (x - x.mean()) / x.std()

x = np.c_[np.ones(x.shape[0]), x]

x
#Tunning model with GRADIENT DESCENT



alpha = 0.1 #Step size

iterations = 3000 #No. of iterations

m = y.size #No. of data points

np.random.seed(4) #Setting the seed

theta = np.random.rand(2) #Picking some random values to start with



def gradient_descent(x, y, theta, iterations, alpha):

    past_costs = []

    past_thetas = [theta]

    for i in range(iterations):

        prediction = np.dot(x, theta)

        error = prediction - y

        cost = 1/(2*m) * np.dot(error.T, error)

        past_costs.append(cost)

        theta = theta - (alpha * (1/m) * np.dot(x.T, error))

        past_thetas.append(theta)

        

    return past_thetas, past_costs



past_thetas, past_costs = gradient_descent(x, y, theta, iterations, alpha)

theta = past_thetas[-1]



#Print the results...

print("Gradient Descent: {:.2f}, {:.2f}".format(theta[0], theta[1]))
#Plotting the cost function...

plt.title('Cost Function J')

plt.xlabel('No. of iterations')

plt.ylabel('Cost')

plt.plot(past_costs)

plt.show()
#predicting january(1988-2009) AQI across india



import numpy as np

newB=[ 214.47, 1.18]



def rmse(y,y_pred):

    rmse= (np.sqrt(np.mean((y-y_pred)**2)))

    return rmse

   

y_pred=x.dot(newB)

dt = pd.DataFrame({'Actual': y, 'Predicted': y_pred})  

x=pd.concat([df, dt], axis=1)

x
#testing the accuracy of the model



from sklearn import metrics

print(np.sqrt(metrics.mean_squared_error(y,y_pred)))
x_axis=x.year

y_axis=x.Actual

y1_axis=x.Predicted

plt.plot(x_axis,y_axis)

plt.plot(x_axis,y1_axis)

plt.title("Actual vs Predicted",fontsize=20)

plt.legend(["actual ","predicted"])

plt.xlabel("YEAR",fontsize=20)

plt.ylabel("AQI",fontsize=20)

plt.tick_params(labelsize=20)

plt.show()
#improving the accuracy by splitting the data on heavy variations



df= dd[['year','AQI']]



#huge variations aqi accures on year 2009-2010 (by moving average graph)

df=df[df.year>2010]

df.describe()
#applying boxplot

import seaborn as sns

sns.boxplot(x=df['AQI'])



df = df[np.isfinite(df['AQI'])]

df=df[df.AQI >101]

df=df[df.AQI <107]

df.head()
cols =['year']

y = df['AQI']

x=df[cols]



plt.scatter(x,y)

plt.show()
x = (x - x.mean()) / x.std()

x = np.c_[np.ones(x.shape[0]), x]

x
#Tunning model with GRADIENT DESCENT



alpha = 0.1 #Step size

iterations = 3000 #No. of iterations

m = y.size #No. of data points

np.random.seed(4) #Setting the seed

theta = np.random.rand(2) #Picking some random values to start with



def gradient_descent(x, y, theta, iterations, alpha):

    past_costs = []

    past_thetas = [theta]

    for i in range(iterations):

        prediction = np.dot(x, theta)

        error = prediction - y

        cost = 1/(2*m) * np.dot(error.T, error)

        past_costs.append(cost)

        theta = theta - (alpha * (1/m) * np.dot(x.T, error))

        past_thetas.append(theta)

        

    return past_thetas, past_costs



past_thetas, past_costs = gradient_descent(x, y, theta, iterations, alpha)

theta = past_thetas[-1]



#Print the results...

print("Gradient Descent: {:.2f}, {:.2f}".format(theta[0], theta[1]))
#Plotting the cost function...

plt.title('Cost Function J',fontsize=28)

plt.xlabel('No. of iterations',fontsize=25)

plt.ylabel('Cost',fontsize=25)

plt.plot(past_costs)

plt.tick_params(labelsize=30)

plt.show()
#prediction of january(2013-2015) across india

import numpy as np

newB=[ 103.59, -2.74]



def rmse(y,y_pred):

    rmse= np.sqrt(sum(y-y_pred))

    return rmse

   

y_pred=x.dot(newB)



dt = pd.DataFrame({'Actual': y, 'Predicted': y_pred})  

x=pd.concat([df, dt], axis=1)

x
#testing the accuracy of the model



from sklearn import metrics

print(np.sqrt(metrics.mean_squared_error(y,y_pred)))
#plotting the actual and predicted results

x_axis=x.year

y_axis=x.Actual

y1_axis=x.Predicted

plt.plot(x_axis,y_axis)

plt.plot(x_axis,y1_axis)

plt.title("Actual vs Predicted",fontsize=20)

plt.legend(["actual ","predicted"])

plt.xlabel("YEAR",fontsize=20)

plt.ylabel("AQI",fontsize=20)

plt.tick_params(labelsize=20)

plt.show()
#Prediction for the future 

from sklearn.preprocessing import MinMaxScaler
#feeding in the x value-years

data=[[-1,2016],[-1,2017],[-1,2018],[-1,2019],[-1,2020]]
#normalization

scaler=MinMaxScaler(feature_range=(-1,1))

scaler.fit(data)

x=scaler.transform(data)
#calculations

newB=[103.59,-2.74]

ypred=-(x.dot(newB))

#AQI for the year 2020

print("AQI for the year 2020===>",ypred[-1])