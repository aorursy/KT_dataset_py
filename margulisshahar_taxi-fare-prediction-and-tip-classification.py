# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings
from IPython.display import Image 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from itertools import combinations
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from pandas.tools.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
warnings.filterwarnings('ignore')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
Image(filename='../input/data-for-green-taxi-note-book/GreenTaxi.png')
#reading taxi df
df_taxi= pd.read_csv('../input/data-for-green-taxi-note-book/2016_Green_Taxi_Trip_Data.csv')
#reading weather df
df_weather= pd.read_csv('../input/data-for-green-taxi-note-book/New_York_Weather_2016.csv',parse_dates=['pickup_datetime'],usecols=["pickup_datetime","icon"])

df_taxi.columns
dfColumns=[col.strip().upper() for col in df_taxi.columns]
df_taxi.columns=dfColumns
print("Green Taxi DF with NA values:")
print(df_taxi.columns[df_taxi.isna().any()].tolist())
print("Counting NA values per recognized columns with NA:")
print("PICKUP_LONGITUDE NA Valus:"+ str(df_taxi.PICKUP_LONGITUDE.isna().sum()))
print("PICKUP_LATITUDE NA Valus:"+ str(df_taxi.PICKUP_LATITUDE.isna().sum()))
print("DROPOFF_LONGITUDE NA Valus:"+str(df_taxi.DROPOFF_LONGITUDE.isna().sum()))
print("DROPOFF_LATITUDE NA Valus:"+str(df_taxi.DROPOFF_LATITUDE.isna().sum()))
print("EHAIL_FEE NA Valus:"+str(df_taxi.EHAIL_FEE.isna().sum()))
print("TRIP_TYPE NA Valus:"+str(df_taxi.TRIP_TYPE.isna().sum()))
print("PULOCATIONID NA Valus:"+str(df_taxi.PULOCATIONID.isna().sum()))
print("DOLOCATIONID NA Valus:"+str(df_taxi.DOLOCATIONID.isna().sum()))


#Removing columns with huge amount of NA values
df_taxi=df_taxi.drop(["VENDORID","STORE_AND_FWD_FLAG","PICKUP_LONGITUDE","PICKUP_LATITUDE","DROPOFF_LONGITUDE","DROPOFF_LATITUDE","EHAIL_FEE","PULOCATIONID","DOLOCATIONID"], axis=1)
#In Addition removing unnecessary columns
df_taxi=df_taxi.drop(["FARE_AMOUNT","EXTRA","MTA_TAX","TOLLS_AMOUNT","IMPROVEMENT_SURCHARGE"], axis=1)
print("Current columns with NA values")
print(df_taxi.columns[df_taxi.isna().any()].tolist())
#Removing records where TRIP_TYPE is NA
df_taxi=df_taxi.dropna()
#renaming columns
df_taxi = df_taxi.rename(columns={'RATECODEID': 'RATECODE_ID'})

df_taxi.describe().style.apply(lambda x: ["background: yellow" if v <= 0  else "" for v in x], axis = 1)
print("RATECODE_ID unique values:"+str(df_taxi.RATECODE_ID.unique()))
print("PASSENGER_COUNT unique values:"+str(df_taxi.PASSENGER_COUNT.unique()))
print("TRIP_DISTANCE unique values:"+str(df_taxi.TRIP_DISTANCE.unique()))
print("TIP_AMOUNT unique values:"+str(df_taxi.TIP_AMOUNT.unique()))
print("TOTAL_AMOUNT unique values:"+str(df_taxi.TOTAL_AMOUNT.unique()))
print("PAYMENT_TYPE unique values:"+str(df_taxi.PAYMENT_TYPE.unique()))
print("TRIP_TYPE unique values:"+str(df_taxi.TRIP_TYPE.unique()))
print("RATECODE_ID values count:")
print(df_taxi.RATECODE_ID.value_counts())
print("PASSENGER_COUNT values count:")
print(df_taxi.PASSENGER_COUNT.value_counts())
print("Num of TRIP_DISTANCE not positive values :"+str(len(df_taxi.TRIP_DISTANCE[df_taxi.TRIP_DISTANCE<=0])))
print("Num of TIP_AMOUNT negative values :"+str(len(df_taxi.TIP_AMOUNT[df_taxi.TIP_AMOUNT<0])))
print("Num of TOTAL_AMOUNT less than 2.5:"+str(len(df_taxi.TOTAL_AMOUNT[df_taxi.TOTAL_AMOUNT< 2.5])))
print(len(df_taxi.TIP_AMOUNT[df_taxi.TIP_AMOUNT<0]))
print("PAYMENT_TYPE values count:")
print(df_taxi.PAYMENT_TYPE.value_counts())
print("Number of payment type cash and tip given :"+str(len(df_taxi[(df_taxi["PAYMENT_TYPE"] == 2) & (df_taxi["TIP_AMOUNT"] > 0)])))

# cleaning some more records according findings above
df_taxi = df_taxi[(df_taxi.RATECODE_ID!=99) & (df_taxi.RATECODE_ID!=6)]
df_taxi = df_taxi[(df_taxi.PASSENGER_COUNT>0) & (df_taxi.PASSENGER_COUNT<7)]
df_taxi = df_taxi[(df_taxi.TRIP_DISTANCE>0) ]
df_taxi = df_taxi[(df_taxi.TIP_AMOUNT>=0) ]
df_taxi = df_taxi[(df_taxi.TOTAL_AMOUNT>=2.5) ]
df_taxi = df_taxi[(df_taxi.PAYMENT_TYPE<3)]
#record with payment type cash and tip amount >0 is not valid
df_taxi=df_taxi[(df_taxi["PAYMENT_TYPE"]==1) | ((df_taxi["PAYMENT_TYPE"] == 2) & (df_taxi["TIP_AMOUNT"] == 0))]


#the function gets df columns to format and formattype,and converts the columns to date type
def convertColumnsToDate(dataFrame,columns,dateFormat):
    for col in columns:
        dataFrame[col]=pd.to_datetime(dataFrame[col], format=dateFormat)
print("taxi df data types")
print(df_taxi.dtypes)
#converting date columns to date type
convertColumnsToDate(df_taxi,["LPEP_PICKUP_DATETIME","LPEP_DROPOFF_DATETIME"],"%m/%d/%Y %I:%M:%S %p")
#removing records before 2016
df_weather=df_weather[df_weather["pickup_datetime"]>='2016-01-01']
print("Columns with NA values:"+str(df_weather.columns[df_weather.isna().any()].tolist()))
df_weather=df_weather.dropna()
df_weather.columns=["DATE","WEATHER"]
print(df_weather.WEATHER.unique())
#removing unknown weather
df_weather=df_weather[df_weather.WEATHER!="unknown"]
#removing duplicate records (we want 1 record per day month hour-recognized some duplicates)
df_weather["STRING_DATE"]=df_weather.DATE
df_weather["STRING_DATE"]=df_weather.STRING_DATE.apply(lambda x:str(x))
df_weather["STRING_DATE"]=df_weather.STRING_DATE.apply(lambda x:x[:-6])
df_weather['MATCH_WITH_PREVIOUS'] = df_weather.STRING_DATE.eq(df_weather.STRING_DATE.shift())
df_weather=df_weather[df_weather["MATCH_WITH_PREVIOUS"]==False]
df_weather=df_weather.drop(['STRING_DATE', 'MATCH_WITH_PREVIOUS'], axis=1)
df_weather.head()
#Converting numeric categorical lables to string lables
df_taxi["RATECODE_ID"]=df_taxi["RATECODE_ID"].map( {1: 'StandardRate',2: 'JFK',3: 'NewWark',4: 'NassauOrWestchester',5: 'NegotiatedFare',6: 'GroupRide'} )
df_taxi["PAYMENT_TYPE"]=df_taxi["PAYMENT_TYPE"].map( {1: 'CreditCard',2: 'Cash'} )
df_taxi["TRIP_TYPE"]=df_taxi["TRIP_TYPE"].map( {1: 'StreetHail',2: 'Dispatch'} )
#converting miles to KM
df_taxi["TRIP_DISTANCE"]=df_taxi["TRIP_DISTANCE"].apply(lambda x:(x/0.62137119))


df_taxi.head()
#adding columns
df_taxi["PICKUP_MONTH"]=df_taxi["LPEP_PICKUP_DATETIME"].dt.month
df_taxi["PICKUP_DAY_OF_MONTH"]=df_taxi["LPEP_PICKUP_DATETIME"].dt.day
df_taxi["PICKUP_HOUR"]=df_taxi["LPEP_PICKUP_DATETIME"].dt.hour
df_taxi["PICKUP_DAY_OF_WEEK"]=df_taxi["LPEP_PICKUP_DATETIME"].dt.dayofweek
df_taxi["PICKUP_DAY_OF_WEEK"]=df_taxi["PICKUP_DAY_OF_WEEK"].map( {0: 'MONDAY',1: 'TUESDAY',2: 'WEDNESDAY', 3: 'THURSDAY',4: 'FRIDAY', 5: 'SATURDAY',6: 'SUNDAY'} ).astype(str)

def isAirPortTrip(x):
    if((x=="NewWark") | (x=="JFK")):
        return 1
    else:
        return 0
df_taxi["IS_AIRPORT_TRIP"]=df_taxi["RATECODE_ID"].apply(lambda x:isAirPortTrip(x))
#the function gets day of week.return 1 if day is weekend (saturday or sunday) otherwise 0
def isWeekEnd(x):
    if((x=="SATURDAY") | (x=="SUNDAY")):
        return 1
    else:
        return 0
df_taxi["IS_WEEK_END"]=df_taxi["PICKUP_DAY_OF_WEEK"].apply(lambda x:isWeekEnd(x))
df_taxi['TRIP_MINUTES'] = (df_taxi['LPEP_DROPOFF_DATETIME'] - df_taxi['LPEP_PICKUP_DATETIME'])
df_taxi['TRIP_MINUTES'] = df_taxi['TRIP_MINUTES']/np.timedelta64(1,'m')

print("Num of TRIP_MINUTES negative values :"+str(len(df_taxi.TRIP_MINUTES[df_taxi.TRIP_MINUTES<0])))
#removing revords with trip minutes negative values
df_taxi = df_taxi[(df_taxi.TRIP_MINUTES>0)]
#the function gets hour in day,return 1 if rush hour (16-20) otherwise 0

def isRushHours(x):
    if((x>=16) & (x<=20)):
        return 1
    else:
        return 0
df_taxi["IS_RUSH_HOURS"]=df_taxi["PICKUP_HOUR"].apply(lambda x:isRushHours(x))

#the function gets hour in day,return 1 if night hour (20-6) otherwise 0

def isNightHours(x):
    if((x>=20) | (x<=6)):
        return 1
    else:
        return 0
df_taxi["IS_NIGHT_HOURS"]=df_taxi["PICKUP_HOUR"].apply(lambda x:isNightHours(x))
#the function gets tip amount,return 1 if tip amount > 0 otherwise 0

def didGaveTip(x):
    if(x>0):
        return 1
    else:
        return 0
df_taxi["DID_GAVE_TIP"]=df_taxi["TIP_AMOUNT"].apply(lambda x:didGaveTip(x))
df_taxi["TIP_PCT"]=(df_taxi["TIP_AMOUNT"]/df_taxi["TOTAL_AMOUNT"])*100
df_taxi.head()
# adding month,day,hour columns for merging with taxi df
df_weather["MONTH"]=df_weather["DATE"].dt.month
df_weather["DAY"]=df_weather["DATE"].dt.day
df_weather["HOUR"]=df_weather["DATE"].dt.hour
print("Weather records per type:")
print(df_weather.groupby(['WEATHER']).size())
#removing records with type fog/sleet
df_weather=df_weather[(df_weather['WEATHER']!='fog') & (df_weather['WEATHER']!='sleet')]
#combine wether types
df_weather['WEATHER'] = df_weather['WEATHER'].map( {'rain': 'rain','snow': 'snow','clear': 'reg', 'cloudy': 'reg','hazy': 'reg', 'mostlycloudy': 'reg','partlycloudy': 'reg'} ).astype(str)
mergedDf = pd.merge(df_taxi, df_weather,  how='inner', left_on=['PICKUP_MONTH','PICKUP_DAY_OF_MONTH','PICKUP_HOUR'], right_on = ['MONTH','DAY','HOUR'])

mergedDf.head()
#droping  duplicate columns
mergedDf=mergedDf.drop(["LPEP_PICKUP_DATETIME","LPEP_DROPOFF_DATETIME","DATE","MONTH","DAY","HOUR"], axis=1)

#mean price by day of week
ax=mergedDf.groupby(['PICKUP_DAY_OF_WEEK']).mean().reindex(['SUNDAY','MONDAY','TUESDAY','WEDNESDAY','THURSDAY','FRIDAY','SATURDAY'])['TOTAL_AMOUNT'].plot.bar(figsize=(12, 5),rot=0)
ax.set_title('Total amount mean by day of week')
ax.set_ylabel('Total amount mean')
#mean price by month
ax=mergedDf.groupby(['PICKUP_MONTH']).mean()['TOTAL_AMOUNT'].plot.bar(figsize=(12, 5),rot=0)
ax.set_title('Total amount mean by month')
ax.set_ylabel('Total amount mean')
#mean price by pay type
ax=mergedDf.groupby(['PAYMENT_TYPE']).mean()['TOTAL_AMOUNT'].plot.bar(figsize=(12, 5),rot=0)
ax.set_title('Total amount mean by payment type')
ax.set_ylabel('Total amount mean')
#mean avg by hour
ax=mergedDf.groupby(['PICKUP_HOUR']).mean()['TOTAL_AMOUNT'].plot.bar(figsize=(12, 5),rot=0)
ax.set_title('Total amount mean by pickup hour')
ax.set_ylabel('Total amount mean')
ax=mergedDf.groupby(['PICKUP_HOUR']).mean()['TRIP_DISTANCE'].plot.bar(figsize=(12, 5),rot=0)
ax.set_title('Trip distance mean by pickup hour')
ax.set_ylabel('Distance mean')
# by night hours
ax=mergedDf.groupby(['IS_NIGHT_HOURS']).mean()['TOTAL_AMOUNT'].plot.bar(figsize=(12, 5),rot=0)
ax.set_title('Total amount mean by is night hour')
ax.set_ylabel('Total amount mean')

sns.factorplot('PICKUP_HOUR', 
                  'TOTAL_AMOUNT',  
                    estimator = np.mean, 
                    data = mergedDf, 
                   size = 8, 
                   aspect = 2, 
                    ci=None,
                   legend_out=False)
plt.title("Total amount by pickup hour")
sns.factorplot('PICKUP_HOUR', 
                  'TRIP_DISTANCE',  
                    estimator = np.mean, 
                    data = mergedDf, 
                   size = 8, 
                   aspect = 2, 
                    ci=None,
                   legend_out=False)
plt.title("Distance by pickup hour")
sns.factorplot('PASSENGER_COUNT', 
                  'TOTAL_AMOUNT', 
                    estimator = np.mean, 
                    data = mergedDf, 
                   size = 8, 
                   aspect = 2, 
                    ci=None,
                   legend_out=False)
plt.title("Total amount by passenger count")
sns.factorplot('PICKUP_HOUR', 
                 'TOTAL_AMOUNT', 
                   hue = 'WEATHER', 
                  estimator = np.mean, 
                   data = mergedDf, 
                   size = 8, 
                   aspect = 2, 
                    ci=None,
                   legend_out=False)
plt.title("Total amount vs pickup hour by weather")

sns.factorplot('PICKUP_MONTH', 
                 'TOTAL_AMOUNT', 
                   hue = 'PICKUP_DAY_OF_WEEK', 
                  estimator = np.mean, 
                   data = mergedDf, 
                   size = 8, 
                   aspect = 2, 
                    ci=None,
                   legend_out=False)
plt.title("Total amount vs pickup month by day of week")
print("Distance mean by ratecode")
mergedDf.groupby(['RATECODE_ID'])['TRIP_DISTANCE'].mean()
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))

bplot1 = axes[0].boxplot(mergedDf.TRIP_DISTANCE,
                          vert=True,  
                          patch_artist=True)   
axes[0].set_ylim(0, mergedDf.TRIP_DISTANCE.max())
axes[0].set_title("Trip distance box plot")
bplot2 = axes[1].boxplot(mergedDf.TRIP_DISTANCE,
                         notch=True, 
                         vert=True,  
                          patch_artist=True)  
axes[1].set_title("Zoomed in Trip distance box plot")

axes[1].set_ylim(0, 30)
mergedDf= mergedDf[mergedDf["TRIP_DISTANCE"]<30]
print("Trip minutes mean by ratecode")
mergedDf.groupby(['RATECODE_ID'])['TRIP_MINUTES'].mean()
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))

bplot1 = axes[0].boxplot(mergedDf.TRIP_MINUTES,
                          vert=True,  
                          patch_artist=True)  
axes[0].set_ylim(0, mergedDf.TRIP_MINUTES.max())
axes[0].set_title("Trip minutes box plot")
bplot2 = axes[1].boxplot(mergedDf.TRIP_MINUTES,
                         notch=True, 
                         vert=True,  
                          patch_artist=True)  
axes[1].set_title("Zoomed in Trip minutes box plot")

axes[1].set_ylim(0, 80)
mergedDf= mergedDf[mergedDf["TRIP_MINUTES"]<80]
print("Total amount mean by ratecode")
mergedDf.groupby(['RATECODE_ID'])['TOTAL_AMOUNT'].mean()
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))

bplot1 = axes[0].boxplot(mergedDf.TOTAL_AMOUNT,
                          vert=True,   
                          patch_artist=True)   
axes[0].set_ylim(0, mergedDf.TOTAL_AMOUNT.max())
axes[0].set_title("Total amount box plot")
bplot2 = axes[1].boxplot(mergedDf.TOTAL_AMOUNT,
                         notch=True,  
                         vert=True,   
                          patch_artist=True)   
axes[1].set_title("Zoomed in Total amount box plot")

axes[1].set_ylim(0, 70)
mergedDf= mergedDf[mergedDf["TOTAL_AMOUNT"]<70]
fig=plt.figure(figsize=(10,10))
ax=fig.gca()
ax.hist(mergedDf.TOTAL_AMOUNT,range=(0, 70))
#ax.set_xlim(np.arange(0, 60, step=10))
ax.set_xticks(np.arange(0, 70, step=5))
ax.set_title("Total amount histogram")

plt.show()
fig=plt.figure(figsize=(10,10))
ax=fig.gca()
ax.hist(mergedDf.TRIP_MINUTES,range=(0, 70))
#ax.set_xlim(np.arange(0, 60, step=10))
ax.set_xticks(np.arange(0, 70, step=2))
ax.set_title("Trip minutes histogram")

plt.show()
fig=plt.figure(figsize=(10,10))
ax=fig.gca()
ax.hist(mergedDf.TRIP_DISTANCE,range=(0, 70))
ax.set_xticks(np.arange(0, 70, step=5))
ax.set_title("Trip distance histogram")

plt.show()

#move down
sns.pairplot(x_vars=['TRIP_DISTANCE'], y_vars=['TRIP_MINUTES'], data=mergedDf, hue ="WEATHER", size=5)
plt.title("Trip distance vs Trip minutes by weather")

features=['PASSENGER_COUNT', 'TRIP_DISTANCE',
       'TOTAL_AMOUNT', 
       'IS_WEEK_END', 'TRIP_MINUTES', 'IS_RUSH_HOURS', 'IS_NIGHT_HOURS',
      ]
sns.set_style()
corr = mergedDf[features].corr()
sns.heatmap(corr,cmap="RdYlBu",vmin=-1,vmax=1)
plt.title("correlation heat map")
features=['PASSENGER_COUNT', 'TRIP_DISTANCE',
       'TOTAL_AMOUNT', 
       'IS_WEEK_END', 'TRIP_MINUTES', 'IS_RUSH_HOURS', 'IS_NIGHT_HOURS',
      ]
mask = np.zeros_like(mergedDf[features].corr(), dtype=np.bool) 
mask[np.triu_indices_from(mask)] = True 

f, ax = plt.subplots(figsize=(16, 12))
plt.title('Correlation Matrix',fontsize=25)

sns.heatmap(mergedDf[features].corr(),vmax=1.0,square=True,cmap="RdYlBu", 
            linecolor='w',annot=True,mask=mask,cbar_kws={"shrink": .75})
mergedDf_dummies=pd.get_dummies(mergedDf,columns=['WEATHER','PASSENGER_COUNT'])
mergedDf_dummies=pd.get_dummies(mergedDf_dummies,columns=['TRIP_TYPE','PAYMENT_TYPE'], drop_first=True)

mergedDf_dummies.head()
mergedDf_dummies = mergedDf_dummies.rename(columns={'WEATHER_reg': 'REG_WEATHER','WEATHER_rain': 'RAIN_WEATHER','WEATHER_snow': 'SNOW_WEATHER','TRIP_TYPE_StreetHail': 'IS_STREETHAIL','PAYMENT_TYPE_CreditCard': 'IS_CREDIT'})
mergedDf_dummies.columns
X = mergedDf_dummies.drop("TOTAL_AMOUNT",axis=1)
y = mergedDf_dummies["TOTAL_AMOUNT"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)
X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=100)
print("train df size="+str(len(X_train))+", valid df size="+ str(len(X_valid))+", test df size="+str(len(X_test))+ ", full df size="+ str(len(mergedDf_dummies)))
plt.subplot(1, 2, 1)
plt.title("Full Data normalized PICKUP_DAY_OF_WEEK")

pd.value_counts(mergedDf_dummies['PICKUP_DAY_OF_WEEK'].values,normalize=True).reindex(['SUNDAY','MONDAY','TUESDAY','WEDNESDAY','THURSDAY','FRIDAY','SATURDAY']).plot.bar(figsize=(12, 5))
plt.subplot(1, 2, 2)
plt.title("Train Data normalized PICKUP_DAY_OF_WEEK")

pd.value_counts(X_train['PICKUP_DAY_OF_WEEK'].values,normalize=True).reindex(['SUNDAY','MONDAY','TUESDAY','WEDNESDAY','THURSDAY','FRIDAY','SATURDAY']).plot.bar(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.title("Full Data normalized PICKUP_MONTH")

pd.value_counts(mergedDf_dummies['PICKUP_MONTH'].values,normalize=True).reindex(index = range(1,13)).plot.bar(figsize=(12, 5))
plt.subplot(1, 2, 2)
plt.title("Train Data normalized PICKUP_MONTH")

pd.value_counts(X_train['PICKUP_MONTH'].values,normalize=True).reindex(index = range(1,13)).plot.bar(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.title("Full Data normalized PICKUP_DAY_OF_MONTH")

pd.value_counts(mergedDf_dummies['PICKUP_DAY_OF_MONTH'].values,normalize=True).reindex(index = range(1,32)).plot.bar(figsize=(15,15))
plt.subplot(1, 2,2)
plt.title("Train Data normalized PICKUP_DAY_OF_MONTH")

pd.value_counts(X_train['PICKUP_DAY_OF_MONTH'].values,normalize=True).reindex(index = range(1,32)).plot.bar(figsize=(15,15))

plt.subplot(1, 2, 1)
plt.title("Full Data normalized PICKUP_HOUR")

pd.value_counts(mergedDf_dummies['PICKUP_HOUR'].values,normalize=True).reindex(index = range(0,24)).plot.bar(figsize=(15,15))

plt.subplot(1, 2, 2)
plt.title("Train Data normalized PICKUP_HOUR")

pd.value_counts(X_train['PICKUP_HOUR'].values,normalize=True).reindex(index = range(0,24)).plot.bar(figsize=(15,15))



plt.figure(figsize=(12,4))
plt.subplot(1, 2, 1)
plt.title("Full Data normalized TOTAL_AMOUNT")

plt.hist(y,bins=40);
plt.subplot(1, 2, 2)
plt.title("Train Data normalized TOTAL_AMOUNT")

plt.hist(y_train,bins=40);
modelInput=[['TRIP_DISTANCE'],
           ['IS_WEEK_END'], 
           ['TRIP_MINUTES'],
           ['IS_RUSH_HOURS'],
           ['IS_NIGHT_HOURS'],
            ['IS_AIRPORT_TRIP'],
           ['REG_WEATHER','RAIN_WEATHER', 'SNOW_WEATHER'],
           ['PASSENGER_COUNT_1','PASSENGER_COUNT_2', 'PASSENGER_COUNT_3','PASSENGER_COUNT_4', 'PASSENGER_COUNT_5','PASSENGER_COUNT_6'], 
           ['IS_STREETHAIL'],
           ['IS_CREDIT']]
#function that gets a list->returns all cobinations of list (example: for [1,2,3] will return 1,2,3,[1,3],[1,2],[2,3],[1,2,3])
def sublists(input):
    for length in range(1,len(input) + 1):
        yield from combinations(input, length)
#apply function on all features
modelsFeatures=list(sublists(modelInput))
#creating data frame the will contain scores per each model
summaryDf = pd.DataFrame(columns=['ModelColumns','MAE','MSE','RMSE','RSQUARE'])