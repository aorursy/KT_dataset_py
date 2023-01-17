# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



#Common Model Algorithms

from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process

from xgboost import XGBClassifier

from xgboost import XGBRegressor



#Common Model Helpers

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn import feature_selection

from sklearn.model_selection import KFold

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn import model_selection



#Visualization

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

import seaborn as sns

from pandas.plotting import scatter_matrix



#Configure Visualization Defaults

#%matplotlib inline = show plots in Jupyter Notebook browser

%matplotlib inline

mpl.style.use('ggplot')

sns.set_style('white')

pylab.rcParams['figure.figsize'] = 12,8



import pandas as pd

import numpy as np



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
gps_log= pd.read_csv('../input/ybs-21/2017-09-01.csv',parse_dates=['Time'])
gps_log.head()
#check out the null values in gps_log

gps_log.isnull().sum()
#laoding the busstio list inot pandas dataframe

bus_stops=pd.read_csv('../input/ybs-21/YBS21Bus_Stop_List.csv')
bus_stops.info
gps_log.columns
to_drop=['ID','Address','Alarm State', 'Device State',  'Fuel(%)', 'Fuel(L)', 'Main Battery(℃)', 'GPS Signal', 'GSM Signal']#
data_log=gps_log.drop(columns=to_drop,axis=1)
data_log.shape
data_log.head()
BusList=data_log['Asset Name'].unique()
BusList
df622=data_log[(data_log.Time.dt.hour>=6) & (data_log.Time.dt.hour<=22)]
df622.shape
df622=df622.rename(index=str,columns={'Asset Name':'BusNo','Asset Status':'EngineStatus','Speed(km/h)':'Speed','Mileage(km)':'Mileage','Direction(°)':'Direction','Longitude(°)':'Longitude','Latitude(°)':'Latitude'})
df622.head()
BusList=df622['BusNo'].unique()

BusList
df622.shape
df1=df622.loc[df622.BusNo==BusList[0]]
df1.shape
df2=df622.loc[df622.BusNo==BusList[1]]
df2.shape
df1.isnull().sum()
df1[['Time','EngineStatus','Direction','Speed']]
#Encode EngienStatus using LabelEncoder

label = LabelEncoder() 

df1['EngineStatus_Code'] = label.fit_transform(df1['EngineStatus'])
df1['EngineStatus_Code'].value_counts()
df1.drop(df1[df1.EngineStatus_Code== 0].index,inplace=True)
df1.EngineStatus_Code
#time conversion for further processing

df1['Time']=pd.to_datetime(df1['Time'])
#Splitting Time column into days,week, month, year, hour, minute, second

df1['WeekDay'] = df1.Time.dt.weekday

df1['Month'] = df1.Time.dt.month

df1['Year'] = df1.Time.dt.year

df1['Hour'] = df1.Time.dt.hour

df1['Minute'] = df1.Time.dt.minute
df1['WeekEND'] =( df1.WeekDay //5==1).astype(float)
df1
#calc_distance is a function to calculate distance between adjicent coordinates 

from itertools import tee

from haversine import haversine

#create pairwise function to make a pair from two consetutive rows

def pairwise(iterable):

    a, b = tee(iterable)

    next(b, None)

    return zip(a, b)



dist=[0]



# Loop through each row in the data frame using pairwise



for (i1, row1), (i2, row2) in pairwise(df1.iterrows()):

      #Assign latitude and longitude as origin/departure points

    LatOrigin = row1['Latitude'] 

    LongOrigin = row1['Longitude']

    origin = (LatOrigin,LongOrigin)



      #Assign latitude and longitude from the next row as the destination point

    LatDest = row2['Latitude']   # Save value as lat

    LongDest = row2['Longitude'] # Save value as lat

    destination = (LatDest,LongDest)

    

    result= haversine(origin, destination)



    dist.append(result)
df1['Distance']=dist
df1.head()
plt.figure(figsize=[16,12])



#Speed Box Plot to learn distribution of speed

plt.subplot(131)

plt.boxplot(x=df1['Speed'], showmeans = True, meanline = True)

plt.title('Speed Boxplot')

plt.ylabel('(km/h)')



#Disatance Box Plot to learn distribution

plt.subplot(132)

plt.boxplot(x=df1['Distance'], showmeans = True, meanline = True)

plt.title('Distacne Boxplot')

plt.ylabel('(km)')



#Disatance Box Plot to learn distribution

plt.subplot(133)

plt.boxplot(x=df1['Direction'], showmeans = True, meanline = True)

plt.title('Direction Boxplot')

plt.ylabel('(measured from pure north)')

#Speed(km/h)vs Time(Hour) plot

sns.set_color_codes()

axis1 = sns.lineplot(x='Hour',y='Speed', data=df1)

axis1.set_title('Speed vs Time')
#Speed vs Distance

sns.set_color_codes()

axis2 = sns.lineplot(x='Speed',y='Distance', data=df1)

axis2.set_title('Speed vs Distance')

plt.figure(figsize = (20,5))

sns.boxplot(df1.Speed)

plt.show()

df1.loc[df1.Speed <= 2 ]
#Find out Speed Distribution



df1.Speed.groupby(pd.cut(df1.Speed, np.arange(0,100,10))).count().plot(kind = 'barh')

plt.xlabel('Count')

plt.ylabel('Speed (Km/H)')

plt.show()



#correlation heatmap of dataset

def correlation_heatmap(df1):

    _ , ax = plt.subplots(figsize =(14, 12))

    colormap = sns.diverging_palette(220, 10, as_cmap = True)

    _ = sns.heatmap(

        df1.corr(), 

        cmap = colormap,

        square=True, 

        cbar_kws={'shrink':.9 }, 

        ax=ax,

        annot=True, 

        linewidths=0.1,vmax=1.0, linecolor='white',

        annot_kws={'fontsize':12 }

    )

    plt.title('ybs-21 Correlation of Features', y=1.05, size=15)



correlation_heatmap(df1)
df1['Speed'].mean()

def cal_traffic(row):

    if row.Speed<=10:

        return 'Heavy'

    elif row.Speed>10 and row.Speed<20:

        return 'Modreate'

    elif row.Speed>=20 and row.Speed<35:

        return 'Normal'

    else:

        return 'Free'

df1['Traffic'] = df1.apply(cal_traffic, axis=1)
df1.loc[df1.Speed >= 30]
df1['Traffic_Code']=label.fit_transform(df1['Traffic'])
tc=df1['Traffic_Code'].unique()
print(tc)
correlation_heatmap(df1)
df1
drop={'EngineStatus','Traffic','BusNo', 'Time','WeekDay'}

df1=df1.drop(columns=drop,axis=1)
df1[df1["EngineStatus_Code"]==1]
import xgboost as xgb

X= df1[['Mileage','Direction','Distance','WeekEND']]

y=df1['Speed']

def train_test_model (df1):

    #X,y = df1.loc[:,df1.columns != 'Speed'], df1.loc[:,'Speed']

    #Use Train Test Split First

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

    # fit model no training data

    #XBG_Hyper_params = objective='reg:linear',colsample_bytree=0.3,learning_rate=0.1,max_depth=5,alpha=10,n_estimators=10

    model = XGBRegressor(objective='reg:linear',learning_rate=0.1,max_depth=5,alpha=10,n_estimators=10)

    model.fit(X_train, y_train)

    # make predictions for test data

    y_pred = model.predict(data=X_test)

    predictions = [round(value) for value in y_pred]

    # evaluate predictions

    accuracy = metrics.accuracy_score(y_test, predictions)

    print("Accuracy: %.2f%%" % (accuracy * 100.0))

    print("Standard Deviation:%.2f" % (y_pred.std()*100))
train_test_model(df1)
#X,y = df1.loc[:,df1.columns != 'Traffic_Code'], df1.loc[:,'Traffic_Code']
def KFold_model(df1):

    #X,y = df1.loc[:,df1.columns != 'Traffic_Code'], df1.loc[:,'Traffic_Code']

    model = xgb.XGBRegressor()

    kfold = model_selection.KFold(n_splits=10, random_state=7)

    results = model_selection.cross_val_score(model, X, y, cv=kfold)

    print("Accuracy: %.2f " % (results.mean()*100))

    print("Standard Deviation:%.2f" % (results.std()*100))

    print(results)
KFold_model(df1)
df_1=df1[:2000]

df_2=df1[2000:]
KFold_model(df_1)
train_test_model(df_1)