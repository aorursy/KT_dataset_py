
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
from pandas import Series,DataFrame
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set_style("whitegrid")
import statistics
import random as ran
from scipy import stats
from datetime import datetime
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,mean_squared_error
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures 
from statsmodels.formula.api import ols
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
import statsmodels.api as sm
def past(dff,lo,lt,ci,fc):               #gives the past details
    lo1=lo-ci
    lo2=lo+ci
    lt1=lt-ci
    lt2=lt+ci
    lo3=[]
    lt3=[]
    fc1=[]
    time=[]
    for i in np.arange(len(dff[fc])):
        if (dff["Latitude"][i]>lt1 and  dff["Latitude"][i]<lt2) and (dff["Longitude"][i]>lo1 and dff["Longitude"][i]<lo2):
            lt3=np.append(lt3,dff["Latitude"][i])
            lo3=np.append(lo3,dff["Longitude"][i])
            fc1=np.append(fc1,dff[fc][i])
            time=np.append(time,dff["Date"][i])
    dff1=DataFrame([])
    dff1["Date"]=time
    dff1["Lattitude"]=lt3
    dff1["Longitude"]=lo3
    dff1[fc]=fc1
    return(dff1)               

def dte(lst):                               #converts date formate
    r=[]
    for i in np.arange(len(lst)):
        y=int(lst[i][6:10])
        m=int(lst[i][0:2])
        d=int(lst[i][3:5])
        dt=datetime(y,m,d)
        r=np.append(r,dt)
    return(r)
df=pd.read_csv("../input/earthquake/earthquake1.csv")
df.head(2)
df.shape
df.columns
df.dtypes
df['Type'].value_counts()
#row=['Magnitude Type','Source', 'Location Source', 'Magnitude Source', 'Status']
#for i in row:
#        value=df[i].value_counts()
#        print(value)
df1=df[["Date","Time","Latitude","Longitude","Depth","Magnitude"]]
df1.head(2)
#there are the only factors which we are considering for the prediction of the earthquake
#Data PreProcessing
n1=len(df['Date'])
p1=[];a1=df['Date'];day=[];
for no in range(n1):
    p1=a1[no]
    p3=(p1[3:5])
    day=np.append(day,p3)
df['day']=day
df['day']=df['day'].replace(to_replace='5-',value='30')
df['day']=df['day'].replace(to_replace='1-',value='31')
df['day']=df['day'].astype(int)    
df['Location Source'].value_counts()
#Max number of earthquake location is US and ISCGEM
df2=df[['Date', 'Time', 'Latitude','Type', 'Longitude','Magnitude','Source','Location Source','Magnitude Source','day']]
multi_group=df2[:1500].groupby(['day','Source','Location Source',])[['Magnitude']].count()
print(multi_group)
from mpl_toolkits.basemap import Basemap
m = Basemap(projection='cyl',llcrnrlat=-80,urcrnrlat=80, llcrnrlon=-180,urcrnrlon=180,lat_ts=20,resolution='c')
df_u=df2[:1000]
longitudes = df_u["Longitude"].tolist()
latitudes = df_u["Latitude"].tolist()
x,y = m(longitudes,latitudes)
fig = plt.figure(figsize=(12,10))
plt.title("All affected areas")
m.plot(x, y, "p", markersize = 2, color = 'blue')
m.drawcoastlines()
m.fillcontinents(color='coral',lake_color='aqua')
m.drawmapboundary()
m.drawcountries()
plt.show()

int(df1["Date"][0][6:11])
df=df1
n1=len(df['Date'])
s1=[];
d1=df['Date']
year=[];month=[]
for no in range(n1):
    s1=d1[no]
    s2=int(s1[6:10])
    s3=int(s1[0:2])
    year=np.append(year,s2)
    month=np.append(month,s3)
df['year']=year
df['month']=month
df['month']=df['month'].replace(to_replace=23.0,value=1.0)
df['month']=df['month'].replace(to_replace=28.0,value=2.0)
df['month']=df['month'].replace(to_replace=13.0,value=3.0)
change_mon = {1.0: 'Jan', 2.0: 'Feb', 3.0: 'Mar', 4.0: 'Apr', 5.0: 'May',6.0: 'Jun', 7.0: 'Jul', 8.0: 'Aug',
              9.0: 'Sep', 10.0: 'Oct', 11.0: 'Nov', 12.0: 'Dec'}
df['month'] = df['month'].apply(lambda x: change_mon[x])

df1=df[["Date","Time","Latitude","Longitude","Depth","Magnitude","year","month"]]
#plt.plot(df1['month'].value_counts())
co1=df1.groupby('month')['Magnitude'].count()
#co1.plot(figsize=(20,5))
df_m=(DataFrame([co1.values,[4,8,12,2,1,7,6,3,5,11,8,9]],index=["freq","mon"])).T
df_m1=df_m.sort_values(by=['mon'])
df_m1["prob"]=df_m1["freq"]/sum(df_m1["freq"])
df_m1.index=np.arange(1,13,1)
df_m1["prob"].plot(figsize=(20,5))
#so if any how i come to know that in which year how many quakes are going to occur than we can obtain the expected quakes in each mon
def past(dff,lo,lt,ci,fc):
    lo1=lo-ci
    lo2=lo+ci
    lt1=lt-ci
    lt2=lt+ci
    lo3=[]
    lt3=[]
    fc1=[]
    time=[]
    for i in np.arange(len(dff[fc])):
        if (dff["Latitude"][i]>lt1 and  dff["Latitude"][i]<lt2) and (dff["Longitude"][i]>lo1 and dff["Longitude"][i]<lo2):
            lt3=np.append(lt3,dff["Latitude"][i])
            lo3=np.append(lo3,dff["Longitude"][i])
            fc1=np.append(fc1,dff[fc][i])
            time=np.append(time,dff["Date"][i])
    dff1=DataFrame([])
    dff1["Date"]=time
    dff1["Lattitude"]=lt3
    dff1["Longitude"]=lo3
    dff1[fc]=fc1
    return(dff1)               
df_tem=past(df1,82,25,5,"Magnitude")
df_tem.head()                    #so this is the data frame of quakes in varanasi since 1966
plt.plot(df["Latitude"],df["Longitude"],"o")
df5050=past(df,50,50,10,"Magnitude")
df5050.head(2)
#my_date = datetime.strptime(df5050["Date"][0], "%Y-%m-%d")
lat= (list(map(int, (df["Latitude"]))))
lon= (list(map(int, (df["Longitude"]))))
x = datetime(1920, 5, 17)
y=datetime(1921, 5, 17)
print(x-y)
df5050["Date"][0][3:5]
def dte(lst):
    r=[]
    for i in np.arange(len(lst)):
        y=int(lst[i][6:10])
        m=int(lst[i][0:2])
        d=int(lst[i][3:5])
        dt=datetime(y,m,d)
        r=np.append(r,dt)
    return(r)
df5050["ndate"]=dte(df5050["Date"].values)
df5050=df5050.sort_values(by=['ndate'], ascending=True)
np.min(df5050["ndate"])
plt.plot(df5050["ndate"].values,df5050["Magnitude"].values)
#clearly no trend obtained
df5050_t=(df5050["ndate"]-df5050["ndate"].shift(1)) #the difference between consicutive quakes
df5050_t[1:]
#now from here i can have two approches for tracing patterns of occurance of quakes
#time_series 
#model_building 
#model_building_approch 
#we have to determine the number of previous quakes to be taken under consideration for future prediction (this will be optimised in future)
#int((str(df5050_t[1]))[:-14])
df5050_t1=[]
for i in np.arange(1,len(df5050_t)):
    df5050_t1=np.append(df5050_t1,int((str(df5050_t[i]))[:-14]))

#plt.plot(df5050_t[1:])
plt.plot(DataFrame(df5050_t1)[0].rolling(window=5).mean(),color="black")
#(DataFrame(df5050_t1)[0].rolling(window=3).mean())



#visualization_part







#prediction_part
#first of all we aim to predict that at what day how many quakes are expected
co=df1.groupby('year')['Magnitude'].count()
co.plot(figsize=(20,5))
df_c=DataFrame([(np.unique(year)),co],index=["year","count"]).T
sns.lmplot("year","count",df_c,order=1)
#clearly the frequency ofquakes per year is increasing per year
#so the question is how much quakes to be expected in next years
p1=0
p2=0
p3=0
p4=0
for j in np.arange(len(df_c["year"])):
    if (df_c["year"][j]>1969) and (df_c["year"][j]<=1979):
        p1=p1+(df_c["count"][j])
    elif (df_c["year"][j]>1979) and (df_c["year"][j]<=1989):
        p2=p2+(df_c["count"][j])
    elif (df_c["year"][j]>1989) and (df_c["year"][j]<=1999):
        p3=p3+(df_c["count"][j])
    elif (df_c["year"][j]>1999) and (df_c["year"][j]<=2009):
        p4=p4+(df_c["count"][j])
p=[p1,p2,p3,p4]
plt.plot(np.arange(1970,2010,10),p)
#cleary the linear trend in the increase of frequency of quakes was observed'
model_fr=LinearRegression()
y=DataFrame(p).values
x=DataFrame(np.arange(1970,2010,10)).values
model_fr.fit(x.reshape(-1,1),y.reshape(-1,1))
y_pre=model_fr.predict(x.reshape(-1,1))
DataFrame([DataFrame(y_pre)[0].values,DataFrame(y)[0].values,DataFrame(x)[0].values],index=["predicted","observed","year"]).T
ee=model_fr.predict([[2010]])  #so in year 2010 to 2020 we are expecting approximatly this frequency of quakes
ee
df_co=DataFrame(co)
t_s=0
for i in np.arange(len(df_co["Magnitude"])):
    if df_co.index[i]>2010:
        t_s=t_s+(df_co["Magnitude"].values)[i]
t_s           #these are the quakes which occured till year 2016
int(ee-t_s)              # so this much quakes are yet to occur till 2020(including 2020) after 2016
y_20=model_fr.predict([[2020]])
y_20                       #this is the frequency of quakes we are expecting in 2020 
m20=[]
for i in df_m1["prob"]:
    m20=np.append(m20,int(y_20*i))
df_m1["m_20"]=m20
df_m1["m_20"]               #so we got to know that in which mon of 2020 how many quakes are expected
#so now for the prediction of quakes per day we consider the history  of quakes at that area for last 100 days or few years