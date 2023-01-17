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
#Import packages 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns 
import csv
import os
import datetime as dt
#Load 1 sample of energy consumption readings 
#determine the type of each columns
data1 = pd.read_csv('/kaggle/input/london/Power-Networks-LCL-June2015(withAcornGps)v2_1.csv')
data2 = pd.read_csv('/kaggle/input/london/Power-Networks-LCL-June2015(withAcornGps)v2_2.csv')
data3 = pd.read_csv('/kaggle/input/london/Power-Networks-LCL-June2015(withAcornGps)v2_3.csv')
data4 = pd.read_csv('/kaggle/input/smartmeter-energy-consumption-data/Power-Networks-LCL-June2015(withAcornGps)v2_4.csv')
data5 = pd.read_csv('/kaggle/input/smartmeter-energy-consumption-data/Power-Networks-LCL-June2015(withAcornGps)v2_5.csv')
data6 = pd.read_csv('/kaggle/input/smartmeter-energy-consumption-data/Power-Networks-LCL-June2015(withAcornGps)v2_6.csv')
data7 = pd.read_csv('/kaggle/input/smartmeter-energy-consumption-data/Power-Networks-LCL-June2015(withAcornGps)v2_7.csv')
data8 = pd.read_csv('/kaggle/input/smartmeter-energy-consumption-data/Power-Networks-LCL-June2015(withAcornGps)v2_8.csv')
data9 = pd.read_csv('/kaggle/input/smartmeter-energy-consumption-data/Power-Networks-LCL-June2015(withAcornGps)v2_9.csv')
data10 = pd.read_csv('/kaggle/input/smartmeter-energy-consumption-data/Power-Networks-LCL-June2015(withAcornGps)v2_10.csv')
data11 = pd.read_csv('/kaggle/input/smartmeter-energy-consumption-data/Power-Networks-LCL-June2015(withAcornGps)v2_11.csv')
data12 = pd.read_csv('/kaggle/input/smartmeter-energy-consumption-data/Power-Networks-LCL-June2015(withAcornGps)v2_12.csv')
data13 = pd.read_csv('/kaggle/input/smartmeter-energy-consumption-data/Power-Networks-LCL-June2015(withAcornGps)v2_13.csv')
data14 = pd.read_csv('/kaggle/input/smartmeter-energy-consumption-data/Power-Networks-LCL-June2015(withAcornGps)v2_14.csv')
data15 = pd.read_csv('/kaggle/input/smartmeter-energy-consumption-data/Power-Networks-LCL-June2015(withAcornGps)v2_15.csv')
data16 = pd.read_csv('/kaggle/input/smartmeter-energy-consumption-data/Power-Networks-LCL-June2015(withAcornGps)v2_16.csv')
data17= pd.read_csv('/kaggle/input/smartmeter-energy-consumption-data/Power-Networks-LCL-June2015(withAcornGps)v2_17.csv')
data18 = pd.read_csv('/kaggle/input/smartmeter-energy-consumption-data/Power-Networks-LCL-June2015(withAcornGps)v2_18.csv')
data19= pd.read_csv('/kaggle/input/smartmeter-energy-consumption-data/Power-Networks-LCL-June2015(withAcornGps)v2_19.csv')
data20= pd.read_csv('/kaggle/input/smartmeter-energy-consumption-data/Power-Networks-LCL-June2015(withAcornGps)v2_20.csv')
data21= pd.read_csv('/kaggle/input/smartmeter-energy-consumption-data/Power-Networks-LCL-June2015(withAcornGps)v2_21.csv')
data22= pd.read_csv('/kaggle/input/smartmeter-energy-consumption-data/Power-Networks-LCL-June2015(withAcornGps)v2_22.csv')
data23= pd.read_csv('/kaggle/input/smartmeter-energy-consumption-data/Power-Networks-LCL-June2015(withAcornGps)v2_23.csv')
data24= pd.read_csv('/kaggle/input/smartmeter-energy-consumption-data/Power-Networks-LCL-June2015(withAcornGps)v2_24.csv')
data25= pd.read_csv('/kaggle/input/smartmeter-energy-consumption-data/Power-Networks-LCL-June2015(withAcornGps)v2_25.csv')
data26= pd.read_csv('/kaggle/input/smartmeter-energy-consumption-data/Power-Networks-LCL-June2015(withAcornGps)v2_26.csv')
data27= pd.read_csv('/kaggle/input/smartmeter-energy-consumption-data/Power-Networks-LCL-June2015(withAcornGps)v2_27.csv')
data28= pd.read_csv('/kaggle/input/smartmeter-energy-consumption-data/Power-Networks-LCL-June2015(withAcornGps)v2_28.csv')
data29= pd.read_csv('/kaggle/input/smartmeter-energy-consumption-data/Power-Networks-LCL-June2015(withAcornGps)v2_29.csv')
data30= pd.read_csv('/kaggle/input/smartmeter-energy-consumption-data/Power-Networks-LCL-June2015(withAcornGps)v2_30.csv')
data  = pd.concat([data1, data2,data3,data4,data5,data6,data7,data8,data9,data10,data11,data12,data13,data14,data15,data16,data17,data18,data19,data20,data21,data22,data23,data24,data25,data26,data27,data28,data29,data30], axis=0)

print(len(data.LCLid.unique()))

data[data.LCLid == 'MAC000002'].iloc[0, :]  
data[data.LCLid == 'MAC001140'].iloc[-1, :] 
print("shape",data.shape)
print("dimension",data.ndim)
data.dtypes
keep_col=['LCLid','DateTime','KWH/hh (per half hour) ']
new_data = data[keep_col]
new_data.to_csv("partie1.csv", index=False)
data = pd.read_csv('partie1.csv')
data.head()
#Convert the column consumption
#Step1 we have to change the name of this column
data.rename(columns={"KWH/hh (per half hour) ":"KWH/hh"}, inplace=True)
#Step2 There is a value Null in the data base, we can't convert it to string so we have to delete it
data=data[data["KWH/hh"] != 'Null']
#Step3 save
#data.to_csv("partie2.csv",index=False)
#data= pd.read_csv('partie2.csv')
#Step4 convert to float
data['KWH/hh'].astype(float)
#convert date time
data["DateTime"]=pd.to_datetime(data['DateTime'])
data.dtypes
data['Date'] = data['DateTime'].dt.date
data['Year'] = data['DateTime'].dt.year
data['Month'] = data['DateTime'].dt.month
data['Day'] = data['DateTime'].dt.day
data['Hour'] = data['DateTime'].dt.hour
data['Minute'] = data['DateTime'].dt.minute
data['Second'] = data['DateTime'].dt.second
#classer les date selon weekday 0:'Monday', 1:'Tuesday', 2:'Wednesday', 3:'Thursday', 4:'Friday', 5:'Saturday', 6:'Sunday'
import datetime as dt
data['weekday'] = data['DateTime'].dt.dayofweek
#1: 'Late Night', 2: 'Early Morning', 3: 'Morning', 4: 'Noon', 5: 'Evening', 6: 'Night'
data['period'] = (data['DateTime'].dt.hour % 24 + 4) // 4
import datetime

def season_of_date(date):
    
    year = date.year
    thresholds = {'autumn': datetime.date(year, 9, 1),
                  'summer': datetime.date(year, 6, 1),
                  'spring': datetime.date(year, 3, 1)}
    for season, threshold in thresholds.items():
        if date >= threshold:
            return season
    return 'winter'

#verify that the dataset has the attribute DateTime !
data['season'] = data.DateTime.map(season_of_date)
data['KWH/hh'] = data['KWH/hh'].astype('float')

LCLid=data.LCLid.unique() 
data['KWH/hh'] = data['KWH/hh'].astype('float')    
consommation=data.groupby('LCLid')['KWH/hh'].sum()
consommation_min=data.groupby('LCLid')['KWH/hh'].min()
consommation_max=data.groupby('LCLid')['KWH/hh'].max()
consommation_mean=data.groupby('LCLid')['KWH/hh'].mean()
consommation_evening=data.groupby(['LCLid','period'])['KWH/hh'].mean()

#creation de dataframe
list_of_tuples = list(zip(LCLid,consommation,consommation_min,consommation_max,consommation_mean)) 
data2= pd.DataFrame(list_of_tuples, columns = ['LCLid', 'Consumption','Consumption_min','Consumption_max','Consumption_mean'])  
data2
#Period
#mean_saison
d=data.groupby(['LCLid','season'])['KWH/hh'].mean().reset_index()
winter=d[d.season=='winter']
autumn=d[d.season=='autumn']
summer=d[d.season=='summer']
spring=d[d.season=='spring']
winter.rename(columns={"KWH/hh": "avg cons Winter"}, inplace=True)
autumn.rename(columns={"KWH/hh": "avg cons Autumn"}, inplace=True)
summer.rename(columns={"KWH/hh": "avg cons Summer"}, inplace=True)
spring.rename(columns={"KWH/hh": "avg cons Spring"}, inplace=True)
winter=winter[['LCLid','avg cons Winter']]
autumn=autumn[['LCLid','avg cons Autumn']]
summer=summer[['LCLid','avg cons Summer']]
spring=spring[['LCLid','avg cons Spring']]
data2=pd.merge(data2,winter,on='LCLid')
data2=pd.merge(data2,autumn,on='LCLid')
data2=pd.merge(data2,summer,on='LCLid')
data2=pd.merge(data2,spring,on='LCLid')
print (len(data2.LCLid.unique()))
p=data.groupby(['LCLid','period'])['KWH/hh'].mean().reset_index()
p1=p[p.period ==1]
p11=p[p.period ==2]
p111=p[p.period ==3]
p1111=p[p.period ==4]
p11111=p[p.period ==5]
p111111=p[p.period ==6]
p2=p1[['LCLid','KWH/hh']]
p22=p11[['LCLid','KWH/hh']]
p222=p111[['LCLid','KWH/hh']]
p2222=p1111[['LCLid','KWH/hh']]
p22222=p11111[['LCLid','KWH/hh']]
p222222=p111111[['LCLid','KWH/hh']]
p2.rename(columns={"KWH/hh": "avg cons Late Night"}, inplace=True)
p22.rename(columns={"KWH/hh": "avg cons Early Morning"}, inplace=True)
p222.rename(columns={"KWH/hh" : "avg cons Morning"}, inplace=True)
p2222.rename(columns={"KWH/hh" : "avg cons Noon"}, inplace=True)
p22222.rename(columns={"KWH/hh" : "avg cons Evening"} , inplace =True)
p222222.rename(columns={"KWH/hh" : "avg cons Night"} , inplace=True)

data2=pd.merge(data2,p2,on='LCLid')
data2=pd.merge(data2,p22,on='LCLid')
data2=pd.merge(data2,p222,on='LCLid')
data2=pd.merge(data2,p2222,on='LCLid')
data2=pd.merge(data2,p22222,on='LCLid')
data2=pd.merge(data2,p222222,on='LCLid')
data2
d=data.groupby(['LCLid','season'])['KWH/hh'].min().reset_index()
winter=d[d.season=='winter']
autumn=d[d.season=='autumn']
summer=d[d.season=='summer']
spring=d[d.season=='spring']
winter.rename(columns={"KWH/hh": "max cons Winter"}, inplace=True)
autumn.rename(columns={"KWH/hh": "max cons Autumn"}, inplace=True)
summer.rename(columns={"KWH/hh": "max cons Summer"}, inplace=True)
spring.rename(columns={"KWH/hh": "max cons Spring"}, inplace=True)
winter=winter[['LCLid','max cons Winter']]
autumn=autumn[['LCLid','max cons Autumn']]
summer=summer[['LCLid','max cons Summer']]
spring=spring[['LCLid','max cons Spring']]
data2=pd.merge(data2,winter,on='LCLid')
data2=pd.merge(data2,autumn,on='LCLid')
data2=pd.merge(data2,spring,on='LCLid')
data2=pd.merge(data2,summer,on='LCLid')
data2
#les weekends et weekdays
#classer les date selon weekday 0:'Monday', 1:'Tuesday', 2:'Wednesday', 3:'Thursday', 4:'Friday', 5:'Saturday', 6:'Sunday'
Weekdays=data.groupby(['LCLid','weekday'])['KWH/hh'].mean().reset_index()

Mondays=Weekdays[Weekdays.weekday==0]
Mondays.rename(columns={"KWH/hh": 'avg cons Mondays'}, inplace=True)
Mondays=Mondays[['LCLid','avg cons Mondays']]
data2=pd.merge(data2,Mondays,on='LCLid')

Tuesdays=Weekdays[Weekdays.weekday==1]
Tuesdays.rename(columns={"KWH/hh" : 'avg cons Tuesday'} , inplace=True)
Tuesdays=Tuesdays[['LCLid','avg cons Tuesday']]
data2=pd.merge(data2,Tuesdays,on='LCLid')

Wednesdays=Weekdays[Weekdays.weekday==2]
Wednesdays.rename(columns={"KWH/hh" : 'avg cons Wednesday'} , inplace=True)
Wednesdays=Wednesdays[['LCLid','avg cons Wednesday']]
data2=pd.merge(data2,Wednesdays,on='LCLid')

Thursdays=Weekdays[Weekdays.weekday==3]
Thursdays.rename(columns={"KWH/hh" : 'avg cons Thursdays'} , inplace=True)
Thursdays=Thursdays[['LCLid','avg cons Thursdays']]
data2=pd.merge(data2,Thursdays,on='LCLid')

Fridays=Weekdays[Weekdays.weekday==4]
Fridays.rename(columns={"KWH/hh" : 'avg cons Fridays'} , inplace=True)
Fridays=Fridays[['LCLid','avg cons Fridays']]
data2=pd.merge(data2,Fridays,on='LCLid')

Saturdays=Weekdays[Weekdays.weekday==5]
Saturdays.rename(columns={"KWH/hh": 'avg cons Saturdays'}, inplace=True)
Saturdays=Saturdays[['LCLid','avg cons Saturdays']]
data2=pd.merge(data2,Saturdays,on='LCLid')

Sundays=Weekdays[Weekdays.weekday==6]
Sundays.rename(columns={"KWH/hh": 'avg cons Sundays'}, inplace=True)
Sundays=Sundays[['LCLid','avg cons Sundays']]
data2=pd.merge(data2,Sundays,on='LCLid')

Weekdays=data.groupby(['LCLid','weekday'])['KWH/hh'].min().reset_index()
Weekdays_min=Weekdays.groupby(['LCLid'])['KWH/hh'].min().reset_index()
Weekdays_min.rename(columns={'KWH/hh':'min cons in Days'},inplace=True)
data2=pd.merge(data2,Weekdays_min)

Weekdays=data.groupby(['LCLid','weekday'])['KWH/hh'].max().reset_index()
Weekdays_max=Weekdays.groupby(['LCLid'])['KWH/hh'].max().reset_index()
Weekdays_max.rename(columns={'KWH/hh':'max cons in Days'},inplace=True)
data2=pd.merge(data2,Weekdays_max)


#min_January_December
p=data.groupby(['LCLid','Month'])['KWH/hh'].min().reset_index()

January=p[p.Month==1]
January.rename(columns={'KWH/hh': 'min cons January'},inplace=True)
January=January[['LCLid','min cons January']]
data2=pd.merge(data2,January,on='LCLid')

July=p[p.Month==7]
July.rename(columns={'KWH/hh': 'min cons July'},inplace=True)
July=July[['LCLid','min cons July']]
data2=pd.merge(data2,July,on='LCLid')

December=p[p.Month==12]
December.rename(columns={'KWH/hh': 'min cons December'},inplace=True)
December=December[['LCLid','min cons December']]
data2=pd.merge(data2,December,on='LCLid')

#min_Monthly_Per_Consumer

m=p.groupby(['LCLid'])['KWH/hh'].min().reset_index()
m.rename(columns={'KWH/hh' : 'min cons in Months'},inplace=True)
data2=pd.merge(data2,m)

#max_Monthly_Per_Consumer

m=p.groupby(['LCLid'])['KWH/hh'].max().reset_index()
m.rename(columns={'KWH/hh' : 'max cons in Months'},inplace=True)
data2=pd.merge(data2,m)

#mean_Monthly_Per_Consumer

m=p.groupby(['LCLid'])['KWH/hh'].mean().reset_index()
m.rename(columns={'KWH/hh' : 'mean cons in Months'},inplace=True)
data2=pd.merge(data2,m)

data2['Consumption']=data2['Consumption']/max(data2['Consumption'])

data2.columns
len(data2.LCLid)
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(data2.drop(columns=['LCLid']))

Nc = range(1, 20)
kmeans = [KMeans(n_clusters=i) for i in Nc]
score = [kmeans[i].fit(X).score(X) for i in range(len(kmeans))]

plt.plot(Nc,score)
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Elbow Curve')
plt.show()






#Consumption, periods of the day and weekdays

keep_col=['Consumption','avg cons Late Night', 'avg cons Early Morning','avg cons Morning', 'avg cons Noon', 'avg cons Evening','avg cons Night'] 
X = data2[keep_col]

#implement k-means
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=6)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)
plt.scatter(X['avg cons Late Night'], X['avg cons Early Morning'], c=y_kmeans, s=50, cmap='viridis')

plt.legend()
plt.xlabel('avg cons Late Night')
plt.ylabel('avg cons Early Morning')

ax = plt.axes(projection='3d')
ax.scatter3D(X['avg cons Late Night'], X['avg cons Early Morning'],X['avg cons Evening'], c=y_kmeans, s=50, cmap='viridis');
plt.figure(figsize=(4.,35.))

ax.legend()
ax.set_zlabel('avg cons Late Night')
ax.set_xlabel('avg cons Early Morning')
ax.set_ylabel('avg cons Evening')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 1], centers[:, 0], c='black', s=200, alpha=0.5)

clustered_data = pd.concat([data2,pd.DataFrame(kmeans.predict(X))],axis=1).sort_values(by=[0])
clustered_data = clustered_data.rename(columns={0: 'Cluster'})
clustered_data.head()
d=clustered_data.copy()
cluster0=d[d.Cluster==0]
cluster1=d[d.Cluster==1]
cluster2=d[d.Cluster==2]
cluster3=d[d.Cluster==3]
cluster4=d[d.Cluster==4]
cluster5=d[d.Cluster==5]

mac0=cluster0['LCLid']
mac1=cluster1['LCLid']
mac2=cluster2['LCLid']
mac3=cluster3['LCLid']
mac4=cluster4['LCLid']
mac5=cluster5['LCLid']

l0=len(mac0)
l1=len(mac1)
l2=len(mac2)
l3=len(mac3)
l4=len(mac4)
l5=len(mac5)
lt=l0+l1+l2+l3+l4+l5

p0=round(l0/lt*100,1)
p1=round(l1/lt*100,1)
p2=round(l2/lt*100,1)
p3=round(l3/lt*100,1)
p4=round(l4/lt*100,1)
p5=round(l5/lt*100,1)

data10=pd.merge(data,cluster0,how='inner',on='LCLid')
data11=pd.merge(data,cluster1,how='inner',on='LCLid')
data12=pd.merge(data,cluster2,how='inner',on='LCLid')
data13=pd.merge(data,cluster3,how='inner',on='LCLid')
data14=pd.merge(data,cluster4,how='inner',on='LCLid')
data15=pd.merge(data,cluster5,how='inner',on='LCLid')

p0=round(l0/lt*100,1)
p1=round(l1/lt*100,1)
p2=round(l2/lt*100,1)
p3=round(l3/lt*100,1)
p4=round(l4/lt*100,1)
p5=round(l5/lt*100,1)

data10=data10.groupby(['Day'])['KWH/hh'].mean().reset_index()
data10.rename(columns={'KWH/hh':"cluster0"+':'+ str(p0)+'%'}, inplace=True)
plt.figure(figsize=(20,5))
data10.set_index('Day',inplace = True)
sns.lineplot(data=data10,palette=['m'])

data11=data11.groupby(['Day'])['KWH/hh'].mean().reset_index()
data11.rename(columns={'KWH/hh':"cluster1"+':'+str(p1)+'%'}, inplace=True)
data11.set_index('Day',inplace = True)
sns.lineplot(data=data11, palette=['red'],linewidth=2.5)

data12=data12.groupby(['Day'])['KWH/hh'].mean().reset_index()
data12.rename(columns={'KWH/hh':"cluster2"+':'+str(p2)+'%'}, inplace=True)
data12.set_index('Day',inplace = True)
sns.lineplot(data=data12,palette=['g'])

data13=data13.groupby(['Day'])['KWH/hh'].mean().reset_index()
data13.rename(columns={'KWH/hh':"cluster3"+':'+str(p3)+'%'}, inplace=True)
data13.set_index('Day',inplace = True)
sns.lineplot(data=data13,palette=['magenta'])

data14=data14.groupby(['Day'])['KWH/hh'].mean().reset_index()
data14.rename(columns={'KWH/hh':"cluster4"+':'+str(p4)+'%'}, inplace=True)
data14.set_index('Day',inplace = True)
sns.lineplot(data=data14,palette=['blue'])

data15=data15.groupby(['Day'])['KWH/hh'].mean().reset_index()
data15.rename(columns={'KWH/hh':"cluster5"+':'+str(p5)+'%'}, inplace=True)
data15.set_index('Day',inplace = True)
sns.lineplot(data=data15,palette=['black'])

plt.rc('figure', titlesize=10)
plt.title('Load Diagram')
data10=pd.merge(data,cluster0,how='inner',on='LCLid')
data11=pd.merge(data,cluster1,how='inner',on='LCLid')
data12=pd.merge(data,cluster2,how='inner',on='LCLid')
data13=pd.merge(data,cluster3,how='inner',on='LCLid')
data14=pd.merge(data,cluster4,how='inner',on='LCLid')
data15=pd.merge(data,cluster5,how='inner',on='LCLid')

data10=data10.groupby(['Hour'])['KWH/hh'].mean().reset_index()
data10.rename(columns={'KWH/hh':"cluster0"}, inplace=True)
plt.figure(figsize=(20,5))
data10.set_index('Hour',inplace = True)
sns.lineplot(data=data10,palette=['m'])

data11=data11.groupby(['Hour'])['KWH/hh'].mean().reset_index()
data11.rename(columns={'KWH/hh':"cluster1"}, inplace=True)
data11.set_index('Hour',inplace = True)
sns.lineplot(data=data11, palette=['red'],linewidth=2.5)

data12=data12.groupby(['Hour'])['KWH/hh'].mean().reset_index()
data12.rename(columns={'KWH/hh':"cluster2"}, inplace=True)
data12.set_index('Hour',inplace = True)
sns.lineplot(data=data12,palette=['g'])

data13=data13.groupby(['Hour'])['KWH/hh'].mean().reset_index()
data13.rename(columns={'KWH/hh':"cluster3"}, inplace=True)
data13.set_index('Hour',inplace = True)
sns.lineplot(data=data13,palette=['magenta'])

data14=data14.groupby(['Hour'])['KWH/hh'].mean().reset_index()
data14.rename(columns={'KWH/hh':"cluster4"}, inplace=True)
data14.set_index('Hour',inplace = True)
sns.lineplot(data=data14,palette=['blue'])

data15=data15.groupby(['Hour'])['KWH/hh'].mean().reset_index()
data15.rename(columns={'KWH/hh':"cluster5"}, inplace=True)
data15.set_index('Hour',inplace = True)
sns.lineplot(data=data15,palette=['black'])

plt.rc('figure', titlesize=10)
plt.title('Load Diagram in Hours')
data10=pd.merge(data,cluster0,how='inner',on='LCLid')
data11=pd.merge(data,cluster1,how='inner',on='LCLid')
data12=pd.merge(data,cluster2,how='inner',on='LCLid')
data13=pd.merge(data,cluster3,how='inner',on='LCLid')
data14=pd.merge(data,cluster4,how='inner',on='LCLid')
data15=pd.merge(data,cluster5,how='inner',on='LCLid')

data10=data10.groupby(['Date'])['KWH/hh'].mean().reset_index()
data10.rename(columns={'KWH/hh':"cluster0"}, inplace=True)
plt.figure(figsize=(20,5))
data10.set_index('Date',inplace = True)
sns.lineplot(data=data10,palette=['m'])

data11=data11.groupby(['Date'])['KWH/hh'].mean().reset_index()
data11.rename(columns={'KWH/hh':"cluster1"}, inplace=True)
data11.set_index('Date',inplace = True)
sns.lineplot(data=data11, palette=['red'],linewidth=2.5)

data12=data12.groupby(['Date'])['KWH/hh'].mean().reset_index()
data12.rename(columns={'KWH/hh':"cluster2"}, inplace=True)
data12.set_index('Date',inplace = True)
sns.lineplot(data=data12,palette=['g'])

data13=data13.groupby(['Date'])['KWH/hh'].mean().reset_index()
data13.rename(columns={'KWH/hh':"cluster3"}, inplace=True)
data13.set_index('Date',inplace = True)
sns.lineplot(data=data13,palette=['magenta'])

data14=data14.groupby(['Date'])['KWH/hh'].mean().reset_index()
data14.rename(columns={'KWH/hh':"cluster4"}, inplace=True)
data14.set_index('Date',inplace = True)
sns.lineplot(data=data14,palette=['blue'])

data15=data15.groupby(['Date'])['KWH/hh'].mean().reset_index()
data15.rename(columns={'KWH/hh':"cluster5"}, inplace=True)
data15.set_index('Date',inplace = True)
sns.lineplot(data=data15,palette=['black'])

plt.rc('figure', titlesize=10)
plt.title('Load Diagram in Date')
data10=pd.merge(data,cluster0,how='inner',on='LCLid')
data11=pd.merge(data,cluster1,how='inner',on='LCLid')
data12=pd.merge(data,cluster2,how='inner',on='LCLid')
data13=pd.merge(data,cluster3,how='inner',on='LCLid')
data14=pd.merge(data,cluster4,how='inner',on='LCLid')
data15=pd.merge(data,cluster5,how='inner',on='LCLid')

data10=data10.groupby(['LCLid','Day'])['KWH/hh'].mean().reset_index()
plt.figure(figsize=(20,5))
for i in mac0:
 #to get the consumption of one consumer by his ID
 
 to_show=data10[data10.LCLid == i]
 #change the index by month to be able to implement lineplot
 to_show.set_index('Day',inplace = True)
 to_show=to_show['KWH/hh']
 sns.lineplot(data=to_show)
plt.title('Consumption In Cluster 0 Per Day')
plt.ylabel('KWh/hh')
plt.xlabel('days')


data10=pd.merge(data,cluster0,how='inner',on='LCLid')
data11=pd.merge(data,cluster1,how='inner',on='LCLid')
data12=pd.merge(data,cluster2,how='inner',on='LCLid')
data13=pd.merge(data,cluster3,how='inner',on='LCLid')
data14=pd.merge(data,cluster4,how='inner',on='LCLid')
data15=pd.merge(data,cluster5,how='inner',on='LCLid')



data11=data11.groupby(['LCLid','Day'])['KWH/hh'].mean().reset_index()
plt.figure(figsize=(20,5))
for i in mac1:
 #to get the consumption of one consumer by his ID
 
 to_show=data11[data11.LCLid == i]
 #change the index by month to be able to implement lineplot
 to_show.set_index('Day',inplace = True)
 to_show=to_show['KWH/hh']
 sns.lineplot(data=to_show)
plt.title('Consumption In Cluster 1 Per Day')
plt.ylabel('KWh/hh')
plt.xlabel('days')


data10=pd.merge(data,cluster0,how='inner',on='LCLid')
data11=pd.merge(data,cluster1,how='inner',on='LCLid')
data12=pd.merge(data,cluster2,how='inner',on='LCLid')
data13=pd.merge(data,cluster3,how='inner',on='LCLid')
data14=pd.merge(data,cluster4,how='inner',on='LCLid')
data15=pd.merge(data,cluster5,how='inner',on='LCLid')

data12=data12.groupby(['LCLid','Day'])['KWH/hh'].mean().reset_index()
plt.figure(figsize=(20,5))
for i in mac2:
 #to get the consumption of one consumer by his ID
 
 to_show=data12[data12.LCLid == i]
 #change the index by month to be able to implement lineplot
 to_show.set_index('Day',inplace = True)
 to_show=to_show['KWH/hh']
 sns.lineplot(data=to_show)
plt.title('Consumption In Cluster 2 Per Day')
plt.ylabel('KWh/hh')
plt.xlabel('days')


data10=pd.merge(data,cluster0,how='inner',on='LCLid')
data11=pd.merge(data,cluster1,how='inner',on='LCLid')
data12=pd.merge(data,cluster2,how='inner',on='LCLid')
data13=pd.merge(data,cluster3,how='inner',on='LCLid')
data14=pd.merge(data,cluster4,how='inner',on='LCLid')
data15=pd.merge(data,cluster5,how='inner',on='LCLid')
data13=data13.groupby(['LCLid','Day'])['KWH/hh'].mean().reset_index()
plt.figure(figsize=(20,5))
for i in mac3:
 #to get the consumption of one consumer by his ID
 
 to_show=data13[data13.LCLid == i]
 #change the index by month to be able to implement lineplot
 to_show.set_index('Day',inplace = True)
 to_show=to_show['KWH/hh']
 sns.lineplot(data=to_show)
plt.title('Consumption In Cluster 3 Per Day')
plt.ylabel('KWh/hh')
plt.xlabel('days')

data10=pd.merge(data,cluster0,how='inner',on='LCLid')
data11=pd.merge(data,cluster1,how='inner',on='LCLid')
data12=pd.merge(data,cluster2,how='inner',on='LCLid')
data13=pd.merge(data,cluster3,how='inner',on='LCLid')
data14=pd.merge(data,cluster4,how='inner',on='LCLid')
data15=pd.merge(data,cluster5,how='inner',on='LCLid')

data14=data14.groupby(['LCLid','Day'])['KWH/hh'].mean().reset_index()
plt.figure(figsize=(20,5))
for i in mac4:
 #to get the consumption of one consumer by his ID
 
 to_show=data14[data14.LCLid == i]
 #change the index by month to be able to implement lineplot
 to_show.set_index('Day',inplace = True)
 to_show=to_show['KWH/hh']
 sns.lineplot(data=to_show)
plt.title('Consumption In Cluster 4 Per Day')
plt.ylabel('KWh/hh')
plt.xlabel('days')



data10=pd.merge(data,cluster0,how='inner',on='LCLid')
data11=pd.merge(data,cluster1,how='inner',on='LCLid')
data12=pd.merge(data,cluster2,how='inner',on='LCLid')
data13=pd.merge(data,cluster3,how='inner',on='LCLid')
data14=pd.merge(data,cluster4,how='inner',on='LCLid')
data15=pd.merge(data,cluster5,how='inner',on='LCLid')

data15=data15.groupby(['LCLid','Day'])['KWH/hh'].mean().reset_index()
plt.figure(figsize=(20,5))
for i in mac5:
 #to get the consumption of one consumer by his ID
 
 to_show=data15[data15.LCLid == i]
 #change the index by month to be able to implement lineplot
 to_show.set_index('Day',inplace = True)
 to_show=to_show['KWH/hh']
 sns.lineplot(data=to_show)
plt.title('Consumption In Cluster 5 Per Day')
plt.ylabel('KWh/hh')
plt.xlabel('days')

data10=pd.merge(data,cluster0,how='inner',on='LCLid')
data11=pd.merge(data,cluster1,how='inner',on='LCLid')
data12=pd.merge(data,cluster2,how='inner',on='LCLid')
data13=pd.merge(data,cluster3,how='inner',on='LCLid')
data14=pd.merge(data,cluster4,how='inner',on='LCLid')
data15=pd.merge(data,cluster5,how='inner',on='LCLid')

data10=data10.groupby(['LCLid','Date'])['KWH/hh'].mean().reset_index()
plt.figure(figsize=(20,5))
for i in mac0:
 #to get the consumption of one consumer by his ID
 
 to_show=data10[data10.LCLid == i]
 #change the index by month to be able to implement lineplot
 to_show.set_index('Date',inplace = True)
 to_show=to_show['KWH/hh']
 sns.lineplot(data=to_show)
plt.title('Consumption In Cluster 0 Per Date')
plt.ylabel('KWh/hh')
plt.xlabel('Date')


data11=data11.groupby(['LCLid','Date'])['KWH/hh'].mean().reset_index()
plt.figure(figsize=(20,5))
for i in mac1:
 #to get the consumption of one consumer by his ID
 
 to_show=data11[data11.LCLid == i]
 #change the index by month to be able to implement lineplot
 to_show.set_index('Date',inplace = True)
 to_show=to_show['KWH/hh']
 sns.lineplot(data=to_show)
plt.title('Consumption In Cluster 1 Per Date')
plt.ylabel('KWh/hh')
plt.xlabel('Date')

data12=data12.groupby(['LCLid','Date'])['KWH/hh'].mean().reset_index()
plt.figure(figsize=(20,5))
for i in mac2:
 #to get the consumption of one consumer by his ID
 
 to_show=data12[data12.LCLid == i]
 #change the index by month to be able to implement lineplot
 to_show.set_index('Date',inplace = True)
 to_show=to_show['KWH/hh']
 sns.lineplot(data=to_show)
plt.title('Consumption In Cluster 2 Per Date')
plt.ylabel('KWh/hh')
plt.xlabel('Date')

data13=data13.groupby(['LCLid','Date'])['KWH/hh'].mean().reset_index()
plt.figure(figsize=(20,5))
for i in mac3:
 #to get the consumption of one consumer by his ID
 
 to_show=data13[data13.LCLid == i]
 #change the index by month to be able to implement lineplot
 to_show.set_index('Date',inplace = True)
 to_show=to_show['KWH/hh']
 sns.lineplot(data=to_show)
plt.title('Consumption In Cluster 3 Per Date')
plt.ylabel('KWh/hh')
plt.xlabel('Date')



data14=data14.groupby(['LCLid','Date'])['KWH/hh'].mean().reset_index()
plt.figure(figsize=(20,5))
for i in mac4:
 #to get the consumption of one consumer by his ID
 
 to_show=data14[data14.LCLid == i]
 #change the index by month to be able to implement lineplot
 to_show.set_index('Date',inplace = True)
 to_show=to_show['KWH/hh']
 sns.lineplot(data=to_show)
plt.title('Consumption In Cluster 4 Per Date')
plt.ylabel('KWh/hh')
plt.xlabel('Date')

data15=data15.groupby(['LCLid','Date'])['KWH/hh'].mean().reset_index()
plt.figure(figsize=(20,5))
for i in mac5:
 #to get the consumption of one consumer by his ID
 
 to_show=data15[data15.LCLid == i]
 #change the index by month to be able to implement lineplot
 to_show.set_index('Date',inplace = True)
 to_show=to_show['KWH/hh']
 sns.lineplot(data=to_show)
plt.title('Consumption In Cluster 5 Per Date')
plt.ylabel('KWh/hh')
plt.xlabel('Date')


data10=pd.merge(data,cluster0,how='inner',on='LCLid')
data11=pd.merge(data,cluster1,how='inner',on='LCLid')
data12=pd.merge(data,cluster2,how='inner',on='LCLid')
data13=pd.merge(data,cluster3,how='inner',on='LCLid')
data14=pd.merge(data,cluster4,how='inner',on='LCLid')
data15=pd.merge(data,cluster5,how='inner',on='LCLid')

data10=data10.groupby(['LCLid','Hour'])['KWH/hh'].mean().reset_index()
plt.figure(figsize=(20,5))
for i in mac0:
 #to get the consumption of one consumer by his ID
 
 to_show=data10[data10.LCLid == i]
 #change the index by month to be able to implement lineplot
 to_show.set_index('Hour',inplace = True)
 to_show=to_show['KWH/hh']
 sns.lineplot(data=to_show)
plt.title('Consumption In Cluster 0 Per Hour')
plt.ylabel('KWh/hh')
plt.xlabel('Hour')

data11=data11.groupby(['LCLid','Hour'])['KWH/hh'].mean().reset_index()
plt.figure(figsize=(20,5))
for i in mac1:
 #to get the consumption of one consumer by his ID
 
 to_show=data11[data11.LCLid == i]
 #change the index by month to be able to implement lineplot
 to_show.set_index('Hour',inplace = True)
 to_show=to_show['KWH/hh']
 sns.lineplot(data=to_show)
plt.title('Consumption In Cluster 1 Per Hour')
plt.ylabel('KWh/hh')
plt.xlabel('Hour')

data12=data12.groupby(['LCLid','Hour'])['KWH/hh'].mean().reset_index()
plt.figure(figsize=(20,5))
for i in mac2:
 #to get the consumption of one consumer by his ID
 
 to_show=data12[data12.LCLid == i]
 #change the index by month to be able to implement lineplot
 to_show.set_index('Hour',inplace = True)
 to_show=to_show['KWH/hh']
 sns.lineplot(data=to_show)
plt.title('Consumption In Cluster 2 Per Hour')
plt.ylabel('KWh/hh')
plt.xlabel('Hour')

data13=data13.groupby(['LCLid','Hour'])['KWH/hh'].mean().reset_index()
plt.figure(figsize=(20,5))
for i in mac3:
 #to get the consumption of one consumer by his ID
 
 to_show=data13[data13.LCLid == i]
 #change the index by month to be able to implement lineplot
 to_show.set_index('Hour',inplace = True)
 to_show=to_show['KWH/hh']
 sns.lineplot(data=to_show)
plt.title('Consumption In Cluster 3 Per Hour')
plt.ylabel('KWh/hh')
plt.xlabel('Hour')

data14=data14.groupby(['LCLid','Hour'])['KWH/hh'].mean().reset_index()
plt.figure(figsize=(20,5))
for i in mac4:
 #to get the consumption of one consumer by his ID
 
 to_show=data14[data14.LCLid == i]
 #change the index by month to be able to implement lineplot
 to_show.set_index('Hour',inplace = True)
 to_show=to_show['KWH/hh']
 sns.lineplot(data=to_show)
plt.title('Consumption In Cluster 4 Per Hour')
plt.ylabel('KWh/hh')
plt.xlabel('Hour')


data15=data15.groupby(['LCLid','Hour'])['KWH/hh'].mean().reset_index()
plt.figure(figsize=(20,5))
for i in mac5:
 #to get the consumption of one consumer by his ID
 
 to_show=data15[data15.LCLid == i]
 #change the index by month to be able to implement lineplot
 to_show.set_index('Hour',inplace = True)
 to_show=to_show['KWH/hh']
 sns.lineplot(data=to_show)
plt.title('Consumption In Cluster 5 Per Hour')
plt.ylabel('KWh/hh')
plt.xlabel('Hour')