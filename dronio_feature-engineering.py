import pandas as pd

import matplotlib.pyplot as plt

from sklearn import preprocessing

from mpl_toolkits.basemap import Basemap

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from sklearn import preprocessing

import warnings

warnings.filterwarnings('ignore')

import random as rn



from sklearn.cross_validation import train_test_split



from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestRegressor

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn import svm

data = pd.read_csv('../input/epa_co_daily_summary.csv')
data.head(1)
(data.isnull().sum()/len(data)*100).sort_values(ascending=False)
#delete missing attributes using a function 
def del_data_func(data,columns):
    for column_name in columns: del data[column_name]
del_list = data[['method_code','aqi','local_site_name','cbsa_name','parameter_code',
                 'units_of_measure','parameter_name']]
del_data_func (data, del_list) 

#look for correlations
correlation = data.corr(method='pearson')
plt.figure(figsize=(25,10))
sns.heatmap(correlation, vmax=1, square=True,  annot=True ) 
plt.show()
def gnumeric_func (data, columns):
  data[columns] = data[columns].apply(lambda x: pd.factorize(x)[0])
  return data
column_list = data.columns[6:8]
gnumeric_func (data, column_list)
data['county_name'] = data['county_name'].factorize()[0]
data['pollutant_standard'] = data['pollutant_standard'].factorize()[0]
data['event_type'] = data['event_type'].factorize()[0]
data['method_name'] = data['method_name'].factorize()[0]
data['address'] = data['address'].factorize()[0]
data['state_name'] = data['state_name'].factorize()[0]
data['county_name'] = data['county_name'].factorize()[0]
data['city_name'] = data['city_name'].factorize()[0]

data["datum1"] = data["datum"].apply(lambda x: 1 if x==1 else 0)
data["datum2"] = data["datum"].apply(lambda x: 1 if x==2 else 0)
data["datum3"] = data["datum"].apply(lambda x: 1 if x==3 else 0)
data["datum0"] = data["datum"].apply(lambda x: 1 if x==0 else 0)
data['sample_duration0'] = data['sample_duration'].apply(lambda x: 1 if x==1 else 0)
data['sample_duration1'] = data['sample_duration'].apply(lambda x: 1 if x==0 else 0)
data['season'] = data['date_local'].apply(lambda x: 'winter' if (x[5:7] =='01' or x[5:7] =='02' or x[5:7] =='12') else x)
data['season'] = data['season'].apply(lambda x: 'autumn' if (x[5:7] =='09' or x[5:7] =='10' or x[5:7] =='11') else x)
data['season'] = data['season'].apply(lambda x: 'summer' if (x[5:7] =='06' or x[5:7] =='07' or x[5:7] =='08') else x)
data['season'] = data['season'].apply(lambda x: 'spring' if (x[5:7] =='03' or x[5:7] =='04' or x[5:7] =='05') else x)
data['season'].replace("winter",1,inplace= True)
data['season'].replace("spring",2,inplace = True)
data['season'].replace("summer",3,inplace=True)
data['season'].replace("autumn",4,inplace=True)
data["winter"] = data["season"].apply(lambda x: 1 if x==1 else 0)
data["spring"] = data["season"].apply(lambda x: 1 if x==2 else 0)
data["summer"] = data["season"].apply(lambda x: 1 if x==3 else 0)
data["autumn"] = data["season"].apply(lambda x: 1 if x==4 else 0)
data['date_local'] = data['date_local'].map(lambda x: str(x)[:4])
data["1990"] = data["date_local"].apply(lambda x: 1 if x=="1990" else 0)
data["1991"] = data["date_local"].apply(lambda x: 1 if x=="1991" else 0)
data["1992"] = data["date_local"].apply(lambda x: 1 if x=="1992" else 0)
data["1993"] = data["date_local"].apply(lambda x: 1 if x=="1993" else 0)
data["1994"] = data["date_local"].apply(lambda x: 1 if x=="1994" else 0)
data["1995"] = data["date_local"].apply(lambda x: 1 if x=="1995" else 0)
data["1996"] = data["date_local"].apply(lambda x: 1 if x=="1996" else 0)
data["1997"] = data["date_local"].apply(lambda x: 1 if x=="1997" else 0)
data["1998"] = data["date_local"].apply(lambda x: 1 if x=="1998" else 0)
data["1999"] = data["date_local"].apply(lambda x: 1 if x=="1999" else 0)
data["2000"] = data["date_local"].apply(lambda x: 1 if x=="2000" else 0)
data["2001"] = data["date_local"].apply(lambda x: 1 if x=="2001" else 0)
data["2002"] = data["date_local"].apply(lambda x: 1 if x=="2002" else 0)
data["2003"] = data["date_local"].apply(lambda x: 1 if x=="2003" else 0)
data["2004"] = data["date_local"].apply(lambda x: 1 if x=="2004" else 0)
data["2005"] = data["date_local"].apply(lambda x: 1 if x=="2005" else 0)
data["2006"] = data["date_local"].apply(lambda x: 1 if x=="2006" else 0)
data["2007"] = data["date_local"].apply(lambda x: 1 if x=="2007" else 0)
data["2008"] = data["date_local"].apply(lambda x: 1 if x=="2008" else 0)
data["2009"] = data["date_local"].apply(lambda x: 1 if x=="2009" else 0)
data["2010"] = data["date_local"].apply(lambda x: 1 if x=="2010" else 0)
data["2011"] = data["date_local"].apply(lambda x: 1 if x=="2011" else 0)
data["2012"] = data["date_local"].apply(lambda x: 1 if x=="2012" else 0)
data["2013"] = data["date_local"].apply(lambda x: 1 if x=="2013" else 0)
data["2014"] = data["date_local"].apply(lambda x: 1 if x=="2014" else 0)
data["2015"] = data["date_local"].apply(lambda x: 1 if x=="2015" else 0)
data["2016"] = data["date_local"].apply(lambda x: 1 if x=="2016" else 0)
data["2017"] = data["date_local"].apply(lambda x: 1 if x=="2017" else 0)
correlation = data.corr(method='pearson')
plt.figure(figsize=(40,40))
sns.heatmap(correlation, vmax=1, square=True,  annot=True ) 
plt.show()
graph = plt.figure(figsize=(20, 10))
graph = data.groupby('date_local')['arithmetic_mean'].mean()
graph = graph.sort_values(ascending=False)
graph.plot (kind="bar",color='blue', fontsize = 15)
plt.grid(b=True, which='both', color='white',linestyle='-')
plt.axhline(y=0.8, xmin=2, xmax=0, linewidth=2, color = 'red', label = 'cc')
plt.show ()
#graph = plt.figure(figsize=(20, 10))
#graph = data.groupby(['state_name'])['arithmetic_mean'].mean()
#graph = graph.sort_values(ascending=False)
#graph.plot (kind="bar",color='blue', fontsize = 15)
#plt.grid(b=True, which='both', color='white',linestyle='-')
#plt.axhline(y=1.2, xmin=2, xmax=0, linewidth=2, color = 'red', label = 'cc')
#plt.show ();
m = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
        projection='lcc',lat_1=33,lat_2=45,lon_0=-95)
longitudes = data["longitude"].tolist()
latitudes = data["latitude"].tolist()

x,y = m(longitudes,latitudes)

fig = plt.figure(figsize=(12,10))
plt.title("Polution areas")
m.plot(x, y, "o", markersize = 3, color = 'red')

m.drawcoastlines()
m.fillcontinents(color='white',lake_color='aqua')
m.drawmapboundary()
m.drawstates()
m.drawcountries()
plt.show()    
    
X = data[['state_code','county_code','site_num','poc'
         ,'latitude','longitude','datum'
         ,'observation_count','date_local',
          'observation_percent','first_max_hour','city_name','state_name','county_name','pollutant_standard'
         ,'event_type','method_name','address','first_max_value',
         '1990','1991' 
        ,'1992','1993','1994','1995','1996','1997','1998','1999','2000','2001','2002',
        '2003','2004','2005','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015','2016','2017',
        'season']]
Y = data[['arithmetic_mean']]
X_train, X_test, Y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=11)
from sklearn.linear_model           import LinearRegression
clf = LinearRegression()
li_1=clf.fit(X_train,Y_train)
print("LinearRegression")
print(li_1.score(X_test,y_test))
data.info()
