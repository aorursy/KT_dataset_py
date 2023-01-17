import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
%matplotlib inline
import statistics
import statsmodels.api as sm
from scipy import stats
import seaborn as sns; sns.set(style = "white", color_codes = True)
import plotly
plotly.tools.set_credentials_file(username='Olivia_Z', api_key='gQ5vg2hAX5tU6ZSQhFnt')
import plotly.plotly as py
df = pd.read_csv('../input/location-areacode/directory.csv')
##View the dataset columns
df.head()
#rename_columns
df.rename(columns={'Store Number':'Store_Number'},inplace = True)
df.rename(columns={'Store Name':'Store_Name'},inplace = True)
df.rename(columns={'Ownership Type':'Ownership_Type'},inplace = True)
df.rename(columns={'Street Address':'Street_Address'},inplace = True)
df.rename(columns={'Ownership Type':'Ownership_Type'},inplace = True)
df.rename(columns={'State/Province':'State_Province'},inplace = True)
df
##Number of records having complete data for each column
df.notnull().sum()
df.notnull().sum()*100/df.shape[0]
#drop columns in order to imporve efficiency 
to_drop =['Brand','Store_Number','Store_Name','Street_Address','Phone Number','Timezone']
df.drop(columns=to_drop,inplace=True)
#drop duplicate rows
df = df.drop_duplicates()
len(df)
#Which country has the largest number of starbucks stores?
df.Country.describe()
#Which city has the largest number of starbucks stores?
print(df.City.describe())
## The top ten countries with large number of Starbucks
df.Country.value_counts().head(10)
#count of category occurence in data
df.Country.value_counts().head(10).plot(kind = "pie")
plt.xlabel('Countries')
plt.ylabel('Number of stores')
#ax = fig.add_subplot(111)
plt.title("Top 10 Countries with Most Number of Starbucks Stores")
#The top ten cities with large number of Starbucks
df.City.value_counts().head(10)
#count of category occurence in data
df.City.value_counts().head(10).plot(kind = "pie")

plt.xlabel('Cities')
plt.ylabel('Number of stores')
#ax = fig.add_subplot(111)
plt.title("Top 10 Cities with Most Number of Starbucks Stores")
##Creat a new dataset for plot the choropleth Map
df1=df.Country.value_counts()
df1 = pd.DataFrame(df1)
with open('../input/alphacou/Country alpha_3.csv') as code:
    table = pd.read_table(code,sep='-',index_col=0,header = None,names=['code'])
dfc=pd.DataFrame(table)
dfc=dfc[:73]
dfc['Number_stores']= np.asarray(df1['Country'])
dfc['code'] = dfc.index
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode,iplot
init_notebook_mode(connected=True)
data = dict(type='choropleth',
            locations = dfc['code'],
            z = dfc['Number_stores'],
            text = dfc['code'], colorbar = dict(autotick = False,
            tickprefix = 'Number of stores',
            title = 'Number of Starbucks stores'),
            colorscale=[[0, 'rgb(224,255,255)'],
                       [0.01, 'rgb(166,206,227)'], [0.02, 'rgb(31,120,180)'],
                       [0.03, 'rgb(178,223,138)'], [0.05, 'rgb(51,160,44)'],
                       [0.10, 'rgb(251,154,153)'], [0.20, 'rgb(255,255,0)'],
                       [1, 'rgb(227,26,28)']], reversescale = False)

layout = dict(title='Number of Starbucks stores all over the World',
geo = dict(showframe = True, projection={'type':'Mercator'}))

choromap = dict(data = [data], layout = layout)
iplot(choromap, validate=False)
## The type of Starbucks stores all over the world
df_w=df.Ownership_Type.value_counts()
print(df_w)
df_us = df.loc[df['Country']=='US']
df_us.Ownership_Type.value_counts()
df_cn = df.loc[df['Country']=='CN']
df_cn.Ownership_Type.value_counts()
df_ca = df.loc[df['Country']=='CA']
df_ca.Ownership_Type.value_counts()
df_type = df.Ownership_Type.value_counts()
df_fran= df.loc[df["Ownership_Type"] == "Franchise"]
import pycountry
input_countries = df['Country']
countries = {}
for country in pycountry.countries:
    countries[country.alpha_2] = country.alpha_3

codes = [countries.get(country,'Unknown code') for country in input_countries]
pd.Series(codes)
df.loc[:,'Alpha_3'] = pd.Series(codes)
df_ty = df.groupby('Alpha_3').Ownership_Type.value_counts()
df_ty = pd.DataFrame(df_ty)

df_ty.rename(columns={'Ownership_Type':'Number'}, inplace = True) 
df_ty2 = df_ty.Number.apply(pd.Series)

df_ty['codes'] = df_ty.index
df_ty1 = df_ty.codes.apply(pd.Series)
df_ty2 = df_ty2.join(df_ty1, lsuffix='_df_ty2', rsuffix='_df_ty1')
df_ty2.columns
df_ty2.rename(columns={'0_df_ty2':'Number_of_type'}, inplace = True) 
df_ty2.rename(columns={'0_df_ty1':'code'}, inplace = True)
df_ty2.rename(columns={1:'type'}, inplace = True)
#
df_comowner = df_ty2.loc[df_ty2['type']=="Company Owned"]
df_com = df_comowner.sort_values(by=['Number_of_type'],ascending=False)
df_com[:10].plot(kind="bar")
plt.show()
#
df_licen = df_ty2.loc[df_ty2['type']=="Licensed"]
df_licen = df_licen.sort_values(by=['Number_of_type'],ascending=False)
df_licen[:10].plot(kind="bar")

df_franch = df_ty2.loc[df_ty2['type']=="Franchise"] 
df_franch = df_franch.sort_values(by=['Number_of_type'],ascending=False)
df_franch[:10].plot(kind="bar")
#
df_join = df_ty2.loc[df_ty2['type']=="Joint Venture"] 
df_join = df_join.sort_values(by=['Number_of_type'],ascending=False)
df_join[:10].plot(kind="bar")
# get the sub-dataset of US
df_us = df.loc[df['Country']=='US']

#add new columns to data frame "How many stores in each states"
df_us.State_Province.value_counts().head(10).plot(kind = "bar")

## CA dataset
df_us_CA = df_us[df_us['State_Province']=='CA']
df_us_CA.City.value_counts().head(30).plot(kind="bar")
df_us_CA1 = df_us_CA[df_us_CA["City"]=='San Diego']
to_drop1 =['Country','State_Province','Ownership_Type','Postcode','City','Alpha_3']
CA = df_us_CA1.drop(columns = to_drop1)
from scipy.spatial import distance
CA.index.name = 'ID'
CA = CA.values

##Function of disatnce measurements
import math
from math import radians, cos, sin, asin, sqrt
def distance_stores(lon1,lat1,lon2,lat2):
    lon1= math.radians(lon1)
    lat1= math.radians(lat1)
    lon2= math.radians(lon2)
    lat2= math.radians(lat2)
    dlon = lon2-lon1
    dlat = lat2-lat1
    a = sin(dlat/2)**2+cos(lat1)*cos(lat2)*sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    km = 6371 * c
    return km
all = []
for i in CA: 
    for j in CA[1:]: 
        all.append(distance_stores(i[0],i[1],j[0],j[1]))
len(all)
all = pd.DataFrame(all)
test = all[:127]
test.loc[:,'p1_Lon'] = pd.DataFrame(CA[:,0])
test.loc[:,'p1_Lat'] = pd.DataFrame(CA[:,1])
test.loc[:,'p2_Lon'] = pd.DataFrame(CA[1:,0])
test.loc[:,'p2_Lat'] = pd.DataFrame(CA[1:,1])
test = test[:127]
test.columns=['Distance','p1_Lon','p1_Lat','p2_Lon','p2_Lat']
San = test.sort_values(by=['Distance'])
size = (12,12)
sns.set(rc={'figure.figsize':(8,8)})
sns.scatterplot(x = 'p1_Lon',y='p1_Lat',data = San) 
test_mean= np.mean(test)
test_std = np.std(test)
plt.hist(San['Distance'])
#coords = distance.cdist(CA, CA,'euclidean')
#min_dis = np.min(coords[np.nonzero(coords)])
#itmeindex = np.where(coords==min)
#itemindex = np.asarray(itmeindex)
#df_min = pd.DataFrame(itemindex).T
#df_min.columns = ['ID1','ID2']
#CA1.index.name = 'ID1'
#df_min_1 = pd.DataFrame(df_min['ID1'])
#close = pd.merge(CA1,df_min_1,how='inner',on=['ID1'])
#close.columns=['ID','Lon','Lat']
#close_dis = distance.cdist(close,close,'euclidean')
#pd.DataFrame(close_dis)
## WA dataset
df_us_wa = df_us[df_us['State_Province']=='WA']
df_us_wa.City.value_counts().head(30).plot(kind="bar")
df_us_wa1 = df_us_wa[df_us_wa["City"]=='Seattle']
to_drop1 =['Country','State_Province','Ownership_Type','Postcode','City','Alpha_3']
sea = df_us_wa1.drop(columns = to_drop1)
from scipy.spatial import distance
sea.index.name = 'ID'
sea = sea.values
len(sea)
seattle = []
for i in sea: 
    for j in sea[1:]: 
        seattle.append(distance_stores(i[0],i[1],j[0],j[1]))
len(seattle)
seattle1 = pd.DataFrame(seattle)
seattle1 = seattle1[:142]
seattle1.loc[:,'p1_Lon'] = pd.DataFrame(sea[:,0])
seattle1.loc[:,'p1_Lat'] = pd.DataFrame(sea[:,1])
seattle1.loc[:,'p2_Lon'] = pd.DataFrame(sea[1:,0])
seattle1.loc[:,'p2_Lat'] = pd.DataFrame(sea[1:,1])
seattle1 =seattle1 [:142]
seattle1.columns=['Distance','p1_Lon','p1_Lat','p2_Lon','p2_Lat']
#seattle_dis = seattle1.sort_values(by=['Distance'])
#plt.figure(figsize=(10,10))
#plt.scatter(x = 'p1_Lon',y='p1_Lat',data =seattle_dis )
#plt.show()
sea_mean= np.mean(seattle1)
sea_std = np.std(seattle1)
plt.hist(seattle1['Distance'])
## NYC dataset
df_us_ny = df_us[df_us['State_Province']=='NY']
df_us_ny.City.value_counts().head(30).plot(kind="bar")
df_us_nyc = df_us_ny[df_us_ny["City"]=='New York']
to_drop1 =['Country','State_Province','Ownership_Type','Postcode','City','Alpha_3']
nyc = df_us_nyc.drop(columns = to_drop1)
from scipy.spatial import distance
nyc.index.name = 'ID'
nyc = nyc.values
len(nyc)
nyc_list = []
for i in nyc: 
    for j in nyc[1:]: 
        nyc_list.append(distance_stores(i[0],i[1],j[0],j[1]))
len(nyc_list)
nyc1 = pd.DataFrame(nyc_list)
nyc1 = nyc1[:207]
nyc1.loc[:,'p1_Lon'] = pd.DataFrame(nyc[:,0])
nyc1.loc[:,'p1_Lat'] = pd.DataFrame(nyc[:,1])
nyc1.loc[:,'p2_Lon'] = pd.DataFrame(nyc[1:,0])
nyc1.loc[:,'p2_Lat'] = pd.DataFrame(nyc[1:,1])
nyc1 =nyc1 [:207]
nyc1.columns=['Distance','p1_Lon','p1_Lat','p2_Lon','p2_Lat']

nyc_dis = nyc1.sort_values(by=['Distance'])
plt.figure(figsize=(10,10))
plt.scatter(x = 'p1_Lon',y='p1_Lat',data =nyc_dis )
plt.show()
nyc_mean= np.mean(nyc1)
nyc_std = np.std(nyc1)
plt.hist(nyc1['Distance'])
## TX dataset
df_us_TX = df_us[df_us['State_Province']=='TX']
df_us_TX.City.value_counts().head(30).plot(kind="bar")
df_us_TX1 = df_us_TX[df_us_TX["City"]=='Houston']
to_drop1 =['Country','State_Province','Ownership_Type','Postcode','City','Alpha_3']
HU = df_us_TX1.drop(columns = to_drop1)
from scipy.spatial import distance
HU.index.name = 'ID'
HU = HU.values
len(HU)
all_HU = []
for i in HU: 
    for j in HU[1:]: 
        all_HU.append(distance_stores(i[0],i[1],j[0],j[1]))
len(all_HU)
all_HU = pd.DataFrame(all_HU)
HU_TX = all_HU[:147]
HU_TX.loc[:,'p1_Lon'] = pd.DataFrame(HU[:,0])
HU_TX.loc[:,'p1_Lat'] = pd.DataFrame(HU[:,1])
HU_TX.loc[:,'p2_Lon'] = pd.DataFrame(HU[1:,0])
HU_TX.loc[:,'p2_Lat'] = pd.DataFrame(HU[1:,1])
HU_TX = HU_TX[:147]
HU_TX.columns=['Distance','p1_Lon','p1_Lat','p2_Lon','p2_Lat']
HU1 = HU_TX.sort_values(by=['Distance'])
size = (12,12)
sns.set(rc={'figure.figsize':(8,8)})
sns.scatterplot(x = 'p1_Lon',y='p1_Lat',data = HU1) 
HU_mean= np.mean(HU_TX)
HU_std = np.std(HU_TX)
plt.hist(HU_TX['Distance'])
# get the sub-dataset of Canada
df_ca = df.loc[df['Country']=='CA']
#add new columns to data frame "How many stores in each states"
df_ca.State_Province.value_counts().head(10).plot(kind = "bar")
## ON dataset
df_ca_on = df_ca[df_ca['State_Province']=='ON']
df_ca_on.City.value_counts().head(30).plot(kind="bar")
df_ca_on1 = df_ca_on[df_ca_on["City"]=='Toronto']
to_drop1 =['Country','State_Province','Ownership_Type','Postcode','City','Alpha_3']
ON = df_ca_on1.drop(columns = to_drop1)
from scipy.spatial import distance
ON.index.name = 'ID'
ON = ON.values
len(ON)
all_tor = []
for i in ON: 
    for j in ON[1:]: 
        all_tor.append(distance_stores(i[0],i[1],j[0],j[1]))
len(all_tor)
all_tor = pd.DataFrame(all_tor)
tor = all_tor[:183]
tor.loc[:,'p1_Lon'] = pd.DataFrame(ON[:,0])
tor.loc[:,'p1_Lat'] = pd.DataFrame(ON[:,1])
tor.loc[:,'p2_Lon'] = pd.DataFrame(ON[1:,0])
tor.loc[:,'p2_Lat'] = pd.DataFrame(ON[1:,1])
tor = tor[:183]
tor.columns=['Distance','p1_Lon','p1_Lat','p2_Lon','p2_Lat']
len(tor)
a_tor = tor.sort_values(by=['Distance'])
plt.figure(figsize=(10,10))
plt.scatter(x = 'p1_Lon',y='p1_Lat',data = a_tor)
plt.show()
tor_mean= np.mean(tor)
tor_std = np.std(tor)
plt.hist(a_tor['Distance'])

# get the sub-dataset of China
df_cn = df.loc[df['Country']=='CN']
#add new columns to data frame "How many stores in each states"
df_cn.State_Province.value_counts().head(10).plot(kind = "bar")
## ON dataset
df_cn_sh = df_cn[df_cn['State_Province']== '31']
#df_cn_sh.City.value_counts().head(10).plot(kind="bar")
to_drop1 =['Country','State_Province','Ownership_Type','Postcode','City','Alpha_3']
SH = df_cn_sh.drop(columns = to_drop1)
from scipy.spatial import distance
SH.index.name = 'ID'
SH = SH.values
len(SH)
all_sh = []
for i in SH: 
    for j in SH[1:]: 
        all_sh.append(distance_stores(i[0],i[1],j[0],j[1]))
len(all_sh)
all_sh = pd.DataFrame(all_sh)
sh = all_sh[:447]
sh.loc[:,'p1_Lon'] = pd.DataFrame(SH[:,0])
sh.loc[:,'p1_Lat'] = pd.DataFrame(SH[:,1])
sh.loc[:,'p2_Lon'] = pd.DataFrame(SH[1:,0])
sh.loc[:,'p2_Lat'] = pd.DataFrame(SH[1:,1])
sh = sh[:447]
sh.columns=['Distance','p1_Lon','p1_Lat','p2_Lon','p2_Lat']
len(sh)
a_sh = sh.sort_values(by=['Distance'])
plt.figure(figsize=(10,10))
sns.jointplot(x = 'p1_Lon',y='p1_Lat',data = a_sh)
plt.show()
sh_mean= np.mean(sh)
sh_std = np.std(sh)
plt.hist(a_sh['Distance'])
plt.xlim(xmin=0,xmax=80)

#San Diego
plt.subplot(221)
San = test.sort_values(by=['Distance'])
plt.scatter(x = 'p1_Lon',y='p1_Lat',data = San) 
plt.title("San Diego")

plt.ylabel("Lontitude")

# Seattle
plt.subplot(222)
seattle_dis = seattle1.sort_values(by=['Distance'])
plt.scatter(x = 'p1_Lon',y='p1_Lat',data =seattle_dis )
plt.title("Seattle")



#new york
plt.subplot(223)
nyc_dis = nyc1.sort_values(by=['Distance'])
plt.scatter(x = 'p1_Lon',y='p1_Lat',data =nyc_dis )
plt.title("New York City")
plt.xlabel("Latitude")
plt.ylabel("Lontitude")

# huston
plt.subplot(224)
HU1 = HU_TX.sort_values(by=['Distance'])
plt.scatter(x = 'p1_Lon',y='p1_Lat',data = HU1) 
plt.title("Huston")
plt.xlabel("Latitude")
#Toronto
plt.subplot(1,2,1)
a_tor = tor.sort_values(by=['Distance'])
plt.scatter(x = 'p1_Lon',y='p1_Lat',data = a_tor)
plt.title("Toronto")
plt.xlabel("Latitude")
plt.ylabel("Lontitude")

#Shanghai
plt.subplot(1,2,2)
a_sh = sh.sort_values(by=['Distance'])
plt.scatter(x = 'p1_Lon',y='p1_Lat',data = a_sh)
plt.title("Shanghai")
plt.xlabel("Latitude")

#San Diego
plt.subplot(221)
test_mean= np.mean(test)
test_std = np.std(test)
plt.hist(San['Distance'])
plt.title("San Diego")

plt.ylabel("Number of stores")

#Seattle
plt.subplot(222)
sea_mean= np.mean(seattle1)
sea_std = np.std(seattle1)
plt.hist(seattle1['Distance'])
plt.title("Seattle")


#NYC
plt.subplot(223)
nyc_mean= np.mean(nyc1)
nyc_std = np.std(nyc1)
plt.hist(nyc1['Distance'])
plt.title("New York City")
plt.xlabel("Distance")
plt.ylabel("Number of stores")
#Houston
plt.subplot(224)
HU_mean= np.mean(HU_TX)
HU_std = np.std(HU_TX)
plt.hist(HU_TX['Distance'])
plt.title("Huston")
plt.xlabel("Distance")

#Toronto
plt.subplot(1,2,1)
tor_mean= np.mean(tor)
tor_std = np.std(tor)
plt.hist(a_tor['Distance'])
plt.title("Toroton")
plt.xlabel("Distance")
plt.ylabel("Number of stores")
#Shanghai
plt.subplot(1,2,2)
sh_mean= np.mean(sh)
sh_std = np.std(sh)
plt.hist(a_sh.Distance[:280])
plt.xlim(xmin=0,xmax=15)
plt.title("Shanghai")
plt.xlabel("Distance")

