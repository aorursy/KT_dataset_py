import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from mpl_toolkits.basemap import Basemap

from geopy.geocoders import Nominatim

plt.style.use('seaborn')

%matplotlib inline

sns.set_style('whitegrid')
plt.rcParams['figure.figsize']=[15,8]

plt.rcParams['xtick.labelsize']=13

plt.rcParams['axes.titlesize']=15
brazil=pd.read_csv('../input/touristData.csv',encoding='cp1252')
brazil.info()
brazil.head()
plt.subplot(121)

brazil.groupby('Year')['Count'].sum().plot.bar()

plt.title('Visitors count by Year')

plt.subplot(122)

brazil.groupby('Month')['Count'].sum().plot.bar()

plt.title('Visitors count by Month')



ax=plt.subplot(121)

pd.DataFrame(brazil.groupby(['Continent','Year'])['Count'].sum().sort_values().unstack().transpose()).plot.area(stacked=1,cmap='gnuplot',ax=ax )

plt.xticks(rotation=90)

plt.suptitle('Count of visitors by Origin: \n Continents',fontsize=20)

plt.legend(bbox_to_anchor=(0.7, 0.12, 1.,-0.28),ncol=2, mode="expand",fontsize =12)

ax=plt.subplot(122)

pd.DataFrame(brazil.groupby(['Continent','Month'])['Count'].sum().sort_values().unstack().transpose()).plot.area(stacked=1,cmap='gnuplot',ax=ax)

plt.xticks(rotation=90)

ax.legend_.remove()

plt.title('')
ax=plt.subplot(121)

pd.DataFrame(brazil.groupby(['WayIn','Year'])['Count'].sum().sort_values().unstack().transpose()).plot.area(stacked=1,cmap='gnuplot',ax=ax )

plt.legend(bbox_to_anchor=(0.5, 0.12, 1.,-0.27),ncol=2, mode="expand",fontsize =12)

ax=plt.subplot(122)

pd.DataFrame(brazil.groupby(['WayIn','Month'])['Count'].sum().sort_values().unstack().transpose()).plot.area(stacked=1,cmap='gnuplot',ax=ax)

plt.xticks(rotation=90)

ax.legend_.remove()

plt.suptitle('Count of visitors by Way of Transport',fontsize=20)

top5=brazil.groupby(['State'])['Count'].sum().sort_values().nlargest(5).index
pd.DataFrame(brazil.groupby(['State','Year'])['Count'].sum()).sort_values(by='Count').unstack().transpose()[top5].plot.area(stacked=1

,cmap='gnuplot')

ax=plt.gca()                                                            

pd.DataFrame(brazil.groupby(['State','Year'])['Count'].sum()).sort_values(by='Count').unstack().transpose()[top5].plot(stacked=1

                                                                       ,ax=ax,lw=1,linestyle='solid',color='black',legend=False)

plt.suptitle('Count of visitors by Destination',fontsize=18)

brazil.groupby('Continent')['Count'].sum().sort_values().plot.bar()

plt.title('Count of Visitors By origin : \n Continent')
brazil.groupby('Country')['Count'].sum().sort_values().plot.bar()

plt.title('Count of Visitors By origin : \n Country')
brazil.groupby('State')['Count'].sum().sort_values().plot.bar()

plt.title('Most visited Brazilian states')
brazil.groupby('WayIn')['Count'].sum().sort_values().plot.bar()

plt.title('Most used Ways of transport')
pd.DataFrame(brazil.groupby(['WayIn','Continent'])['Count'].sum()).sort_values(by='Count').unstack().transpose().plot.bar(stacked=1)

plt.title('Count of Visitors : \n Continent and WayIn Breakdown')
fig,axes=plt.subplots(nrows=1,ncols=3,figsize=(18,8))

for Cont,i in zip(['Africa','Europe','Asia'],[0,1,2]):

    brazil[brazil['Continent']==Cont].groupby('Country').sum()['Count'].sort_values().plot.bar(ax=axes[i])

    axes[i].set_title('Most visitors Coming from :\n '+ Cont,fontsize=15)



fig,axes=plt.subplots(nrows=1,ncols=3,figsize=(18,8))

for Cont,i in zip(['Central America and Caribbean','North America','South America'],[0,1,2]):

    brazil[brazil['Continent']==Cont].groupby('Country').sum()['Count'].sort_values().plot.bar(ax=axes[i])

    axes[i].set_title('Most visitors Coming from :\n '+ Cont,fontsize=15)
brazil[brazil['Continent']=='Oceania'].groupby('Country').sum()['Count'].sort_values().plot.bar()

plt.title('Most vistiors coming from Oceania')
'''States=brazil['State'].unique()

StateDict=dict()

for State in States:

    try:

        geolocator = Nominatim()

        location = geolocator.geocode(State)

        print(State+' :'+str((location.latitude, location.longitude)))

        StateDict[State]=(location.latitude, location.longitude)

    except: print( '!!!'+State +' Not Found')''' #Code used to genreate the coordinates , but it's not working on kaggle notebooks.
StateDict={'Acre': (32.9281747, 35.0756366),

 'Amapá': (1.3545442, -51.9161977),

 'Amazonas': (-3.3306543, -60.6592703),

 'Bahia': (-12.285251, -41.9294776),

 'Ceará': (-5.3264703, -39.7156073),

 'Distrito Federal': (-15.7754462, -47.7970891),

 'Mato Grosso do Sul': (-19.5852564, -54.4794731),

 'Minas Gerais': (-18.5264844, -44.1588654),

 'Paraná': (-24.4842187, -51.8148872),

 'Pará': (-4.7493933, -52.8973006),

 'Pernambuco': (-8.4116316, -37.5919699),

 'Rio Grande do Norte': (-5.6781175, -36.4781776),

 'Rio Grande do Sul': (-29.8425284, -53.7680577),

 'Rio de Janeiro': (-22.9110137, -43.2093727),

 'Roraima': (2.135138, -61.3631922),

 'Santa Catarina': (21.14133975, -100.070929691885),

 'São Paulo': (-23.5506507, -46.6333824)}
States=pd.DataFrame(StateDict).transpose()
brazil=brazil.set_index('State').join(States).reset_index()
brazil.columns = ['State', 'Continent', 'Country', 'WayIn', 'Year', 'Month', 'Count', 'Lat',

       'Lon']
byState=brazil .groupby(['State']).sum().drop(['Year','Lat','Lon'],axis=1)

byState=byState.join(States).dropna()
byState.columns=['Count','Lat','Lon']
byState.loc['Acre']['Lat']=-9.9754

byState.loc['Acre']['Lon']=-67.8249

byState.loc['Santa Catarina']['Lat']=-27.2423

byState.loc['Santa Catarina']['Lon']=-50.2189
byState
plt.figure(figsize=(18,15))

m=Basemap(llcrnrlat=-55.401805,llcrnrlon=-92.269176,urcrnrlat=13.884615,urcrnrlon=-27.581676)

m.drawcoastlines(color='grey')

m.drawmapboundary(fill_color='w')

m.fillcontinents(color='green')

m.drawcountries(color='lightgrey')

y=byState['Lat']

x=byState['Lon']

x,y=m(x,y)



m.scatter(x,y,s=byState['Count']/6000 ,marker='o',zorder=10,alpha=0.5,color='red')



for State,x,y in zip(byState.index,x,y): 

    plt.text(x,y,State,color='k',size=12,fontweight='bold',zorder=11,rotation=0)

plt.title('Most Popular brazilian states for Tourists',fontsize=15,verticalalignment='bottom');
'''Countries=brazil[brazil['Continent']==Continent]['Country'].unique()

CountryDict=dict()

for Country in Countries:

    try:

        geolocator = Nominatim()

        location = geolocator.geocode(Country)

        print(Country+' :'+str((location.latitude, location.longitude)))

        CountryDict[Country]=(location.latitude, location.longitude)

    except: print( '!!!'+Country +' Not Found')'''
def Plot_Map(Continent,ulat,ulon,llat,llon) :

    Countries=brazil[brazil['Continent']==Continent]['Country'].unique()

    #CountryDict=dict()

    #for Country in Countries:

    #    try:

    #       geolocator = Nominatim()

    #       location = geolocator.geocode(Country)

    #       print(Country+' :'+str((location.latitude, location.longitude)))

    #       CountryDict[Country]=(location.latitude, location.longitude)

    #except: print( '!!!'+Country +' Not Found')

    

    byCountry=brazil.groupby('Country').sum().drop(['Year','Lat','Lon'],axis=1)

    byCountry=byCountry.transpose()

    countryDict=pd.DataFrame(CountryDict).transpose()

    byCountry=pd.DataFrame(byCountry).transpose().join(countryDict)

    try:

        byCountry.dropna(inplace=True)

        byCountry.drop('Russia',inplace=True)

    except:pass

    byCountry.columns=['Count','Lat','Lon']

    plt.figure(figsize=(20,20))

    m=Basemap(urcrnrlat=ulat,urcrnrlon=ulon,llcrnrlat=llat,llcrnrlon=llon,resolution='i')

    m.drawcoastlines(color='grey')

    m.drawmapboundary(fill_color='w')

    m.fillcontinents(color='green')

    m.drawcountries(color='lightgrey')

    y=byCountry['Lat']

    x=byCountry['Lon']

    x,y=m(x,y)



    m.scatter(x,y,s=byCountry['Count']/2000 ,marker='o',zorder=10,alpha=0.5,color='red')



    for Country,X,Y in zip(byCountry.dropna().index,x,y): 

        plt.text(X,Y,Country,color='k',size=12,fontweight='bold',zorder=11)

    plt.title('Number of Foreign visitors by Origin Country \n : '+ Continent,fontsize=30);

CountryDict={'Austria': (47.2000338, 13.199959),

 'Belgium': (50.6407351, 4.66696),

 'Czech Republic': (49.8167003, 15.4749544),

 'Denmark': (55.670249, 10.3333283),

 'Finland': (63.2467777, 25.9209164),

 'France': (46.603354, 1.8883335),

 'Germany': (51.0834196, 10.4234469),

 'Greece': (38.9953683, 21.9877132),

 'Hungary': (47.1817585, 19.5060937),

 'Ireland': (52.865196, -7.9794599),

 'Italy': (42.6384261, 12.674297),

 'Netherlands': (52.2379891, 5.53460738161551),

 'Norway': (64.5731537, 11.5280364395482),

 'Poland': (52.0977181, 19.0258159),

 'Portugal': (40.033265, -7.8896263),

 'Russia': (64.6863136, 97.7453061),

 'Spain': (40.0028028, -4.003104),

 'Sweden': (59.6749712, 14.5208584),

 'Switzerland': (46.7985624, 8.2319736),

 'United Kingdom': (54.7023545, -3.2765753)}



Plot_Map('Europe',ulon=33.113459,llat=35.658913,llon=-15.724432,ulat=71.546417)
CountryDict={'China': (35.000074, 104.999927),

 'India': (22.3511148, 78.6677428),

 'Iraque': (33.0955793, 44.1749775),

 'Israel': (30.8760272, 35.0015196),

 'Japan': (36.5748441, 139.2394179),

 'Republic of Korea': (36.5581914, 127.9408564),

 'Saudi Arabia': (25.6242618, 42.3528328)}



Plot_Map('Asia',ulon=146.113459,llat=05.658913,llon=28.724432,ulat=53.546417)
CountryDict={'Angola': (-11.8775768, 17.5691241),

 'Cape Verde': (16.0000552, -24.0083947),

 'Nigeria': (9.6000359, 7.9999721),

 'South Africa': (-28.8166236, 24.991639)}



Plot_Map('Africa',ulon=50.113459,llat=-32.580484,llon=-24.265264,ulat=35.546417)
CountryDict={'Argentina': (-34.9964963, -64.9672817),

 'Bolivia': (-17.0568696, -64.9912286),

 'Canada': (61.0666922, -107.9917071),

 'Chile': (-31.7613365, -71.3187697),

 'Colombia': (2.8894434, -73.783892),

 'Ecuador': (-1.3397668, -79.3666965),

 'French Guiana': (4.0039882, -52.999998),

 'Guiana': (4.8417097, -58.6416891),

 'Mexico': (19.4326009, -99.1333416),

 'Paraguay': (-23.3165935, -58.1693445),

 'Peru': (-6.8699697, -75.0458515),

 'Suriname': (4.1413025, -56.0771187),

 'United States': (39.7837304, -100.4458825),

 'Uruguay': (-32.8755548, -56.0201525),

 'Venezuela': (8.0018709, -66.1109318)}



Countries=brazil[(brazil['Continent']=='South America')| (brazil['Continent']=='North America')]['Country'].unique()

'''CountryDict=dict()

    for Country in Countries:

    try:

        geolocator = Nominatim()

        location= geolocator.geocode(Country)

        print(Country+' :'+str((location.latitude, location.longitude)))

        CountryDict[Country]=(location.latitude, location.longitude)

    except: print( '!!!'+Country +' Not Found')'''



byCountry=brazil.groupby('Country').sum().drop(['Year','Lat','Lon'],axis=1)

byCountry=byCountry.transpose()

CountryDict=pd.DataFrame(CountryDict).transpose()

byCountry=pd.DataFrame(byCountry).transpose().join(CountryDict)

try:

    byCountry.dropna(inplace=True)

    byCountry.drop('Russia',inplace=True)

except:pass

byCountry.columns=['Count','Lat','Lon']

plt.figure(figsize=(20,20))

m=Basemap(llcrnrlat=-60,llcrnrlon=-130,urcrnrlat=70,urcrnrlon=-25,resolution='i')

m.drawcoastlines(color='grey')

m.drawmapboundary(fill_color='w')

m.fillcontinents(color='green')

m.drawcountries(color='lightgrey')

y=byCountry['Lat']

x=byCountry['Lon']

x,y=m(x,y)



m.scatter(x,y,s=byCountry['Count']/2000 ,marker='o',zorder=10,alpha=0.5,color='red')



for Country,X,Y in zip(byCountry.dropna().index,x,y): 

    plt.text(X,Y,Country,color='k',size=12,fontweight='bold',zorder=11)

plt.title('Number of Foreign visitors by Origin Country \n : Amercia',fontsize=30);
