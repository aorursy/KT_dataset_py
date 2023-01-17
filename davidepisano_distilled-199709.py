import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
nome = 'Davide'
cognome = 'Pisano'
import random
def parametri(nome, cognome):
    seed_alpha = 0
    seed_emiss = 0
    for c in nome:
        seed_alpha += ord(c)
    for c in cognome:
        seed_emiss += ord(c)

    random.seed(a=seed_alpha*seed_emiss, version=2)
    
    emiss = round(0.780*(1.01+0.05*(random.random())), 3)
    alpha = round(0.300*(1+0.05*(2*random.random()-1)), 3)
        
    return alpha, emiss
alpha, emiss = parametri(nome, cognome)
print('Albedo: ' + str(alpha))
print('Emissività: ' + str(emiss))
So = 1367
Save = So/4
sigma = 5.67e-8
Tp = 0 # definizione della variabile
Tp = (Save*(1-alpha)/(1-0.5*emiss)/sigma)**0.25
Ta = ((Tp**4)/2)**0.25
print('Temperatura atmosfera  : ' + str(Ta- 273.15) + '°C')
print('Temperatura del pianeta: ' + str(Tp- 273.15) + '°C')
emiss = np.linspace(0,1,101)
# Tp =
Tp_C=(Save*(1-alpha)/(1-0.5*emiss)/sigma)**0.25-273.15
Tp_C
plt.figure(figsize=(16,8));
plt.plot(100*emiss, Tp_C);

plt.xlabel('emissivita [%]');
plt.ylabel('Tp [°C]');

plt.xticks(np.arange(0, 101, 2));
plt.yticks(np.arange(-20, 35, 2));

plt.grid();
temperatures = pd.read_csv('../input/climate-change-earth-surface-temperature-data/GlobalLandTemperaturesByCity.csv')
temperatures['dt'] = pd.to_datetime(temperatures['dt'])
def country_city(nome, cognome, df):
    seed = 0
    for c in (nome+cognome):
        seed += ord(c)

    countries = ['Spain', 'Germany', 'France', 'Italy', 'Netherlands', 'United Kingdom']
    country = countries[int(seed%len(countries))]
    cities = df.loc[df.Country==country].City.unique()
    city = cities[int(seed%len(cities))]
    
    return country, city
country, city = country_city(nome, cognome, temperatures)
print(city + ', ' + country)
temperatures.shape
data = temperatures.loc[temperatures.City == city]
data.shape
data.head
data = data.set_index( 'dt' )
temp = data.AverageTemperature
temp
plt.figure(figsize=(16,8));
plt.plot(temp);
plt.grid();
plt.title('Temperatura media mensile di ' + city + ', ' + country);
plt.xlabel('data [anni]');
plt.ylabel('temperatura [°C]');
#plt.savefig('temp.png', dpi=300);
climate = temp.rolling(30*12).mean()
plt.figure(figsize=(16,8))
#plt.plot(cagliari_temp, label='media mensile');
plt.plot(climate, label='climate');
plt.grid();
plt.title('Temperatura media mensile di ' + city);
plt.xlabel('data [anni]');
plt.ylabel('temperatura [°C]');
plt.legend();
year = 2012
year_30 = year - 30
temp_year = temp.loc[temp.index.year == year].values
temp_year_30 = temp.loc[temp.index.year == year_30].values
print(temp_year)
print(temp_year_30)
plt.figure(figsize=(16,8))
plt.plot(temp_year, label=year);
plt.plot(temp_year_30, label=year_30);
plt.grid();
plt.title('Temperatura mensile di ' + city + ', ' + country);
plt.xlabel('data [anni]');
plt.ylabel('temperatura [°C]');
plt.xticks(np.arange(12), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.legend();
#temp.groupby(temp.index.month).rolling(30).mean().head()
climate_month = []
for i in range(1,13):
    climate_month.append(temp.loc[temp.index.month == i].rolling(30).mean())
len(climate_month)
climate_month[11]
climate_year = np.zeros(12)
climate_year_30 = np.zeros(12)
for i in range(12):
    climate_year[i] = climate_month[i].loc[climate_month[i].index.year == year].values
    climate_year_30[i] = climate_month[i].loc[climate_month[i].index.year == year_30].values
climate_year_30
plt.figure(figsize=(16,8))
plt.plot(climate_year, label=year);
plt.plot(climate_year_30, label=year_30);
plt.grid();
plt.title('Clima mensile di ' + city + ', ' + country);
plt.xlabel('mese');
plt.ylabel('temperatura [°C]');
plt.xticks(np.arange(12), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.legend();
print(city, country, (climate_year - climate_year_30).mean(), (climate_year - climate_year_30).max())
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
PrimaryBiomes_names=['Humid',
                     'Intermediate',
                     'Semi-arid',
                     'Desert']
def correzione_clima(nome, cognome):
    seed = 0
    for c in (nome+cognome):
        seed += ord(c)
    gamma = 1 + ((seed%10)/100)
    
    return gamma
gamma = correzione_clima(nome, cognome)
print('Coefficiente di correzione del clima al 2100: ' + str(gamma))
lat = np.load('../input/clima-globale/lat.npy')
lon = np.load('../input/clima-globale/lon.npy')
lsmask = np.load('../input/clima-globale/lsmask.npy')
Tclima2020 = np.load('../input/clima-globale/Tclima2020.npy')
Pclima2020 = np.load('../input/clima-globale/Pclima2020.npy')
Tclima2100 = np.load('../input/clima-globale/Tclima2100.npy')
Tclima2100 = Tclima2100*gamma
Pclima2100 = np.load('../input/clima-globale/Pclima2100.npy')
Tclima2100 *= gamma
Pclima2100 *= gamma

Tclima = Tclima2020
Pclima = Pclima2020
#Tclima = Tclima2100
#Pclima = Pclima2100
nlat = len(lat)
nlon = len(lon)
nlat_2 = int(nlat / 2)

# calcola dal clima i campi che servono per individure i biomi
P_annual = np.zeros([nlat,nlon])
T_annual = np.zeros([nlat,nlon])
T_summer = np.zeros([nlat,nlon])

P_annual = np.sum(Pclima, axis=0)
T_annual = np.mean(Tclima, axis=0)
# calcola i biomi primari
PrimaryBiomes = np.zeros([nlat,nlon])
print(PrimaryBiomes.shape)
PrimaryBiomes[P_annual>=1.5] = 1
PrimaryBiomes[(P_annual>=0.62)&(P_annual<1.5)] = 2
PrimaryBiomes[(P_annual>=0.25)&(P_annual<0.62)] = 3
PrimaryBiomes[P_annual<0.25] = 4
plt.figure(figsize=(16, 8))
m = Basemap(projection='cyl', resolution='c')
#m = Basemap(projection='ortho', resolution='l', lat_0=39, lon_0=9)


xx, yy = np.meshgrid(lon, lat);

m.contourf(xx, yy, PrimaryBiomes, cmap=plt.cm.get_cmap('rainbow', 4), latlon=True);
#plt.colorbar(np.arange(1,5), PrimaryBiomes_names);
cbar = plt.colorbar();
cbar.set_ticks(list())

for index, label in enumerate(PrimaryBiomes_names):
    x = 1.5
    y = (2 * index + 1) / 8
    cbar.ax.text(x, y, label)

m.drawlsmask(resolution='c', land_color='none', ocean_color='w', zorder=25);
m.drawcoastlines(zorder=26);
Area = np.zeros([nlat,nlon])
dx = 360./nlon*60*1.852
for i in range(nlat):
    a = np.cos(lat[i]*np.pi/180.)*dx**2
    Area[i,:] = a
Area_land = Area*lsmask
areas = np.zeros(4)
for i in range(4):
    mask=(PrimaryBiomes==(i+1))
    a=np.sum(Area_land[mask])
    print(PrimaryBiomes_names[i]+' area km^2:',np.int(a))
    areas[i] = np.int(a)
areas_2020 = areas
#areas_2100 = areas
areas_delta = areas_2100 - areas_2020

areas_2020


areas_2100
areas_2020


for i in range(4):
    a = areas_delta[i]
    print(PrimaryBiomes_names[i]+' area km^2:',np.int(a))
