import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib as mpl

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')

import mpl_toolkits.basemap as mb

from mpl_toolkits.basemap import Basemap



print('* Libraries *')

print('Numpy Version     : ', np.__version__)

print('Pandas Version    : ', pd.__version__)

print('Matplotlib Version: ', mpl.__version__)

print('Seaborn Version   : ', sns.__version__)

print('Basemap Version   : ', mb.__version__)



#seaborn options

sns.set_style('white')



#pandas options

pd.options.display.max_rows = 100

pd.options.display.max_columns = 100
def plotRegFeatures0(reg):

    

    plt.figure(figsize=(20,15))

    plt.suptitle('Region %s' % reg['RegionName'].unique(), fontsize=20)

     

    plt.subplot(2,2,1)

    plt.bar(reg['Date'], reg['DailyTPC'], color=('orange'))

    plt.xlabel('Date')

    plt.ylabel('Total Cases')

    plt.xticks(rotation=90, fontsize=8)

    plt.yticks(fontsize=8)

    plt.title('New Positive Cases by day', fontsize=15)

    

    plt.subplot(2,2,2)

    plt.bar(reg['Date'], reg['DailyTeP'], color=('skyblue'))

    plt.xlabel('Date')

    plt.ylabel('Total Tests')

    plt.xticks(rotation=90, fontsize=8)

    plt.yticks(fontsize=8)

    plt.title('Tests Performed by day', fontsize=15)

    

    plt.subplot(2,2,3)

    

    plt.bar(reg['Date'], reg['DailyRec'], color=('green'))

    plt.xlabel('Date')

    plt.ylabel('Total Cases')

    plt.xticks(rotation=90, fontsize=8)

    plt.yticks(fontsize=8)

    plt.title('Recovered Cases by day', fontsize=15)

    

    plt.subplot(2,2,4)

    

    plt.bar(reg['Date'], reg['DailyDea'], color=('red'))

    plt.xlabel('Date', fontsize=8)

    plt.ylabel('Total Cases')

    plt.xticks(rotation=90, fontsize=8)

    plt.yticks(fontsize=8)

    plt.title('Deaths Cases by day', fontsize=15)

    

    plt.subplots_adjust(hspace=0.4)

    plt.show()

    

def plotRegFeatures1(reg):

    plt.figure(figsize=(16,10))

    x = reg['Date']

    y0 = reg['TotalPositiveCases']

    y1 = reg['HospitalizedPatients']

    y2 = reg['IntensiveCarePatients']

    plt.bar(x, y0, label='TotalPositiveCases', color='orange')

    plt.bar(x, y1, label='HospitalizedCases', color='red')

    plt.bar(x, y2, label='IntensiveCarePatients', color='darkred')

    plt.xticks(rotation=90, fontsize=8)

    plt.yticks(fontsize=8)

    plt.legend()

    plt.title('TotalPositives vs Hospitalized vs IntensiveCare', fontsize=16)

    plt.suptitle('Region %s' % reg['RegionName'].unique(), fontsize=20)

    plt.show()

    

    

def plotPercentRegFeatures(reg):

    plt.figure(figsize=(16,10))

    x = reg['Date']

    y0 = reg['DailyTCTRatio']

    y1 = reg['DailyHPTRatio']

    y2 = reg['DailyICTRatio']

    plt.bar(x, y0, label='TotalPositiveCases/Tests', color='orange')

    plt.bar(x, y1, label='HospitalizedPatients/Tests', color='red')

    plt.bar(x, y2, label='IntensiveCare/Tests', color='darkred')

    plt.xticks(rotation=90, fontsize=8)

    plt.legend()

    plt.title('TotalPositiveCases vs Hospitalized vs IntensiveCare Patients % TestsPerformed', fontsize=16)

    plt.suptitle('Region %s' % reg['RegionName'].unique(), fontsize=20)

    plt.show()
dfR = pd.read_csv('/kaggle/input/covid19-in-italy/covid19_italy_region.csv')

dfP = pd.read_csv('/kaggle/input/covid19-in-italy/covid19_italy_province.csv')

print('Full Region Dataset Shape:', dfR.shape)

print('Full Province Dataset Shape:', dfP.shape)
dfR['Date'] = pd.to_datetime(dfR['Date'])

dfR['Date'] = dfR['Date'].dt.strftime('%m/%d/%Y')

dfP['Date'] = pd.to_datetime(dfP['Date'])

dfP['Date'] = dfP['Date'].dt.strftime('%m/%d/%Y')
dfR.tail(21)
labels = ['SNo', 'Country', 'RegionCode']



regions = ['Abruzzo','Basilicata',

           'Calabria','Campania',

           'Emilia Romagna','Friuli Venezia Giulia',

           'Lazio','Liguria',

           'Lombardia','Marche',

           'Molise','Piemonte',

           'P.A. Bolzano', 'P.A. Trento',

           'Puglia','Sardegna',

           'Sicilia','Toscana',

           'Umbria',"Valle d'Aosta",'Veneto']



d = {}



for reg in regions:

    d[reg] = pd.DataFrame()

    d[reg] = dfR.loc[dfR['RegionName'] == reg]

    d[reg].drop(labels=labels, axis=1, inplace=True)

# Generate new features

    dailyTPC = d[reg]['TotalPositiveCases'].diff()

    dailyTeP = d[reg]['TestsPerformed'].diff()

    dailyRec = d[reg]['Recovered'].diff()

    dailyDea = d[reg]['Deaths'].diff()

    TCTRatio = (d[reg]['TotalPositiveCases'] * 100) / d[reg]['TestsPerformed']

    THTRatio = (d[reg]['TotalHospitalizedPatients'] * 100) / d[reg]['TestsPerformed']

    HPTRatio = (d[reg]['HospitalizedPatients'] * 100) / d[reg]['TestsPerformed']

    ICTRatio = (d[reg]['IntensiveCarePatients'] * 100) / d[reg]['TestsPerformed']

    

    d[reg].insert(14, 'DailyTPC', dailyTPC)

    d[reg].insert(15, 'DailyTeP', dailyTeP)

    d[reg].insert(16, 'DailyRec', dailyRec)

    d[reg].insert(17, 'DailyDea', dailyDea)

    d[reg].insert(18, 'DailyTCTRatio', TCTRatio)

    d[reg].insert(19, 'DailyTHTRatio', THTRatio)

    d[reg].insert(20, 'DailyHPTRatio', HPTRatio)

    d[reg].insert(21, 'DailyICTRatio', ICTRatio)
regLast = dfR[-21:].copy() #Last Day!



covidR=regLast[['RegionName',

                'HospitalizedPatients',

                'IntensiveCarePatients', 

                'TotalHospitalizedPatients',

                'HomeConfinement', 

                'CurrentPositiveCases', 

                'NewPositiveCases',

                'Recovered', 

                'Deaths', 

                'TotalPositiveCases', 

                'TestsPerformed']]



covidR.sort_values(by='TotalPositiveCases',ascending=False,inplace=True)

covidR.style.background_gradient(cmap='YlOrRd')
for reg in regions:

    plotRegFeatures0(d[reg])
#for r in d:

#    plotRegFeatures1(d[r])
#for r in d:

#    plotPercentRegFeatures(d[r])
dfP = dfP[dfP.ProvinceName != 'In fase di definizione/aggiornamento']

proLast = dfP[-107:].copy() #Last Day!





lat = proLast['Latitude'][:]

lon = proLast['Longitude'][:]

lat = np.array(lat)

lon = np.array(lon)



fig=plt.figure()

ax=fig.add_axes([1.0,1.0,2.8,2.8])

m = Basemap(llcrnrlon=5.,llcrnrlat=35.,urcrnrlon=20.,urcrnrlat=48.,

            rsphere=(6378137.00,6356752.3142),

            resolution='l',projection='merc',

            lat_0=40.,lon_0=-20.,lat_ts=20.)



m.drawmapboundary(fill_color='black', linewidth=0)

m.fillcontinents(color='grey', alpha=0.3)

m.drawcountries(linewidth=0.1, color='black')

m.drawcoastlines(linewidth=0.1, color="white")



x, y = m(lon, lat)

m.scatter(x,y,proLast['TotalPositiveCases']/10,marker='o',color='r')



ax.set_title('Diffusion across the provinces', fontsize=20)

plt.text( 7, 7,'Data from https://github.com/pcm-dpc/COVID-19', ha='left', va='bottom', size=8, color='black' )

plt.show()
covidP=proLast[['RegionName',

                'ProvinceName',

               'TotalPositiveCases', 

]]



covidP.sort_values(by='TotalPositiveCases',ascending=False,inplace=True)

covidP.style.background_gradient(cmap='YlOrRd')