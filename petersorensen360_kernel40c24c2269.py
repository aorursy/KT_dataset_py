# import necessary modules

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt

import os

import warnings

from datetime import datetime

from scipy import stats

from scipy.stats import norm, skew, probplot 



warnings.filterwarnings('ignore')
#print(os.listdir("../kaggle-Covid19/covid19-global-forecasting-week-2"))

dftrain = pd.read_csv('../input/covid19-global-forecasting-week-2/train.csv', parse_dates=['Date']).sort_values(by=['Country_Region', 'Date'])

dftest    = pd.read_csv('../input/covid19-global-forecasting-week-2/test.csv', parse_dates=['Date']).sort_values(by=['Country_Region', 'Date'])

#dftrain.head()



ppp_tabel = pd.read_csv('../input/country-ppp/Country_PPP.csv', sep='\s+')#.sort_values(by=['Country'])

ppp_tabel.drop('Id', 1,inplace=True)

ppp_tabel["Country"].replace( '_',' ', regex=True,inplace=True)  # _ var indført for at få den til at læse

ppp_tabel["Country"].replace( 'United States','US', regex=True,inplace=True)  # _ var indført for at få den til at læse

ppp_tabel.rename(columns={'Country':'Country_Region'},inplace=True)

ppp_tabel.sort_values('Country_Region',inplace=True)

#ppp_tabel
dftrain['Dayofyear'] = dftrain['Date'].dt.dayofyear

dftest['Dayofyear'] = dftest['Date'].dt.dayofyear

dftest['Expo'] = dftest['Dayofyear']-89.5 # var 86

#ppp_tabel["Country_Region"]



dftest = dftest.merge(dftrain[['Country_Region','Province_State','Date','ConfirmedCases','Fatalities']], on=['Country_Region','Province_State','Date'], how='left', indicator=True)

print("dftest columns =",dftest.columns)

dftest.head(60)

#dftrain = dftrain.loc[dftrain['Country_Region'] == 'Denmark']



dftrain['Province_State'].fillna(dftrain['Country_Region'], inplace=True)

dftest ['Province_State_orig'] = dftest ['Province_State']

dftest ['Province_State'].fillna(dftest['Country_Region'], inplace=True)



dftrain.sort_values(by =['Country_Region', 'Province_State','Date'], inplace=True)

dftrain[['NewCases','NewFatalities']] = dftrain.groupby(['Country_Region', 'Province_State'])[['ConfirmedCases','Fatalities']].transform(lambda x: x.diff()) 

dftrain['FatalityBasis'] = dftrain.groupby(['Country_Region', 'Province_State'])[['ConfirmedCases']].transform(lambda x: x.shift(10)) 



dftrain = dftrain.loc[dftrain['Dayofyear'] > 80]



to_sum = ['Country_Region','Province_State','ConfirmedCases','Fatalities']

lastinfo = dftrain.groupby(['Country_Region','Province_State']).tail(1)[to_sum]

lastinfo.rename(columns={'ConfirmedCases':'ConfirmedCases_init','Fatalities':'Fatalities_init'},inplace=True)



to_sum = ['ConfirmedCases','NewCases','FatalityBasis']

grouped = dftrain.groupby(['Country_Region','Province_State']).tail(4)

grouped_gem = dftrain.groupby(['Country_Region','Province_State'])[to_sum].mean()

grouped_gem.reset_index(inplace=True)

grouped_gem.rename(columns={'ConfirmedCases':'ConfirmedCases_base','Fatalities':'Fatalities_base'

                                ,'NewCases':'NewCases_base'},inplace=True)

grouped_gem = grouped_gem.merge(lastinfo, on=['Country_Region','Province_State'], how='outer', indicator=True)

                       

to_sum = ['NewCases','NewFatalities','FatalityBasis']

grouped2 = grouped.groupby(['Country_Region'])[to_sum].sum()

grouped2['FatalityPct'] = 100*grouped2['NewFatalities']/grouped2['FatalityBasis']



grouped2.rename(columns={'NewCases':'NewCases2','NewFatalities':'NewFatalities2'

                         ,'FatalityBasis':'FatalityBasis2','FatalityPct':'FatalityPct2'},inplace=True)





with_ppp = pd.merge(grouped2, ppp_tabel, on=['Country_Region'], how='outer', indicator=True)

missing = with_ppp.loc[with_ppp['ppp'].isnull()]

#missing



#grouped_gem.tail(6)

print("grouped_gem =",grouped_gem.columns)

#dftest.head(60)

#lastinfo.head(60)

dftrain.tail(60)

grouped=dftrain.groupby(['Country_Region','Province_State']).tail(8)

grouped=grouped.groupby(['Country_Region','Province_State']).head(4)

grouped.drop(['FatalityBasis'],axis=1,inplace=True)



to_sum = ['NewCases','NewFatalities']

grouped1 = grouped.groupby(['Country_Region'])[to_sum].sum()



grouped1.rename(columns={'NewCases':'NewCases1','NewFatalities':'NewFatalities1'}, inplace=True)



print("grouped1 columns =",grouped1.columns)

print("grouped2 columns =",grouped2.columns)

grouped = pd.merge(grouped1, grouped2, on=['Country_Region'])

grouped['CasesIncreasePct'] = 100*(grouped['NewCases2']/grouped['NewCases1']-1)

mask = grouped['CasesIncreasePct'] > 140

grouped.loc[mask,'CasesIncreasePct'] = 140

mask = grouped['CasesIncreasePct'] < 0

grouped.loc[mask,'CasesIncreasePct'] = 0

mask = grouped['CasesIncreasePct'].isnull()

grouped.loc[mask,'CasesIncreasePct'] = 0

grouped['Factor'] = (grouped['CasesIncreasePct']/100+1)**0.25

#grouped['FatalityIncreasePct'] = 100*(grouped['NewFatalities2']/grouped['NewFatalities1']-1)



grouped = pd.merge(grouped, ppp_tabel, on=['Country_Region'])

#grouped['ppp'].isnull().sum()



grouped['ppp'] = grouped['ppp']/10000.

mask = (grouped['FatalityPct2'] > 9) & (grouped['ppp'] <= 1)

grouped.loc[mask,'FatalityPct2'] = 5

mask = (grouped['FatalityPct2'] < 5) & (grouped['ppp'] <= 1)

grouped.loc[mask,'FatalityPct2'] = 5

mask = (grouped['FatalityPct2'] > 6) & (grouped['ppp'] >= 7)

grouped.loc[mask,'FatalityPct2'] = 6

mask = (grouped['FatalityPct2'] < 1.5) & (grouped['ppp'] >= 7)

grouped.loc[mask,'FatalityPct2'] = 1.5

mask = (grouped['FatalityPct2'] > (9.5 - 0.43*grouped['ppp'])) & (grouped['ppp'] > 1) & (grouped['ppp'] < 7)

grouped.loc[mask,'FatalityPct2'] = (9.5 - 0.43*grouped['ppp'])

mask = (grouped['FatalityPct2'] < (5.6 - 0.5*grouped['ppp'])) & (grouped['ppp'] > 1) & (grouped['ppp'] < 7)

grouped.loc[mask,'FatalityPct2'] = (5.6 - 0.5*grouped['ppp'])

mask = (grouped['FatalityPct2'].isnull()) &  (grouped['ppp'] <= 1)

grouped.loc[mask,'FatalityPct2'] = 7

mask = (grouped['FatalityPct2'].isnull()) &  (grouped['ppp'] >= 7)

grouped.loc[mask,'FatalityPct2'] = 4

mask = (grouped['FatalityPct2'].isnull()) & (grouped['ppp'] > 1) & (grouped['ppp'] < 7)

grouped.loc[mask,'FatalityPct2'] = (7.5 - 0.5*grouped['ppp'])



grouped.tail(6)

print("grouped columns =",grouped.columns)
#grouped_gem.head(5)

print("grouped_gem columns =",grouped_gem.columns)
#grouped[['Country_Region','FatalityPct2','Factor']].head(5)

dftest.drop('_merge',axis=1,inplace= True)

dftest = dftest.merge(grouped[['Country_Region','FatalityPct2','Factor']], on=['Country_Region'], how='left')

dftest = dftest.merge(grouped_gem[['Province_State','Country_Region','ConfirmedCases_base','ConfirmedCases_init','NewCases_base','Fatalities_init','FatalityBasis']], on=['Province_State','Country_Region'], how='left')



dftest['ConfirmedCases_shift'] = dftest.groupby(['Country_Region', 'Province_State'])[['ConfirmedCases']].transform(lambda x: x.shift(1)) 



mask = dftest['ConfirmedCases'].isnull()

# find new cases

dftest.loc[mask,'NewCases'] = dftest.loc[mask,'NewCases_base']*(dftest.loc[mask,'Factor']**dftest.loc[mask,'Expo'])



#dftest.loc[mask,'Confirmed'] = dftest.loc[mask,'FatalityBasis2']*(dftest.loc[mask,'Factor']**dftest.loc[mask,'Expo'])

dftest['NewCases_cum'] = dftest.groupby(['Country_Region', 'Province_State'])[['NewCases']].cumsum() #transform(lambda x: x.shift(1)) 

dftest.loc[mask,'ConfirmedCases'] = dftest.loc[mask,'ConfirmedCases_init'] + dftest.loc[mask,'NewCases_cum']



mask3 = dftest['ConfirmedCases'] > 400000

dftest.loc[mask3,'FatalityPct2'] = dftest.loc[mask3,'FatalityPct2']*0.7

mask4 = dftest['ConfirmedCases'] > 800000

dftest.loc[mask4,'FatalityPct2'] = dftest.loc[mask4,'FatalityPct2']*0.7

dftest['FatalityBasis'] = dftest.groupby(['Country_Region', 'Province_State'])[

                                                ['ConfirmedCases']].transform(lambda x: x.shift(10)) 

dftest.loc[mask,'NewFatalities'] = dftest.loc[mask,'FatalityBasis'] * dftest.loc[mask,'FatalityPct2']/100

mask2 = dftest['NewFatalities'] >1000

dftest.loc[mask2,'NewFatalities'] = 1000

print("MASK2",mask2.sum())



dftest['NewFatalities_cum'] = dftest.groupby(['Country_Region', 'Province_State'])[['NewFatalities']].cumsum() #transform(lambda x: x.shift(1)) 

dftest.loc[mask,'Fatalities'] = dftest.loc[mask,'Fatalities_init'] + dftest.loc[mask,'NewFatalities_cum']





print("dftest columns =",dftest.columns)

#dftest[['Fatalities','NewFatalities','NewFatalities_cum','FatalityBasis','FatalityPct2']].head(60)

#dftest = dftest.loc[dftest['Country_Region'] == 'United Kingdom']

dftest.head(5)
# Forbered aflevering 

dftest.drop(['Dayofyear',

       'Expo','FatalityPct2', 'Factor',

       'ConfirmedCases_base', 'ConfirmedCases_init',

       'NewCases_base', 'Fatalities_init', 'FatalityBasis',

       'ConfirmedCases_shift',

       'NewCases', 'NewCases_cum', 'NewFatalities','NewFatalities_cum'],axis=1,inplace=True)

final = dftest.groupby(['Country_Region','Province_State']).tail(1)

dftest.drop(['Province_State'],axis=1,inplace=True)

dftest.rename(columns={'Province_State_orig':'Province_State'},inplace=True)


#final = final.loc[final['Country_Region'] == 'Denmark']

final.tail(60)
dftest.tail(60)

#grouped.loc[grouped['ppp'] > 120].tail(60)


plotgrouped = grouped.loc[grouped['FatalityPct2'] > 0.2]

plotgrouped = plotgrouped.loc[grouped['NewFatalities2'] > 30]

#print(plotgrouped)

plt.figure(figsize=(15,10))

plt.subplots_adjust(wspace=0.2, hspace=0.2)

#

ylabels = ['FatalityPct2','CasesIncreasePct']

ys = [plotgrouped['FatalityPct2'],plotgrouped['CasesIncreasePct']]

loglin = ['log','linear']

for iy, y in enumerate(ys):

    plt.subplot(2,2,1+iy)

    plt.xticks(rotation=30)

    plt.xlabel('ppp')

    plt.ylabel(ylabels[iy])

    plt.yscale(loglin[iy])

 #   plt.xscale('log')

    plt.plot(plotgrouped['ppp'],y,'*')

#    plt.legend(allcountries_ordered[:11])

#

#plt.plot(x, y, 'o', color='black');

#plt.show()
dftest.drop(['Province_State','Country_Region','Date'],axis=1,inplace=True)

print("dftest columns =",dftest.columns)

        

dftest.ForecastId = dftest.ForecastId.astype('int')



dftest['ConfirmedCases'] = dftest['ConfirmedCases'].round().astype(int)

dftest['Fatalities'] = dftest['Fatalities'].round().astype(int)



dftest.to_csv('submission.csv', index=False)
