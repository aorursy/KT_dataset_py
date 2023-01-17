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
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
df3=pd.read_csv("/kaggle/input/world-happiness/2017.csv")
df5=pd.read_csv('/kaggle/input/world-happiness/2019.csv')
df2=pd.read_csv('/kaggle/input/world-happiness/2016.csv')
df1=pd.read_csv('/kaggle/input/world-happiness/2015.csv')
df4=pd.read_csv('/kaggle/input/world-happiness/2018.csv')
df1.head(10)
df1.columns
df1.info()
df1['Country'].nunique()
df1['Region'].unique()
w_europe=df1[df1['Region']=='Western Europe']
w_europe.info()
w_europe.plot(x='Country', y=['Happiness Score','Economy (GDP per Capita)'], kind="bar",figsize=(10,8))
plt.xticks(rotation='vertical')
from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
data=dict(type='choropleth',
          locations=w_europe['Country'],
          locationmode='country names',
          colorscale="Jet",
          text=w_europe['Country'],
          z=w_europe['Happiness Score'],
          colorbar={'title':'Happiness score'})

layout=dict(title='2015 Western Europe Happiness Score',geo={'scope':'europe'})
choromap=go.Figure(data=[data],layout=layout)
iplot(choromap)
data=dict(type='choropleth',
          locations=df1['Country'],
          locationmode='country names',
          colorscale="blues",
          text=df1['Country'],
          z=df1['Happiness Score'],
          colorbar={'title':'Happiness score'})

layout=dict(title='2015 World Happiness Score',geo=dict(showframe=True,projection={'type':'natural earth'}))
choromap=go.Figure(data=[data],layout=layout)
iplot(choromap)


data=dict(type='choropleth',
          locations=df1['Country'],
          locationmode='country names',
          colorscale="blues",
          text=df1['Country'],
          z=df1['Freedom'],
          colorbar={'title':'Freedom score'})

layout=dict(title='2015 World Freedom Score',geo=dict(showframe=True,projection={'type':'natural earth'}))
choromap=go.Figure(data=[data],layout=layout)
iplot(choromap)
plt.figure(figsize=(12,8))
sns.heatmap(df1.corr(),cmap='coolwarm',annot=True)
a,b=plt.ylim()
a+=0.5
b-=0.5
plt.ylim(a,b)
df5.info()
df5['Country or region'].unique()
df5=df5.replace('Congo (Brazzaville)','Republic of the Congo')
df5=df5.replace('Congo (Kinshasa)','Democratic Republic of the Congo')
data=dict(type='choropleth',
          locations=df1['Country'],
          locationmode='country names',
          colorscale="Jet",
          text=df1['Country'],
          z=df1['Happiness Score'],
          colorbar={'title':'Happiness score'})

layout=dict(title='2015 World Happiness Score',geo=dict(showframe=True,projection={'type':'natural earth'}))
choromap=go.Figure(data=[data],layout=layout)
iplot(choromap)

data=dict(type='choropleth',
          locations=df5['Country or region'],
          locationmode='country names',
          colorscale="Jet",
          text=df5['Country or region'],
          z=df5['Score'],
          colorbar={'title':'Happiness score'})

layout=dict(title='2019 World Happiness Score',geo=dict(showframe=True,projection={'type':'natural earth'}))
choromap=go.Figure(data=[data],layout=layout)
iplot(choromap)


import pycountry

input_countries = df1['Country'].unique()

countries = {}
for country in pycountry.countries:
    countries[country.name] = country.alpha_3

df1_codes = [countries.get(country, 'Unknown code') for country in input_countries]

print(df1_codes)
    
df1['Country'].unique()
df1['Country code']=['CHE', 'ISL', 'DNK', 'NOR', 'CAN', 'FIN', 'NLD', 'SWE', 'NZL', 'AUS', 'ISR', 'CRI',
           'AUT', 'MEX', 'USA', 'BRA', 'LUX', 'IRL', 'BEL', 'ARE', 'GBR', 'OMN', 'VEN', 'SGP',
           'PAN', 'DEU', 'CHL', 'QAT', 'FRA', 'ARG', 'CZE', 'URY', 'COL', 'THA', 'SAU', 'ESP',
           'MLT', 'TWN', 'KWT', 'SUR', 'TTO', 'SLV', 'GTM', 'UZB', 'SVK', 'JPN', 'KOR','ECU',
           'BHR', 'ITA', 'BOL', 'MDA', 'PRY', 'KAZ', 'SVN', 'LTU', 'NIC', 'PER', 'BLR', 'POL', 
           'MYS', 'HRV', 'LBY', 'RUS', 'JAM', 'NaN','CYP', 'DZA', 'KOS', 'TKM', 'MUS', 'HKG', 'EST',
           'IDN', 'VNM', 'TUR', 'KGZ', 'NGA', 'BTN', 'AZE', 'PAK', 'JOR', 'MNE', 'CHN', 'ZMB', 
           'ROU', 'SRB', 'PRT', 'LVA', 'PHL', 'SOM', 'MAR', 'MKD', 'MOZ', 'ALB', 'BIH', 'LSO',
           'DOM', 'LAO', 'MNG', 'SWZ', 'GRC', 'LBN', 'HUN', 'HND', 'TJK', 'TUN', 'PSE', 'BGD',
           'IRN', 'UKR', 'IRQ', 'ZAF', 'GHA', 'ZWE', 'LBR', 'IND', 'SDN', 'HTI', 'COD', 'NPL',
           'ETH', 'SLE', 'MRT', 'KEN', 'DJI', 'ARM', 'BWA', 'MMR', 'GEO', 'MWI', 'LKA', 'CMR',
           'BGR', 'EGY', 'YEM', 'AGO', 'MLI', 'COG', 'COM', 'UGA', 'SEN', 'GAB', 'NER', 'KHM',
           'TZA', 'MDG', 'CAF', 'TCD', 'GIN', 'CIV', 'BFA', 'AFG', 'RWA', 'BEN', 
           'SYR', 'BDI', 'TGO']
df1=df1[['Country','Country code', 'Region', 'Happiness Rank', 'Happiness Score',
       'Standard Error', 'Economy (GDP per Capita)', 'Family',
       'Health (Life Expectancy)', 'Freedom', 'Trust (Government Corruption)',
       'Generosity', 'Dystopia Residual']]
df1.head()
df5['Country or region'].unique()
df1['Country'].nunique()
n=0
for country in df1['Country'].unique() :
    if country not in df5['Country or region'].unique():
        n+=1
        print( country,False)
print(n)
n=0
for country in df5['Country or region'].unique() :
    if country not in df1['Country'].unique():
        n+=1
        print( country,False)
print(n)
#dropping countries that do not belong to both datasets
df1_1=df1.drop(df1[df1['Country']=='Oman'].index)
df1_1=df1_1.drop(df1_1[df1_1['Country']=='Suriname'].index)
df1_1=df1_1.drop(df1_1[df1_1['Country']=='Djibouti'].index)
df1_1=df1_1.drop(df1_1[df1_1['Country']=='Angola'].index)
len(df1_1)
df1_1['Country code']=['CHE', 'ISL', 'DNK', 'NOR', 'CAN', 'FIN', 'NLD', 'SWE', 'NZL', 'AUS', 'ISR', 'CRI',
           'AUT', 'MEX', 'USA', 'BRA', 'LUX', 'IRL', 'BEL', 'ARE', 'GBR', 'VEN', 'SGP',
           'PAN', 'DEU', 'CHL', 'QAT', 'FRA', 'ARG', 'CZE', 'URY', 'COL', 'THA', 'SAU', 'ESP',
           'MLT', 'TWN', 'KWT', 'TTO', 'SLV', 'GTM', 'UZB', 'SVK', 'JPN', 'KOR','ECU',
           'BHR', 'ITA', 'BOL', 'MDA', 'PRY', 'KAZ', 'SVN', 'LTU', 'NIC', 'PER', 'BLR', 'POL', 
           'MYS', 'HRV', 'LBY', 'RUS', 'JAM', 'NaN','CYP', 'DZA', 'KOS', 'TKM', 'MUS', 'HKG', 'EST',
           'IDN', 'VNM', 'TUR', 'KGZ', 'NGA', 'BTN', 'AZE', 'PAK', 'JOR', 'MNE', 'CHN', 'ZMB', 
           'ROU', 'SRB', 'PRT', 'LVA', 'PHL', 'SOM', 'MAR', 'MKD', 'MOZ', 'ALB', 'BIH', 'LSO',
           'DOM', 'LAO', 'MNG', 'SWZ', 'GRC', 'LBN', 'HUN', 'HND', 'TJK', 'TUN', 'PSE', 'BGD',
           'IRN', 'UKR', 'IRQ', 'ZAF', 'GHA', 'ZWE', 'LBR', 'IND', 'SDN', 'HTI', 'COD', 'NPL',
           'ETH', 'SLE', 'MRT', 'KEN', 'ARM', 'BWA', 'MMR', 'GEO', 'MWI', 'LKA', 'CMR',
           'BGR', 'EGY', 'YEM', 'MLI', 'COG', 'COM', 'UGA', 'SEN', 'GAB', 'NER', 'KHM',
           'TZA', 'MDG', 'CAF', 'TCD', 'GIN', 'CIV', 'BFA', 'AFG', 'RWA', 'BEN', 
           'SYR', 'BDI', 'TGO']
df5_1=df5.drop(df5[df5['Country or region']=='Namibia'].index)
df5_1=df5_1.drop(df5_1[df5_1['Country or region']=='Gambia'].index)
df1_1['Country'].nunique()
df5_1['Country or region'].nunique()
n=0
for country in df1_1['Country'].unique() :
    if country not in df5_1['Country or region'].unique():
        n+=1
        print( country,False)
print(n)
n=0
for country in df5_1['Country or region'].unique() :
    if country not in df1_1['Country'].unique():
        n+=1
        print( country,False)
print(n)
#renaming similar country names
df5_1=df5_1.replace('Trinidad & Tobago','Trinidad and Tobago')
df5_1=df5_1.replace(['Northern Cyprus','North Macedonia'],['North Cyprus','Macedonia'])
df5_1=df5_1.replace('South Sudan','Sudan')
df1_1=df1_1.replace(['Congo (Brazzaville)','Somaliland region','Congo (Kinshasa)'],
                 ['Republic of the Congo','Somalia','Democratic Republic of the Congo'])
                
n=0
for country in df1_1['Country'].unique() :
    if country not in df5_1['Country or region'].unique():
        n+=1
        print( country,False)
print(n)
n=0
for country in df5_1['Country or region'].unique() :
    if country not in df1_1['Country'].unique():
        n+=1
        print( country,False)
print(n)
diff=df1_1.sort_values('Country')[['Country','Country code', 'Happiness Score']]
diff.head()
diff.info()
diff.info()
sorted2=df5_1.sort_values('Country or region')
df5_1.head(50)
sorted2.head(30)
sorted2.info()
diff['Country'].unique()==sorted2['Country or region'].unique()
l=diff['Country'].unique()
l
l2=sorted2['Country or region'].unique()
l2
l==l2
diff=diff.reset_index(drop=True)
diff.head(10)
sorted2=sorted2.reset_index(drop=True)
sorted2.head(30)
diff["Happiness score 2019"]=sorted2['Score']
sorted2['Score']
diff.info()
diff=diff.rename(columns={'Happiness Score':'Happiness score 2015'})
diff['% difference']=round(((diff['Happiness score 2019']-diff['Happiness score 2015'])/
                     diff['Happiness score 2015']*100),2)
diff.head(5)
diff_pos=diff[diff['% difference']>=0][['Country','Country code','% difference']].sort_values(by='% difference',
                                                                               axis=0)
diff_pos.tail(5)
diff_pos.apply(len)
diff_neg=diff[diff['% difference']<0][['Country','Country code','% difference']].sort_values(by='% difference',
                                                                               axis=0)
diff_neg.head()
diff_neg.apply(len)
diff_pos.plot(x='Country',y='% difference', kind='bar',figsize=(20,10))
print(f"Percentage of countries whose Happiness score has increased for more than 20% is:{len(diff_pos[diff_pos['% difference']>20])/len(diff_pos)*100}")
diff_neg.plot(x='Country',y='% difference', kind='bar',figsize=(20,10))
diff_neg.head(20)
print(f"Percentage of countries whose Happiness score has decreased for more than 20% is:{len(diff_neg[diff_neg['% difference']<-20])/len(diff_neg)*100}")
data=dict(type='choropleth',
          locations=diff_pos['Country code'],
          colorscale="reds",
          text=diff_pos['Country'],
          z=diff_pos['% difference'],
          colorbar={'title':'% difference legend '})

layout=dict(title='% increase in Happiness Score 2015 to 2019',geo=dict(showframe=True,projection={'type':'natural earth'}))
choromap=go.Figure(data=[data],layout=layout)
iplot(choromap)

data=dict(type='choropleth',
          locations=diff_neg['Country code'],
          colorscale="greens_r",
          text=diff_neg['Country'],
          z=diff_neg['% difference'],
          colorbar={'title':'% difference legend '})

layout=dict(title='% decrease in Happiness Score 2015 to 2019',geo=dict(showframe=True,projection={'type':'natural earth'}))
choromap=go.Figure(data=[data],layout=layout)
iplot(choromap)
df1.columns

african=['DZA',
         'AGO',
'SHN',
'BEN',
'BWA',
'BFA',
'BDI',
'CMR',
'CPV',
'CAF',
'TCD',
'COM',
'COG',
'COD',
'DJI',
'EGY',
'GNQ',
'ERI',
'SWZ',
'ETH',
'GAB',
'GMB',
'GHA',
'GIN',
'GNB',
'CIV',
'KEN',
'LSO',
'LBR',
'LBY',
'MDG',
'MWI',
'MLI',
'MRT',
'MUS',
'MYT',
'MAR',
'MOZ',
'NAM',
'NER',
'NGA',
'STP',
'REU',
'RWA',
'STP',
'SEN',
'SYC',
'SLE',
'SOM',
'ZAF',
'SSD',
'SHN',
'SDN',
'SWZ',
'TZA',
'TGO',
'TUN',
'UGA',
'COD',
'ZMB',
'TZA',
'ZWE']

len(african)
european=['AND',
'AUT',
 'ALB',         
'BLR',
'BEL',
'BIH',
'BGR',
'HRV',
'CYP',
'CZE',
'DNK',
'EST',
'FRO',
'FIN',
'FRA',
'DEU',
'GIB',
'GRC',
'HUN',
'ISL',
'IRL',
'IMN',
'ITA',
'XKX',
 'KOS',         
'LVA',
'LIE',
'LTU',
'LUX',
'MKD',
'MLT',
'MDA',
'MCO',
'MNE',
'NLD',
'NOR',
'POL',
'PRT',
'ROU',
'RUS',
'SMR',
'SRB',
'SVK',
'SVN',
'ESP',
'SWE',
'CHE',
'UKR',
'GBR',
'VAT']

len(european)
n_american=['AIA',
'ATG',
'ABW',
'BHS',
'BRB',
'BLZ',
'BMU',
'BES',
'VGB',
'CAN',
'CYM',
'CRI',
'CUB',
'CUW',
'DMA',
'DOM',
'SLV',
'GRL',
'GRD',
'GLP',
'GTM',
'HTI',
'HND',
'JAM',
'MTQ',
'MEX',
'SPM',
'MSR',
'ANT',
'KNA',
'NIC',
'PAN',
'PRI',
'BES',
'SXM',
'KNA',
'LCA',
'SPM',
'VCT',
'TTO',
'TCA',
'USA',
'VIR'
]
len(n_american)
s_american=['ARG',
'BOL',
'BRA',
'CHL',
'COL',
'ECU',
'FLK',
'GUF',
'GUY',
'PRY',
'PER',
'SUR',
'URY',
'VEN'
]
len(s_american)
asian=['AFG',
'ARM',
'AZE',
'BHR',
'BGD',
'BTN',
'BRN',
'KHM',
'CHN',
'CXR',
'CCK',
'IOT',
'GEO',
'HKG',
'IND',
'IDN',
'IRN',
'IRQ',
'ISR',
'JPN',
'JOR',
'KAZ',
'KWT',
'KGZ',
'LAO',
'LBN',
'MAC',
'MYS',
'MDV',
'MNG',
'MMR',
'NPL',
'PRK',
'OMN',
'PAK',
'PSE',
'PHL',
'QAT',
'SAU',
'SGP',
'KOR',
'LKA',
'SYR',
'TWN',
'TJK',
'THA',
'TUR',
'TKM',
'ARE',
'UZB',
'VNM',
'YEM'
]
len(asian)
australian=['ASM',
'AUS',
'NZL',
'COK',
'TLS',
'FSM',
'FJI',
'PYF',
'GUM',
'KIR',
'MNP',
'MHL',
'UMI',
'NRU',
'NCL',
'NZL',
'NIU',
'NFK',
'PLW',
'PNG',
'MNP',
'WSM',
'SLB',
'TKL',
'TON',
'TUV',
'VUT',
'UMI',
'WLF'
]
len(australian)
def continent(code):
    if code in african:
        return 'AF'
    elif code in european:
        return 'EU'
    elif code in n_american:
        return 'NA'
    elif code in s_american:
        return 'SA'
    elif code in australian:
        return 'OC'
    elif code in asian:
        return 'AS'
df1['Continent']=df1['Country code'].apply(continent)
df1.head()
df1=df1[['Country', 'Country code','Continent', 'Region', 'Happiness Rank',
       'Happiness Score', 'Standard Error', 'Economy (GDP per Capita)',
       'Family', 'Health (Life Expectancy)', 'Freedom',
       'Trust (Government Corruption)', 'Generosity', 'Dystopia Residual']]
df1.info()
df1[df1['Continent'].isnull()]
df1.drop( df1[ df1['Country'] == 'North Cyprus' ].index , inplace=True)
len(df1)
input_countries = df2['Country'].unique()

countries = {}
for country in pycountry.countries:
    countries[country.name] = country.alpha_3

df2_codes = [countries.get(country, 'Unknown code') for country in input_countries]

print(df2_codes)
    
df2['Country'].unique()
df2['Country code']=['DNK', 'CHE', 'ISL', 'NOR', 'FIN', 'CAN', 'NLD', 'NZL', 'AUS', 'SWE', 'ISR', 
                     'AUT', 'USA', 'CRI', 'PRI', 'DEU', 'BRA', 'BEL', 'IRL', 'LUX', 'MEX', 'SGP',
                     'GBR', 'CHL', 'PAN', 'ARG', 'CZE', 'ARE', 'URY', 'MLT', 'COL', 'FRA', 'THA', 
                     'SAU', 'TWN', 'QAT', 'ESP', 'DZA', 'GTM', 'SUR', 'KWT', 'BHR', 'TTO', 'VEN',
                     'SVK', 'SLV', 'MYS', 'NIC', 'UZB', 'ITA', 'ECU', 'BLZ', 'JPN', 'KAZ', 'MDA',
                     'RUS', 'POL', 'KOR', 'BOL', 'LTU', 'BLR', 'NaN', 'SVN', 'PER', 'TKM', 'MUS', 
                     'LBY', 'LVA', 'CYP', 'PRY', 'ROU', 'EST', 'JAM', 'HRV', 'HKG', 'SOM', 'KOS', 
                     'TUR', 'IDN', 'JOR', 'AZE', 'PHL', 'CHN', 'BTN', 'KGZ', 'SRB', 'BIH', 'MNE',
                     'DOM', 'MAR', 'HUN', 'PAK', 'LBN', 'PRT', 'MKD', 'VNM', 'SOM', 'TUN', 'GRC',
                     'TJK', 'MNG', 'LAO', 'NGA', 'HND', 'IRN', 'ZMB', 'NPL', 'PSE', 'ALB', 'BGD',
                     'SLE', 'IRQ', 'NAM', 'CMR', 'ETH', 'ZAF', 'LKA', 'IND', 'MMR', 'EGY', 'ARM', 
                     'KEN', 'UKR', 'GHA', 'COD', 'GEO', 'COG', 'SEN', 'BGR', 'MRT', 'ZWE', 'MWI',
                     'SDN', 'GAB', 'MLI', 'HTI', 'BWA', 'COM', 'CIV', 'KHM', 'AGO', 'NER', 'SSD', 
                     'TCD', 'BFA', 'UGA', 'YEM', 'MDG', 'TZA', 'LBR', 'GIN', 'RWA', 'BEN', 'AFG', 
                     'TGO', 'SYR', 'BDI']
df2['Continent']=df2['Country code'].apply(continent)
df2=df2[['Country', 'Country code', 'Continent','Region', 'Happiness Rank', 'Happiness Score',
       'Lower Confidence Interval', 'Upper Confidence Interval',
       'Economy (GDP per Capita)', 'Family', 'Health (Life Expectancy)',
       'Freedom', 'Trust (Government Corruption)', 'Generosity',
       'Dystopia Residual']]
df2.head()
df2[df2['Continent'].isnull()]
df2.drop( df2[ df2['Country'] == 'North Cyprus' ].index , inplace=True)
df2.info()
df2[df2['Continent']=='None']
input_countries = df3['Country'].unique()

countries = {}
for country in pycountry.countries:
    countries[country.name] = country.alpha_3

df3_codes = [countries.get(country, 'Unknown code') for country in input_countries]

print(df3_codes)
    
df3['Country'].unique()
df3['Country code']=['NOR', 'DNK', 'ISL', 'CHE', 'FIN', 'NLD', 'CAN', 'NZL', 'SWE', 'AUS', 'ISR',
                     'CRI', 'AUT', 'USA', 'IRL', 'DEU', 'BEL', 'LUX', 'GBR', 'CHL', 'ARE', 'BRA',
                     'CZE', 'ARG', 'MEX', 'SGP', 'MLT', 'URY', 'GTM', 'PAN', 'FRA', 'THA', 'TWN',
                     'ESP', 'QAT', 'COL', 'SAU', 'TTO', 'KWT', 'SVK', 'BHR', 'MYS', 'NIC', 'ECU', 
                     'SLV', 'POL', 'UZB', 'ITA', 'RUS', 'BLZ', 'JPN', 'LTU', 'DZA', 'LVA', 'KOR',
                     'MDA', 'ROU', 'BOL', 'TKM', 'KAZ', 'NaN', 'SVN', 'PER', 'MUS', 'CYP', 'EST',
                     'BLR', 'LBY', 'TUR', 'PRY', 'HKG', 'PHL', 'SRB', 'JOR', 'HUN', 'JAM', 'HRV',
                     'KOS', 'CHN', 'PAK', 'IDN', 'VEN', 'MNE', 'MAR', 'AZE', 'DOM', 'GRC', 'LBN', 
                     'PRT', 'BIH', 'HND', 'MKD', 'SOM', 'VNM', 'NGA', 'TJK', 'BTN', 'KGZ', 'NPL', 
                     'MNG', 'ZAF', 'TUN', 'PSE', 'EGY', 'BGR', 'SLE', 'CMR', 'IRN', 'ALB', 'BGD', 
                     'NAM', 'KEN', 'MOZ', 'MMR', 'SEN', 'ZMB', 'IRQ', 'GAB', 'ETH', 'LKA', 'ARM',
                     'IND', 'MRT', 'COG', 'GEO', 'COD', 'MLI', 'CIV', 'KHM', 'SDN', 'GHA', 'UKR',
                     'UGA', 'BFA', 'NER', 'MWI', 'TCD', 'ZWE', 'LSO', 'AGO', 'AFG', 'BWA', 'BEN',
                     'MDG', 'HTI', 'YEM', 'SSD', 'LBR', 'GIN', 'TGO', 'RWA', 'SYR', 'TZA', 'BDI',
                     'CAF']
df3['Continent']=df3['Country code'].apply(continent)
df3=df3[['Country','Country code','Continent','Happiness.Rank', 'Happiness.Score', 'Whisker.high',
       'Whisker.low', 'Economy..GDP.per.Capita.', 'Family','Health..Life.Expectancy.', 'Freedom',
        'Generosity','Trust..Government.Corruption.', 'Dystopia.Residual']]
df3.head()
df3[df3['Continent'].isnull()]
df3.drop( df3[ df3['Country'] == 'North Cyprus' ].index , inplace=True)
df3.info()
input_countries = df4['Country or region'].unique()

countries = {}
for country in pycountry.countries:
    countries[country.name] = country.alpha_3

df4_codes = [countries.get(country, 'Unknown code') for country in input_countries]

print(df4_codes)
    
df4['Country or region'].unique()
df4['Country code']=['FIN', 'NOR', 'DNK', 'ISL', 'CHE', 'NLD', 'CAN', 'NZL', 'SWE', 'AUS', 'GBR',
                     'AUT', 'CRI', 'IRL', 'DEU', 'BEL', 'LUX', 'USA', 'ISR', 'ARE', 'CZE','MLT',
                     'FRA', 'MEX', 'CHL', 'TWN', 'PAN', 'BRA', 'ARG', 'GTM', 'URY', 'QAT', 'SAU',
                     'SGP', 'MYS', 'ESP', 'COL', 'TTO', 'SVK', 'SLV', 'NIC', 'POL', 'BHR', 'UZB', 
                     'KWT', 'THA', 'ITA', 'ECU', 'BLZ', 'LTU', 'SVN', 'ROU', 'LVA', 'JPN', 'MUS',
                     'JAM', 'KOR', 'NaN', 'RUS', 'KAZ', 'CYP', 'BOL', 'EST', 'PRY', 'PER', 'KOS',
                     'MDA', 'TKM', 'HUN', 'LBY', 'PHL', 'HND', 'BLR', 'TUR', 'PAK', 'HKG', 'PRT',
                     'SRB', 'GRC', 'LBN', 'MNE', 'HRV', 'DOM', 'DZA', 'MAR', 'CHN', 'AZE', 'TJK',
                     'MKD', 'JOR', 'NGA', 'KGZ', 'BIH', 'MNG', 'VNM', 'IDN', 'BTN', 'SOM', 'CMR', 
                     'BGR', 'NPL', 'VEN', 'GAB', 'PSE', 'ZAF', 'ZAF', 'IRN', 'GHA', 'SEN', 'LAO',
                     'TUN', 'ALB', 'SLE', 'COG', 'BGD', 'LKA', 'IRQ', 'MLI', 'NAM', 'KHM', 'BFA',
                     'EGY', 'MOZ', 'KEN', 'ZMB', 'MRT', 'ETH', 'GEO', 'ARM', 'MMR', 'TCD', 'COD',
                     'IND', 'NER', 'UGA', 'BEN', 'SDN', 'UKR', 'TGO', 'GIN', 'LSO', 'AGO', 'MDG', 
                     'ZWE', 'AFG', 'BWA', 'MWI', 'HTI', 'LBR', 'SYR', 'RWA', 'YEM', 'TZA', 'SSD',
                     'CAF', 'BDI']
df4['Continent']=df4['Country code'].apply(continent)
df4=df4[['Country or region','Country code', 'Continent','Overall rank','Score', 'GDP per capita',
       'Social support', 'Healthy life expectancy',
       'Freedom to make life choices', 'Generosity',
       'Perceptions of corruption']]
df4[df4['Continent'].isnull()]
df4.drop( df4[ df4['Country or region'] == 'Northern Cyprus' ].index , inplace=True)
df4.info()
input_countries = df5['Country or region'].unique()

countries = {}
for country in pycountry.countries:
    countries[country.name] = country.alpha_3

df5_codes = [countries.get(country, 'Unknown code') for country in input_countries]

print(df5_codes)
    
df5['Country or region'].unique()
df5['Country code']=['FIN', 'DNK', 'NOR', 'ISL', 'NLD', 'CHE', 'SWE', 'NZL', 'CAN', 'AUT', 'AUS',
                     'CRI', 'ISR', 'LUX', 'GBR', 'IRL', 'DEU', 'BEL', 'USA', 'CZE', 'ARE', 'MLT',
                     'MEX', 'FRA', 'TWN', 'CHL', 'GTM', 'SAU', 'QAT', 'ESP', 'PAN', 'BRA', 'URY',
                     'SGP', 'SLV', 'ITA', 'BHR', 'SVK', 'TTO', 'POL', 'UZB', 'LTU', 'COL', 'SVN',
                     'NIC', 'KOS', 'ARG', 'ROU', 'CYP', 'ECU', 'KWT', 'THA', 'LVA', 'KOR', 'EST',
                     'JAM', 'MUS', 'JPN', 'HND', 'KAZ', 'BOL', 'HUN', 'PRY', 'NaN', 'PER', 'PRT', 
                     'PAK', 'RUS', 'PHL', 'SRB', 'MDA', 'LBY', 'MNE', 'TJK', 'HRV', 'HKG', 'DOM',
                     'BIH', 'TUR', 'MYS', 'BLR', 'GRC', 'MNG', 'MKD', 'NGA', 'KGZ', 'TKM', 'DZA',
                     'MAR', 'AZE', 'LBN', 'IDN', 'CHN', 'VNM', 'BTN', 'CMR', 'BGR', 'GHA', 'CIV', 
                     'NPL', 'JOR', 'BEN', 'COG', 'GAB', 'LAO', 'ZAF', 'ALB', 'VEN', 'KHM', 'PSE', 
                     'SEN', 'SOM', 'NAM', 'NER', 'BFA', 'ARM', 'IRN', 'GIN', 'GEO', 'GMB', 'KEN',
                     'MRT', 'MOZ', 'TUN', 'BGD', 'IRQ', 'COD', 'MLI', 'SLE', 'LKA', 'MMR', 'TCD',
                     'UKR', 'ETH', 'SWZ', 'UGA', 'EGY', 'ZMB', 'TGO', 'IND', 'LBR', 'COM', 'MDG', 
                     'LSO', 'BDI', 'ZWE', 'HTI', 'BWA', 'SYR', 'MWI', 'YEM', 'RWA', 'TZA', 'AFG',
                     'CAF', 'SSD']
df5['Continent']=df5['Country code'].apply(continent)
df5=df5[['Country or region','Country code', 'Continent','Overall rank','Score', 'GDP per capita',
       'Social support', 'Healthy life expectancy',
       'Freedom to make life choices', 'Generosity',
       'Perceptions of corruption']]
df5[df5['Continent'].isnull()]
df5.drop( df5[ df5['Country or region'] == 'Northern Cyprus' ].index , inplace=True)
#making sure columns have same names
df3=df3.rename(columns={'Happiness.Score':'Happiness Score'})
df4=df4.rename(columns={'Score':'Happiness Score'})
#creating first dataframe for 2015
score_15=df1[['Country code','Continent','Happiness Score']]
len(score_15)
l=['2015']*157
score_15['Year']=l
score_15
score_16=df2[['Country code','Continent','Happiness Score']]
len(score_16)
l=['2016']*156
score_16['Year']=l
score_16
score_16.info()
score_17=df3[['Country code','Continent','Happiness Score']]
len(score_17)
l=['2017']*154
score_17['Year']=l
score_17
score_18=df4[['Country code','Continent','Happiness Score']]
len(score_18)
l=['2018']*155
score_18['Year']=l
score_18.info()
df5=df5.rename(columns={'Score':'Happiness Score'})
score_19=df5[['Country code','Continent','Happiness Score']]
len(score_19)
l=['2019']*155
score_19['Year']=l
score_19
data=[score_15,score_16,score_17,score_18,score_19]
cont_score=pd.concat(data,axis=0)
cont_score.info()
score_15.groupby('Continent').mean()
plt.figure(figsize=(12,8))
sns.barplot(x='Continent',y='Happiness Score',hue='Year',data=cont_score,errwidth=0)
sorted(round(cont_score.groupby('Continent')['Happiness Score'].mean(),2))

def display_figures(ax):
    l=[ 6.13,6.11,7.29, 5.26, 6.13, 4.29]
    i=0
    for p in ax.patches:
        h=p.get_height()
        if (h>0):
            value=l[i]
            ax.text(p.get_x()+p.get_width()/2,h+0.08, value, ha='center')
            i=i+1
            
plt.figure(figsize=(12,8))
ax=sns.barplot(x='Continent',y='Happiness Score',data=cont_score,errwidth=0)
display_figures(ax)

plt.figure(figsize=(12,8))
sns.boxplot( hue='Year', y='Happiness Score', x='Continent',data=cont_score)
stat=cont_score.groupby('Continent').describe()
stat[('Happiness score','range')]=stat[('Happiness Score','max')]-stat[('Happiness Score','min')]
stat[('Happiness score','IQR')]=stat[('Happiness Score','75%')]-stat[('Happiness Score','50%')]
stat
cont_score
score_15[score_15['Happiness Score']==score_15['Happiness Score'].max()]
score_15[score_15['Happiness Score']==score_15['Happiness Score'].min()]
score_16[score_16['Happiness Score']==score_16['Happiness Score'].max()]
score_16[score_16['Happiness Score']==score_16['Happiness Score'].min()]
score_17[score_17['Happiness Score']==score_17['Happiness Score'].max()]
score_17[score_17['Happiness Score']==score_17['Happiness Score'].min()]
score_18[score_18['Happiness Score']==score_18['Happiness Score'].max()]
score_18[score_18['Happiness Score']==score_18['Happiness Score'].min()]
score_19[score_19['Happiness Score']==score_19['Happiness Score'].max()]
score_19[score_19['Happiness Score']==score_19['Happiness Score'].min()]
