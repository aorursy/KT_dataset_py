import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import re
energy = pd.read_excel('../input/energy-indicators/Energy_Indicators.xls',skiprows=17, usecols=[2,3,4,5], skip_footer=38)
energy = energy.replace({'...': np.nan})
energy.columns = ['Country', 'Energy Supply', 'Energy Supply per Capita', '% Renewable']
energy = energy.replace({"Republic of Korea": "South Korea","United States of America": "United States",
                        "United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
                        "China, Hong Kong Special Administrative Region": "Hong Kong",
                        'Australia1':'Australia',
                        'Bolivia (Plurinational State of)':'Bolivia',
                        'China2':'China',
                        'China, Hong Kong Special Administrative Region3':'China, Hong Kong Special Administrative Region',
                        'China, Macao Special Administrative Region4':'China, Macao Special Administrative Region',
                        'Denmark5':'Denmark',
                        'Falkland Islands (Malvinas)':'Falkland Islands',
                        'France6':'France',
                        'Greenland7':'Greenland',
                        'Indonesia8':'Indonesia',
                        'Iran (Islamic Republic of)':'Iran',
                        'Italy9':'Italy',
                        'Japan10':'Japan',
                        'Kuwait11':'Kuwait',
                        'Micronesia (Federated States of)':'Micronesia',
                        'Netherlands12':'Netherlands',
                        'Portugal13':'Portugal',
                        'Saudi Arabia14':'Saudi Arabia',
                        'Serbia15':'Serbia',
                        'Sint Maarten (Dutch part)':'Sint Maarten',
                        'Spain16':'Spain',
                        'Switzerland17': 'Switzerland',
                        'Ukraine18':'Ukraine','United States of America20':'United States of America',
                        'Venezuela (Bolivarian Republic of)':'Venezuela',
                        'United Kingdom of Great Britain and Northern Ireland19':'United Kingdom of Great Britain and Northern Ireland',})
energy['Energy Supply'] *= 1000000
energy.head()
GDP =  pd.read_csv('../input/world-bank-data/world_bank.csv', skiprows=[0,1,2,3])
GDP.replace({"Korea, Rep.": "South Korea","Iran, Islamic Rep.": "Iran","Hong Kong SAR, China": "Hong Kong"})
GDP = GDP.rename(columns={'Country Name': 'Country'})
# trim to just last 10 years
GDP = GDP.loc[:,['Country', 'Country Code', 'Indicator Name', 'Indicator Code','2006', '2007','2008', '2009',
                  '2010', '2011', '2012', '2013', '2014', '2015','2016']]
GDP.head()
ScimEn = pd.read_excel('../input/publications/scimagojr-3.xlsx')
ScimEn = ScimEn.loc[:15]
ScimEn.head()
merge1 = pd.merge(ScimEn, energy, how='left', on='Country', left_on=None, right_on=None,
         left_index=False, right_index=False, sort=True,
         suffixes=('_x', '_y'), copy=True, indicator=False)
finalForm = pd.merge(merge1, GDP, how='left', on= 'Country')

finalForm = finalForm.drop(['H index','Country Code','Indicator Name','Indicator Code'],axis=1)
finalForm =  finalForm.rename(columns={'Rank':'Research Rank','2006':'GDP 2006', '2007':'GDP 2007','2008': 'GDP 2008', '2009':'GDP 2009','2010':'GDP 2010', '2011':'GDP 2011', '2012':'GDP 2012', '2013':'GDP 2013', '2014':'GDP 2014', '2015':'GDP 2015','2016':'GDP 2016'})
finalForm.sort_values(by='Research Rank', inplace=True)
finalForm.set_index('Country', inplace=True)
finalForm.head()
finalForm.loc[:,['GDP 2006', 'GDP 2007', 'GDP 2008', 'GDP 2009', 'GDP 2010', 'GDP 2011', 'GDP 2012', 'GDP 2013', 'GDP 2014', 'GDP 2015','GDP 2016']].mean(axis=1).sort_values(ascending=False)
mean = finalForm.loc[:,'Energy Supply per Capita'].mean()
print(str(mean) +' Gigajoules per capita is the global average')
df = finalForm.loc[:,'% Renewable'].sort_values(ascending=False)
df
ratio = finalForm.loc[:,'Self-citations'] / finalForm.loc[:,'Citations']
ratio = ratio.sort_values()
print(ratio)