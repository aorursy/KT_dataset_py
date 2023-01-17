import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns; sns.set(style="ticks", color_codes=True)

import plotly_express as px
import os

print(os.listdir("../input"))
immigData = pd.read_excel('../input/immigration-to-canada-ibm-dataset/Canada.xlsx',

                     sheet_name='Canada by Citizenship',

                     skiprows = range(20),

                     skipfooter = 2)

immigData.sample(4)

countries = pd.read_csv('../input/countries-of-the-world/countries of the world.csv'

                    )

countries.sample(4)
print(immigData.columns)
print(countries.columns)
immigData=immigData.drop(['Type', 'Coverage','AREA', 'AreaName','REG','RegName','DEV','DevName'],axis=1)

immigData['TotalImmig']=immigData.sum(axis=1)

immigData=immigData[['OdName','TotalImmig']]

immigData=immigData.rename(columns={"OdName": "Country"})

#immigData=immigData.set_index("Country")

immigData.sample(4)
toConvTofloat=['Pop. Density (per sq. mi.)', 'Coastline (coast/area ratio)',

       'Net migration', 'Infant mortality (per 1000 births)',

       'GDP ($ per capita)', 'Literacy (%)', 'Phones (per 1000)', 'Arable (%)',

       'Crops (%)', 'Other (%)', 'Climate', 'Birthrate', 'Deathrate',

       'Agriculture', 'Industry', 'Service']

for col in toConvTofloat:

    countries[col]=countries[col].astype(str).str.replace(',','.')

    countries[col]=countries[col].astype(float)

    

countries.sample(4)
immigData['Country']=immigData['Country'].astype(str)

countries['Country']=countries['Country'].astype(str)

countries['Country'] = countries['Country'].str.strip()

countries['Country'] = countries['Country'].str.replace('&','and')

immigData['Country'] = immigData['Country'].str.strip()
#get The non common Countries 

IS=pd.DataFrame()

IS['d']= immigData['Country'].isin(countries['Country'])

immigData['Country'].loc[ IS.index[IS['d']==False]].tolist()
changeCont = {	

			 'Bahamas, The': 'Bahamas',

			 'Cape Verde':'Cabo Verde',

			 'Central African Rep.':'Central African Republic',

			 'Congo, Repub. of the' :'Congo',

			 'Israel':'State of Palestine',

			 'Congo, Dem. Rep.' :'Democratic Republic of the Congo',

			 'Gambia, The':'Gambia',

			 'Montserrat' :'Montenegro'

			 

			 

			 

			 }

			 

chgangeImmi={	'Bolivia (Plurinational State of)': 'Bolivia',

			'Brunei Darussalam':'Brunei' ,

             'Israel':'State of Palestine',

             "Côte d'Ivoire":"Cote d'Ivoire",

             'Syrian Arab Republic':'Syria',

             'Russian Federation':'Russia',

			#'China, Hong Kong Special Administrative Region':'China',

			"Democratic People's Republic of Korea":"Korea, South",

			'Iran (Islamic Republic of)':'Iran' ,

			"Lao People's Democratic Republic":'Laos',

			'Republic of Korea':'Korea, North',

			'Republic of Moldova':'Moldova',

			'The former Yugoslav Republic of Macedonia':'Macedonia',

			'United Kingdom of Great Britain and Northern Ireland':'United Kingdom',

			'United Republic of Tanzania':'Tanzania',

			'United States of America':'United States',

			'Venezuela (Bolivarian Republic of)':'Venezuela',

			'Myanmar':'Burma',

            'Viet Nam':'Vietnam'

}


tempcountries=countries['Country']

tempIcountries=immigData['Country']

countries['Country'] = countries['Country'].map(changeCont, na_action='ignore').fillna(tempcountries)

immigData['Country'] = immigData['Country'].map(chgangeImmi, na_action='ignore').fillna(tempIcountries)



comp=immigData.merge(countries,how='inner',left_on = 'Country', right_on ='Country')#,left_index=True,right_index=True)

comp.shape
IS=pd.DataFrame()

IS['d']= immigData['Country'].isin(countries['Country'])

immigData['Country'].loc[ IS.index[IS['d']==False]].tolist()
comp[comp['Country']=='China']
plt.rcParams['figure.figsize'] = (18, 9)



sns.countplot( comp['Region'], palette = 'hsv')

plt.title('Regions', fontsize = 20, fontweight = 100)

plt.xticks(rotation = 90)

plt.show()
#get a scaled version of the data

from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()

comp=comp.fillna(comp.mean())

scaled=pd.DataFrame( min_max_scaler.fit_transform(comp.drop(['Country','Region'],axis=1)) ,columns= comp.drop(['Country','Region'],axis=1).columns)

comp['%Immig']=(comp['TotalImmig'].div( comp['Population']) ) .astype(float)

corr = comp.corr()



f,ax = plt.subplots(figsize=(12,6))

sns.heatmap(corr, annot = True, linewidths=1.5 , fmt = '.2f',ax=ax)

plt.show()
sns.pairplot(comp[['TotalImmig','%Immig','Population','GDP ($ per capita)','Crops (%)','Literacy (%)','Industry','Service']] ,  diag_kind='kde', size=2);
sns.pairplot(comp[['TotalImmig','%Immig','Population','GDP ($ per capita)','Crops (%)','Literacy (%)','Industry','Service']] ,kind="reg",  diag_kind='kde', size=2);
top10=comp.sort_values(by='%Immig',ascending=False).head(10) 

top10.set_index('Country')[['TotalImmig','%Immig','Population','GDP ($ per capita)','Crops (%)','Literacy (%)','Industry','Service']].reset_index()


grid=sns.pairplot(top10[['Country','TotalImmig','%Immig','Population','GDP ($ per capita)','Crops (%)','Literacy (%)','Industry','Service']] , hue='Country', diag_kind='kde', size=2);

handles = grid._legend_data.values()

labels = grid._legend_data.keys()

grid.fig.legend(handles=handles, labels=labels, loc='lower center', ncol=10)