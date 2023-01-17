# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns# plotting libraries

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import_india=pd.read_csv('/kaggle/input/india-trade-data/2018-2010_import.csv')# reading files

export_india=pd.read_csv('/kaggle/input/india-trade-data/2018-2010_export.csv')
print("No of Country were we are importing Comodities are "+str(len(import_india['country'].unique())))

print("No of Country were we are Exporting Comodities are "+str(len(export_india['country'].unique())))
import pandas_profiling# trick and everyone should use it

# import_india.profile_report()
import_india.head(10)

import_india.dtypes

import_india.isnull().sum()
import_india.dropna(inplace=True)

import_india.drop_duplicates(keep="first",inplace=True) # removving all the duplicate rows
country_list=list(import_india.country.unique())

import_india.loc[import_india.country=='UNSPECIFIED']
country_group=import_india.groupby('country')

ls=[]

for country_name in country_list:

    ls.append([country_name, country_group.get_group(str(country_name)).value.sum() ])



total = pd.DataFrame(ls, columns = ['country', 'total_imports']) 



total.loc[total.total_imports==0] # query to check th names of country that have total imports equal to  zero in India

# This shows that all countries have exported their commodities to India

largest_importers_dataframe=total.nlargest(5,['total_imports'])

largest_importers_dataframe['total_imports']=largest_importers_dataframe['total_imports']/1000 # to convert value of import from MILLION $ to BILLION $



plt.figure(figsize=(10,10))

sns.set_style('whitegrid')

largest_importers_bar=sns.barplot(x=largest_importers_dataframe['country'],y=largest_importers_dataframe['total_imports'])

plt.xlabel('COUNTRIES',size=18)

plt.ylabel('Total Imports in Billion $',size=18)

plt.title('LARGEST IMPORTERS TO INDIA 2010-2018',SIZE=20)
top_5_importers_sum=largest_importers_dataframe.total_imports.sum() # sum of top 5 importers in billion dollars from 2010 to 2018



rest_of_the_world=total.sort_values('total_imports',ascending=False)[5:].total_imports.sum()/1000 # sum of Import from rest of the world(leaving top 5) from 2010 to 2018 in billion dollars



labels=['Rest of the world','Top 5']



colors = ['#99ff99','#ffcc99']



sizes=[rest_of_the_world,top_5_importers_sum]



explode = [ 0.1, 0.3]

plt.rcParams['figure.figsize'] = (9, 9)

plt.pie(sizes, labels = labels, colors = colors, explode = explode, shadow = True,autopct='%1.2f')

plt.title('Top 5 Importers vs Rest of The world', fontsize = 20)

plt.legend()

plt.show()
total.nlargest(5,['total_imports'])
china=country_group.get_group('CHINA P RP')

imports_per_year_china=[]



for i in list(china['year'].unique())[::-1]:

    imports_per_year_china.append(china.sort_values('year').loc[china.year==i].value.sum()/1000)



year_china=list(china['year'].unique())[::-1]





plt.plot(year_china,imports_per_year_china,alpha=1,color='red',marker='o')# alpha is for line brightness alpha=0 is transparent and alpha=1 is opaque



plt.xlabel('YEAR',size=17)

plt.ylabel('IMPORTS IN BILLION $',size=17)

plt.fill_between(year_china,imports_per_year_china,facecolor='#99ff99')

plt.title('TRADE TRENDS',size=20)
fig, ax = plt.subplots()

###################################################################################

UAE=country_group.get_group('U ARAB EMTS')

imports_per_year_uae=[]



for i in list(UAE['year'].unique())[::-1]:

    imports_per_year_uae.append(UAE.sort_values('year').loc[UAE.year==i].value.sum()/1000)



year_uae=list(UAE['year'].unique())[::-1]



##################################################################################



saudi=country_group.get_group('SAUDI ARAB')

imports_per_year_saudi=[]



for i in list(saudi['year'].unique())[::-1]:

    imports_per_year_saudi.append(saudi.sort_values('year').loc[saudi.year==i].value.sum()/1000)



year_saudi=list(saudi['year'].unique())[::-1]



####################################################################################



uae_line, = ax.plot(year_uae,imports_per_year_uae,alpha=0.99,color='green',marker='o',label='United Arab Emirates')# alpha is for line brightness alpha=0 is transparent and alpha=1 is opaque



saudi_line, = ax.plot(year_saudi,imports_per_year_saudi,alpha=0.7,color='blue',marker='o',label='Saudi Arabia')



plt.xlabel('YEAR',size=17)

plt.ylabel('IMPORTS IN BILLION $',size=17)

plt.title('TRADE TRENDS',size=20)





ax.legend()

plt.show()
china_detail=country_group.get_group('CHINA P RP')



c=china_detail.groupby('HSCode')



item_value_china=[]



for item in list(set(china_detail.HSCode)):# taking HScode only once(used set) from china_detail dataframe

    item_value_china.append([item,

                             round(c.get_group(item).value.sum()/1000,5),# divide by 1000 to convert to Billion $ and 5 is to round off the result to first 5 digits after decimal

                             list(china_detail.loc[china_detail.HSCode==item].Commodity)[0]]),

          

df_china = pd.DataFrame(item_value_china, columns = ['HScode','Total_value', 'Name'])      



print(df_china.sort_values('Total_value',ascending=False)[:5])# top 5 imported items from China

df_china=df_china.sort_values('Total_value',ascending=False)



rest_of_the_imports_china=df_china.sort_values('Total_value',ascending=False)[5:].Total_value.sum() # sum of Import from rest of the world(leaving top 5) from 2010 to 2018 in billion dollars



labels=['Rest of the imports','ELECTRICAL MACHINERY AND EQUIPMENT AND PARTS','NUCLEAR REACTORS, BOILERS, MACHINERY'

,'ORGANIC CHEMICALS','FERTILISERS','PROJECT GOODS; SOME SPECIAL USES']



colors = ['#99ff99','#ffcc99','#66b3ff','#33ff33','#cc9966','#d279d2']



sizes=[rest_of_the_imports_china,df_china.Total_value[83],df_china.Total_value[82],df_china.Total_value[28],df_china.Total_value[30],df_china.Total_value[96]]

# these numbers 83 ,82,28,30,96 are the indexes of items in df_china dataframe



explode = [ 0.03, 0.03,0.1,0.1,0.1,0.3]

plt.rcParams['figure.figsize'] = (9, 9)

plt.pie(sizes, labels = labels, colors = colors, explode = explode, shadow = True,autopct='%1.2f')

plt.title('Top Imported Items from China', fontsize = 20)

plt.show()
export_india.head()

export_india.dtypes

export_india.isnull().sum()

export_india.dropna(inplace=True)

export_india.drop_duplicates(keep="first",inplace=True) # removving all the duplicate rows

country_export_list=list(export_india.country.unique())

# country_export_list
export_india.loc[export_india.country=='UNSPECIFIED']
country_export_group=export_india.groupby('country')

ls=[]

for country_name in country_export_list:

    ls.append([country_name, country_export_group.get_group(str(country_name)).value.sum() ])



total_exports = pd.DataFrame(ls, columns = ['country', 'total_exports']) 

# total_exports.sort_values(by='total_exports',ascending=0)





total_exports.loc[total_exports.total_exports==0] # query to check th names of country that have total imports equal to  zero in India

# This shows that all countries have exported their commodities to India

largest_exporters_dataframe=total_exports.nlargest(5,['total_exports'])

largest_exporters_dataframe['total_exports']=largest_exporters_dataframe['total_exports']/1000 # to convert value of import from MILLION $ to BILLION $



plt.figure(figsize=(10,10))

sns.set_style('whitegrid')

largest_exporters_bar=sns.barplot(x=largest_exporters_dataframe['country'],y=largest_exporters_dataframe['total_exports'])

plt.xlabel('COUNTRIES',size=18)

plt.ylabel('Total exports in Billion $',size=18)

plt.title('LARGEST EXPORTERS OF INDIA 2010-2018',SIZE=20)
usa_detail=country_export_group.get_group('U S A')



u=usa_detail.groupby('HSCode')



item_value_usa=[]



for item in list(set(usa_detail.HSCode)):# taking HScode only once(used set) from usa_detail dataframe

    item_value_usa.append([item,

                             round(u.get_group(item).value.sum()/1000,5),# divide by 1000 to convert to Billion $ and 5 is to round off the result to first 5 digits after decimal

                             list(usa_detail.loc[usa_detail.HSCode==item].Commodity)[0]]),

          

df_usa = pd.DataFrame(item_value_usa, columns = ['HScode','Total_value', 'Name'])      



print(df_usa.sort_values('Total_value',ascending=False)[:5])# top 5 imported items from USA

df_usa=df_usa.sort_values('Total_value',ascending=False)



rest_of_the_imports_usa=df_usa.sort_values('Total_value',ascending=False)[5:].Total_value.sum() # sum of Import from rest of the world(leaving top 5) from 2010 to 2018 in billion dollars



labels=['Rest of the imports','NATURAL OR CULTURED PEARLS','PHARMACEUTICAL PRODUCTS'

,'MINERAL FUELS, MINERAL OILS AND PRODUCTS','OTHER MADE UP TEXTILE ARTICLES','NUCLEAR REACTORS, BOILERS, MACHINERY']



colors = ['#99ff99','#ffcc99','#cc9966','#d279d2','#66b3ff','#33ff33']



sizes=[rest_of_the_imports_usa,df_usa.Total_value[70],df_usa.Total_value[29],df_usa.Total_value[26],df_usa.Total_value[62],df_usa.Total_value[82]]



explode = [ 0.03, 0.03,0.1,0.1,0.1,0.3]

plt.rcParams['figure.figsize'] = (9, 9)

plt.pie(sizes, labels = labels, colors = colors, explode = explode, shadow = True,autopct='%1.2f')

plt.title('Top Imported Items to USA', fontsize = 20)

plt.show()
total_imports_per_year=import_india.groupby('year').agg({'value':'sum'})

total_exports_per_year=export_india.groupby('year').agg({'value':'sum'})



trade_deficit=[round(list(total_imports_per_year.value/1000)[i]-list(total_exports_per_year.value/1000)[i],2) for i in range(len(total_exports_per_year.index))]

#divided by 1000 to calculate trade deficit in Billion dollars

trade_deficit=pd.Series(trade_deficit,index=total_exports_per_year.index)

sns.set_style('whitegrid')

trade_deficit.plot(kind='bar',colors=['green','red'])

plt.xlabel('COUNTRIES',size=18)

plt.ylabel('TRADE DEFICIT in Billion $',size=18)

plt.title('YEARS',SIZE=20)