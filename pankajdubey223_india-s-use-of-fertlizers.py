# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import seaborn as sns

import matplotlib.pyplot as plt
# Load The Data from CSV to pandas format

data = pd.read_csv('/kaggle/input/fertilizers-by-product-fao/FertilizersProduct.csv',encoding='ISO-8859-1')

data.head()
#Visualising The Last 5 Row of Data

data.tail()
# Counting All the Country In dataset

data["Area"].nunique()
India_data = data.loc[data.Area == 'India'] #seprate the data 

data = data.loc[data.Area != 'India']
India_data.head()
# Drop irrelevant columns

India_data.drop(['Area Code','Item Code', 'Element Code', 'Year Code', 'Flag'],inplace=True,axis=1)
India_data.head() # visualisisng the first five row
India_data.Element.value_counts()
India_data.Item.value_counts()
data.Item.value_counts()
data.Item.nunique()
India_data.Item.nunique()
agr_usage = India_data.loc[India_data.Element == 'Agricultural Use']

agr_usage.sort_values(by=['Value'], ascending=False).head()
plt.figure(figsize=(25,25))

sns.barplot(x='Year',

    y='Value',

    hue='Item',

    data=agr_usage

)
plt.figure(figsize=(20,15))

sns.lineplot(x= 'Year',y = 'Value',hue = 'Item',data=agr_usage)

plt.show()
usage_global = data.loc[(data.Element == 'Agricultural Use')  & (data.Year == 2017)]

countries = data.Area.unique()

cdf = []

adf = []

for country in countries:

    df_aux = usage_global.loc[usage_global.Area == country]

    amount = df_aux.Value.sum()

    cdf.append(country)

    adf.append(amount)

df_fert = pd.DataFrame({'Country': cdf, 'Amount': adf})

df_fert = df_fert.sort_values(by=['Amount'], ascending=False)
plt.figure(figsize=(25,25))

sns.barplot(data= df_fert.iloc[:15,:],x = "Country",y = 'Amount')

plt.show()
fert_prod = India_data.loc[India_data.Element == 'Production']

fert_prod.sort_values(by=['Value'], ascending=False).head()
plt.figure(figsize=(10,10))

sns.barplot(data= fert_prod.iloc[:,:10],y = "Item",x = 'Value')

plt.show()
fert_expo = India_data.loc[India_data.Element == 'Export Quantity']

fert_expo.sort_values(by=['Value'], ascending=False).head()
plt.figure(figsize=(10,10))

sns.barplot(data= fert_expo.iloc[:,:10],y = "Item",x = 'Value')

plt.show()
plt.figure(figsize=(10,10))

sns.barplot(data= fert_expo,y = "Item",x = 'Year')

plt.show()
urea_export = India_data.loc[(India_data.Item == 'Urea') & (India_data.Element == 'Export Value')]

urea_export.head()
plt.figure(figsize=(10,10))

sns.barplot(data = urea_export, x="Year", y="Value", label='Exportation of Urea in India (x1000 US$)')

plt.legend()

plt.show()

import math



millnames = ['',' Thousand',' Million',' Billion',' Trillion']



def millify(n):

    n = float(n)

    millidx = max(0,min(len(millnames)-1,

                        int(math.floor(0 if n == 0 else math.log10(abs(n))/3))))



    return '{:.0f}{}'.format(n / 10**(3 * millidx), millnames[millidx])
total = urea_export['Value'].sum()

print (f'India exported US$ {millify(total*1000)} of Urea')
total_export = India_data.loc[(India_data.Element == 'Export Value')]

total_export.head()
india_export = total_export['Value'].sum()

#total = {millify(total*1000)}

print (f'India exported US$ {millify(india_export*1000)} of total fertilizers')
total_import = India_data.loc[(India_data.Element == 'Import Value')]

total_import.head()
india_import = total_import['Value'].sum()

print (f'India imported US$ {millify(india_import*1000)} of total fertilizers')
urea_import = India_data.loc[(India_data.Item == 'Urea') & (India_data.Element == 'Import Value')]

urea_import.head()
total = urea_import['Value'].sum()

print (f'India imported {millify(total*1000)} US$ of Urea')
millify((india_import-india_export)*1000)
prod_value = India_data.loc[(India_data.Element == 'Production')]

prod_value.head()
total = prod_value['Value'].sum()

#print(type(total))

print (f'India produced  {millify(total*1000)} tons of total fertilizers')
total_export_world = data.loc[(data.Element == 'Export Value')]

total_export_world.head()
world_export = total_export_world['Value'].sum()

#world_export = {millify(total*1000)}

print (f'Whole World exported US$ {millify(world_export*1000)} of total fertilizers')
shared_export = round((india_export/world_export)*100,2)



print(f'Total shared percentage of india in export on world label: {shared_export}%')
total_import_world = data.loc[(data.Element == 'Import Value')]

total_import_world.head()
world_import = total_import_world['Value'].sum()

#world_export = {millify(total*1000)}

print (f'Whole World imported US$ {millify(world_import*1000)} of total fertilizers')
shared_import = round((india_import/world_import)*100,3)



print(f'Total shared  percentage of india in import on world label: {shared_import}%')
#plt.figure(figsize=(8,8))

plt.pie(x = [world_import,india_import],labels = ['Total World Import Fertilizers',"Total India's Import Fertilizers "],shadow=True,

    labeldistance=1.1,

    startangle=None,

    radius=2,autopct= '%.2f%%')

plt.show()
plt.pie(x = [world_export,india_export],labels = ['Total World Export Fertilizers',"Total India's Export Fertilizers "],shadow=True,

    labeldistance=1.1,

    startangle=None,

    radius=2,autopct= '%.2f%%')

plt.show()
world_prod = data.loc[data.Element=='Production']

world_prod = world_prod['Value'].sum()
india_prod = India_data.loc[(India_data.Element == 'Production')]

india_prod.head()
india_prod = india_prod['Value'].sum()

type(india_prod)
plt.pie(x = [world_prod,india_prod],labels = ['Total World Production in Fertilizers',"Total India's Production in Fertilizers "],shadow=True,

    labeldistance=1.1,

    startangle=None,

    radius=2,autopct= '%.2f%%')

plt.show()
india_urea = India_data.loc[(India_data.Item == 'Urea') & (India_data.Element == 'Production') & (India_data.Area == 'India')]
india_urea.head()
india_urea = india_urea['Value'].sum()

india_urea
world_urea = data.loc[(data.Item == 'Urea') & (data.Element == 'Production')]
world_urea.head()
world_urea = world_urea['Value'].sum()
plt.pie(x = [world_urea,india_urea],labels = ['Total World Production in Urea',"Total India's Production in Urea "],shadow=True,

    labeldistance=1.1,

    startangle=None,

    radius=5,autopct= '%.2f%%')

plt.show()