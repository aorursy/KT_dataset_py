import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import warnings

warnings.filterwarnings('ignore')

sns.set_style('whitegrid')

import_data = pd.read_csv('/kaggle/input/india-trade-data/2018-2010_import.csv')

export_data = pd.read_csv('/kaggle/input/india-trade-data/2018-2010_export.csv')

def remove_0_val(blob):

    if blob == 0:

        return False

    else:

        return True

import_data = import_data[import_data['value'].apply(remove_0_val)]

export_data = export_data[export_data['value'].apply(remove_0_val)]
import_data_temp = import_data.copy(deep=True)

export_data_temp = export_data.copy(deep=True)

import_data_temp['commodity_sum'] = import_data_temp['value'].groupby(import_data_temp['Commodity']).transform('sum')

export_data_temp['commodity_sum'] = export_data_temp['value'].groupby(export_data_temp['Commodity']).transform('sum')

import_data_temp.drop(['value','country','year','HSCode'],axis=1,inplace=True)

export_data_temp.drop(['value','country','year','HSCode'],axis=1,inplace=True)

import_data_temp.drop_duplicates(inplace=True)

export_data_temp.drop_duplicates(inplace=True)

import_data_temp.sort_values(by='commodity_sum',inplace=True,ascending=False)

export_data_temp.sort_values(by='commodity_sum',inplace=True,ascending=False)

import_data_temp['Commodity'] = import_data_temp['Commodity'].apply(lambda x:x.split(',')[0].split()[0])

fig, ax = plt.subplots(1, 1, figsize = (16, 10), dpi=500)

splot = sns.barplot(data=import_data_temp.head(7),y='Commodity',x='commodity_sum')

temp1 = import_data_temp['Commodity'].tolist()

ax.set_ylabel('Commodity')

ax.set_xlabel('Cumulative value for 2010-2018(million US$)')

plt.title('Highest valued imported goods to India for 2010-2018 according to aggregate value')

plt.show()
export_data_temp['Commodity'] = export_data_temp['Commodity'].apply(lambda x:x.split(',')[0].split()[0])

fig, ax = plt.subplots(1, 1, figsize = (16, 10), dpi=500)

splot = sns.barplot(data=export_data_temp.head(7),y='Commodity',x='commodity_sum')

temp1 = export_data_temp['Commodity'].tolist()

ax.set_ylabel('Commodity')

ax.set_xlabel('Cumulative value for 2010-2018(million US$)')

plt.title('Highest valued exported goods from India for 2010-2018 according to aggregate value')

plt.show()
most_imports_from = {}

most_exports_from = {}

import_data['temp1'] = import_data['value'].groupby(import_data['country']).transform('sum')

export_data['temp1'] = export_data['value'].groupby(export_data['country']).transform('sum')

for _ in range(2010,2019):

    x_1 = import_data[import_data['year']==_].sort_values('value',ascending=False)[:5][['country','temp1']][:5]

    for i in range(5):

        x_2 = x_1.iloc[i]

        if x_2.country in most_imports_from :

            most_imports_from[x_2.country] += x_2.temp1

        else:

            most_imports_from[x_2.country] = x_2.temp1

for _ in range(2010,2019):

    x_1 = export_data[export_data['year']==_].sort_values('value',ascending=False)[:5][['country','temp1']][:5]

    for i in range(5):

        x_2 = x_1.iloc[i]

        if x_2.country in most_exports_from :

            most_exports_from[x_2.country] += x_2.temp1

        else:

            most_exports_from[x_2.country] = x_2.temp1
plt.figure(figsize=(16,8))

temp1 = []

temp2 = []

for i,j in most_imports_from.items():

    temp1.append(i)

    temp2.append(j)

fig, ax = plt.subplots(1, 1, figsize = (16, 10), dpi=500)

splot = sns.barplot(x=temp1,y=temp2)

ax.set_xlabel('Countries')

ax.set_ylabel('Value per year(million US$)')

plt.title('Countries that imported most goods to India(for 2010-2018) according to aggregate value')

for i,p in enumerate(splot.patches):

    temp_x = str(temp2[i])[:11]

    splot.annotate(temp_x, (p.get_x() + p.get_width() / 2., p.get_height())

               , ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

plt.show()
plt.figure(figsize=(16,8))

temp1 = []

temp2 = []

for i,j in most_exports_from.items():

    temp1.append(i)

    temp2.append(j)

fig, ax = plt.subplots(1, 1, figsize = (16, 10), dpi=500)

splot = sns.barplot(x=temp1,y=temp2)

ax.set_xlabel('Countries')

ax.set_ylabel('Value per year(million US$)')

plt.title('Countries that exported most goods from India(for 2010-2018) according to aggregate value')

for i,p in enumerate(splot.patches):

    temp_x = str(temp2[i])[:11]

    splot.annotate(temp_x, (p.get_x() + p.get_width() / 2., p.get_height())

               , ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

import_data.drop(['temp1'],axis=1,inplace=True)

export_data.drop(['temp1'],axis=1,inplace=True)

plt.show()
import_commodity_food = [

    'PRODUCTS OF ANIMAL ORIGIN, NOT ELSEWHERE SPECIFIED OR INCLUDED.',

    'EDIBLE VEGETABLES AND CERTAIN ROOTS AND TUBERS.',

    'EDIBLE FRUIT AND NUTS; PEEL OR CITRUS FRUIT OR MELONS.',

    'COFFEE, TEA, MATE AND SPICES.',

    'PRODUCTS OF THE MILLING INDUSTRY; MALT; STARCHES; INULIN; WHEAT GLUTEN.',

    'OIL SEEDS AND OLEA. FRUITS; MISC. GRAINS, SEEDS AND FRUIT; INDUSTRIAL OR MEDICINAL PLANTS; STRAW AND FODDER.',

    'LAC; GUMS, RESINS AND OTHER VEGETABLE SAPS AND EXTRACTS.',

    'PREPARATIONS OF VEGETABLES, FRUIT, NUTS OR OTHER PARTS OF PLANTS.',

    'BEVERAGES, SPIRITS AND VINEGAR.',

    'FISH AND CRUSTACEANS, MOLLUSCS AND OTHER AQUATIC INVERTABRATES.',

    "DAIRY PRODUCE; BIRDS' EGGS; NATURAL HONEY; EDIBLE PROD. OF ANIMAL ORIGIN, NOT ELSEWHERE SPEC. OR INCLUDED.",

    'RESIDUES AND WASTE FROM THE FOOD INDUSTRIES; PREPARED ANIMAL FODER.',

    'SUGARS AND SUGAR CONFECTIONERY.', 'MEAT AND EDIBLE MEAT OFFAL.',

    'PREPARATIONS OF CEREALS, FLOUR, STARCH OR MILK; PASTRYCOOKS PRODUCTS.',

    'MISCELLANEOUS EDIBLE PREPARATIONS.',

    'MANUFACTURES OF STRAW, OF ESPARTO OR OF OTHER PLAITING MATERIALS; BASKETWARE AND WICKERWORK.',

    'PREPARATIONS OF MEAT, OF FISH OR OF CRUSTACEANS, MOLLUSCS OR OTHER AQUATIC INVERTEBRATES',

]
def separate_food(blob):

    if blob in import_commodity_food:

        return True

    else:

        return False

import_data_food =  import_data[import_data['Commodity'].apply(separate_food)]

import_data_food_value = import_data_food.copy(deep=True)

import_data_food_value.dropna(axis=0,inplace=True)

del import_data_food_value['country']

import_data_food_value['total_value'] = import_data_food_value['value'].groupby(import_data_food_value['Commodity']).transform('sum')

del import_data_food_value['value']

top_10_food_commodities_import = []

commodity_food_import_dict = {}

for i in import_data_food_value['year'].unique():

    temp = import_data_food_value[import_data_food_value['year']==i].copy(deep=True)

    temp['Value'] = temp['total_value'].groupby(temp['Commodity']).transform('sum')

    del temp['total_value']

    temp.drop_duplicates(inplace=True)

    temp.sort_values('Value',inplace=True,ascending=False)

    top_10_food_commodities_import.append(temp['Commodity'].tolist()[0:10])

for i in top_10_food_commodities_import:

    for j in i:

        if j in commodity_food_import_dict:

            commodity_food_import_dict[j] += 1

        else:

            commodity_food_import_dict[j] = 1
temp = import_data_food.copy(deep=True)

temp['Value_year'] = temp['value'].groupby(temp['year']).transform('sum')

del temp['HSCode']

del temp['Commodity']

del temp['value']

del temp['country']

temp.drop_duplicates(inplace=True)

temp.sort_values('year',ascending=True,inplace=True)

fig, ax = plt.subplots(1, 1, figsize = (16, 10), dpi=500)

splot = sns.barplot(x='year',y='Value_year',data=temp)

ax.set_xlabel('Years')

ax.set_ylabel('Value per year(million US$)')

plt.title('Total value of food imported to India per year')

for i,p in enumerate(splot.patches):

    temp_x = str(temp.Value_year.iloc[i])[:9]

    splot.annotate(temp_x, (p.get_x() + p.get_width() / 2., p.get_height())

               , ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

plt.show()
import_data_nuclear = import_data[import_data['Commodity']=='NUCLEAR REACTORS, BOILERS, MACHINERY AND MECHANICAL APPLIANCES; PARTS THEREOF.'].copy(deep=True)

import_data_nuclear.dropna(axis=0,inplace=True)

del import_data_nuclear['Commodity']

top_10_nuclear_commodities_import = []

commodity_nuclear_import_dict = {}

for i in import_data_nuclear['year'].unique():

    #plt.figure(i-2010,figsize=(14,6))

    temp = import_data_nuclear[import_data_nuclear['year']==i].copy(deep=True)

    temp.sort_values('value',inplace=True,ascending=False)

    top_10_nuclear_commodities_import.append(temp['country'].tolist()[0:10])

    for j in temp['country'].tolist()[0:10]:

        if j in commodity_nuclear_import_dict:

            commodity_nuclear_import_dict[j] += 1

        else:

            commodity_nuclear_import_dict[j] = 0

temp = import_data_nuclear.copy(deep=True)

temp['Value'] = temp['value'].groupby(temp['year']).transform('sum')

del temp['value']

del temp['HSCode']

del temp['country']

temp.drop_duplicates(inplace=True)

temp.sort_values('year',ascending=True,inplace=True)

fig, ax = plt.subplots(1, 1, figsize = (16, 10), dpi=500)

splot = sns.barplot(x=temp.year,y=temp.Value)

ax.set_xlabel('Years')

ax.set_ylabel('Value per year(million US$)')

plt.title('Total value of nuclear commodity imported to India per year')

for i,p in enumerate(splot.patches):

    temp_x = str(temp.Value.iloc[i])[:9]

    splot.annotate(temp_x, (p.get_x() + p.get_width() / 2., p.get_height())

               , ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

plt.show()
import_vehicle = import_data[import_data['Commodity']=='VEHICLES OTHER THAN RAILWAY OR TRAMWAY ROLLING STOCK, AND PARTS AND ACCESSORIES THEREOF.'].copy(deep=True)

del import_vehicle['Commodity']

del import_vehicle['HSCode']

del import_vehicle['country']

import_vehicle['Value'] = import_vehicle['value'].groupby(import_vehicle['year']).transform('sum')

del import_vehicle['value']

import_vehicle.drop_duplicates(inplace=True)

import_vehicle.sort_values('year',ascending=True,inplace=True)

fig, ax = plt.subplots(1, 1, figsize = (16, 10), dpi=500)

splot = sns.barplot(x=import_vehicle.year,y=import_vehicle.Value)

ax.set_xlabel('Years')

ax.set_ylabel('Value per year(million US$)')

plt.title('Total value of vehicle imported in India per year')

for i,p in enumerate(splot.patches):

    temp_x = str(import_vehicle.Value.iloc[i])[:9]

    splot.annotate(temp_x, (p.get_x() + p.get_width() / 2., p.get_height())

               , ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

plt.show()
export_vehicle = export_data[export_data['Commodity']=='VEHICLES OTHER THAN RAILWAY OR TRAMWAY ROLLING STOCK, AND PARTS AND ACCESSORIES THEREOF.'].copy(deep=True)

del export_vehicle['Commodity']

del export_vehicle['HSCode']

del export_vehicle['country']

export_vehicle['Value'] = export_vehicle['value'].groupby(export_vehicle['year']).transform('sum')

del export_vehicle['value']

export_vehicle.drop_duplicates(inplace=True)

export_vehicle.sort_values('year',ascending=True,inplace=True)

fig, ax = plt.subplots(1, 1, figsize = (16, 10), dpi=500)

splot = sns.barplot(x=export_vehicle.year,y=export_vehicle.Value)

ax.set_xlabel('Years')

ax.set_ylabel('Value per year(million US$)')

plt.title('Total value of vehicle exported from India per year')

for i,p in enumerate(splot.patches):

    temp_x = str(export_vehicle.Value.iloc[i])[:9]

    splot.annotate(temp_x, (p.get_x() + p.get_width() / 2., p.get_height())

               , ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

plt.show()
export_nuclear = export_data[export_data['Commodity']=='NUCLEAR REACTORS, BOILERS, MACHINERY AND MECHANICAL APPLIANCES; PARTS THEREOF.'].copy(deep=True)

export_nuclear['Value'] = export_nuclear['value'].groupby(export_nuclear['year']).transform('sum')

del export_nuclear['value']

del export_nuclear['HSCode']

del export_nuclear['country']

export_nuclear.drop_duplicates(inplace=True)

export_nuclear.sort_values('year',ascending=True,inplace=True)

fig, ax = plt.subplots(1, 1, figsize = (16, 10), dpi=500)

splot = sns.barplot(x=export_nuclear.year,y=export_nuclear.Value)

ax.set_xlabel('Years')

ax.set_ylabel('Value per year(million US$)')

plt.title('Total value of Nuclear commodity Exports from India per year')

for i,p in enumerate(splot.patches):

    temp_x = str(export_nuclear.Value.iloc[i])[:9]

    splot.annotate(temp_x, (p.get_x() + p.get_width() / 2., p.get_height())

               , ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

plt.show()
imports_from_usa = import_data[import_data['country']=='U S A']

imports_from_usa['Value_year'] = imports_from_usa['value'].groupby(imports_from_usa['year']).transform('sum')

usa_imports_year_value = imports_from_usa[['Value_year','year']]

usa_imports_year_value.drop_duplicates(inplace=True)

usa_imports_year_value.sort_values('year',ascending=True,inplace=True)

fig, ax = plt.subplots(1, 1, figsize = (16, 10), dpi=500)

splot = sns.barplot(x=usa_imports_year_value.year,y=usa_imports_year_value.Value_year)

ax.set_xlabel('Years')

ax.set_ylabel('Value per year(million US$)')

plt.title('Total value of imported goods from USA per year(India)')

for i,p in enumerate(splot.patches):

    temp_x = str(usa_imports_year_value.Value_year.iloc[i])[:9]

    splot.annotate(temp_x, (p.get_x() + p.get_width() / 2., p.get_height())

               , ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

plt.show()
del imports_from_usa['Value_year']

del imports_from_usa['country']

del imports_from_usa['HSCode']

commodity_usa_import = {}

for i in range(2010,2019):

        temp = imports_from_usa[imports_from_usa['year']==i]

        temp.sort_values('value',ascending=False,inplace=True)

        commodity_usa_import[i] = (temp.iloc[0].value,temp.iloc[0].Commodity)

temp1 = [i for i in commodity_usa_import.keys()]

temp2 = [i[1] for i in commodity_usa_import.values()]

temp3 = [i[0] for i in commodity_usa_import.values()]

fig, ax = plt.subplots(1, 1, figsize = (16, 10), dpi=500)

splot = sns.barplot(y=temp3,x=temp1)

ax.set_xlabel('Years')

ax.set_ylabel('Value per year(million US$)')

plt.title('Most valued imported commodity per year from USA(India)')

for i,p in enumerate(splot.patches):

    temp_x = temp2[i].split(',')[0]

    splot.annotate(temp_x, (p.get_x() + p.get_width() / 2., p.get_height()/2)

               , ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points',rotation=90)

plt.show()
exports_to_usa = export_data[export_data['country']=='U S A']

del exports_to_usa['HSCode']

del exports_to_usa['country']

exports_to_usa['Value_year'] = exports_to_usa['value'].groupby(exports_to_usa['year']).transform('sum')

usa_exports_year_value = exports_to_usa[['Value_year','year']]

usa_exports_year_value.drop_duplicates(inplace=True)

usa_exports_year_value.sort_values('year',ascending=True,inplace=True)

fig, ax = plt.subplots(1, 1, figsize = (16, 10), dpi=500)

splot = sns.barplot(x=usa_exports_year_value.year,y=usa_exports_year_value.Value_year)

ax.set_xlabel('Years')

ax.set_ylabel('Value per year(million US$)')

plt.title('Total value of exported goods to USA per year(India)')

for i,p in enumerate(splot.patches):

    temp_x = str(usa_exports_year_value.Value_year.iloc[i])[:9]

    splot.annotate(temp_x, (p.get_x() + p.get_width() / 2., p.get_height())

               , ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

plt.show()
del exports_to_usa['Value_year']

commodity_usa_export = {}

for i in range(2010,2019):

        temp = exports_to_usa[exports_to_usa['year']==i]

        temp.sort_values('value',ascending=False,inplace=True)

        commodity_usa_export[i] = (temp.iloc[0].value,temp.iloc[0].Commodity)

temp1 = [i for i in commodity_usa_import.keys()]

temp2 = [i[1] for i in commodity_usa_import.values()]

temp3 = [i[0] for i in commodity_usa_import.values()]

fig, ax = plt.subplots(1, 1, figsize = (16, 10), dpi=500)

splot = sns.barplot(y=temp3,x=temp1)

ax.set_xlabel('Years')

ax.set_ylabel('Value per year(million US$)')

plt.title('Most valued exported commodity per year to USA(India)')

for i,p in enumerate(splot.patches):

    temp_x = temp2[i].split(',')[0]

    splot.annotate(temp_x, (p.get_x() + p.get_width() / 2., p.get_height()/2)

               , ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points',rotation=90)

plt.show()