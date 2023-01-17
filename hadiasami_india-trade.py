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
export_data = pd.read_csv("../input/india-trade-data/2018-2010_export.csv")

import_data = pd.read_csv("../input/india-trade-data/2018-2010_import.csv")
export_data.head(5)
import_data.head(5)
export_data.shape
export_data.isnull().sum()
export_data.shape
export_data.shape
export_data.isnull().sum()
import_data.isnull().sum()
import_data.shape
mean_value = export_data.value.mean()

export_data.value.fillna(mean_value, inplace = True )

export_data.drop_duplicates(keep="first",inplace=True) 

export_data['country']= export_data['country'].apply(lambda x : np.NaN if x == "UNSPECIFIED" else x)

export_data = export_data[export_data.value!=0]

export_data.dropna(inplace=True)
mean_value = import_data.value.mean()

import_data.value.fillna(mean_value, inplace = True )

import_data.drop_duplicates(keep="first",inplace=True) 

import_data['country']= import_data['country'].apply(lambda x : np.NaN if x == "UNSPECIFIED" else x)

import_data = import_data[import_data.value!=0]

import_data.dropna(inplace=True)
import_data.shape
import_data.isnull().sum()
#export_data.loc[export_data.value == export_data.value.max()]
#export_data.loc[export_data.value == export_data.value.max()].Commodity
#export_data.year.loc[export_data.Commodity == 'NATURAL OR CULTURED PEARLS,PRECIOUS OR SEMIPRE']   
#export_data.loc[export_data.Commodity == 'MEAT AND EDIBLE MEAT OFFAL.'].country.unique()

import pandas as pd

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
total_imports_per_year=import_data.groupby('year').agg({'value':'sum'})

total_exports_per_year=export_data.groupby('year').agg({'value':'sum'})

deficit = export_data.groupby('year').agg({'value':'sum'}) - import_data.groupby('year').agg({'value':'sum'})



t = total_imports_per_year.join(total_exports_per_year, lsuffix='_IMP', rsuffix='_EXP')

r = t.join(deficit)

sns.lineplot(data=r)





export_data.describe()
country_export_list=list(export_data.country.unique())

country_import_list=list(import_data.country.unique())
import pandas as pd

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

country_export_group=export_data.groupby('country')

country_import_group=import_data.groupby('country')



ls_export=[]

ls_import = []

for country_name in country_export_list:

    ls_export.append([country_name, country_export_group.get_group(str(country_name)).value.sum() ])

for country_name in country_import_list:

    ls_import.append([country_name, country_import_group.get_group(str(country_name)).value.sum() ])

total_exports = pd.DataFrame(ls_export, columns = ['country', 'total_exports']) 

total_imports = pd.DataFrame(ls_import, columns = ['country', 'total_imports']) 



largest_exporters_dataframe=total_exports.nlargest(5,['total_exports'])

largest_importers_dataframe=total_imports.nlargest(5,['total_imports'])



fig = plt.figure(figsize=(17,10))

ax = fig.add_subplot(221)

sns.barplot(largest_exporters_dataframe['country'],largest_exporters_dataframe['total_exports'])

plt.xlabel('COUNTRIES',size=12)

plt.ylabel('Total exports in Million $',size=12)

plt.title('LARGEST EXPORTERS OF INDIA 2010-2018',SIZE=15)

ax = fig.add_subplot(222)

sns.barplot(largest_importers_dataframe['country'],largest_importers_dataframe['total_imports'])



plt.xlabel('COUNTRIES',size=12)

plt.ylabel('Total imports in Million $',size=12)

plt.title('LARGEST IMPORTERS OF INDIA 2010-2018',SIZE=15)



plt.show()

smallest_exporters_dataframe=total_exports.nsmallest(5,['total_exports'])

smallest_importers_dataframe=total_imports.nsmallest(5,['total_imports'])



fig = plt.figure(figsize=(17,10))

ax = fig.add_subplot(221)

sns.barplot(smallest_exporters_dataframe['country'],smallest_exporters_dataframe['total_exports'])

plt.xlabel('COUNTRIES',size=12)

plt.ylabel('Total exports in Million $',size=12)

plt.title('SMALLEST EXPORTERS OF INDIA 2010-2018',SIZE=15)

ax = fig.add_subplot(222)

sns.barplot(smallest_importers_dataframe['country'],smallest_importers_dataframe['total_imports'])



plt.xlabel('COUNTRIES',size=12)

plt.ylabel('Total imports in Million $',size=12)

plt.title('SMALLEST IMPORTERS OF INDIA 2010-2018',SIZE=15)



plt.show()

# commodity_export_list=list(export_data.Commodity.unique())

# commodity_export_group=export_data.groupby('Commodity')

# ls=[]

# for commodity_name in commodity_export_list:

#     ls.append([commodity_name, commodity_export_group.get_group(str(commodity_name)).value.sum() ])



# total_exports = pd.DataFrame(ls, columns = ['Commodity', 'total_exports']) 





# largest_exporters_dataframe=total_exports.nlargest(5,['total_exports'])

# largest_exporters_dataframe.Commodity[22] = 'MINERAL FUELS&OILS'

# largest_exporters_dataframe.Commodity[62] = 'PEARLS'

# largest_exporters_dataframe.Commodity[75] = 'VEHICLES'

# largest_exporters_dataframe.Commodity[72] = 'NUCLEAR REACTORS'

# largest_exporters_dataframe.Commodity[24] = 'ORGANIC CHEMICALS'

# print(largest_exporters_dataframe.Commodity)
commodity_export_list=list(export_data.Commodity.unique())

commodity_export_group=export_data.groupby('Commodity')

ls=[]

for commodity_name in commodity_export_list:

    ls.append([commodity_name, commodity_export_group.get_group(str(commodity_name)).value.sum() ])



total_exports = pd.DataFrame(ls, columns = ['Commodity', 'total_exports']) 





largest_commodities_dataframe=total_exports.nlargest(5,['total_exports'])



plt.figure(figsize=(10,10))

sns.set_style('whitegrid')

largest_commodities_bar=sns.barplot(y=largest_commodities_dataframe['Commodity'],x=largest_commodities_dataframe['total_exports'])

plt.ylabel('commodities',size=20)

plt.xlabel('Total exports in Million $',size=18)

plt.title('LARGEST EXPORT commodities OF INDIA 2010-2018',SIZE=20)


export_USA = export_data.loc[export_data.country == 'U S A']

import_USA = import_data.loc[import_data.country == 'U S A']

export_USA.value.sum()

export_USA_total = export_USA.groupby('year').agg({'value': "sum"})

import_USA_total = import_USA.groupby('year').agg({'value': "sum"})

t = import_USA_total.join(export_USA_total, lsuffix='_IMP', rsuffix='_EXP')

sns.lineplot(data=t)
expensive_import = import_data[import_data.value>3000]

expensive_import5=expensive_import.nlargest(20,['value'])

expensive_import.shape
expensive_import = import_data[import_data.value>1000]

expensive_import5=expensive_import.nlargest(100,['value'])

plt.figure(figsize=(10,10))

sns.set_style('whitegrid')

expensive_import_commodities_bar=sns.barplot(y=expensive_import5['Commodity'],x=expensive_import5['value'])

plt.ylabel('commodities')

plt.xlabel('Million $')

plt.title('Expensive Import Commodity OF INDIA 2010-2018',SIZE=20)
expensive_export = export_data[export_data.value>1000]

expensive_export5=expensive_export.nlargest(50,['value'])

plt.figure(figsize=(10,10))

expensive_export_commodities_bar=sns.barplot(y=expensive_export5['Commodity'],x=expensive_export5['value'])

plt.ylabel('commodities')

plt.xlabel('Million $')

plt.title('Expensive Export Commodity OF INDIA 2010-2018',SIZE=20)