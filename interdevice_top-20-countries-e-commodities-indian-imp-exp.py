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
db_import = pd.read_csv("/kaggle/input/india-trade-data/2018-2010_import.csv")

db_export = pd.read_csv("/kaggle/input/india-trade-data/2018-2010_export.csv")
import matplotlib.pyplot as plt

%matplotlib inline

%config InlineBackend.figure_format = 'retina'
db_export.isnull().sum()
db_export['value'].fillna(0.00, inplace=True)
db_import.isnull().sum()
db_import['value'].fillna(0.00, inplace=True)
db_import.isnull().sum()
db_import_country = db_import.groupby('country', as_index=False).agg({'value': 'sum'})

db_import_country.head()
db_export_country = db_export.groupby('country', as_index=False).agg({'value': 'sum'})
db_import_export = pd.merge(db_import_country, db_export_country, on='country')
db_import_export.rename(columns={'value_x':'import', 'value_y':'export'}, inplace=True)

db_import_export.head()
db_import_export['total'] = db_import_export['import']+db_import_export['export']

db_import_export.head()
exp_imp_grafh = db_import_export.sort_values(by=['total'], ascending=False).head(20)

exp_imp_grafh.head()
N = 20

impCountry = exp_imp_grafh['country']

impValue = exp_imp_grafh['import']

expoValue = exp_imp_grafh['export']

ind = np.arange(N)

width = 0.85



p1 = plt.bar(ind, impValue, width)

p2 = plt.bar(ind, expoValue, width, bottom=impValue)



plt.ylabel('Values Us$')

plt.title('Scores by Country')

plt.xticks(ind, impCountry, rotation='vertical')

plt.yticks(np.arange(500, exp_imp_grafh['total'].max(), 50000))

plt.legend((p1[0], p2[0]), ('Import', 'Export'))



plt.show()
db_export_commodity = db_export.groupby('Commodity', as_index=False).agg({'value': 'sum'})

db_export_commodity.head()
db_import_commodity = db_import.groupby('Commodity', as_index=False).agg({'value': 'sum'})

db_import_commodity.head()
db_imp_exp_commodity = pd.merge(db_import_commodity, db_export_commodity, on='Commodity')

db_imp_exp_commodity.head()
db_imp_exp_commodity.rename(columns={'value_x':'import', 'value_y':'export'}, inplace=True)

db_imp_exp_commodity.head()
db_imp_exp_commodity['total'] = db_imp_exp_commodity['import']+db_imp_exp_commodity['export']

db_imp_exp_commodity.head()
exp_imp_commodity_grafh = db_imp_exp_commodity.sort_values(by=['total'], ascending=False).head(20)

exp_imp_commodity_grafh.head()
N = 20

impCommodity = exp_imp_commodity_grafh['Commodity']

impValue = exp_imp_commodity_grafh['import']

expoValue = exp_imp_commodity_grafh['export']

ind = np.arange(N)

width = 0.85



p1 = plt.bar(ind, impValue, width)

p2 = plt.bar(ind, expoValue, width, bottom=impValue)



plt.ylabel('Values Us$')

plt.title('Scores by Commodity')

plt.xticks(ind, impCommodity, rotation='vertical')

plt.yticks(np.arange(5000, exp_imp_commodity_grafh['total'].max(), 500000))

plt.legend((p1[0], p2[0]), ('Import', 'Export'))



plt.show()
listCountry = exp_imp_grafh['country'].values

listCountry
evolution_country = db_import[db_import['country'].isin(listCountry)]

evolution_country.head()
evolution_country_grafh = evolution_country.groupby(['country','year'], as_index=False).agg({'value': 'sum'})

evolution_country_grafh.head()
def contries():

    for i in listCountry:

        a = evolution_country_grafh.loc[evolution_country_grafh['country']==i]

        x = a['year']

        y = a['value']

        z = a['country'].unique()

        plt.xlabel('Years')

        plt.ylabel('Values Us$')

        plt.title('Import - Countries for years')

        plt.legend(loc='best', bbox_to_anchor=(1.45, 1.2), shadow=True, ncol=1)

        plt.plot(x, y, label=z)
contries()
evolution_country_exp = db_export[db_export['country'].isin(listCountry)]

evolution_country_exp.head()
evolution_country_exp_grafh = evolution_country_exp.groupby(['country','year'], as_index=False).agg({'value': 'sum'})

evolution_country_exp_grafh.head()
def contries1():

    for i in listCountry:

        a = evolution_country_exp_grafh.loc[evolution_country_grafh['country']==i]

        x = a['year']

        y = a['value']

        z = a['country'].unique()

        plt.xlabel('Years')

        plt.ylabel('Values Us$')

        plt.title('Export - Countries for years')

        plt.legend(loc='best', bbox_to_anchor=(1.45, 1.2), shadow=True, ncol=1)

        plt.plot(x, y, label=z)
contries1()
listCommodity = exp_imp_commodity_grafh['Commodity'].values

listCommodity
evolution_commodity = db_import[db_import['Commodity'].isin(listCommodity)]

evolution_commodity.head()
evolution_commodity_grafh = evolution_commodity.groupby(['Commodity','year'], as_index=False).agg({'value': 'sum'})

evolution_commodity_grafh.head()
def commodity():

    for i in listCommodity:

        a = evolution_commodity_grafh.loc[evolution_commodity_grafh['Commodity']==i]

        x = a['year']

        y = a['value']

        z = a['Commodity'].unique()

        plt.xlabel('Years')

        plt.ylabel('Values Us$')

        plt.title('Import - Commodity for years')

        plt.legend(loc='best', bbox_to_anchor=(1.00, 1.00), shadow=True, ncol=1, fontsize=10)

        plt.plot(x, y, label=z)
commodity()
evolution_exp_commodity = db_export[db_export['Commodity'].isin(listCommodity)]

evolution_exp_commodity.head()
evolution_exp_commodity_grafh = evolution_exp_commodity.groupby(['Commodity','year'], as_index=False).agg({'value': 'sum'})

evolution_exp_commodity_grafh.head()
def commodity1():

    for i in listCommodity:

        a = evolution_exp_commodity_grafh.loc[evolution_exp_commodity_grafh['Commodity']==i]

        x = a['year']

        y = a['value']

        z = a['Commodity'].unique()

        plt.xlabel('Years')

        plt.ylabel('Values Us$')

        plt.title('Export - Commodity for years')

        plt.legend(loc='best', bbox_to_anchor=(1.00, 1.00), shadow=True, ncol=1, fontsize=10)

        plt.plot(x, y, label=z)
commodity1()