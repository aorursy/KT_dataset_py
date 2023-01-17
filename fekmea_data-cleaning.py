import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pandas_profiling

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('../input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
cgb = pd.read_csv('../input/miniwiki/countries-government-budget.csv')

cgb.tail()
cgb.rename(columns={

    'Rank': 'rank',

    'Country': 'country',

    'Revenues': 'revenues',

    'Expenditures': 'expenditures',

    'Surplus (or deficit)': 'Surplus_or_deficit',

    'Surplus percentage of GDP': 'Surplus_per_GDP',

    'Year': 'year'

}, inplace=True)
cgb.replace({'-': None}, inplace = True)



cgb.dropna(subset=['rank'], how= 'any', inplace=True)



cgb_2017 = cgb[cgb['year'].str.contains('2017')]



cgb_2017 = cgb_2017.iloc[:,:-1]



cgb_2017.Surplus_per_GDP = cgb_2017.Surplus_per_GDP.str.replace('%', '')



 

cgb_2017['Surplus_or_deficit'].replace(',','.', regex=True , inplace= True)

cgb_2017['Surplus_or_deficit'].replace('−','-', regex=True , inplace= True)

cgb_2017['Surplus_per_GDP'].replace(',','.', regex=True , inplace= True)

cgb_2017['Surplus_per_GDP'].replace('−','-', regex=True , inplace= True)

cgb_2017.replace({'-': None}, inplace = True)

cgb_2017.dropna(subset=['Surplus_per_GDP'], how= 'any', inplace=True)

cgb_2017.astype({'rank':'int32',

                 'revenues':'float32',

                 'expenditures':'float32',

                 'Surplus_or_deficit':'float32',

                 'Surplus_per_GDP':'float32'}).dtypes
#missing data

total = cgb_2017.isnull().sum().sort_values(ascending=False)

percent = (cgb_2017.isnull().sum()/cgb_2017.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)
pandas_profiling.ProfileReport(cgb_2017)
import os

if not os.path.exists('../'):

    os.makedirs('../')

    print('output create')

else:

    print('exst')

for dirname, _, filenames in os.walk('../'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
cgb_2017.to_csv('../output/countries-government-budget_2017.csv', index=False)
cgb_test = pd.read_csv('../output/countries-government-budget_2017.csv')

cgb_test