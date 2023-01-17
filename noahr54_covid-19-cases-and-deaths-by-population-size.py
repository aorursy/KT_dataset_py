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
import plotly.express as px

import pandas as pd

import numpy as np
URL = 'https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv'

URL_pop = 'https://www2.census.gov/programs-surveys/popest/datasets/2010-2019/counties/totals/co-est2019-alldata.csv'
data_pop = pd.read_csv(URL_pop, encoding = 'latin-1')

data_pop = data_pop[data_pop.STNAME != data_pop.CTYNAME]

data_pop.CTYNAME = data_pop.CTYNAME.str.split().str[0]

data_pop.set_index(data_pop['COUNTY'], inplace = True)

data_pop['STATE'] = data_pop['STATE'].map(lambda a: '0' + str(a) if len(str(a)) == 1 else str(a))

data_pop['COUNTY'] = data_pop['COUNTY'].map(lambda a: '00' + str(a) if len(str(a)) == 1 else ('0' + str(a)) if len(str(a)) == 2 else str(a))

data_pop['fips'] = data_pop.STATE + data_pop.COUNTY

data_pop.fips = pd.to_numeric(data_pop.fips)

data_pop.set_index(data_pop.fips, inplace = True, drop = True)
data = pd.read_csv(URL)



data.dropna(axis = 0, inplace = True)

data.fips = pd.to_numeric(data.fips, errors='coerce', downcast = 'integer')

by_county = data.groupby('fips').max()

by_county.head()

by_county = by_county.join(data_pop['POPESTIMATE2019'])
by_county['pop_case_ratio'] = by_county['cases']/(by_county['POPESTIMATE2019']/10000)

by_county.pop_case_ratio.fillna(value = 0, inplace = True)

by_county.pop_case_ratio = by_county.pop_case_ratio.round(decimals = 2)

#by_county.pop_case_ratio.dtype
fig = px.scatter(by_county, x = 'POPESTIMATE2019', y  = 'cases',size = 'pop_case_ratio', hover_data = ['county','state'], color = 'state', log_x = True, log_y = True)

#fig.write_html('case_ratio.html')

fig.show()
by_county['pop_death_ratio'] = by_county['deaths']/(by_county['POPESTIMATE2019']/10000)

by_county.pop_death_ratio.fillna(value = 0, inplace = True)

by_county.pop_death_ratio = by_county.pop_death_ratio.round(decimals = 2)
fig = px.scatter(by_county, x = 'POPESTIMATE2019', y  = 'deaths',size = 'pop_death_ratio', hover_data = ['county','state'], color = 'state', log_x = True, log_y = True)

#fig.write_html('case_ratio.html')

fig.show()