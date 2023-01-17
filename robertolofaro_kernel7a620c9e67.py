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



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

pd.set_option('display.max_colwidth', 100)

input_data = pd.read_csv('../input/selected-indicators-from-world-bank-20002019/facttable.csv').drop('Unnamed: 0',axis=1)

category_country = pd.read_csv('../input/selected-indicators-from-world-bank-20002019/dimension_country.csv')

category_indicator = pd.read_csv('../input/selected-indicators-from-world-bank-20002019/dimension_indicator.csv')
input_data.columns
selected_countries = ['AUT','BEL','BGR','CYP','CZE','DEU','DNK','ESP','EST' \

             ,'FIN','FRA','GRC','HRV','HUN','IRL','ITA','LTU','LUX' \

             ,'LVA','MLT','NLD','POL','PRT','ROU','SVK','SVN','SWE' \

             , 'GBR'

            ]
selected_indicators = ['NY.GDP.MKTP.KD.ZG','SP.URB.TOTL.IN.ZS']
selected_years = ['2001','2002','2003','2004','2005','2006','2007','2008','2009','2010','2011','2013','2015','2016','2017','2018']
model_data = input_data[(input_data['Country Code'].isin(selected_countries))&(input_data['Indicator Code'].isin(selected_indicators))]
model_data.columns
model_data.head()
df_countries = pd.DataFrame(model_data['Country Code'].unique())

df_countries.columns = ['Code']

table_countries = pd.merge(category_country,df_countries,left_on='Code',right_on='Code', how='right').drop("Unnamed: 0",axis=1)

table_countries
df_indicators = pd.DataFrame(model_data['Indicator Code'].unique())

df_indicators.columns = ['Code']

table_indicators = pd.merge(category_indicator,df_indicators,left_on='Code',right_on='Code', how='right').drop("Unnamed: 0",axis=1)

table_indicators
gdp_data = model_data[model_data['Indicator Code']=='NY.GDP.MKTP.KD.ZG'].drop(['Indicator Code','2019'], axis=1)

gdp_data.columns[1:]
gdp_data.set_index('Country Code')
urbanization_data = model_data[model_data['Indicator Code']=='SP.URB.TOTL.IN.ZS'].drop(['Indicator Code','2019'], axis=1)

urbanization_data.columns[1:]
urbanization_data
selected_four = ['DEU','ESP','FRA','ITA']

gdp_data[gdp_data['Country Code'].isin(selected_four)]

urbanization_data[urbanization_data['Country Code'].isin(selected_four)]