# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/commodity_trade_statistics_data.csv")
df.head()
df.count()
df.isnull().sum()
df = df.dropna(how='any').reset_index(drop=True)  
df['commodity'].unique().shape
gr_country_year_flow = pd.DataFrame({'trade_usd' : df.groupby( ["country_or_area","year","flow"] )["trade_usd"].sum()}).reset_index()
gr_country_year_flow.head()
def balance_imp_exp(gr_country_year_flow):
    gr_export = gr_country_year_flow[gr_country_year_flow['flow'] == 'Export'].reset_index(drop = True)
    gr_import = gr_country_year_flow[gr_country_year_flow['flow'] == 'Import'].reset_index(drop = True)
    del gr_export['flow']
    gr_export.rename(columns = {'trade_usd':'export_usd'}, inplace = True)
    gr_export['import_usd'] = gr_import['trade_usd']
    import_export_country_year = gr_export
    import_export_country_year['balance_import_export'] = import_export_country_year['export_usd'] - import_export_country_year['import_usd']
    return import_export_country_year
import_export_country_year = balance_imp_exp(gr_country_year_flow)
balance_imp_exp(gr_country_year_flow).head()
import_export_country_year['country_or_area'].unique().shape
countries = import_export_country_year['country_or_area'].unique()
countries
def sorted_balance_year(year, import_export_country_year):
    sorted_balance = import_export_country_year[import_export_country_year['year'] == year].sort_values(by=['balance_import_export'], ascending=False)
    return sorted_balance
sorted_balance_2016 = sorted_balance_year(2016, import_export_country_year)
plot = sorted_balance_2016[:20].plot(x='country_or_area' , y='balance_import_export', kind='bar', legend = False, figsize=(20, 10))
def plot_country_revenue(country, import_export_country_year):
    data_country = import_export_country_year[import_export_country_year['country_or_area'] == country].sort_values(by=['year'], ascending=True)
    plot = data_country.plot(x='year' , y='balance_import_export', kind='bar', legend = False, color = np.where(data_country['balance_import_export']<0, 'red', 'black'), figsize=(20, 12))
plot_country_revenue('France', import_export_country_year)
def info_country(country, df):
    info = df[df['country_or_area'] == country].reset_index(drop = True)
    return info
info_France = info_country('France', df)
info_France.head()
info_France.drop(columns = ['comm_code', 'commodity', 'weight_kg', 'quantity_name', 'quantity'], inplace = True)
info_France_category = pd.DataFrame({'trade_usd' : info_France.groupby( ["country_or_area","year","flow", "category"] )["trade_usd"].sum()}).reset_index()
info_France_category.head()
balance_France = balance_imp_exp(info_France_category)
balance_France.head()
balance_France[balance_France['balance_import_export'] == max(balance_France['balance_import_export'])]
plot = balance_France[balance_France['category'] == '88_aircraft_spacecraft_and_parts_thereof'].plot(x='year' , y='balance_import_export', kind='bar', legend = False, color = 'black', figsize=(20, 12))
plot = plt.pie(balance_France[balance_France['year'] == 2016]['balance_import_export'].sort_values(ascending = False)[0:11], labels = balance_France[balance_France['year'] == 2016].sort_values(by = 'balance_import_export', ascending = False)['category'][0:11], autopct='%1.1f%%')
