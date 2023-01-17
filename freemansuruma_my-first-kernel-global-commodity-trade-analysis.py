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
#read in data for analysis
trade_data = pd.read_csv('../input/commodity_trade_statistics_data.csv')
trade_data.head(10)
#overview statistics of the data
trade_data.describe()
#simple plot to see the trend of global trading volume of the period in the dataset
per_year_data = trade_data.sort_values('year',ascending=True)
plt.plot(per_year_data.groupby('year').sum()['trade_usd'])
plt.grid(True)
plt.title('Total volume traded per year in USD')
plt.show();
#create dataframe that shows for each year in dataset whether there was a trading surplus or trading deficit, and by how much
exports = per_year_data.loc[per_year_data['flow']=='Export'].groupby('year').sum()['trade_usd']
imports = per_year_data.loc[per_year_data['flow']=='Import'].groupby('year').sum()['trade_usd']
current_account = pd.DataFrame()
current_account['Exports'] = exports
current_account['Imports'] = imports
current_account['Current Account'] = current_account['Exports'] - current_account['Imports']
current_account['Surplus/Deficit'] = ''

for index,row in current_account.iterrows():
    if row['Current Account']>0:
        current_account.at[index, 'Surplus/Deficit'] = 'Surplus'
    elif row['Current Account']<0:
        current_account.at[index, 'Surplus/Deficit'] = 'Deficit'
#'${:,.2f}'.format
current_account.head(10)
total_surplus_years = current_account['Surplus/Deficit'].loc[current_account['Surplus/Deficit']=='Surplus'].count()
total_deficit_years = current_account['Surplus/Deficit'].loc[current_account['Surplus/Deficit']=='Deficit'].count()

print(f'Total number of years in period with trade surplus: {total_surplus_years}')
print(f'Total number of years in period with trade deficit: {total_deficit_years}')
#create dataframe that shows each country's major export and import and the respective amount
trade_categories = trade_data.loc[(trade_data['category'] != 'all_commodities') & (trade_data['category'] != '99_commodities_not_specified_according_to_kind')]
per_country_data = trade_categories.groupby(['country_or_area','flow','category']).sum()['trade_usd']

countries = trade_data['country_or_area'].unique()
major_exports = []
major_exports_usd = []
major_imports = []
major_imports_usd = []
no_data = 'Not available'
for country in countries:
    try:
        major_export = per_country_data[country]['Export'].idxmax()
        major_import = per_country_data[country]['Import'].idxmax()
        major_exports.append(major_export)
        major_imports.append(major_import)
    
        major_exports_usd.append(per_country_data[country]['Export'].max())
        major_imports_usd.append(per_country_data[country]['Import'].max())
        
    except KeyError:
        major_exports.append(no_data)
        major_imports.append(no_data)
        major_exports_usd.append(no_data)
        major_imports_usd.append(no_data)

imports_exports = pd.DataFrame({'Country':countries, 'Major Export':major_exports, 'Export Amount':major_exports_usd, 'Major Import':major_imports, 'Import Amount':major_imports_usd})
imports_exports.head(15)
most_usd = trade_data.groupby('country_or_area').sum()['trade_usd'].idxmax()
most_weight = trade_data.groupby('country_or_area').sum()['weight_kg'].idxmax()

print(f'Country with highest trading amount in USD: {most_usd}')
print(f'Country with highest trading weight: {most_weight}')
#pie chart showing top traded categories
top10_categories = trade_categories.groupby('category').sum()['trade_usd'].sort_values(ascending=False)[0:10]
plt.pie(top10_categories,labels=top10_categories.index,shadow=True,autopct='%1.1f%%')
plt.title('Top 10 traded categories')
plt.show()
