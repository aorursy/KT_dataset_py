import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
%matplotlib inline
from plotly.offline  import download_plotlyjs,init_notebook_mode,plot, iplot
import seaborn as sns
import cufflinks as cf
init_notebook_mode(connected = True)
cf.go_offline()
print(os.listdir("../input"))
# Read these 6 files into data frames
import_2014 = pd.read_csv("../input/PC_Import_2014_2015.csv")
import_2015 = pd.read_csv("../input/PC_Import_2015_2016.csv")
import_2016 = pd.read_csv("../input/PC_Import_2016_2017.csv")
export_2014 = pd.read_csv("../input/PC_Export_2014_2015.csv")
export_2015 = pd.read_csv("../input/PC_Export_2015_2016.csv")
export_2016 = pd.read_csv("../input/PC_Export_2016_2017.csv")
# Add a column Year to respectiv files and populate that 
import_2014['Year'] = 2014
import_2015['Year'] = 2015
import_2016['Year'] = 2016
export_2014['Year'] = 2014
export_2015['Year'] = 2015
export_2016['Year'] = 2016
# add all the files row wise and make a single for imports
imports = import_2014
imports = imports.append(import_2015, ignore_index=True)
imports = imports.append(import_2016, ignore_index = True)
imports.head()
exports = export_2014
exports = exports.append(export_2015, ignore_index = True)
exports = exports.append(export_2016, ignore_index = True)
exports.head()
# add all the files row wise and make a single for imports
imports = import_2014
imports = imports.append(import_2015, ignore_index=True)
imports = imports.append(import_2016, ignore_index = True)
imports.head()
# add all the files row wise and make a single for exports
exports = export_2014
exports = exports.append(export_2015, ignore_index = True)
exports = exports.append(export_2016, ignore_index = True)
exports.head()
all_imports = imports[['pc_description', 'value', 'Year']].groupby(['Year']).sum()
all_imports.plot(kind = 'bar')
plt.show()
all_imports
round((imports[imports['Year'] == 2014]['value'].sum() - imports[imports['Year'] == 2016]['value'].sum())/imports[imports['Year'] == 2014]['value'].sum()*100)

all_exports = exports[['pc_description', 'value', 'Year']].groupby(['Year']).sum()
all_exports.plot(kind = 'bar')
plt.show()
round((exports[exports['Year'] == 2014]['value'].sum() - exports[exports['Year'] == 2016]['value'].sum())/exports[exports['Year'] == 2014]['value'].sum()*100)

table = pd.pivot_table(data = import_2014, index = 'country_name', values = 'value' , aggfunc = np.sum).reset_index()
table = table.sort_values(by = 'value', ascending = False).reset_index(drop=True)
table.head(25).iplot(kind = 'pie', labels= 'country_name', values= 'value', title = '2014 Imports' )
table = pd.pivot_table(data = import_2015, index = 'country_name', values = 'value' , aggfunc = np.sum).reset_index()
table = table.sort_values(by = 'value', ascending = False).reset_index(drop=True)
table.head(25).iplot(kind = 'pie', labels= 'country_name', values= 'value', title = ' 2015 Imports')
table = pd.pivot_table(data = import_2016, index = 'country_name', values = 'value' , aggfunc = np.sum).reset_index()
table = table.sort_values(by = 'value', ascending = False).reset_index(drop=True)
table.head(25).iplot(kind = 'pie', labels= 'country_name', values= 'value', title = '2016 Imports' )
table = pd.pivot_table(data = imports, index = 'country_name', values = 'value' , aggfunc = np.sum).reset_index()
table = table.sort_values(by = 'value', ascending = False).reset_index(drop=True)
table.head(25).iplot(kind = 'pie', labels= 'country_name', values= 'value', title = 'All Imports 2014 to 2016 in M USD' )
top_ten_countries = table.head(10)['country_name']
top_ten_countries
commodity_value = imports[['pc_description', 'value' ]].sort_values('value', ascending = False)
commodity_value = commodity_value.groupby('pc_description').sum()
commodity_value = commodity_value.sort_values('value', ascending = False)
commodity_value = commodity_value.reset_index()
commodity_value.head(25).iplot(kind = 'bar', x = 'pc_description', y = 'value', title = "TOP 25 Products imported in M USD")
export_value = exports[['pc_description', 'value' ]].sort_values('value', ascending = False)
export_value = export_value.groupby('pc_description').sum()
export_value = export_value.sort_values('value', ascending = False)
export_value =export_value.reset_index()
export_value.head(25).iplot(kind = 'bar', x = 'pc_description', y = 'value', title = "TOP 25 Products Exported")
all_imports = all_imports.reset_index()
all_imports = all_imports.rename(columns={'Year': 'Year', 'value': 'Imports M USD'})
all_imports
all_exports = all_exports.reset_index()
all_exports = all_exports.rename(columns={'Year': 'Year', 'value': 'Exports M USD'})
all_exports
import_export = pd.merge(all_imports, all_exports, on=['Year'], how='outer')
import_export['trade Balance'] = import_export[ 'Exports M USD'] - import_export['Imports M USD']
import_export.reset_index()
import_export.iplot(kind = 'bar', x = 'Year', y = ['Imports M USD','Exports M USD' ], title = "Trade Balance M USD")
# Cumulative trade deficit
import_export['trade Balance'].sum()
# Imports table
country_imports = imports[['country_name', 'value']]
country_imports = country_imports.groupby('country_name').sum()
country_imports = country_imports.reset_index()
# Exports Table
country_exports = exports[['country_name', 'value']]
country_exports = country_exports.groupby('country_name').sum()
country_exports = country_exports.reset_index()
import_export_balancesheet = pd.merge(country_imports, country_exports, on="country_name")
import_export_balancesheet.columns = ['country_name', 'imports', 'exports']
import_export_balancesheet['total_trade'] = import_export_balancesheet['imports'] +import_export_balancesheet['exports']
import_export_balancesheet['trade_balance'] = import_export_balancesheet['exports'] -import_export_balancesheet['imports']
import_export_balancesheet.sort_values('exports',ascending= False).head(10).iplot(kind ='bar', x= 'country_name',
                                                    y =['imports', 'exports','trade_balance'], title = "Top 10 Highest exports countries and Trade Balance")
import_export_balancesheet.sort_values('imports',ascending= False).head(10).iplot(kind ='bar', x= 'country_name',
                                                    y =['imports', 'exports','trade_balance'], title = "Top 10 Highest Imports countries and Trade Balance")
import_export_balancesheet.sort_values('trade_balance',ascending= False).head(10).iplot(kind ='bar', x= 'country_name',
                                                    y =['trade_balance'], title = "Top 5 Trade Surplus Countries")
import_export_balancesheet.sort_values('trade_balance',ascending= True).head(10).iplot(kind ='bar', x= 'country_name',
                                                    y =['trade_balance'], title = "Top 5 Trade Deficit Countries")
import_export_balancesheet.sort_values('total_trade', ascending = False).head(10).iplot(kind = 'bar', x = 'country_name', y = ['total_trade', 'trade_balance'],
                                                                                       
                                                                                       
                                                                                       title = "Top 10 Highest trading countries")
#What We import from China
china_imports = imports[imports['country_name'] == 'China P Rp']
china_imports = china_imports[['pc_description','value']]
china_imports = china_imports.groupby('pc_description').sum().sort_values('value',ascending = False)
china_imports.head(10).iplot(kind = 'bar', title = ' Business with China - Imports')
#What We export to China
china_exports = exports[exports['country_name'] == 'China P Rp']
china_exports = china_exports[['pc_description','value']]
china_exports = china_exports.groupby('pc_description').sum().sort_values('value',ascending = False)
china_exports.head(10).iplot(kind = 'bar', title = 'Business with china exports')
#What We import from USA
usa_imports = imports[imports['country_name'] == 'U S A']
usa_imports = usa_imports[['pc_description','value']]
usa_imports = usa_imports.groupby('pc_description').sum().sort_values('value',ascending = False)
usa_imports.head(10).iplot(kind = 'bar', title = 'Business with United States - Imports')
#What We export to USA
usa_exports =exports[exports['country_name'] == 'U S A']
usa_exports = usa_exports[['pc_description','value']]
usa_exports = usa_exports.groupby('pc_description').sum().sort_values('value',ascending = False)
usa_exports.head(10).iplot(kind = 'bar', title = 'Business with United States - Exports')
#What We import from UAE
uae_imports = imports[imports['country_name'] == 'U Arab Emts']
uae_imports = uae_imports[['pc_description','value']]
uae_imports = uae_imports.groupby('pc_description').sum().sort_values('value',ascending = False)
uae_imports.head(10).iplot(kind = 'bar', title = 'Business with UAE - Imports')
#What We exportt to UAE
uae_exports = exports[exports['country_name'] == 'U Arab Emts']
uae_exports = uae_exports[['pc_description','value']]
uae_exports = uae_exports.groupby('pc_description').sum().sort_values('value',ascending = False)
uae_exports.head(10).iplot(kind = 'bar', title = 'Business with UAE - Exports')
#What We import from Saudi
saudi_imports = imports[imports['country_name'] == 'Saudi Arab']
saudi_imports = saudi_imports[['pc_description','value']]
saudi_imports = saudi_imports.groupby('pc_description').sum().sort_values('value',ascending = False)
saudi_imports.head(10).iplot(kind = 'bar', title = 'Business with Saudi Arabia - Imports')
#What We import from Saudi
saudi_exports = exports[exports['country_name'] == 'Saudi Arab']
saudi_exports = saudi_exports[['pc_description','value']]
saudi_exports = saudi_exports.groupby('pc_description').sum().sort_values('value',ascending = False)
saudi_exports.head(10).iplot(kind = 'bar', title = 'Business with Saudi Arabia - Exports')


