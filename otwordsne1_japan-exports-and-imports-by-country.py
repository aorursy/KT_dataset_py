import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
country = pd.read_csv("../input/country_eng.csv")

ym_latest = pd.read_csv("../input/ym_latest.csv")

yr_latest = pd.read_csv("../input/year_latest.csv")
yr_data = pd.merge(country,yr_latest, how = 'left' , on = ['Country'])

yr_data.head()
hs2 = pd.read_csv("../input/hs2_eng.csv")

hs4 = pd.read_csv("../input/hs4_eng.csv")

hs6 = pd.read_csv("../input/hs6_eng.csv")

hs9 = pd.read_csv("../input/hs9_eng.csv")

hs_list = [hs2,hs4,hs6,hs9]



for hs in hs_list:

     hs_num = hs.columns.values[0]

     print(hs_num)

     yr_data = pd.merge(yr_data, hs, how = 'left', on = [hs_num])

del yr_data['hs2'] 

del yr_data['hs4'] 

del yr_data['hs6'] 

del yr_data['hs9']

#yr_data.columns

yr_data.head()
exports_groupby_country = yr_data[yr_data['exp_imp'] == 1].groupby(['Country_name', 'Year'], as_index = False)

countries = yr_data['Country_name'].unique()



export_values = exports_groupby_country.aggregate(np.sum)

#export_values
#Identify top 10 countries export to

top_export_country = yr_data[yr_data['exp_imp'] == 1].groupby(['Country_name'], as_index = False)

export = top_export_country[['Country_name', 'exp_imp', 'VY']].aggregate(np.sum)

#print(type(export.VY[1]))

export.sort_values(by = 'VY',inplace = True, ascending = False)

#export[0:10]

#export.tail()

top_export_countries = export['Country_name'][0:10]

top_export_countries
def plot_trade_country(df, country):

    for x in country:

        country_trade = df[df['Country_name'] == x]

        plt.plot(country_trade['Year'], country_trade['VY'], label = x)

    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    plt.title("Exports by Country")

    plt.show()
#The plotting format is from Kai Sasaki's notebook https://www.kaggle.com/lewuathe/d/zanjibar/japan-trade-statistics/japan-trade-trend-from-1988-to-2015/comments

#plot_trade_country(country_export, top_countries )

for c in top_export_countries:

    country_export = export_values[export_values['Country_name'] == c]

    plt.plot(country_export['Year'], country_export['VY'], label = c)

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.title("Exports by Country")

plt.show()
import_groupby_country = yr_data[yr_data['exp_imp'] == 2].groupby(['Country_name', 'Year'], as_index = False)

imp_countries = yr_data['Country_name'].unique()



import_values = import_groupby_country.aggregate(np.sum)

#import_values
#Identify top 10 countries import from

top_import_country = yr_data[yr_data['exp_imp'] == 2].groupby(['Country_name'], as_index = False)

import_c = top_import_country[['Country_name', 'exp_imp', 'VY']].aggregate(np.sum)



import_c.sort_values(by = 'VY',inplace = True, ascending = False)

top_import_countries = import_c['Country_name'][0:10]

top_import_countries
from pylab import *

cla()

cla()
for c in top_import_countries:

    country_import = import_values[import_values['Country_name'] == c]

    plt.plot(country_import['Year'], country_import['VY'], label = c)

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.title("Imports by Country")

plt.show()
print(len(yr_data))

print(len(yr_data[yr_data['exp_imp'] == 2]))

print(len(yr_data[yr_data['exp_imp'] == 1]))
