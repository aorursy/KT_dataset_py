# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# The trade statistics records for each year from 1988 to 2016

df = pd.read_csv('../input/year_latest.csv')



df.head()
# The area and country codes

country_df = pd.read_csv('../input/country_eng.csv')



country_df.head()
# Joined trade records with Country name

joined_df = pd.merge(df, country_df, on=['Country'])



# Plot VY transition

def plot_vys(column_value, column_name):

    data = vys[vys[column_name] == column_value]

    plt.plot(data['Year'], data['VY'], label=column_value)

    

areas = np.unique(country_df['Area'].values)

    

grouped_by_area = joined_df[['Year', 'VY', 'Area']].groupby(['Area', 'Year'], as_index=False)

vys = grouped_by_area.aggregate(np.sum)



plt.figure(figsize=(7.5, 8))

for area in areas:

    plot_vys(area, 'Area')

    

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.show()
hs2_df = pd.read_csv('../input/hs2_eng.csv')

middle_east_df = joined_df[joined_df['Area'] == 'Middle_East']



grouped_by_hs2 = joined_df[['Year', 'VY', 'hs2']].groupby(['hs2','Year'], as_index=False)



vys = grouped_by_hs2.aggregate(np.sum)

vys = pd.merge(vys, hs2_df, on=['hs2'])



main_goods = vys[vys['VY'] >  0.4 * 1e10]

hs2_names = np.unique(main_goods['hs2_name'].values)



for hs2_name in hs2_names:

    plot_vys(hs2_name, 'hs2_name')



plt.legend()

plt.legend(bbox_to_anchor=(1.0, 1), loc=2, borderaxespad=0.)

plt.show()
areas = np.unique(country_df['Area'].values)

    

grouped_by_area = joined_df[['Year', 'VY', 'Area']].groupby(['Area', 'Year'], as_index=False)

vys = grouped_by_area.aggregate(np.sum)

plt.figure(figsize=(12.0, 8))

g = seaborn.boxplot(x="Area", y="VY", data=vys, palette="PRGn")

g.set_xticklabels(labels=areas, rotation=45)
grouped_by_country = joined_df[['Year', 'VY', 'Country_name']].groupby(['Country_name','Year'], as_index=False)



vys = grouped_by_country.aggregate(np.sum)
plt.figure(figsize=(7.5, 8))



asia_countries = country_df[country_df['Area'] == 'Asia']['Country_name']

    

for country in asia_countries:

    plot_vys(country, 'Country_name')



plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.show()
# Trade values with China

china_df = joined_df[joined_df['Country_name'] == 'People\'s_Republic_of_China']

grouped_by_hs2 = china_df[['Year', 'VY', 'hs2']].groupby(['hs2','Year'], as_index=False)



vys = grouped_by_hs2.aggregate(np.sum)

vys = pd.merge(vys, hs2_df, on=['hs2'])

vys.head()
main_goods = vys[vys['VY'] >  0.15 * 1e10]

hs2_names = np.unique(main_goods['hs2_name'].values)



for hs2_name in hs2_names:

    plot_vys(hs2_name, 'hs2_name')



plt.legend(bbox_to_anchor=(1., 1), loc=2, borderaxespad=0.)

plt.show()
minor_goods = vys[vys['VY'] <  0.1 * 1e6]

hs2_names = np.unique(minor_goods['hs2_name'].values)



pivotted_vys = vys[vys['hs2_name'].isin(hs2_names)]

pivotted_vys = pivotted_vys.pivot('Year' ,'hs2_name', 'VY')

plt.figure(figsize=(12.0, 8))

g = seaborn.heatmap(pivotted_vys, annot=True, linewidths=.5)

g.set_xticklabels(labels=hs2_names, rotation=45)