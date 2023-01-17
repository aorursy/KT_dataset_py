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
df_base = pd.read_csv("../input/base_etablissement_par_tranche_effectif.csv")

df_geo = pd.read_csv("../input/name_geographic_information.csv")

df_salary = pd.read_csv("../input/net_salary_per_town_categories.csv")

df_population = pd.read_csv("../input/population.csv")



print (df_base.head())

print (df_geo.head())

print (df_salary.head())

print (df_population.head())
df_base=df_base.rename(columns = {'CODGEO':'code_insee'})

df_base['code_insee']=df_base['code_insee'].astype(str)

df_geo['code_insee']=df_geo['code_insee'].astype(str)

merged_df_1=df_base.merge(df_geo, how='inner', on=['code_insee'])
df_salary=df_salary.rename(columns = {'CODGEO':'code_insee'})

df_population=df_population.rename(columns = {'CODGEO':'code_insee'})

df_salary['code_insee']=df_salary['code_insee'].astype(str)

df_population['code_insee']=df_population['code_insee'].astype(str)



merged_df_2=df_salary.merge(df_population, how='inner', on=['code_insee'])

merged_df_all=merged_df_1.merge(merged_df_2, how='inner', on=['code_insee'])

merged_df_all.columns
final_df = merged_df_all.drop(columns=['REG','DEP', 'EU_circo', 'code_région', 'nom_région', 'chef.lieu_région', 'numéro_département',

                                      'nom_commune', 'codes_postaux', 'éloignement', 'LIBGEO_x', 'LIBGEO_y', 'LIBGEO', 'nom_département', 'préfecture', 'numéro_circonscription'])

final_df
all_columns = final_df.columns

print (all_columns)

groupby_columns = list(all_columns[:-4])

print (groupby_columns)

final_df_sum_population = final_df.reset_index().groupby(groupby_columns, as_index=False).sum()

final_df_sum_population = final_df_sum_population.drop(columns=['MOCO', 'AGEQ80_17', 'SEXE', 'index', 'NIVGEO', 'code_insee'])

final_df_sum_population.head()
from pylab import rcParams

rcParams['figure.figsize'] = 10,10



num_firms = final_df_sum_population['E14TST']

percentile_upper = (num_firms.quantile(0.9))

percentile_lower = (num_firms.quantile(0.1))

final_df_sum_population_small_town = final_df_sum_population[final_df_sum_population['E14TST'] < percentile_upper]

#final_df_sum_population_small_town = final_df_sum_population[final_df_sum_population['E14TST'] > percentile_lower]



num_firms = final_df_sum_population_small_town['E14TST']

num_population = final_df_sum_population_small_town['NB']

woman_salary = final_df_sum_population_small_town['SNHMF14']

man_salary = final_df_sum_population_small_town['SNHMH14']



plt.subplot(4, 1, 1)

plt.hist(num_firms, bins = 50)

plt.title('number of firms histogram')



plt.subplot(4, 1, 2)

plt.hist(num_population, bins = 50)

plt.title('town population histogram')



plt.subplot(4, 1, 3)

plt.hist(woman_salary, bins = 50)

plt.title('town mean woman salary histogram')



plt.subplot(4, 1, 4)

plt.hist(man_salary, bins = 50)

plt.title('town mean man salary histogram')



plt.tight_layout()

plt.show()
firm_population_data = final_df_sum_population_small_town[['E14TST', 'E14TS1', 'E14TS6', 'E14TS10', 'E14TS20', 'E14TS50', 'E14TS100', 'E14TS200', 'E14TS500']].reset_index(drop=True)

human_population = final_df_sum_population_small_town[['NB']].reset_index(drop=True)

mean_salary_female_overall = final_df_sum_population_small_town[['SNHMF14']].reset_index(drop=True)

mean_salary_male_overall = final_df_sum_population_small_town[['SNHMH14']].reset_index(drop=True)

mean_salary_overall = final_df_sum_population_small_town[['SNHM14']].reset_index(drop=True)

mean_salary_age_overall = final_df_sum_population_small_town[['SNHM1814', 'SNHM2614', 'SNHM5014']].reset_index(drop=True)



# combine firm population column data 

firm_population_data['binned_small'] = firm_population_data['E14TS1'] +  firm_population_data['E14TS6']

firm_population_data['binned_med'] = firm_population_data['E14TS10'] +  firm_population_data['E14TS20']

firm_population_data['binned_large'] = firm_population_data['E14TS50'] +  firm_population_data['E14TS100'] + firm_population_data['E14TS200'] + firm_population_data['E14TS500']

# matrix 1: firm size number, overall salary, human population number

firm_generic_pop = firm_population_data[['binned_small', 'binned_med', 'binned_large', 'E14TST']]

corr_1 = pd.concat([human_population, firm_generic_pop], axis=1)

corr_1 = pd.concat([corr_1, mean_salary_overall], axis=1)

corr_1 = corr_1.rename(index=str, columns={"NB": "population", "E14TST": "total firms", "SNHM14": "mean_wages"})

corr_matrix = corr_1.corr()

corr_matrix.style.background_gradient()

plt.matshow(corr_matrix)

plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns);

plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns);

corr_matrix
final_df_sum_population_large_town = final_df_sum_population[final_df_sum_population['E14TST'] > percentile_upper]

firm_population_data = final_df_sum_population_large_town[['E14TST', 'E14TS1', 'E14TS6', 'E14TS10', 'E14TS20', 'E14TS50', 'E14TS100', 'E14TS200', 'E14TS500']].reset_index(drop=True)

human_population = final_df_sum_population_large_town[['NB']].reset_index(drop=True)

mean_salary_female_overall = final_df_sum_population_large_town[['SNHMF14']].reset_index(drop=True)

mean_salary_male_overall = final_df_sum_population_large_town[['SNHMH14']].reset_index(drop=True)

mean_salary_overall = final_df_sum_population_large_town[['SNHM14']].reset_index(drop=True)

mean_salary_age_overall = final_df_sum_population_large_town[['SNHM1814', 'SNHM2614', 'SNHM5014']].reset_index(drop=True)



# combine firm population column data 

firm_population_data['binned_small'] = firm_population_data['E14TS1'] +  firm_population_data['E14TS6']

firm_population_data['binned_med'] = firm_population_data['E14TS10'] +  firm_population_data['E14TS20']

firm_population_data['binned_large'] = firm_population_data['E14TS50'] +  firm_population_data['E14TS100'] + firm_population_data['E14TS200'] + firm_population_data['E14TS500']



# matrix 1: firm size number, overall salary, human population number

firm_generic_pop = firm_population_data[['binned_small', 'binned_med', 'binned_large', 'E14TST']]

corr_1 = pd.concat([human_population, firm_generic_pop], axis=1)

corr_1 = pd.concat([corr_1, mean_salary_overall], axis=1)

corr_1 = corr_1.rename(index=str, columns={"NB": "population", "E14TST": "total firms", "SNHM14": "mean_wages"})

corr_matrix = corr_1.corr()

corr_matrix.style.background_gradient()

plt.matshow(corr_matrix)

plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns);

plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns);

corr_matrix
women_mean_young = final_df_sum_population['SNHMF1814'].mean()

women_mean_mid = final_df_sum_population['SNHMF2614'].mean()

women_mean_old = final_df_sum_population['SNHMF5014'].mean()

man_mean_young = final_df_sum_population['SNHMH1814'].mean()

man_mean_mid = final_df_sum_population['SNHMH2614'].mean()

man_mean_old = final_df_sum_population['SNHMH5014'].mean()



X = ['age 18-25', 'age 26-50', 'age >50']

y_women = [women_mean_young, women_mean_mid, women_mean_old]

y_men = [man_mean_young, man_mean_mid, man_mean_old]

df = pd.DataFrame(np.c_[y_women, y_men], index=X)

df.plot.bar(rot=0)

plt.title('mean wages of different gender across age group')

plt.legend(['woman', 'man'])

plt.show()
# The code to create the France map was taken from another kernel, but the plots are mine

from mpl_toolkits.basemap import Basemap

import matplotlib.colors as colors



plt.figure(figsize=(20,20))

# Load map of France

map = Basemap(projection='lcc', 

            lat_0=46.2374,

            lon_0=2.375,

            resolution='h',

            llcrnrlon=-4.76, llcrnrlat=41.39,

            urcrnrlon=10.51, urcrnrlat=51.08)



# Draw parallels.

parallels = np.arange(40.,52,2.)

map.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)

# Draw meridians

meridians = np.arange(-6.,10.,2.)

map.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)



map.drawcoastlines()

map.drawcountries()

map.drawmapboundary()

map.drawrivers()



lat_long_data = final_df_sum_population[['latitude', 'longitude', 'SNHM14']].dropna(subset=['latitude', 'longitude'])

lat_long_data = lat_long_data[(lat_long_data['latitude'].astype(str) != '-')]

lat_long_data = lat_long_data[(lat_long_data['longitude'].astype(str) != '-')]

lat_long_data = lat_long_data.sort_values(by=["SNHM14"], ascending=False).head(2000)

lat_ = lat_long_data['latitude'].astype(float).values.tolist()

long_ = lat_long_data['longitude'].astype(float).values.tolist()

print (lat_[0])

print (long_[0])

# scatter these high wage towns

x1, y1 = map(long_,lat_)

map.scatter(x1, y1, c=lat_long_data['SNHM14'],  norm=colors.LogNorm(vmin=1, vmax=max(lat_long_data['SNHM14'])), cmap='hsv')



x_paris, y_paris = map([2.3522], [48.8566])

x_Lyon, y_Lyon = map([4.8357], [45.7640])

x_Marseille, y_Marseille = map([5.3698], [43.2965])

# plot single points of large cities

map.plot(x_paris, y_paris, marker='o', markersize = 20, c='yellow')

map.plot(x_Lyon, y_Lyon, marker='o', markersize = 20, c='yellow')

map.plot(x_Marseille, y_Marseille, marker='o', markersize = 20, c='yellow')

plt.annotate('Paris', (x_paris[0], y_paris[0]), fontsize=30)

plt.annotate('Lyon', (x_Lyon[0],y_Lyon[0]), fontsize=30)

plt.annotate('Marselle', (x_Marseille[0], y_Marseille[0]), fontsize=35)

plt.title('scatter plots of top 2000 mean wage towns', fontsize=40, fontweight='bold', y=1.05)

plt.show()
