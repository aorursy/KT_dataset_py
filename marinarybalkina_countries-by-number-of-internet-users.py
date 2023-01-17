import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import geopandas as gpd
import plotly.express as px

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Reading csv file with data
internet_users = pd.read_csv('../input/list-of-countries-by-number-of-internet-users/List of Countries by number of Internet Users - Sheet1.csv')

# Reading the file with "world geometry"
world_geometry = gpd.read_file('../input/natural-earth/10m_cultural/10m_cultural/ne_10m_admin_0_countries_lakes.shp')
world_geometry = world_geometry[['SOVEREIGNT','geometry']].copy()
world_geometry.rename(columns={'SOVEREIGNT':'Country or Area'}, inplace=True)

# Preparing data
internet_users['Population'] = internet_users['Population'].str.replace(',', '').astype(int)
internet_users['Internet Users'] = internet_users['Internet Users'].str.replace(',', '').astype(int)
internet_users['percent'] = internet_users['Internet Users'] / internet_users['Population'] * 100

country_to_user_percentage = internet_users[['Country or Area', 'percent']].copy()

# Updating the name of some countries
country_to_user_percentage.at[2, 'Country or Area'] = 'United States of America'
country_to_user_percentage.at[53, 'Country or Area'] = 'United Republic of Tanzania'
country_to_user_percentage.at[156, 'Country or Area'] = 'eSwatini'

merged = world_geometry.merge(country_to_user_percentage, how='left', on='Country or Area')
# Creating a map Percentage of Internet Users Around the World
title = str(np.around(merged['percent'].mean(), decimals=2)) + '% population of the World has access to Internet'

plt.style.use('fivethirtyeight')

ax = merged.dropna().plot(column='percent', cmap='YlOrBr' , figsize=(30, 30),
                          scheme='User_Defined',
                          classification_kwds=dict(bins=[10,20,30,40, 50, 60, 70, 80, 90, 100]),
                          edgecolor='black', legend=True)
ax.get_legend().set_bbox_to_anchor((0.15, 0.4))
ax.get_legend().set_title('Percentage (%)')
ax.set_title("Percentage of Internet Users Around the World" , size=30, pad=20)
ax.axis('off')
ax.text(-15, -60, title, horizontalalignment='left', size=15, color='black', weight='semibold')
f = plt.figure(figsize=(7, 10))
f.add_subplot(111)

plt.axes(axisbelow=True)
sort_users_by_population = internet_users.sort_values('Population', ascending=True)
plt.barh(sort_users_by_population['Country or Area'].values[:10], 
         sort_users_by_population['percent'].values[:10],
         color='darkcyan')
plt.xlim(0, 100)
plt.tick_params(size=5, labelsize=13)
plt.xlabel('Percent (%)', fontsize=17)
plt.title('Percentage of Internet Users in the top 10 depopulated countries', fontsize=20)
plt.grid(alpha=1)
plt.savefig('Percentage of Internet Users in the top 10 depopulated countries.png')
f = plt.figure(figsize=(7, 10))
f.add_subplot(111)

sort_users_by_population = internet_users.sort_values('Population', ascending=False)

plt.axes(axisbelow=True)
plt.barh(sort_users_by_population['Country or Area'].values[:10], 
         sort_users_by_population['percent'].values[:10],
         color='darkcyan')
plt.xlim(0, 100)
plt.tick_params(size=5, labelsize=13)
plt.xlabel('Percent (%)', fontsize=17)
plt.title('Percentage of Internet Users in the top 10 populated countries', fontsize=20)
plt.grid(alpha=1)
plt.savefig('Percentage of Internet Users in the top 10 populated countries.png')
f = plt.figure(figsize=(7, 10))
f.add_subplot(111)

sorted_users_by_percent = country_to_user_percentage.sort_values('percent')

plt.axes(axisbelow=True)
plt.barh(sorted_users_by_percent['Country or Area'].values[:10], 
         sorted_users_by_percent['percent'].values[:10],
        color='darkcyan')
plt.xlim(0, 10)
plt.tick_params(size=5, labelsize=13)
plt.xlabel('Percent (%)', fontsize=17)
plt.title('Top 10 Countries with the Smallest Amount of Internet Users', fontsize=20)
plt.grid(alpha=1)
plt.savefig('Top 10 Countries with the Smallest Amount of Internet Users.png')
f = plt.figure(figsize=(7, 10))
f.add_subplot(111)

plt.axes(axisbelow=True)
plt.barh(sorted_users_by_percent['Country or Area'].values[-10:], 
         sorted_users_by_percent['percent'].values[-10:],
         color='darkcyan')
plt.xlim(90, 100)
plt.tick_params(size=5, labelsize=13)
plt.xlabel('Percent (%)', fontsize=17)
plt.title('Top 10 Countries with the Biggest Amount of Internet Users', fontsize=20)
plt.grid(alpha=1)
plt.savefig('Top 10 Countries with the Biggest Amount of Internet Users.png')
fig = px.scatter(country_to_user_percentage, 
                 y=internet_users.loc[:,"percent"], 
                 x=internet_users.loc[:,"Population"],
                 color="Country or Area", hover_name="Country or Area",
                 hover_data=['Country or Area'],
                 color_continuous_scale=px.colors.sequential.Plasma,
                 title='Population vs Percent of Internet Users',
                 size=np.power(country_to_user_percentage["percent"] + 1, 0.3) - 0.5,
                 size_max=30,
                 height=600)
fig.update_coloraxes(colorscale="hot")
fig.update(layout_coloraxis_showscale=False)
fig.update_yaxes(title_text="Percent of Internet Users (%)")
fig.update_xaxes(title_text="Population (people)")
fig.show()
f = plt.figure(figsize=(15, 10))
plt.tight_layout()

amount_of_people = np.sum(internet_users['Internet Users'])
country_to_users = internet_users[['Country or Area', 'Internet Users']].copy()
country_to_users.sort_values(country_to_users.columns[-1], ascending=False, inplace=True)
country_percentage = country_to_users['Internet Users'] / amount_of_people

coef = 0.01
tmp = country_percentage > coef

country_percentage = country_percentage[tmp]
amount_of_others_countries = np.sum(country_to_users[~tmp]['Internet Users'])
country_percentage[country_to_users.__len__()] = 1 - amount_of_others_countries / amount_of_people

labels = [country_to_users[tmp].loc[i, 'Country or Area'] + ' (' + str(country_to_users[tmp].loc[i, 'Internet Users']) + ')'
          for i in range(country_to_users[tmp].__len__())]
labels.append('Other (' + str(amount_of_others_countries) + ')')

ax = f.add_subplot(111) # ',
plt.pie(country_percentage, labels=labels, autopct='%1.1f%%', pctdistance=0.85, labeldistance=1.2, textprops = {'fontsize':10.5})
my_circle = plt.Circle((0, 0), 0.6, color='white')
p = plt.gcf()

p.gca().add_artist(my_circle)
plt.text(0.5,0.5,"World Total Internet Users\n" + str(amount_of_people) + ' people', horizontalalignment='center',verticalalignment='center',transform=ax.transAxes, size=18, alpha = 0.6)


plt.show()
