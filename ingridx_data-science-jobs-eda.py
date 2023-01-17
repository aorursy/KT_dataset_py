import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn
df = pd.read_csv('../input/alldata.csv')
df.head()
df.describe()
most_reviewed_comp = df.groupby(['company'])['reviews'].sum()
most_reviewed_comp = most_reviewed_comp.reset_index(name = 'reviews')
most_reviewed_comp = most_reviewed_comp.sort_values(['reviews'], ascending = False)
most_reviewed_comp.head(10)
most_vacancy = df.groupby(['company'])['position'].count()
most_vacancy = most_vacancy.reset_index(name='position')
most_vacancy = most_vacancy.sort_values(['position'], ascending = False)
most_vacancy.head(20)
most_wanted_positions = pd.DataFrame(df['position'].value_counts())
most_wanted_positions.head(10)
most_cities = pd.DataFrame(df['location'].value_counts())
most_cities.head(10)
cities_plot = most_cities.head(15).plot(kind = 'bar', width = .7, figsize = (10, 6), rot=45, color = 'thistle', title='Cities with the most vacancies')
positions_plot2 = most_wanted_positions.head(10).sort_values(by='position')
positions_plot = positions_plot2.plot(kind = 'barh', figsize = (10, 6), width = 0.65, color = 'lightseagreen', title='Most wanted position')
role_city = df.groupby(['location','company'])[['position']].count()

role_city = role_city.reset_index()

role_city.head()
role_city = role_city.sort_values(['position'], ascending=False).head(10)

role_city.head()
a = seaborn.barplot(x="company", y="position", hue="location", data=role_city, dodge=False);

a.set_xticklabels(role_city['company'],rotation=90)   

a.set_ylabel('No Of Positions',fontsize=12, color='dimgrey')

a.set_xlabel('Company Name',fontsize=12, color='dimgrey')   

a.set_title('Role & City', fontsize=16)
total_jobs = len(df['position'])

total_jobs
total_companies = df['company'].nunique()

total_companies * .3
x = total_jobs * .7

x
most_pos_comp = df.groupby(['company'])[['position']].count()
most_pos_comp = most_pos_comp.sort_values(['position'], ascending = False)
toptiercomp = most_pos_comp.head(664)
toptiercomp['position'].sum()
5185/6964