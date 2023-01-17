import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')
celeb_deaths = pd.read_csv('../input/celebrity_deaths_3.csv')

celeb_deaths.head()
celeb_deaths.info()
celeb_deaths.cause_of_death.value_counts().iloc[:20]
celeb_deaths.death_year.value_counts().sort_index().plot(kind = 'bar')

plt.title('Celebrity Deaths have increased throughout the Years')

plt.ylabel('Death Count')

plt.xlabel('Year')

plt.show()
by_death_year = celeb_deaths.groupby(['death_year']).mean()

print('Overall Average Age of Death:', round(by_death_year.age.mean(),1))
plt.bar(by_death_year.index, by_death_year.fame_score)

plt.title('Average Fame of Celebrities by Death Year')

plt.ylabel('Fame Index')

plt.xlabel('Month')

plt.show()
months_map = {'death_month':{'January':1, 'February':2, 'March':3, 'April':4, 'May':5, 'June':6,\

          'July':7, 'August':8, 'September':9, 'October':10, 'November':11, 'December':12}}



for year in celeb_deaths.death_year.unique():

    deaths_by_year = celeb_deaths[celeb_deaths.death_year == year]

    deaths_by_year = deaths_by_year.replace(months_map)

    deaths_by_year.death_month.value_counts().sort_index().plot(kind = 'line', label = year)

plt.legend(loc = 'lower right',prop = {'size':5.2})

plt.ylabel('Death Count')

plt.xlabel('Month')

plt.yscale('log')

plt.title('2006 - 2016 Death Count by Month')

plt.show()
for year in celeb_deaths.death_year.unique():

    deaths_by_year = celeb_deaths[celeb_deaths.death_year == year]

    mean = deaths_by_year.fame_score.mean()

    std = np.std(deaths_by_year.fame_score)

    deaths_by_year['fame_score_standard'] = (deaths_by_year.fame_score - mean)/std

    top_celebs = deaths_by_year[deaths_by_year.fame_score_standard >= 2]

    top_celebs.sort_values('fame_score_standard', ascending = False, inplace = True)

    print(top_celebs[['name', 'death_year']].head(10))