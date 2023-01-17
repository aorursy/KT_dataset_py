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
df = pd.read_csv('/kaggle/input/golden-globe-awards/golden_globe_awards.csv')
df.head()
df_nominee_year = df.groupby(['nominee', 'year_award'])['win'].sum()
df_nominee_year[df_nominee_year==df_nominee_year.max()]
df_nominee = df.groupby(['nominee'])['win'].sum()
df_nominee[df_nominee==df_nominee.max()]
df_film_year = df.groupby(['film','year_award'])['win'].sum()
df_film_year[df_film_year==df_film_year.max()]
df.head()
df_year = pd.DataFrame()

df_year['total_nominations'] = df.groupby(['year_award'])['win'].count()

df_year['wins'] = df.groupby(['year_award'])['win'].sum()
import matplotlib.pyplot as plt
plt.plot(df_year.index, df_year['total_nominations'], label = 'nominations')

plt.plot(df_year.index, df_year['wins'], label = 'wins')

plt.legend()
import re

pattern = re.compile('[A-z ]+,')
def get_country(film):

    rs = re.findall(pattern, film)

    if(rs):

        return (rs[0]).replace(',','')

    else:

        return ''

get_country('United Kingdom, asas')
df['country']=df[df['category'].str.contains('Best Motion Picture - Foreign Language')]['film'].apply(get_country)
df_country = pd.DataFrame()
df_country=df_country.append(df[df['category'].str.contains('Best Motion Picture - Foreign Language')],ignore_index=True)
df['country'] = df[df['category'].str.contains('Foreign Film - Foreign Language')]['nominee'].apply(get_country)
df_country=df_country.append(df[df['category'].str.contains('Foreign Film - Foreign Language')],ignore_index=True)
df_country
df_country_awards_year = df_country.groupby(['country','year_award'])['win'].sum()
df_country_awards_year[df_country_awards_year==df_country_awards_year.max()]
df_country[df_country['year_award']==1973]
df_country_overall = df_country.groupby(['country'])['win'].sum()
df_country_overall[df_country_overall==df_country_overall.max()]
df_category_year = df.groupby(['category','year_award'])['win'].sum()
df_category_year[df_category_year==df_category_year.max()]
df_category_year[df_category_year==df_category_year.min()]