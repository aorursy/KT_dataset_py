# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/who-suicide-statistics/who_suicide_statistics.csv')

df.head()
#Create column with number of suicides per 100k people

df['rate_per_100k'] = df['suicides_no'] / (df['population'] / 100000)
usa_df = df.loc[df['country']=='United States of America',]

usa_df.reset_index(drop=True,inplace=True)
usa_df.head()
usa_age_s = sns.catplot(x='age',data=usa_df, palette='hls', kind='count',

                order=['5-14 years','15-24 years','25-34 years','35-54 years',

                      '55-74 years','75+ years']).set_xticklabels(rotation=30)
g = sns.catplot(x='sex',data=usa_df, palette='hls', kind='count')
usa_age_s = sns.barplot(x='age',y='population',data=usa_df, palette='winter',

                order=['5-14 years','15-24 years','25-34 years','35-54 years',

                      '55-74 years','75+ years'])

usa_age_s.set_xticklabels(usa_age_s.get_xticklabels(), rotation=30)

usa_age_s.set_xlabel('Age')

usa_age_s.set_ylabel('Number of Suicides')

usa_age_s.set_title('Number of Suicides per Age Group in the USA: 1979-2015')
usa_age_s = sns.barplot(x='age',y='rate_per_100k',data=usa_df, palette='BuGn_r',

                order=['5-14 years','15-24 years','25-34 years','35-54 years',

                      '55-74 years','75+ years'])

usa_age_s.set_xticklabels(usa_age_s.get_xticklabels(), rotation=30)

usa_age_s.set_xlabel('Age')

usa_age_s.set_ylabel('Number of Suicides per 100k')

usa_age_s.set_title('Number of Suicides per Age Group in the USA: 1979-2015')
sns.set_style("whitegrid")

g=sns.catplot(x="sex",y="rate_per_100k", hue="age", kind="bar", data=usa_df, palette='PRGn').set_xticklabels(rotation=90)

(g.fig.suptitle('Number of suicides per 100k: USA'))
sns.set_style("whitegrid")

usa_year_s = sns.barplot(x='year',y='rate_per_100k',data=df, palette='BuGn_r')

                

usa_year_s.set_xticklabels(usa_year_s.get_xticklabels(), rotation=90)

usa_year_s.set_xlabel('Year')

usa_year_s.set_ylabel('Number of Suicides per 100k people')

usa_year_s.set_title('Number of Suicides per Year: USA')
usa_decades = usa_df.loc[usa_df['year'].isin(['1985', '1995', '2005', '2015'])]
sns.set_style("whitegrid")

g=sns.catplot(x="age",y="rate_per_100k",  col='sex', hue="year", kind="bar",palette='PRGn', data=usa_decades,order=['5-14 years','15-24 years','25-34 years','35-54 years',

                      '55-74 years','75+ years']).set_xticklabels(rotation=90)



(g.despine(left=True),g.set_axis_labels("", "Number of Suicides per 100k people"))
sns.set_style("whitegrid")

sns.set_palette('Blues')

g=sns.catplot(x="sex", y="suicides_no", kind="swarm", hue='age', data=usa_df)

(g.fig.suptitle('Number of suicides: USA'))
sns.set_style("whitegrid")

sns.set_palette('Blues')

g=sns.catplot(x="year", y="rate_per_100k", kind="swarm", hue='age', col='sex', data=usa_df).set_xticklabels(rotation=90)

sns.set_style("whitegrid")

sns.set_palette('Blues')

g=sns.catplot(x="year", y="suicides_no", kind="swarm", hue='age', col='sex', data=usa_df).set_xticklabels(rotation=90)
usa_df.columns