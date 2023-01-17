import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df = pd.read_csv('/kaggle/input/suicide-rates-overview-1985-to-2016/master.csv')
df.head()
df.shape
print('There are %s countries in the dataset'%(df['country'].nunique()))

print('The analysis starts from the year %s to %s'%(np.min(df['year']),np.max(df['year'])))
df.isnull().sum()
df['age'].value_counts()
df_usa = df[df['country'] == 'United States']
df_usa = df_usa.reset_index(drop= True)
df_usa.head()
df_usa.drop(columns=['country-year','HDI for year',' gdp_for_year ($) ','gdp_per_capita ($)','generation'], inplace= True)
df_usa.head()
df_usa[(df_usa['year'] == 1985) & (df_usa['age'] == '75+ years')]
df_usa['year'].nunique()
sns.catplot(

    data=df_usa, kind="bar",

    x="year", y="suicides_no", hue="sex",

    ci= None, palette="dark",height=5, aspect= 3

)

plt.title('Suicide Count in USA')

plt.savefig('suicide-count-usa.png')
sns.catplot(

    data=df_usa, kind="point",

    x="year", y="suicides_no", hue="age",

    ci= None, palette="dark",height=5, aspect= 3

)

plt.title('Suicide Count by Age Group')

plt.savefig('suicide-count-age-usa.png')
df_group = df_usa.groupby(['year','sex'])['suicides_no'].mean().unstack().plot() 

plt.savefig('aggregate-mean-allage-usa.png')