import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
df = pd.read_csv('../input/mort.csv')
df.head()
df.info()
df.describe()
pc_cols = ['Location','FIPS','Category',
           '% Change in Mortality Rate, 1980-2014',
           '% Change in Mortality Rate, 1980-2014 (Min)',
           '% Change in Mortality Rate, 1980-2014 (Max)']
pct_chg = df[pc_cols].set_index('Location')
pct_chg.head()
pa = pct_chg.loc['Pennsylvania']
pa
pa_by_cat = pa.reset_index()
pa_by_cat.drop(['Location','FIPS'],axis=1,inplace=True)
pa_by_cat.set_index(['Category'],inplace=True)
pa_by_cat['% Change in Mortality Rate, 1980-2014'].plot(kind='bar', rot=90, figsize=(14,8));
plt.ylabel('% Change in Mort Rate, 1980-2014');
pa_by_cat[pa_by_cat['% Change in Mortality Rate, 1980-2014'] > 0]

pa_cats_over_30 = ['HIV/AIDS and tuberculosis', 'Neurological disorders',
                   'Mental and substance use disorders', 'Maternal disorders']

mort_rates = ['Location','Category','Mortality Rate, 1980*','Mortality Rate, 1985*','Mortality Rate, 1990*','Mortality Rate, 1995*',
              'Mortality Rate, 2000*','Mortality Rate, 2005*','Mortality Rate, 2010*','Mortality Rate, 2014*']

locs = ['United States','Pennsylvania']
pa_us = df[mort_rates].set_index(['Category'])
pa_us_cat = pa_us.loc[pa_cats_over_30]
pa_us_df = pa_us_cat[(pa_us_cat.Location == locs[1]) | (pa_us_cat.Location == locs[0])]
pa_us_df.reset_index(inplace=True)
pa_us_df
temp2 = pa_us_df.melt(id_vars=mort_rates[:2], value_vars=mort_rates[2:])
temp2.head()
# g=sns.factorplot(y='value', x='Location', hue='Category', 
#                  col='variable', col_wrap=4,
#                  data=temp2, kind='bar')

g=sns.factorplot(y='value', x='variable', col='Location', hue='Category',
                 data=temp2, aspect=2, size=5)
g.set_xticklabels(rotation=90)
pcmr = ['Location','Category','% Change in Mortality Rate, 1980-2014']
pa_us_pc = df[pcmr].set_index(['Category'])
pa_us_pc_cat = pa_us_pc.loc[pa_cats_over_30]
pa_us_pc_df = pa_us_pc_cat[(pa_us_pc_cat.Location == locs[1]) | (pa_us_pc_cat.Location == locs[0])]
pa_us_pc_df.reset_index(inplace=True)
pa_us_pc_df
g2=sns.factorplot(y='% Change in Mortality Rate, 1980-2014', 
                  x='Location', hue='Category', 
                  size=5, aspect=3,
                  data=pa_us_pc_df, kind='bar')
