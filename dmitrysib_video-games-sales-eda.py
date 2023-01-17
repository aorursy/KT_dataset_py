import pandas as pd

!pip install downcast

from downcast import reduce

import numpy as np

import matplotlib.pyplot as plt

import scipy.stats as st

import warnings

warnings.simplefilter('ignore')
df = pd.read_csv('../input/video-games/games.csv')
df_list = [df]

for df in df_list:

  display(df.sample(5))

  df.info()

  display(df.isna().sum())

  display(df.describe())
df.columns = df.columns.str.lower()
df = reduce(df)
df.dtypes
df['year_of_release']=df['year_of_release'].fillna(0).astype(int)

df.dtypes
df['user_score'] = pd.to_numeric(df['user_score'], errors='coerce')

df.dtypes
display(df.isna().sum())
df=df.dropna(subset=['name'])
df.fillna(dict.fromkeys(['critic_score', 'user_score'], -10), inplace=True)

df['rating'] = df['rating'].cat.add_categories('unknown').fillna('unknown')
df.shape
df.drop_duplicates(keep=False)
df.shape
df['global_sales']= df.iloc[:, -8:-4].sum(axis=1)

df.sort_values(by='global_sales', ascending=False)
rel_years=df['year_of_release'].value_counts()

rel_years=pd.DataFrame({'year_of_release':rel_years.index, 'games_num':rel_years.values})

rel_years.drop(rel_years[rel_years['year_of_release']==0].index, inplace=True)
def rel_years_chart(df, x, y):

  df.sort_values(by=x, ascending=True, inplace=True)

  ax=df.plot(x=x, y=y, kind='bar', figsize=(23, 5), rot=45)

  ax.get_legend().remove()

  plt.title('Number of games released by year', size=14)

  plt.xlabel('Year')

  plt.ylabel('Number of games')

  plt.show()



rel_years_chart(rel_years, 'year_of_release', 'games_num')
rel_years_10=rel_years.query('year_of_release>2006').copy()

rel_years_chart(rel_years_10, 'year_of_release', 'games_num')
platf_sales=df.groupby('platform')['global_sales'].sum().sort_values(ascending=False).reset_index()
platf_sales_top10=platf_sales.iloc[0:10,0:10]

ax=platf_sales_top10.plot(x='platform', y='global_sales', kind='bar', figsize=(23, 5), rot=360)

ax.get_legend().remove()

plt.title('Sales by platform', size=14)

plt.xlabel('Platform')

plt.ylabel('Worldwide sales, M copies')

plt.show()
top_10_platf=platf_sales_top10['platform'].tolist()

top_10_platf
platf_sales_top10=df.drop(df[df.year_of_release == 0].index)

platf_sales_top10=platf_sales_top10.query('platform==@top_10_platf')

platf_sales_top10['platform']=platf_sales_top10['platform'].astype('object')



platf_sales_top10_plot=platf_sales_top10.groupby(['year_of_release','platform']).count()['global_sales'].reset_index()



fig, axes = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True, figsize=(20,5))

axes_list = [item for sublist in axes for item in sublist]



for platf, selection in platf_sales_top10_plot.groupby('platform'):

  ax = axes_list.pop(0)

  selection.plot(x='year_of_release', y='global_sales', ax=ax, label=platf, legend=False)

  ax.set_title(platf)

  ax.grid(linewidth=0.25)

  ax.set_xlim((1993, 2016))

  ax.set_xlabel("Year")

  ax.set_ylabel("Number of games")

  ax.spines['top'].set_visible(False)

  ax.spines['right'].set_visible(False)

plt.tight_layout()
top_5_platf=['X360', 'Wii', 'PS3', 'PS4', 'PSP']

platf_sales_top5=platf_sales_top10.query('platform==@top_5_platf').copy()
fig, ax = plt.subplots(figsize=(23,5))

platf_sales_top5.groupby(['year_of_release','platform']).count()['global_sales'].unstack().plot(ax=ax)

plt.title('Number of games released by platform', size=14)

plt.xlabel('Year')

plt.ylabel('Number of games')

plt.show()
platf_sales_top5=platf_sales_top5.sort_values(by='platform')

ax=platf_sales_top5.boxplot(column='global_sales', by='platform', vert=False, figsize=(23,5), patch_artist=True, flierprops=dict(markeredgecolor="#e0434b"),

            medianprops=dict(color='#71c451'))

plt.suptitle('')

plt.title('Scatter of global sales by platform', size=14)

plt.xlabel('Worldwide sales, M copies')

plt.show()
platf_sales_top5_cut=platf_sales_top5.drop(platf_sales_top5[platf_sales_top5['global_sales'] > 1.1].index, inplace=True)

ax=platf_sales_top5.boxplot(column='global_sales', by='platform', vert=False, figsize=(23,5), patch_artist=True, flierprops=dict(markeredgecolor="#e0434b"),

            medianprops=dict(color='#71c451'))

plt.suptitle('')

plt.title('Scatter of global sales by platform', size=14)

plt.xlabel('Worldwide sales, M copies')

plt.show()
def score_vs_sales (df_slice, score):

  df_slice.plot(x=score, y='global_sales', kind='scatter', alpha=0.2)

  plt.xlabel('Score')

  plt.ylabel('Worldwide Sales, M copies')

  plt.show()

  print('Correlation - ', df_slice[score].corr(df['global_sales']).round(2))
score_vs_sales(platf_sales_top5.query('platform=="PS3"'), 'critic_score')
score_vs_sales(platf_sales_top5.query('platform=="PS3"'), 'user_score')
score_vs_sales(platf_sales_top5.query('platform!="PS3"'), 'critic_score')
score_vs_sales(platf_sales_top5.query('platform!="PS3"'), 'user_score')
genre=platf_sales_top5.groupby('genre')['global_sales'].sum().reset_index()

ax=genre.sort_values(by='global_sales', ascending=False).plot(x='genre', y='global_sales', kind='bar', figsize=(23, 5), rot=360)

ax.get_legend().remove()

plt.title('Top Selling Genres', size=14)

plt.xlabel('Genre')

plt.ylabel('Worldwide Sales, M copies')

plt.show()
ax=platf_sales_top5.boxplot(column='global_sales', by='genre', vert=False, figsize=(23,7), patch_artist=True, flierprops=dict(markeredgecolor="#e0434b"),

            medianprops=dict(color='#71c451'))

plt.suptitle('')

plt.title('Distribution of Global Sales by Genres', size=14)

plt.ylabel('Worldwide Sales, M copies')

plt.show()
region_sales=platf_sales_top5.pivot_table(index='platform', values=['na_sales', 'eu_sales', 'jp_sales', 'other_sales'], aggfunc='sum').reset_index().sort_values('na_sales', ascending = False)

ax=region_sales.plot(x='platform', y=['na_sales', 'eu_sales', 'jp_sales', 'other_sales'], kind='bar', figsize=(25, 5), rot=360)

ax.legend(['North America', 'Europe', 'Japan', 'Other Countries'])

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2,

            height + 2,

            '{:.2f}'.format(height),

            fontsize=10,

            ha="center")

plt.title('Top Selling Platforms in Different Regions', size=14)

plt.xlabel('Platform')

plt.ylabel('Sales, M copies')

plt.show()
na=platf_sales_top5.pivot_table(index='genre', values='na_sales', aggfunc='sum').reset_index().sort_values('na_sales', ascending = False).head(5)

eu=platf_sales_top5.pivot_table(index='genre', values='eu_sales', aggfunc='sum').reset_index().sort_values('eu_sales', ascending = False).head(5)

jp=platf_sales_top5.pivot_table(index='genre', values='jp_sales', aggfunc='sum').reset_index().sort_values('jp_sales', ascending = False).head(5)

other=platf_sales_top5.pivot_table(index='genre', values='other_sales', aggfunc='sum').reset_index().sort_values('other_sales', ascending = False).head(5)

region_sales=na.copy()

region_sales = region_sales.merge(eu, on='genre', how='outer')

region_sales = region_sales.merge(jp, on='genre', how='outer')

region_sales = region_sales.merge(other, on='genre', how='outer')
ax=region_sales.plot(x='genre', y=['na_sales', 'eu_sales', 'jp_sales', 'other_sales'], kind='bar', figsize=(25, 5), rot=360)

ax.legend(['North America', 'Europe', 'Japan', 'Other Countries'])

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2,

            height + 2,

            '{:.1f}'.format(height),

            fontsize=9,

            ha="center")

plt.title('Top Selling Genres in Different Regions', size=14)

plt.xlabel('Genre')

plt.ylabel('Sales, M copies')

plt.show()
region_sales=platf_sales_top5.pivot_table(index='rating', values=['na_sales', 'eu_sales', 'jp_sales', 'other_sales'], aggfunc='sum').reset_index().sort_values('na_sales', ascending = False)

ax=region_sales.plot(x='rating', y=['na_sales', 'eu_sales', 'jp_sales', 'other_sales'], kind='bar', figsize=(25, 5), rot=360)

ax.legend(['North America', 'Europe', 'Japan', 'Other Countries'])

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2,

            height + 2,

            '{:.1f}'.format(height),

            fontsize=9,

            ha="center")

plt.title('ESRB Rating and Sales in Different Regions', size=14)

plt.xlabel('Rating')

plt.ylabel('Sales, M copies')

plt.show()
hypo=df[df['platform'].isin(['PSP', 'X360']) ]

hypo=hypo.query('user_score != -10')
variance_estimate1 = np.var(hypo.loc[hypo['platform']=='X360', 'user_score'])

print(variance_estimate1)



variance_estimate2 = np.var(hypo.loc[hypo['platform']=='PSP', 'user_score'])

print(variance_estimate2)
alpha = .05

results = st.ttest_ind(hypo.query("platform=='X360'")['user_score'], hypo.query("platform=='PSP'")['user_score'], equal_var = False)

print('p-value: ', results.pvalue)

if (results.pvalue < alpha):

    print("Reject the null-hypothesis")

else:

    print("Fail to reject the null-hypothesis") 
hypo=df[df['genre'].isin(['Action', 'Sports']) ]
variance_estimate1 = np.var(hypo.loc[hypo['genre']=='Action', 'user_score'])

print(variance_estimate1)



variance_estimate2 = np.var(hypo.loc[hypo['genre']=='Sports', 'user_score'])

print(variance_estimate2)
alpha = .05

results = st.ttest_ind(hypo.query("genre=='Action'")['user_score'], hypo.query("genre=='Sports'")['user_score'], equal_var = True)

print('p-value: ', results.pvalue)

if (results.pvalue < alpha):

    print("Reject the null-hypothesis")

else:

    print("Fail to reject the null-hypothesis") 