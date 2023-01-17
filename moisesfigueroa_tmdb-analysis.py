import pandas as pd
import numpy as np
import ast

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import squarify as sq

from IPython.display import display, HTML

display(HTML(data="""
<style>
    div#notebook-container    { width: 95%; }
    div#menubar-container     { width: 65%; }
    div#maintoolbar-container { width: 99%; }
</style>
"""))
data = pd.read_csv('../input/tmdbset/main.csv')
data.head(3)
data.tail(3)
data.info()
data_processed = data.copy()
data_processed.drop(columns=['adult','backdrop_path','homepage','poster_path','popularity','video',
                'status', 'imdb_id'], inplace=True)
data_processed.info()
data_processed.belongs_to_collection = data_processed.belongs_to_collection.apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else pd.NA)
data_processed.spoken_languages	= data_processed.spoken_languages.apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else pd.NA)
#data_processed.loc[data_processed.spoken_languages.apply(lambda x: len(x)) == 0, 'spoken_languages'] = pd.NA
data_processed.genres = data_processed.genres.apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else pd.NA)
#data_processed.loc[data_processed.genres.apply(lambda x: len(x)) == 0, 'genres'] = pd.NA
data_processed.production_companies = data_processed.production_companies.apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else pd.NA)
#data_processed.loc[data_processed.production_companies.apply(lambda x: len(x)) == 0, 'production_companies'] = pd.NA
data_processed.production_countries= data_processed.production_countries.apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else pd.NA)
#data_processed.loc[data_processed.production_countries.apply(lambda x: len(x)) == 0, 'production_countries'] = pd.NA
data_processed.loc[data_processed.revenue == 0, 'revenue'] = pd.NA
data_processed.loc[data_processed.budget == 0, 'budget'] = pd.NA
data_processed.info()
data_processed.budget = data_processed.budget/1000000
data_processed.revenue = data_processed.revenue/1000000
data_processed.info()
data_processed.original_language = data_processed.original_language.astype('category')
data_processed.release_date = data_processed.release_date.astype('datetime64')
data_processed.id = data_processed.id.astype('Int32')
data_processed.runtime = data_processed.runtime.astype('Int16')
data_processed.budget = pd.to_numeric(data_processed.budget, errors='coerce')
data_processed.revenue = pd.to_numeric(data_processed.revenue, errors='coerce')
data_processed.info()
data_processed.drop(data_processed.loc[(data_processed.release_date.dt.year < 1960) | (data_processed.release_date.dt.year > 2019)].index,inplace=True)
data_processed.drop(data_processed.loc[data_processed.release_date.isna()].index, inplace=True)
inflation = pd.read_csv('../input/top-movies-19602020/CPI.csv', index_col='observation_date', parse_dates=True)

inflation = inflation.resample('A',kind='period').mean()

cpi_dict = {}
for x,y in zip(inflation.index.year, inflation.CPIAUCSL):
    cpi_dict[x] = round(y,2)
    
data_processed.budget = data_processed.apply(lambda x: x.budget * (cpi_dict[2020] / cpi_dict[x.release_date.year]), axis=1)
data_processed.revenue = data_processed.apply(lambda x: x.revenue * (cpi_dict[2020] / cpi_dict[x.release_date.year]),axis=1)
data_processed.budget = data_processed.budget.apply(lambda x: round(x,2) if isinstance(x, float) else pd.NA)
data_processed.revenue = data_processed.revenue.apply(lambda x: round(x,2) if isinstance(x, float) else pd.NA)
data_processed['roi'] = data_processed.apply(lambda x: x.revenue - x.budget, axis=1)
data_processed.roi = pd.to_numeric(data_processed.roi, errors='coerce')

data_processed['decade'] = (data_processed.release_date.dt.year // 10) * 10
data_processed['month'] = data_processed.release_date.dt.month
def get_assoc_roi_per_genre(df, groupby):
    df = df.explode('genres')
    df.genres = df.genres.apply(lambda x: [x[key] for key in x.keys()][1] if isinstance(x, dict) else pd.NA)
    df.loc[df.genres=='Science Fiction','genres'] = 'Sci Fi'

    df = pd.DataFrame(df.groupby(groupby).roi.sum())
    return df
df_roi = data_processed[['roi','genres']].copy()

df_roi = get_assoc_roi_per_genre(df_roi, 'genres')

df_roi = df_roi.sort_values('roi',ascending=False)
df_roi
fig, ax = plt.subplots()
fig.set_size_inches(20,10)

mini= min(df_roi.roi)
maxi= max(df_roi.roi)
norm = matplotlib.colors.Normalize(vmin=mini, vmax=maxi)
colors = [matplotlib.cm.cool(norm(value)) for value in df_roi.roi]

sq.plot(sizes=df_roi.roi, label=df_roi.index, ax=ax,ec='black', lw=2,color=colors, text_kwargs={'size':15})

ax.axis('off')
fig.suptitle('Genres associated with the Most ROI - 1960-2019',y=.96, size=35)
plt.show()
df_roi_dec = data_processed[['roi','genres','decade']].copy()

df_roi_dec = get_assoc_roi_per_genre(df_roi_dec, ['decade', 'genres'])

df_roi_dec.drop(df_roi_dec.loc[df_roi_dec.roi < 1, 'roi'].index, inplace=True)

df_roi_dec = df_roi_dec.sort_values('roi',ascending=True)
fig, axes = plt.subplots(3,2)
fig.set_size_inches(30,25)
axes=axes.flatten()

for ax,decade in zip(axes, range(1960, 2011, 10)):
    mini= min(df_roi_dec.loc[decade, 'roi'])
    maxi= max(df_roi_dec.loc[decade, 'roi'])
    norm = matplotlib.colors.Normalize(vmin=mini, vmax=maxi)
    colors = [matplotlib.cm.cool(norm(value)) for value in df_roi_dec.loc[decade, 'roi']]

    sq.plot(sizes=df_roi_dec.loc[decade, 'roi'], label=df_roi_dec.loc[decade, 'roi'].index.get_level_values(0),
            ax=ax,ec='black', lw=2,color=colors, text_kwargs={'size':15})
    ax.axis('off')
    ax.set_title(str(decade)+'s',size=30)
    
fig.suptitle('Genres associated with the Most ROI per decade',x =.5,y=.94, size=40)
plt.show()
def create_ranking_chart(df, onlyshow=''):
    fig,ax = plt.subplots()
    fig.set_size_inches(31,16)

    for x in df.genres.unique():  
        sns.lineplot(data=df[df.genres == x], x='decade', y='ranks', ax=ax,
                     markers=['h'],markeredgecolor='black',markeredgewidth=3,style='genres',lw=8,label=x,legend=False,markersize=35)

    for x in range(1960, 2011, 10):
        for y in range(1,20):
            ax.annotate(y, (x,y),xytext=(0, -1), textcoords='offset points', ha='center', va='center',size = 15, weight='bold',
                        label = df.loc[(df.decade == x) & (df.ranks == y), 'genres'].iat[0],color='white')

    for line, name in zip(ax.lines, df.loc[df.decade == 1960, 'genres']):
        y = line.get_ydata()[0]
        x = line.get_xdata()[0]
        ax.annotate(name,(x,y),xytext=(-25, 0), textcoords='offset points', ha='right',va='center', size=15,label=name)
        y = line.get_ydata()[-1]
        x = line.get_xdata()[-1]
        ax.annotate(name,(x,y),xytext=(25, 0), textcoords='offset points', ha='left',va='center', size=15,label=name)
        
    if not isinstance(onlyshow, str):    
        for elem in ax.lines + ax.texts:
            if elem.get_label() not in onlyshow:
                elem.set_visible(False)


    formatter = ticker.FormatStrFormatter('%ds')
    y = range(1960,2011,10)
    labels = [formatter(x) for x in y]
    ax.set_xticks(range(1960,2011,10))
    ax.set_xticklabels(labels)

    ax.set_yticks([])
    ax.set_ylabel(ylabel='Rank', size=30, labelpad=10)
    ax.set_xlabel(xlabel='Decades', size=30,labelpad=20)
    ax.set_axisbelow(True)
    ax.tick_params(axis='both',labelsize=20)
    ax.grid('x')
    ax.margins(x=.10)
def create_ranking(df):
    df = df.unstack(level=0)
    df.columns = df.columns.droplevel(0)
    df = df.rank(na_option='top').sort_values(1960).reset_index().melt(id_vars='genres', value_name='ranks')
    return df
df_roi_dec = data_processed[['roi','genres','decade']].copy()

df_roi_dec = get_assoc_roi_per_genre(df_roi_dec, ['decade', 'genres'])

df_roi_rank = create_ranking(df_roi_dec)
create_ranking_chart(df_roi_rank)
df_roi_dec = data_processed[['roi','genres','decade']].copy()

df_roi_dec = get_assoc_roi_per_genre(df_roi_dec, ['decade', 'genres'])

df_roi_rank = create_ranking(df_roi_dec)
df_roi_rank.groupby('genres').ranks.describe().sort_values('std')
show = df_roi_rank.groupby('genres').ranks.describe().sort_values('std').head(5).index
create_ranking_chart(df_roi_rank, show)
df_roi_dec = data_processed[['roi','genres','decade']].copy()

df_roi_dec = get_assoc_roi_per_genre(df_roi_dec, ['decade', 'genres'])

df_roi_rank = create_ranking(df_roi_dec) 
show = df_roi_rank.groupby('genres').ranks.describe().sort_values('std').tail(5).index
create_ranking_chart(df_roi_rank, show)
data_processed.sort_values('roi')
df_month = data_processed[['id', 'month', 'roi']].copy()
df_month.groupby('month').roi.describe().sort_values('mean')
df_month_low = data_processed.loc[data_processed.budget <= 5,['id', 'month', 'roi']].copy()
df_month_low['std'] = df_month_low.groupby('month').roi.transform('std')#.describe().sort_values(['count', 'mean'], ascending=[True, False])
df_month_low['mean'] = df_month_low.groupby('month').roi.transform('mean')
df_month_low.apply(lambda x: x if (True) else False, axis=1)
df_month_low.info()
df_month_low = df_month_low.loc[(df_month_low.roi > (df_month_low['mean'] - (df_month_low['std'] * 2))) & (df_month_low.roi < (df_month_low['mean'] + (df_month_low['std']*2)))].sort_values('roi')
df_month_low.groupby('month').roi.describe()
df_month_low
