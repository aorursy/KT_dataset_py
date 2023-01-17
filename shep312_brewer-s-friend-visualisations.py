import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.offline as py
import plotly.figure_factory as ff
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
df = pd.read_csv('../input/recipeData.csv', index_col='BeerID', encoding='latin1')
df.head()
print('Number of recipes =\t\t{} \nNumber of beer styles =\t{}'.format(len(df), len(df['Style'].unique())))
top_n_types = 15
recipe_popularity_as_perc = 100 * df['Style'].value_counts()[:top_n_types] / len(df)

pltly_data = [go.Bar(x=recipe_popularity_as_perc.index,
                     y=recipe_popularity_as_perc.values)]

layout = go.Layout(title='Most popular beer styles',
                   xaxis={'title': 'Style'},
                   yaxis={'title': 'Proportion of recipes (%)'},
                   margin=go.Margin(l=50, r=50, b=150, t=50, pad=4))

fig = go.Figure(data=pltly_data, layout=layout)
py.iplot(fig)

broad_styles = ['Ale', 'IPA', 'Pale Ale', 'Lager', 'Stout', 'Bitter', 'Cider', 'Porter']
df['BroadStyle'] = 'Other'
df['Style'].fillna('Unknown', inplace=True)
for broad_style in broad_styles:
    df.loc[df['Style'].str.contains(broad_style), 'BroadStyle'] = broad_style
style_popularity_as_perc = 100 * df['BroadStyle'].value_counts() / len(df)
style_popularity_as_perc.drop('Other', inplace=True)

pltly_data = [go.Bar(x=style_popularity_as_perc.index,
                     y=style_popularity_as_perc.values)]

layout = go.Layout(title='Most popular general styles',
                   xaxis={'title': 'Style'},
                   yaxis={'title': 'Proportion of recipes (%)'})

fig = go.Figure(data=pltly_data, layout=layout)
py.iplot(fig)
fig, ax = plt.subplots(1, 1, figsize=[12,5])
sns.distplot(df['ABV'], ax=ax)
ax.set_title('ABV distribution')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.get_yaxis().set_visible(False)
plt.show()
strengths = [10, 15, 20, 30, 50]
for abv in strengths:
    print('{} ({:.2f})%\tbeers stronger than {} ABV'.format(sum(df['ABV'] > abv), 100 * sum(df['ABV'] > abv) / len(df), abv))
abv_df = df[df['ABV'] <= 15]
fig, ax = plt.subplots(1, 1, figsize=[12, 5])
sns.violinplot(x='BroadStyle',
               y='ABV',
               data=abv_df,
               ax=ax)
ax.set_xlabel('General beer style')
ax.set_title('ABV by beer style')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show()
df['gravity_change'] = df['OG'] - df['FG']  # Suspect this may be correlated with ABV
df_for_corr = df.drop(['Style', 'BroadStyle', 'StyleID'], axis=1).copy()
categoricals = df_for_corr.columns[df_for_corr.dtypes == 'object']
for categorical in categoricals:
    print('{} has {} unique values'.format(categorical, len(df_for_corr[categorical].unique())))
    if len(df_for_corr[categorical].unique()) > 20:
           df_for_corr.drop(categorical, axis=1, inplace=True)
encoded_df = pd.get_dummies(df_for_corr)
corr_mat = encoded_df.corr()
abv_corrs = corr_mat['ABV'].sort_values()
abv_corrs.drop(['ABV', 'Color', 'IBU'], inplace=True)  # Color and IBU are results rather than parts of the brewing process so drop.
pltly_data = [go.Bar(y=abv_corrs.index,
                     x=abv_corrs.values,
                     orientation='h')]

layout = go.Layout(title='Linear correlations with ABV',
                   xaxis={'title': 'Correlation'},
                   margin=go.Margin(l=200, r=50, b=100, t=100, pad=4)
                  )

fig = go.Figure(data=pltly_data, layout=layout)
py.iplot(fig)
fig, ax = plt.subplots(1, 1, figsize=[10, 7])
sns.heatmap(corr_mat.mask(np.triu(np.ones(corr_mat.shape)).astype(np.bool)), ax=ax, center=0)
plt.show()
