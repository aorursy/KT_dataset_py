import matplotlib as plt
import pandas as pd
import seaborn as sns

df = pd.read_csv('../input/pokemon.csv')
df.head()
print(df.columns)
print(df.describe)
x = df['type1'].value_counts().plot.bar(
    figsize = (12,6),
    fontsize= '20'
)
x.set_title('Pokemon Primary Types', fontsize = '20')
sns.despine(bottom = True, left = True)
x = df['hp'].value_counts().sort_index().plot.line(
    figsize = (12,6),
    fontsize= '20'
)
x.set_title('Pokemon Health Points', fontsize = '20')
sns.despine(bottom = True, left = True)
df['sp_attack'].value_counts().sort_index().plot.line()
df['weight_kg'].value_counts().sort_index().plot.hist()
df.plot.scatter(x='sp_attack', y='sp_defense')
pokemon = df.groupby(['generation']).mean()[['sp_attack', 'sp_defense', 'speed', 'hp']]
pokemon.plot.line()
df.plot.hexbin(x='attack', y='defense', gridsize=15)
#SeabornPlots
sns.countplot(df['generation'])
sns.distplot(df['hp'], kde=True)
sns.jointplot(x='attack', y='defense', data = df)
sns.jointplot(x='attack', y='defense', data = df, kind='hex', gridsize=20)
sns.kdeplot(df['attack'], df['defense'])
df1 = df[['is_legendary', 'attack']]

sns.boxplot(
    x='is_legendary',
    y='attack',
    data=df1
)
#Faceting
g = sns.FacetGrid(df, col = 'is_legendary')
g.map(sns.kdeplot, 'attack')
g = sns.FacetGrid(df, col = 'is_legendary', row = 'generation')
g.map(sns.kdeplot , 'attack')
sns.pairplot(df[['hp', 'attack', 'defense']])
#MultiVariate
sns.lmplot(x='attack', y='defense', hue='is_legendary', 
           markers=['x', 'o'],
           fit_reg=True, data=df)
sns.heatmap(
    df.loc[:, ['hp', 'attack', 'defense','speed']].corr(),
    annot=True
)
#Plotly
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
iplot([go.Scatter(x=df['attack'], y=df['defense'], mode='markers')])