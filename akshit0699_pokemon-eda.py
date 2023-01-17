# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/pokemon-challenge/pokemon.csv')
data.head()
data.info()
data.shape
data.isna().sum()
data.corr()>0.5
import plotly.express as px
fig = px.imshow(data.corr(),x = data.corr().index, y = data.corr().index)
fig
px.scatter(data, "Defense", "Speed", color = "Legendary", size = "HP")
fig = px.density_heatmap(data, x="Defense", y="Attack", marginal_x="histogram", marginal_y="histogram")
fig.show()
px.histogram(data, "Defense", color = "Generation")
px.histogram(data, "Speed", color = "Generation")
data.head()
data["GenesHP"] = data.groupby("Generation")["HP"].transform("mean")
data.groupby(["Type 1", "Type 2"])[["HP", "Attack", "Speed"]].agg(["mean", "min", "max", "count"])
data.head()
data["GenesHP"].value_counts()
data[data['Defense']>190]
data[ (data['Defense']>200) & (data['Attack']>100)]
data.describe()
data.columns
data.info()
data['Type 1'].value_counts()
data['Type 1'].nunique()
data['Type 2'].nunique()
data['Type 2'].value_counts()
data.groupby("Type 1")["Name"].count().sort_values(ascending= False).to_frame()
px.histogram(data, "Type 1")
px.histogram(data, "HP")
px.box(data, x = "Generation", y = "Defense", points = "all")
px.box(data, x = "Generation", y = "Attack", points = "all")
data.dtypes
data['Type 1'] = data['Type 1'].astype('category')
data['Speed'] = data['Speed'].astype('float')

data.isna().sum()
data.shape
data["Type 2"].fillna('unknown', inplace = True)
data.isna().sum()
data.dropna(inplace = True)
data.shape
px.line(data, y= "Defense", x = "Name")
data.groupby(["Type 1", "Type 2"]).count().dropna().unstack(-1)
data[data["Legendary"] == True]
#finding the missing pokemon
data.isna().sum()
pokemon = pd.read_csv('/kaggle/input/pokemon-challenge/pokemon.csv')
pokemon.describe()
pokemon.isna().sum()
pokemon[pokemon['Name'].isnull()]
pokemon.loc[62, "Name"] = 'Primeape'
pokemon.iloc[62:67]
combats = pd.read_csv('/kaggle/input/pokemon-challenge/combats.csv')
combats.head()
combats.info()
# How often did the first_pokemon win?
combats[combats['First_pokemon'] == combats['Winner']].shape
# How often did the second_pokemon win?
combats[combats['Second_pokemon'] == combats['Winner']].shape
# How many winners were there?
wcomb = combats.groupby("Winner").count()
# How many unique first_poke winners were there?
fcomb = combats.groupby("First_pokemon").count()
# How many unique second_poke winners were there?
scomb = combats.groupby("Second_pokemon").count()
the_losing_pokemon = np.setdiff1d(fcomb.index.values, wcomb.index.values)-1
pokemon.iloc[the_losing_pokemon[0], ]
pokemon[pokemon["Name"] == "Pikachu"]
# FEATURE ENGINEERING, WIN PERCENTAGE CALCULATION
fcomb
wcomb.sort_index()
combats['First_pokemon'].nunique()
fcomb['Winner']
scomb['Winner']
wcomb['Total Fights' ] = fcomb['Winner']+ scomb['Winner']
wcomb['Win percentage'] = wcomb['First_pokemon']/wcomb['Total Fights']
combats.groupby('Winner')['Winner'].count()
combats.loc[combats['Winner']== 1]
combats.loc[combats['Winner']== 1].shape
combats.loc[ (combats['First_pokemon']== 1) | (combats['Second_pokemon']== 1) ]
combats.loc[ (combats['First_pokemon']== 1) | (combats['Second_pokemon']== 1) ].shape
pokemon.loc[pokemon['#']== 1]
fcomb
combats_winner = combats.groupby('Winner').count()
combats_first = combats.groupby('First_pokemon').count()
combats_second = combats.groupby('Second_pokemon').count()
#pokemon.iloc[297]
combats['Total_Matches'] = combats_first['Winner']+ combats_second['Winner']
#combats.drop(['Win percentage(first)'], axis =1, inplace = True )
combats['Win percentage'] = (combats_winner['First_pokemon'] /combats['Total_Matches'] )*100
combats[combats['Total_Matches'].isnull()]
res = pd.merge(pokemon, combats, right_index = True, left_on = '#')
res[res['Win percentage'].isnull()]
res.sort_values(by = 'Win percentage', ascending = False).head()
px.scatter(res, "Attack", "Win percentage", color = "Legendary", trendline="ols")
px.scatter(res, "Defense", "Win percentage", color = "Legendary", trendline="ols")
px.scatter(res, "Speed", "Win percentage", color = "Legendary", trendline="ols")
winnings = res.groupby('Type 1')['Win percentage'].mean().reset_index().sort_values(by = 'Win percentage' ,ascending = False)
px.bar(winnings, x = "Type 1", y = "Win percentage")
res.info()
data = dict(
    character=["Eve", "Cain", "Seth", "Enos", "Noam", "Abel", "Awan", "Enoch", "Azura"],
    parent=["", "Eve", "Eve", "Seth", "Seth", "Eve", "Eve", "Awan", "Eve" ],
    value=[10, 14, 12, 10, 2, 6, 6, 4, 4])
pd.DataFrame.from_dict(data)
res.sample(20).groupby("Type 1")[["Attack", "Defense", "Speed", "Win percentage"]].mean()
res.head()
# What makes a good pokemon?
good_pokemon = res[res['Win percentage']>85][['Attack', 'Defense', 'Speed']].mean().to_frame()
px.pie(good_pokemon, values = 0, names = good_pokemon.index)
sliced = res.sample(10)
sliced
melted = pd.melt(sliced, id_vars = ['Name', 'Win percentage'], value_vars = ['Attack', 'Defense'])
melted
melted.pivot(index = "Name", columns = 'variable', values = 'value')
res.shape
pokemon.shape
temp = res[res['Type 2'].isnull()]
temp
res['Type 2'] = np.where(res['Type 2'].isnull(), res['Type 1'], res['Type 2'])
res.sample(20)
res.isnull().sum()
res.fillna(0)
res[res['Total_Matches'].isnull()]
res['Total_Matches'].fillna(0)
res[res['Total_Matches'].isnull()]
res['Total_Matches'] = np.where(res['Total_Matches'].isnull(), 0, res['Total_Matches'])
res.info()
res['Win percentage'] = np.where(res['Win percentage'].isnull(), 0, res['Win percentage'])
res[res['Win percentage']== 0]
res.info()
res1 = res.sample(100)
px.sunburst(res, path = ['Type 1', 'Type 2', 'Name'], values = 'HP',color= 'Win percentage', hover_data=['Attack', 'Defense'],
                  color_continuous_scale= 'Inferno',
                  color_continuous_midpoint=np.average(res['Win percentage']))
df = px.data.tips()
#fig = px.sunburst(df, path=['day', 'time', 'sex'], values='total_bill')
df
