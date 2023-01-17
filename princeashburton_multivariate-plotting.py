

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re 
import seaborn as sns
import os
print(os.listdir("../input"))
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go

# Any results you write to the current directory are saved as output.
pd.set_option('max_columns', None)
df = pd.read_csv("../input/fifa-18-demo-player-dataset/CompleteDataset.csv", index_col=0)

footballers = df.copy()
footballers['Unit'] = df['Value'].str[-1]
footballers['Value (M)'] = np.where(footballers['Unit'] == '0', 0, 
                                    footballers['Value'].str[1:-1].replace(r'[a-zA-Z]',''))
footballers['Value (M)'] = footballers['Value (M)'].astype(float)
footballers['Value (M)'] = np.where(footballers['Unit'] == 'M', 
                                    footballers['Value (M)'], 
                                    footballers['Value (M)']/1000)
footballers = footballers.assign(Value=footballers['Value (M)'],
                                 Position=footballers['Preferred Positions'].str.split().str[0])
footballers.head(100)
df = footballers['Nationality'].value_counts()

iplot([
    go.Choropleth(
    locationmode='country names',
    locations=df.index.values,
    text= df.index,
    z=df.values
    )
])
sns.lmplot(x='Value', y='Overall',hue='Position',
          data=footballers.loc[footballers['Position'].isin(['ST', 'RW','LW'])],
          fit_reg=False)
sns.lmplot(x='Value', y='Overall', markers=['o','x','*'], hue='Position',
          data=footballers.loc[footballers['Position'].isin(['ST','RW','LW'])],
          fit_reg=False)
f = (footballers
         .loc[footballers['Position'].isin(['ST','GK'])]
        .loc[:,['Value', 'Overall','Aggression','Position']]
    )
f= f[f["Overall"] >= 80]
f  = f[f["Overall"] < 85]
f['Aggression'] = f['Aggression'].astype(float)

sns.boxplot(x="Overall", y="Aggression", hue='Position', data=f)
f = (
   footballers.loc[:,['Acceleration','Aggression','Agility','Balance','Ball control']]
    .applymap(lambda v: int(v) if str.isdecimal(v) else np.nan)
    .dropna()
).corr()

sns.heatmap(f, annot=True)
from pandas.plotting import parallel_coordinates

f = (
   footballers.iloc[:, 12:17]
       .loc[footballers['Position'].isin(['ST','GK'])]
       .applymap(lambda v: int(v) if str.isdecimal(v) else np.nan)
       .dropna()
)
f['Position'] = footballers['Position']
f = f.sample(200)

parallel_coordinates(f, 'Position')
pokemon = pd.read_csv("../input/pokemon/Pokemon.csv", index_col=0)
pokemon.head(100)
sns.lmplot(x='Attack', y='Defense', hue='Legendary', markers=['x','o'],
          data=pokemon,
          fit_reg=False)
sns.boxplot(x="Generation",y="Total",hue='Legendary',data=pokemon)
p = (
   pokemon.loc[:, ['HP','Attack','Defense','Sp. Atk','Sp. Def','Speed']]
       
         
).corr()

sns.heatmap(p,annot=True)
p = (
    pokemon[(pokemon['Type 1'].isin(['Fighting','Psychic']))]
 .loc[:, ['Type 1', 'Attack','Sp. Atk', 'Defense','Sp. Def']]
)

parallel_coordinates(p,'Type 1')
