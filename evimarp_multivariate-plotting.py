# Importing cell
import pandas as pd
from pandas.plotting import parallel_coordinates
import seaborn as sns
import re
import numpy as np

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
footballers.head()


sns.lmplot(x='Value', y='Overall', hue='Position', 
           data=footballers.loc[footballers['Position'].isin(['ST', 'RW', 'LW'])], 
           fit_reg=False)
sns.lmplot(x='Value', y='Overall', markers=['o', 'x', '*'], hue='Position',
           data=footballers.loc[footballers['Position'].isin(['ST', 'RW', 'LW'])],
           fit_reg=False
          )
f = (footballers
         .loc[footballers['Position'].isin(['ST', 'GK'])]
         .loc[:, ['Value', 'Overall', 'Aggression', 'Position']]
    )
f = f[f["Overall"] >= 80]
f = f[f["Overall"] < 85]
f['Aggression'] = f['Aggression'].astype(float)

sns.boxplot(x="Overall", y="Aggression", hue='Position', data=f)
f = (
    footballers.loc[:, ['Acceleration', 'Aggression', 'Agility', 'Balance', 'Ball control']]
        .applymap(lambda v: int(v) if str.isdecimal(v) else np.nan)
        .dropna()
).corr()

sns.heatmap(f, annot=True)


f = (
    footballers.iloc[:, 12:17]
        .loc[footballers['Position'].isin(['ST', 'GK'])]
        .applymap(lambda v: int(v) if str.isdecimal(v) else np.nan)
        .dropna()
)
f['Position'] = footballers['Position']
f = f.sample(200)

parallel_coordinates(f, 'Position')
pokemon = pd.read_csv("../input/pokemon/Pokemon.csv", index_col=0)
pokemon.head()
from IPython.display import HTML
HTML("""
<ol>
<li>The three techniques we have covered in this tutorial are faceting, using more visual variables, and summarization.</li>
<br/>
<li>Some examples of visual variables are shape, color, size, x-position, y-position, and grouping. However there are many more that are possible!</li>
<br/>
<li>In data visualization, summarization works by compressing complex data into simpler, easier-to-plot indicators.</li>
</ol>
""")
# Defense vs Attack group by Legendary

sns.lmplot(x='Attack', y='Defense', hue='Legendary',
          data=pokemon, markers=['x', 'o'], fit_reg=False)

# fit_reg param indicate that does not plot a fit linear model
# that is the default behavior
# Generation vs Total for Legendary box example
sns.boxplot(x='Generation', y='Total', hue='Legendary', data=pokemon)
# heatmap example of correlation between HP, Attack, Sp Attack, 
sns.heatmap(pokemon.iloc[:, 4:10].corr(), annot=True)
# parallel coordinates example for HP Attack and Defense variables
# aggregated by Fighting vs Phychic

df = pokemon.iloc[:, 5:9]
df["Type"] = pokemon["Type 1"]

df = df.loc[df.Type.isin(['Fighting', 'Psychic'])]
df.iloc[:, [4,0,2,1,3]]  # sort the columns

parallel_coordinates(df, "Type")