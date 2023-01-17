import pandas as pd
import re
import numpy as np

pd.set_option('max_columns', None)
df = pd.read_csv("../input/fifa-18-demo-player-dataset/CompleteDataset.csv", index_col=0)
df.head(3)
footballers = df.copy()
footballers.head(3)
footballers['Unit'] = df['Value'].str[-1]
footballers['Value (M)'] = np.where(footballers['Unit'] == '0', 0, 
                                    footballers['Value'].str[1:-1].replace(r'[a-zA-Z]',''))
footballers['Value (M)'] = footballers['Value (M)'].astype(float)
footballers['Value (M)'] = np.where(footballers['Unit'] == 'M', 
                                    footballers['Value (M)'], 
                                    footballers['Value (M)']/1000)
footballers = footballers.assign(Value=footballers['Value (M)'],
                                 Position=footballers['Preferred Positions'].str.split().str[0])
footballers.head(3)
footballers.head()
import seaborn as sns

sns.lmplot(x='Value', y='Overall', hue='Position', 
           data=footballers.loc[footballers['Position'].isin(['ST', 'RW', 'LW'])], 
           fit_reg=False)
sns.lmplot(x='Value', y='Overall', markers=['o', 'x', '*'], hue='Position',
           data=footballers.loc[footballers['Position'].isin(['ST', 'RW', 'LW'])],
           fit_reg=False
          )
f1=(footballers.loc[footballers['Position'].isin(['ST','GK'])].loc[:,['Value', 'Overall', 'Aggression', 'Position']])
f1 = f1[f1["Overall"] >= 80]
f1 = f1[f1["Overall"] < 85]
f1['Aggression'] = f1['Aggression'].astype(float)

sns.boxplot(x="Overall", y="Aggression", hue='Position', data=f1)
f = (
    footballers.loc[:, ['Acceleration', 'Aggression', 'Agility', 'Balance', 'Ball control']]
        .applymap(lambda v: int(v) if str.isdecimal(v) else np.nan)
        .dropna()
).corr()

sns.heatmap(f, annot=True)
from pandas.plotting import parallel_coordinates

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
import seaborn as sns

sns.lmplot(x='Attack', y='Defense', hue='Legendary', 
           markers=['x', 'o'],
           fit_reg=False, data=pokemon)
sns.boxplot(x="Generation", y="Total", hue='Legendary', data=pokemon)
sns.heatmap(
    pokemon.loc[:, ['HP', 'Attack', 'Sp. Atk', 'Defense', 'Sp. Def', 'Speed']].corr(),
    annot=True
)
import pandas as pd
from pandas.plotting import parallel_coordinates

p = (pokemon[(pokemon['Type 1'].isin(["Psychic", "Fighting"]))]
         .loc[:, ['Type 1', 'Attack', 'Sp. Atk', 'Defense', 'Sp. Def']]
    )

parallel_coordinates(p, 'Type 1')