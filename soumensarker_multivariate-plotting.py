import pandas as pd

pd.set_option('max_columns', None)

df = pd.read_csv("../input/fifa-18-demo-player-dataset/CompleteDataset.csv", index_col=0)



import re

import numpy as np



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
import seaborn as sns



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