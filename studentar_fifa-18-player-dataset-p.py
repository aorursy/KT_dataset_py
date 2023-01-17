import pandas as pd
pd.set_option('max_columns', None)
df = pd.read_csv("../input/CompleteDataset.csv", index_col=0)

import re
import numpy as np
import seaborn as sns
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
df = footballers[footballers['Position'].isin(['ST','GK'])]
g = sns.FacetGrid(df, col = "Position")
g.map(sns.kdeplot,"Overall")
df = footballers
g = sns.FacetGrid(df, col = "Position", col_wrap = 6)
g.map(sns.kdeplot,"Overall")
df = footballers[footballers['Position'].isin(['ST','GK'])]
df = df[df['Club'].isin(['Real Madrid CF', 'FC Barcelona', 'Atlético Madrid'])]
g = sns.FacetGrid(df, row = "Position", col = "Club")
g.map(sns.violinplot,"Overall")
df = footballers[footballers['Position'].isin(['ST','GK'])]
df = df[df['Club'].isin(['Real Madrid CF', 'FC Barcelona', 'Atlético Madrid'])]
g = sns.FacetGrid(df, row = "Position", col = "Club", 
                  row_order = ['GK','ST'], 
                  col_order = ['Atlético Madrid', 'FC Barcelona', 'Real Madrid CF'])
g.map(sns.violinplot,"Overall")
sns.pairplot(footballers[['Overall', 'Potential', 'Value']])
import seaborn as sns

sns.lmplot(x='Value', y='Overall', hue='Position', 
           data=footballers.loc[footballers['Position'].isin(['ST', 'RW', 'LW'])], 
           fit_reg=False)
sns.lmplot(x = 'Value', y = 'Overall', markers = ['o', 'x', '*'], hue = 'Position', 
           data = footballers.loc[footballers['Position'].isin(['ST','RW', 'LW'])],
           fit_reg = False)
f = (footballers
    .loc[footballers['Position'].isin(['ST','GK'])]
    .loc[:, ['Value', 'Overall', 'Aggression', 'Position']]
    )
f = f[f["Overall"] >= 80]
f = f[f["Overall"] < 85]
f['Aggression'] = f['Aggression'].astype(float)

sns.boxplot(x = "Overall", y = "Aggression", hue = 'Position', data = f)
f = (
    footballers.loc[:, ['Acceleration', 'Aggression', 'Agility', 'Balance', 'Ball control']]
        .applymap(lambda v: int(v) if str.isdecimal(v) else np.nan)
        .dropna()
).corr()

sns.heatmap(f, annot = True)
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



