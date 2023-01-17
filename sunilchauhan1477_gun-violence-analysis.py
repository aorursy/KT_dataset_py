import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt 
import seaborn as sns 
df = pd.read_csv("../input/gun-violence-data_01-2013_03-2018.csv")
df.shape
df.index.values
df.columns.values
df.dtypes
df['killed_or_not'] = np.where(df['n_killed'] > 0, 'YES', 'NOT')
df['killed_or_not'].value_counts()
df.isna().sum()
df.isnull().sum()
df['date']=pd.to_datetime(df['date'])
df['year']=df['date'].dt.year
gp_state = df.groupby(by=['state']).agg({'n_killed':'sum','n_injured':'sum','n_guns_involved':'sum'}).reset_index()
gp_state['n_killed'] =  (gp_state['n_killed'] - min(gp_state['n_killed']))/(max(gp_state['n_killed']) - min(gp_state['n_killed']))
gp_state['n_injured'] =  (gp_state['n_injured'] - min(gp_state['n_injured']))/(max(gp_state['n_injured']) - min(gp_state['n_killed']))
gp_state['n_guns_involved'] =  (gp_state['n_guns_involved'] - min(gp_state['n_guns_involved']))/(max(gp_state['n_guns_involved']) - min(gp_state['n_guns_involved']))
gp_state
g = sns.jointplot("n_injured",
              "n_killed",
              gp_state
              )

g.ax_joint.plot(                # Plot y versus x as lines and/or markers.
               np.linspace(0, 1),
               np.linspace(0, 1)
               )
varsforgrid = ['n_killed', 'n_injured','n_guns_involved']
g = sns.PairGrid(gp_state,
                 vars=varsforgrid,  # Variables in the grid
                 hue='state'       # Variable as per which to map plot aspects to different colors.
                 )
g = g.map_diag(plt.hist)                   
g.map_offdiag(plt.scatter)
g.add_legend();
g = sns.distplot(df['n_killed'], kde=True);
g.axvline(0, color="red", linestyle="--");
plt.xlim(0,5)
plt.ylim(0,1)
gp_year = df.groupby(by=['year']).agg({'n_killed':'sum','n_injured':'sum','n_guns_involved':'sum','killed_or_not':'count'}).reset_index()
gp_year.killed_or_not
gp_year['n_killed'] =  (gp_year['n_killed'] - min(gp_year['n_killed']))/(max(gp_year['n_killed']) - min(gp_year['n_killed']))
gp_year['n_injured'] =  (gp_year['n_injured'] - min(gp_year['n_injured']))/(max(gp_year['n_injured']) - min(gp_year['n_killed']))
gp_year['n_guns_involved'] =  (gp_year['n_guns_involved'] - min(gp_year['n_guns_involved']))/(max(gp_year['n_guns_involved']) - min(gp_year['n_guns_involved']))
varsforgrid = ['n_killed', 'n_injured','n_guns_involved']
g = sns.PairGrid(gp_year,
                 vars=varsforgrid,  # Variables in the grid
                 hue='year'       # Variable as per which to map plot aspects to different colors.
                 )
g = g.map_diag(plt.hist)                   
g.map_offdiag(plt.scatter)
g.add_legend();
sns.boxplot(x='year',y='killed_or_not',data = df)
sns.violinplot("year", "killed_or_not", data=df );     # x-axis has categorical variable