import numpy as np

import pandas as pd 

from matplotlib import pyplot as plt

import seaborn as sns



df = pd.read_csv('../input/fed_campsites.csv', index_col=0)

df.head()
states = df['AddressStateCode'].unique()

print (states)
df2 = pd.DataFrame([])



#calculate the percent of tent-only, standard electric, and standard nonelectric campsites per state

for state in states: 

    pc_tent = df[(df.AddressStateCode == state) & (df.CampsiteType == 'TENT ONLY NONELECTRIC')].count()/df[df.AddressStateCode == state].count()

    pc_sne = df[(df.AddressStateCode == state) & (df.CampsiteType == 'STANDARD NONELECTRIC')].count()/df[df.AddressStateCode == state].count()

    pc_se = df[(df.AddressStateCode == state) & (df.CampsiteType == 'STANDARD ELECTRIC')].count()/df[df.AddressStateCode == state].count()

    temp = pd.DataFrame({'state': state,

                         'frac_tent_only': pc_tent,

                         'frac_standard_nonelec': pc_sne,

                         'frac_standard_elec': pc_se}) 

    df2 = df2.append(temp, ignore_index = True)

    

df2 = df2.drop_duplicates(['state'])

df2 = df2.sort(columns = 'state')

df2 = df2.reset_index()

df2 = df2.drop('index', axis=1)

df2.head()
%matplotlib inline

sns.set(style="white", context="talk")



ax = sns.barplot('state', 'frac_tent_only', data = df2)

ax.set_xticklabels(df2['state'],rotation=45)

plt.show();
agg = df.groupby(['AddressStateCode']).count()

print (agg.loc['NJ'],  agg.loc['NY'], agg.loc['CA'])
df[df.AddressStateCode == 'NJ']
df[df.AddressStateCode == 'NY']
g = sns.lmplot(x = 'frac_standard_nonelec', y = 'frac_tent_only', data = df2, \

               legend=False, hue = 'state', fit_reg=False, size = 5, aspect = 2)

sns.regplot(x = 'frac_standard_nonelec', y = 'frac_tent_only', data = df2, \

            scatter=False, ax=g.axes[0, 0])





box = g.ax.get_position() # get position of figure

g.ax.set_position([box.x0, box.y0, box.width*0.6, box.height]) # resize position



# Put a legend to the right side

g.ax.legend(loc='right', bbox_to_anchor=(1.65, .5), ncol=3)

plt.show();