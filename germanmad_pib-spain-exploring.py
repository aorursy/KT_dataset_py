%matplotlib inline

import pandas as pd

import seaborn as sns
df = pd.read_csv('../input/GDP_capita_spanish_regions.csv')
df.head()
df.plot()
df = df.transpose()
df = df.rename(columns={0:'anda',1:'arag',2:'astu',3:'bale',4:'cana',5:'cant',6:'casl',7:'casm',8:'cata',9:'vale',10:'extr',11:'gali',12:'madr',13:'murc',14:'nava',15:'eusk',16:'rioj',17:'ceut',18:'meli',19:'SPAIN'})
df.shape
df = df[1:]
df.head()
year = 2010

df['SPAIN'][year-2000] 
df['eusk'][year-2000]
df['anda'][year-2000]
df.mean().sort_values()
df.max().sort_values()
df.min().sort_values()
dforder = df.median().sort_values()

dforder
sorted_df = df[['extr','anda','meli','casm','murc','ceut','cana','gali','vale','astu','cant','casl','SPAIN','bale','rioj','arag','cata','nava','eusk','madr']]
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 1.5})
s = sns.pointplot(data=sorted_df)

for item in s.get_xticklabels():

    item.set_rotation(60)
g = sns.stripplot(data=sorted_df)

for item in g.get_xticklabels():

    item.set_rotation(60)
ax = sns.violinplot(sorted_df)

for item in ax.get_xticklabels():

    item.set_rotation(60)