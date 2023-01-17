from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

print(os.listdir('../input'))
#let's initialize the data and see a few info about it
df=pd.read_csv('../input/Africa_1997-2018_Dec15.csv')
df.info()
df.head()
sns.set()
sns.set_context("poster", font_scale=0.5, rc={"lines.linewidth": 15})
ax = sns.catplot(x="YEAR", data=df, aspect=1.5,kind="count" )
ax.set_xticklabels(rotation=60)
sns.set()
sns.set_context("poster", font_scale=0.5, rc={"lines.linewidth": 15})
ax = sns.catplot(x="REGION", data=df, aspect=1.5,kind="count")
ax.set_xticklabels(rotation=90)
sns.set()
sns.set_context("poster", font_scale=0.5, rc={"lines.linewidth": 15})
ax = sns.catplot(x="COUNTRY",data=df,aspect=1.5,kind="count")
ax.set_xticklabels(rotation=90)
#let's confirm which one of DRC and Nigeria has the least conflicts. 
df_nigeria=df.loc[df['COUNTRY'] == 'Nigeria']
print('Nigeria conflict count:',df_nigeria.count())
df_congo=df.loc[df['COUNTRY'] == 'Democratic Republic of Congo']
print('Democratic Republic of Congo:',df_congo.count())
