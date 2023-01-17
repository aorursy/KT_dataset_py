import numpy as np
import seaborn as sns
import pandas as pd
a = 5 + np.random.normal(0, 15, size=100)
b = 55 + np.random.normal(0, 15, size=100)

x = np.append(a,b)
y = x + np.random.normal(0, 10, size=200)

group = ['1']*100 
group.extend(['2']*100)
df = pd.DataFrame({'x':x, 'y':y, 'group' : group})
df.loc[df['group'] == '2', 'y'] = df.loc[df['group'] == '2', 'y'] - 150
sns.lmplot(x='x', y='y', data=df)
sns.lmplot(x='x', y='y', hue='group', data=df)
df = pd.DataFrame({'x':x, 'y':y, 'group' : group})
df.loc[df['group'] == '2', 'y'] = df.loc[df['group'] == '2', 'y'] - 50
sns.lmplot(x='x', y='y', data=df)
sns.lmplot(x='x', y='y', hue='group', data=df)