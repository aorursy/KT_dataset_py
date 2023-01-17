import pandas as pd
import numpy as np
import matplotlib as plotpy
%matplotlib inline

df = pd.read_csv("../input/gun-violence-data_01-2013_03-2018.csv")
df.head()
df.tail()
df.columns
df.describe(include=["O"])
df['state'].value_counts()
df.isnull().sum().sort_values(ascending = False)
ax = df['state'].value_counts().plot.bar(rot=90, figsize=(10,5))
ax1 = df['city_or_county'].value_counts().head(50).plot.bar(rot=90, figsize=(10,5))
df['date'].dtype
df['date'] = pd.to_datetime(df['date'])
df['year'], df['month'], df['day'], df['weekday'] = df['date'].apply(lambda x: x.year), df['date'].apply(lambda x: x.month), df['date'].apply(lambda x: x.day), df['date'].apply(lambda x: x.weekday())
df
ax = df['year'].value_counts().plot.bar(rot=90, figsize=(10,5))


ax2 = df['day'].value_counts().plot.bar(rot=90, figsize=(10,5))

ax3 = df['month'].value_counts().plot.bar(rot=90, figsize=(10,5))

ax4 = df['weekday'].value_counts().plot.bar(rot=0, figsize=(10,5))

df_pivot = df.pivot_table(index = 'state', values='n_killed', aggfunc= np.sum).reset_index().sort_values(by='n_killed', ascending=False)
df_pivot


#df_pivot = df_pivot.to_frame('n_killed').reset_index()


#df_pivot_New = df_pivot.sort_values(by='n_killed', ascending=False)
#df_pivot_New
df_pivot[:10].plot.bar(x='state', y='n_killed')
temperory = df[df['state'] == 'California']["city_or_county"].value_counts()[:20]
temperory
labels = list(temperory.index)
values = list(temperory.values)
print(labels)
print(values)
temperory.plot.bar(x='labels', y='values')
df_pivot_1 = df.pivot_table(index = 'state', values='n_injured', aggfunc= np.sum).reset_index().sort_values(by='n_injured', ascending=False).head(10)

#df_pivot = df_pivot.to_frame('n_injured').reset_index()


#df_pivot_NewVer_1= df_pivot.sort_values(by='n_injured', ascending=False).head(10)
df_pivot_1.plot.bar(x='state', y='n_injured')
df_pivot_NewVer_2 = df.pivot_table(index = 'year', values=['n_killed', 'n_injured'], aggfunc= np.sum).reset_index()

df_pivot_NewVer_2
ax = df_pivot_NewVer_2[['year', 'n_injured', 'n_killed']].set_index('year').plot.bar(rot=90, figsize=(10,5))
df_pivot_NewVer_3 = df.pivot_table(index = 'month', values=['n_killed', 'n_injured'], aggfunc= np.sum)
df_pivot_NewVer_3
ax_new = df_pivot_NewVer_3[['n_injured', 'n_killed']].plot.bar(rot= 0, figsize=(10,5))
ax5 = df['location_description'].value_counts().head(20).plot.barh(rot=0, figsize=(10,5))


