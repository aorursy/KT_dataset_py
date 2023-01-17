import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
# Loading in the data
dc_data = pd.read_csv('../input/fivethirtyeight-comic-characters-dataset/dc-wikia-data.csv')
marvel_data= pd.read_csv('../input/fivethirtyeight-comic-characters-dataset/marvel-wikia-data.csv')
# Printing the first five rows of the data
dc_data.head()
marvel_data.head()
dc_data['ALIGN'].value_counts()
marvel_data['ALIGN'].value_counts()
plt.subplots(1,2, figsize= (10,10))
plt.subplot(121)
sns.countplot(data=marvel_data, x='ALIGN', hue='SEX')
plt.legend(loc='upper right')
plt.subplot(122)

sns.countplot(data=dc_data, x='ALIGN', hue='SEX')
plt.legend(loc='upper right')

# Hero evolution by the decade
dc_data.groupby('YEAR')['SEX'].value_counts().unstack().plot()
marvel_data.groupby('Year')['SEX'].value_counts().unstack().plot()
plt.subplots(1,2, figsize= (10,10))
sns.set_style('darkgrid')
plt.subplot(121)

sns.countplot(data=marvel_data, x='ALIGN', hue='ID')
plt.legend(loc='upper right')
plt.title('Marvel character identities')

plt.subplot(122)

sns.countplot(data=dc_data, x='ALIGN', hue='ID')
plt.legend(loc='upper right')
plt.title('DC character identities')
# Dead People distribution
plt.subplots(1,2, figsize= (10,10))
sns.set_style('white')
plt.subplot(121)

sns.countplot(data=marvel_data[marvel_data['ALIVE']=='Deceased Characters'], x='ALIGN', hue='SEX')
plt.legend(loc='upper right')
plt.title('Marvel Death tolls')

plt.subplot(122)

sns.countplot(data=dc_data[dc_data['ALIVE']=='Deceased Characters'], x='ALIGN', hue='SEX')
plt.legend(loc='upper right')
plt.title('DC Death tolls')
max_dc=dc_data.sort_values('APPEARANCES', ascending=False)
max_dc[['name','APPEARANCES','ID','ALIGN','ALIVE']].set_index('APPEARANCES').head(10)

max_marv=marvel_data.sort_values('APPEARANCES', ascending=False)

max_marv[['name','APPEARANCES','ID','ALIGN','ALIVE']].set_index('APPEARANCES').head(10)
marvel_data[marvel_data['ALIGN']=='Bad Characters'].sort_values('APPEARANCES',ascending=False)
plt.figure(figsize=(16,6))
sns.swarmplot(dc_data['ID'], dc_data['YEAR'], hue=dc_data['ALIGN'])
sns.swarmplot(marvel_data['ID'], marvel_data['Year'], hue=marvel_data['ALIGN'])
