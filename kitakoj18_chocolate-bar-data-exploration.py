import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
chocdf = pd.read_csv('../input/flavors_of_cacao.csv')

chocdf.head()
chocdf.rename(columns = lambda x: x.replace('\n', ' '), inplace = True)
chocdf.columns
chocdf.info()
chocdf['Bean Type'].value_counts()
chocdf.groupby(['Broad Bean Origin', 'Bean Type']).size()
chocdf['Cocoa Percent'] = chocdf['Cocoa Percent'].apply(lambda x: float(x.strip('%')))
sns.distplot(chocdf['Cocoa Percent'])
sns.countplot(chocdf['Rating'])
fig = plt.figure()
fig.suptitle('Relationship Between Cocoa Percentage and Chocolate Ratings', fontweight = 'bold')
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Cocoa Percentage', fontweight = 'bold')
ax.set_ylabel('Rating', fontweight = 'bold')
plt.scatter(chocdf['Cocoa Percent'], chocdf['Rating'])
countrydf = chocdf.groupby('Company Location').filter(lambda x: len(x) > 25)
df2 = pd.DataFrame({col:vals['Rating'] for col,vals in countrydf.groupby('Company Location')})
meds = df2.median()
meds.sort_values(ascending=False, inplace=True)

fig2 = plt.figure(figsize = (25, 10))
ax2 = fig2.add_subplot(111)
sns.boxplot(x='Company Location',y='Rating', data=countrydf, order=meds.index, ax = ax2)
plt.xticks(rotation = 90)
ax2.set_xlabel('Company location', fontdict = {'weight': 'bold', 'size': 16})
ax2.set_ylabel('Rating', fontdict = {'weight': 'bold', 'size': 16})
for label in ax2.get_xticklabels():
    label.set_size(16)
    label.set_weight("bold")
for label in ax2.get_yticklabels():
    label.set_size(16)
    label.set_weight("bold")
