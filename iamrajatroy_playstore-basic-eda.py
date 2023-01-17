import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline

sns.set_style('darkgrid')
apps = pd.read_csv('../input/googleplaystore.csv')
apps.info()
apps.head()
apps['App'].nunique()
plt.figure(figsize=(15,10))
category = apps.groupby('Category')['App'].count()
top_10 = category.sort_values(ascending=False).head(10)
top_10.plot(kind = 'bar')
plt.title('Top 10 Apps by category')
plt.show()
apps.head(2)
apps[apps['Rating'] == apps['Rating'].max()]
apps[pd.isnull(apps['Android Ver'])]
apps.drop(index=10472, inplace=True)
apps['Reviews'] = apps['Reviews'].apply(lambda x : pd.to_numeric(x))
apps[apps['Rating'] == apps['Rating'].max()][apps['Reviews']>=100]
plt.figure(figsize=(15,12))
sns.scatterplot(x='Rating', y='Reviews', data=apps, hue='Category', palette='Set1')
plt.title('Review vs Rating distribution')
plt.show()
#apps[apps['Rating'] == apps['Rating'].max()]['App']
apps[(apps['Rating'] == apps['Rating'].max()) & (apps['Reviews']>=100)]['App'].count()
apps[apps['Category']=='GAME'][(apps['Rating'] == apps['Rating'].max())][['App','Reviews']]
apps[apps['Category']=='GAME'][apps['Rating'] == apps['Rating'].max()][['App']].count()
apps[apps['Category']=='GAME'][apps['Rating'] == apps['Rating'].max()][apps['Type']=='Paid']['App']
plt.figure(figsize=(12,10))
sns.countplot(apps['Content Rating'], palette='rainbow')
plt.title('Apps by content rating')
plt.show()
plt.figure(figsize=(20,10))
sns.countplot(apps[apps['Installs'] == apps['Installs'].max()]['Category'], palette='rainbow')
plt.title('Apps category with most installs')
plt.show()


