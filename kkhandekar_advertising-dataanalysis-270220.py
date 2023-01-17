# Importing Libraries



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

# Loading Dataset

data = pd.read_csv('../input/tvradionewspaperadvertising/Advertising.csv',header='infer')
data.shape
data.isna().sum()
data.head()
data.describe()
corr = data.corr(method='pearson')

fig = plt.figure(figsize=(8,8))

ax = fig.add_subplot(111)

cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)

fig.colorbar(cax)

ticks = np.arange(0,len(data.columns),1)

ax.set_xticks(ticks)

plt.xticks(rotation=90)

ax.set_yticks(ticks)

ax.set_xticklabels(data.columns)

ax.set_yticklabels(data.columns)

plt.show()
plt.style.use('seaborn-deep')

plt.figure(figsize=(10,10))

plt.grid(True)

a = data['TV']

b = data['Sales']

c = data['Radio']

d = data['Newspaper']

plt.hist([a,b,c,d], label=['TV Ads Dist','Sales Ads Dist','Radio Ads Dist','Newspaper Ads Dist'])

plt.legend(loc='upper right')

plt.title('Adverts Distribution')

plt.show()
sns.set(style="darkgrid")

fig = plt.figure()

x1 = data['TV']

x2 = data['Sales']

fig = sns.jointplot(x1, x2, kind="kde", height=7, space=0)

fig.fig.set_size_inches(10,10)

plt.show()
sns.set(style="darkgrid")

fig = plt.figure(figsize=(15,10))

sns.lineplot(data=data, palette="tab10", linewidth=2.5)

plt.show()
