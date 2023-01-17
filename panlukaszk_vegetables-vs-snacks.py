# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df_original = pd.read_csv('../input/8b. AUSNUT 2011-13 AHS Food Nutrient Database.csv',
                         decimal=',')
df_original.head()
veggies = ['tomato','onion','broccoli']
snacks = ['snack']

def apply_category(row):
    if any(row['Food Name'].lower().startswith(veggie) for veggie in veggies):
        return 'veggie'
    elif any(snack in row['Food Name'].lower() for snack in snacks):
        return 'snack'

df_original['category'] = df_original.apply(apply_category, axis=1)
df_original.groupby('category').count()
df_original[df_original['category'] == 'snack'].head()
df_original.columns
df = df_original.rename(index=str, columns={
    'Food Name': 'name',
    'Energy, with dietary fibre (kJ)': 'energy',
    'Vitamin C (mg)': 'vitamin_C',
    'Total fat (g)': 'fat'
})
df = df[df['category'].notna()]
df = df[['name','energy','vitamin_C','category','fat']]
df.sample(5)
import seaborn as sns
sns.scatterplot(data=df, x='energy', y='vitamin_C', hue='category')
# plt.scatter(df['energy'], df['vitamin_C'], col)
from sklearn.cluster import KMeans
X = df[['energy','vitamin_C']]
kmeans = KMeans(n_clusters=2).fit(X)
pred = kmeans.predict(df[['energy','vitamin_C']])
df['kmeans'] = pred
sns.scatterplot(data=df, x='energy', y='vitamin_C', hue='kmeans')
import matplotlib.pyplot as plt

K = range(1,15)
distortions = []
for k in K:
    kmeans_for_k = KMeans(n_clusters=k).fit(X)
    distortion = kmeans_for_k.inertia_
    distortions.append(distortion)
    
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum of Squared Errors')
plt.title('The Elbow Method showing the optimal k')
plt.show()
kmeans = KMeans(n_clusters=6).fit(X)
pred = kmeans.predict(df[['energy','vitamin_C']])
df['kmeansX'] = pred
sns.scatterplot(data=df, x='energy', y='vitamin_C', hue='kmeansX')