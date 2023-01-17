import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/forest-fires-in-brazil/amazon.csv', encoding='ISO-8859-1')

df.head()
df.shape
df.describe()
df.isnull().values.any()
plt.figure(figsize=(15, 5))

sns.lineplot(x=df['year'], y=df['number'], color='red', estimator='sum', ci=None)

plt.title('Forest fire in brazil between 1999-2017')

plt.show()
plt.figure(figsize=(15, 5))

sns.lineplot(x=df['state'], y=df['number'], estimator='sum')

plt.title('Forest fire in a state between 1999-2017')

plt.xticks(rotation=45)

plt.show()
df.groupby('state')['number'].sum().sort_values(ascending=False).head()
df_state = df[df['state'].isin(['Mato Grosso', 'Paraiba', 'Sao Paulo', 'Rio', 'Bahia'])]



plt.figure(figsize=(15, 5))

sns.lineplot(x=df_state['year'], y=df_state['number'], color='red', estimator='sum', ci=None, hue=df_state['state'])

plt.title('Forest fire in brazil between 1999-2017')

plt.show()