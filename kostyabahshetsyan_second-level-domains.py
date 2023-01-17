import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline
df = pd.read_csv('../input/3rd_lev_domains.csv')
df.info()
df['second'] = df['ascension.gov.ac'].map(lambda x: x.split('.')[1])

df['first'] = df['ascension.gov.ac'].map(lambda x: x.split('.')[2])
df.head()
first_count = df.groupby('first').apply(lambda x: len(x)/len(df)*100)

first_count.sort_values(inplace=True, ascending=False)

first_count
plt.figure(figsize=(17,8))

first_count.plot(kind='bar')
domain_count = df.groupby(['second', 'first']).apply(lambda x: len(x)/len(df)*100)

domain_count.sort_values(inplace=True, ascending=False)

domain_count
plt.figure(figsize=(17,8))

domain_count[:20].plot(kind='bar')