%matplotlib inline

# Loading external libraries



import seaborn as sns

import pandas as pd

import matplotlib.pyplot as plt



from collections import Counter



plt.rc('figure', figsize=(10, 5))
df = pd.read_csv('../input/top-1m.csv', header=None, names=['Rank', 'Domain'])



# Expand the sitename into TLD and domain strings

df[['Name', 'TLD']] = df['Domain'].str.split('.', 1, expand=True)



# Note: this is just for demonstration purposes, use Regex to get these values

# e.g.  df['Domain'].str.extract('((?:\w{2-3}\.)?\w+)')



df.head()
p = ''.join(df['Name'])

p = Counter(p)



print('Occurences of each alphanum character in top1m')

sns.barplot(list(p.keys()), list(p.values()), order=sorted(p.keys()))



plt.show()
only_vowels = {i: p[i] for i in 'aeiouy'}

print('Histogram of Vowels in Top 1-Million websites')



sns.barplot(list(only_vowels.keys()), list(only_vowels.values()))

plt.show()
labels = [l.upper() for l in only_vowels.keys()]

sizes = list(only_vowels.values())

plt.pie(sizes, labels=labels, autopct='%1.1f%%', textprops={'fontsize': 'x-large'})

plt.axis('equal')

plt.tight_layout()

plt.show()