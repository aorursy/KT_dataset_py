import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline
pd.set_option('display.max_columns', 500)
df = pd.read_csv('../input/data.csv', low_memory=False)
df.head()
df.tail()
df['sex'].value_counts()
df['sex'].value_counts().plot(kind='bar', title='Sex Distribution')
df['race / ethnicity'].value_counts()
df['race / ethnicity'].value_counts().plot(kind='bar', title='Race / Ethnicity Distribution')
df['interview decision'].value_counts()
df['interview decision'].value_counts().plot(kind='bar', title='Interview Decisions')
by_race = df['interview decision'].groupby(df['race / ethnicity'])
by_race.describe()
df['housing or interview facility'].value_counts()