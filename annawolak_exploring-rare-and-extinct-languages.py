import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_csv('../input/data.csv')
df.head()
usa = df[df['Country codes alpha 3'].str.contains('USA') == True]
usa.head()
pol = df[df['Country codes alpha 3'].str.contains('POL') == True]
pol.head()
pol['Countries'].ix[5]
len(pol['Countries'].ix[5])
clean = df[['Name in English', 'Number of speakers']]
clean.plot(kind='line', figsize=(9,4))
active = clean[clean['Number of speakers'] > 0]
active.plot(figsize=(9,4))
active[active['Number of speakers'] > 100000].plot(kind='bar', figsize=(9,4))