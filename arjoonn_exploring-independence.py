import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

%pylab inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/independence-days.csv')

df.head()
df.info()
df['year_of_freedom'] = df['Year celebrated'].str.split(' ').str[0].astype(int)

plt.figure(figsize=(14, 6))

sns.distplot(df.year_of_freedom)

plt.title('Which years saw the most liberation?')
d = [(k, v) for k, v in dict(df.groupby('year_of_freedom')['Country'].count()).items()]

d.sort(key=lambda x: x[0])

years = [i[0] for i in d]

countries_free = [i[1] for i in d]

plt.plot(years, np.cumsum(countries_free) / sum(countries_free))

plt.xlabel('Year')

plt.ylabel('% of countries which are free')

plt.title('What fraction of the world was free at a given time?')
df['month'] = df['Date of holiday'].str.split(' ').str[1].str[:3]

df['month'].unique()
# Refer to https://en.wikipedia.org/wiki/Iyar

df.loc[df['Date of holiday'].str.contains('Iya')]
df.loc[df['month'] == 'Iya', 'month'] = 'May'
plt.figure(figsize=(14, 6))

order='Jan,Feb,Mar,Apr,May,Jun,Jul,Aug,Sep,Oct,Nov,Dec'.split(',')

sns.countplot(df.month, order=order)

plt.title('Which months usually give birth to freedom?')