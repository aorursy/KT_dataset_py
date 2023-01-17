# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("/kaggle/input/gtd/globalterrorismdb_0718dist.csv", encoding="ISO-8859-1")

df.head()
# No. of terrorist attacks per year

plt.hist(df['iyear'], edgecolor = 'black', linewidth = 0.2)

plt.xlabel("Year")

plt.ylabel("No. of attacks")
#Countries with highest number of terrorist attacks

print(df['country_txt'].value_counts().head())

df['country_txt'].value_counts().head().plot(kind = 'bar')

plt.title("Terrorist attacks for top 5 most affected counntries")

plt.xlabel("Country")

plt.ylabel("Value_Counts")

plt.xticks(rotation = 80)
#Regions with highest number of terrorist attacks

(df['region_txt'].value_counts()).plot(kind = 'bar')

plt.title("Terrorist attacks per region")

plt.xlabel("Region")

plt.ylabel("Value_Counts")

plt.xticks(rotation = 80)
print([i for i in df.columns])
#Maximum number of people killed in a single attack

df.sort_values('nkill',ascending = False)[['nkill','country_txt','iyear','imonth']].head()
#Attack methods employed by the terrorists

(df['attacktype1_txt'].value_counts()).plot(kind = 'bar')

plt.title("Attack methods employed by the terrorists")

plt.xlabel("Attack methods")

plt.ylabel("Value_Counts")

plt.xticks(rotation = 80)
c = df.targtype1_txt.astype('category')

d = dict(enumerate(c.cat.categories))

(df['targtype1'].value_counts()).plot(kind = 'bar')

plt.title("Favoirite Target types of Terrorist")

plt.xlabel("Target Type")

plt.ylabel("Value_Counts")

s ='\n'.join(['%s: %s' % (key, value) for (key, value) in d.items()])

plt.text(25,0, s, fontsize=14)
#Finding Peretrators with highest number of kills

df2 = df.groupby(['gname']).sum()['nkill'].sort_values(ascending = False).head(10)

d = dict(df2)

df2.plot(kind = 'bar')

plt.title("Groups with highest number of kills")
#Terrorist Activities of top ten known terrorist organizations

def day(x):

    if not x%100:

        return False

    return True

l = list(d.keys())

l = l[1:]

df3 = df[df['gname'].isin(l)][['eventid','nkill','gname']]

df3.dropna(inplace = True)

df3['Date'] = [int(i/100000000) for i in list(df3['eventid'])]

df3 = df3[df3['Date'].map(day)]

df3['Date']  = pd.to_datetime(df3['Date'],errors='coerce', format = '%Y')

df3.drop(columns = ['eventid'], inplace = True)

g = sns.lineplot(x = "Date", y = 'nkill',hue = 'gname', data = df3)

g.legend(loc=(1.3,0))
