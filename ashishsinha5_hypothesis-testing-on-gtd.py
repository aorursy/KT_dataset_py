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
print(df['country_txt'].value_counts().head())

df['country_txt'].value_counts().head().plot(kind = 'bar')

plt.title("Terrorist attacks for top 5 most affected counntries")

plt.xlabel("Country")

plt.ylabel("Value_Counts")

plt.xticks(rotation = 80)
dict(df['country_txt'].value_counts())
df.set_index(df['country_txt'])
df2 = df[(df['country_txt'] == 'Cameroon') | (df['country_txt'] == 'Honduras')]

df2.head()
d1 = dict(df2[df2['country_txt'] == 'Honduras'].groupby("iyear").mean()['nkill'].dropna())

d2 = dict(df2[df2['country_txt'] == 'Cameroon'].groupby("iyear").mean()['nkill'].dropna())
d1
d2
#Calculating Average number kills per year

Y1 = sum(d1.values())/len(d1)

Y2 = sum(d2.values())/len(d2)
#Calculating sample variance for the given data

s1_sqr = 0

for i in d1.values():

    s1_sqr += (i - Y1)**2

s1_sqr/=(len(d1) - 1)



s2_sqr = 0

for i in d2.values():

    s2_sqr +=(i - Y2)**2

s2_sqr/=(len(d2)-1)

s2_sqr
#Calculating test Statistic

T = (Y1 - Y2)/(s1_sqr/len(d1) + s2_sqr/len(d2))**0.5
# level of significance alpha = 0.05

alpha = 0.05

degree_of_freedom = len(d1) + len(d2) - 2
from scipy import stats
t = stats.t.ppf(1-alpha,degree_of_freedom)
if abs(T) > t:

    print("Null Hypothesis is regected")

else:

    print("fail to regect Null Hypothesis ")