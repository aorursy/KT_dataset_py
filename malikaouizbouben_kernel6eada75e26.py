# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from scipy import misc
import seaborn as sns
#sns.set()
sns.set(rc={'figure.figsize':(15,15)})
#plt.rcParams['figure.figsize'] = (15, 10)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk("kernel/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv("/kaggle/input/suicide-rates-overview-1985-to-2016/master.csv", index_col = "country")
df
 
country_to_drop = ["Mongolia", "Macau", "Cabo Verde", "Dominica", "Bosnia and Herzegovina", "San Marino", "Saint Kitts and Nevis"]
df = df.drop(country_to_drop, axis=0)
df
 
df = df[df["year"] != 2016]
df
# Any results you write to the current directory are saved as output.
df[['year','suicides/100k pop']]
#x = df['year']
#y = df['suicides/100k pop']

#fig, ax = plt.subplots()
#ax.plot(x, y)

#ax.set(xlabel='year', ylabel='suicides/100k pop',
#       title='Nombre de suicide dans le monde par année')
#ax.grid()


#plt.show()
# le nombre de morts par an
sns.set_style()
sns.set_context('paper') 
g = sns.relplot(x="year", y="suicides/100k pop",dashes=False, markers = True ,kind="line", data=df,height = 10 )
g.fig.autofmt_xdate()
sns.set_style()
sns.set_context('paper') 
ax = sns.barplot(x="year", y="suicides/100k pop", data=df)
# le nombre de mort par an réparti par sex
sns.set_style()
sns.set_context('paper') 
sns.relplot(x="year", y="suicides/100k pop", hue="sex", kind="line", data=df,height = 10);

sns.set_style()
sns.set_context('paper') 
ax = sns.barplot(y="suicides/100k pop", x="sex", data=df)
# le nombre de mort par an réparti par groupe d'âge
sns.set_style()
sns.set_context('paper') 
sns.relplot(x="year", y="suicides/100k pop", hue="age", kind="line", data=df,height = 15);

sns.set_style()
sns.set_context('talk') 
ax = sns.barplot(y="suicides/100k pop", x="age", order = ['75+ years','55-74 years','35-54 years','25-34 years','15-24 years','5-14 years'],data=df)