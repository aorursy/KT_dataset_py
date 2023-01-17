# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/cereal.csv')

df.head(5)
df.describe()
import seaborn as sns

import matplotlib.pyplot as plt
sns.distplot(df['calories'], bins=20, rug=True,hist_kws=dict(edgecolor="k", linewidth=2))

plt.title("'Cereal Calories Distribution'") 
sns.distplot(df['sodium'], bins=30, color = 'green',hist_kws=dict(edgecolor="k", linewidth=1))

plt.plot(edgecolor="black")

plt.title("'Cereal sodium Distribution'") 
from scipy.stats import ttest_ind



%matplotlib inline



from matplotlib import pylab, pyplot

from scipy.stats import probplot
# df.loc[df['type']=='C','sodium']



probplot(df['calories'],dist='norm',plot=pyplot)


cc= df.loc[df['type']=='C','calories']

hc= df.loc[df['type']=='H','calories']
ttest_ind(hc, cc,equal_var=False)
df.loc[df['type']=='C','calories'].mean()
df.loc[df['type']=='H','calories'].mean()
# plotting

print(df.loc[df['type']=='H','calories'].count())

plt.hist([hc,cc], color =['red','grey'],alpha=0.5,label=['Hot Cereal sodium','Cold Cereal sodium'],edgecolor="black")

# plt.hist(cc , color='orange', label='Cold Cereal Calories',edgecolor="black")



plt.xlabel('Amount of Sodium')

plt.ylabel('Count')



plt.legend(loc="upper left")

plt.show()


# plotting

print(df.loc[df['type']=='C','sodium'].count())



plt.hist([df.loc[df['type']=='H','sodium'], df.loc[df['type']=='C','sodium'] ], color=['orange','yellow'], alpha=0.8,label=['Hot Cereal sodium','Cold Cereal sodium'],edgecolor='black')

# plt.hist(df.loc[df['type']=='H','sodium'],edgecolor="black", color ='green',alpha=0.5,label='Hot Cereal sodium')

# plt.hold(True)

# plt.hist(df.loc[df['type']=='C','sodium'] ,edgecolor="black", color='orange', label='Cold Cereal sodium')



plt.xlabel('Amount of Sodium')

plt.ylabel('Count')

plt.legend(loc="upper left")

plt.show()
plt.hist(df['sodium'],edgecolor='black',color='purple',alpha=0.5,linewidth=1)

# sns.distplot(df['sodium'], bins=25,hist_kws=dict(edgecolor="green", linewidth=2))



plt.xlabel('Amount of Sodium')

plt.ylabel('Count')

plt.legend(loc="upper left")

plt.show()
# Manufacturer of cereal

plt.title('Manufacturer of cereal')

# dic = {'A' : 'American Home Food Products', 'G' : 'General Mills', 'K' : 'Kelloggs','N' : 'Nabisco','P' : 'Post','Q' : 'Quaker Oats','R' : 'Ralston Purina'}

sns.countplot(x=df['mfr'],palette="Blues_d", data=df)



plt.show()
g = sns.barplot(x="rating", y="name", data=df.sort_values('rating',ascending=False)[:15])

g.set(xlabel='Rating', ylabel='Name',title = "The best rated cereals")
from scipy.stats import chisquare

chisquare(df['mfr'].value_counts())

# identical values case
chisquare(df['type'].value_counts())