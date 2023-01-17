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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



df=pd.read_csv(r"../input/cities_r2.csv")

print(df.shape)

df.drop(['state_code','dist_code','location'], axis=1, inplace=True)

print(df.dtypes)
df_good_sex_ratio = df.ix[df.sex_ratio>1000,['state_name','name_of_city', 'sex_ratio']]

df_1 = df_good_sex_ratio.groupby('state_name', as_index=False)[['sex_ratio']].count()

fig = plt.figure()

ax = fig.gca()

labels = df_1['state_name']

sizes = df_1['sex_ratio']

colors = ['red', 'green', 'blue','yellow','cyan','salmon','pink','yellowgreen','lightskyblue','lightcoral']

plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')

ax.set_title('states with % of cities out of 493 cities in dataset,having sex ratio>1000 ')

plt.show()
df_good_literacy = df.ix[df.effective_literacy_rate_total>90, ['state_name','name_of_city', 'effective_literacy_rate_total']]

df_2 = df_good_literacy.groupby('state_name')['name_of_city'].count()

df_2.plot.barh()

plt.xlabel('states with number of cities having literacy rate more than 90%')

plt.show()
peices = dict(list(df.groupby('state_name')))
df_jk = peices['JAMMU & KASHMIR']

df_jktop = df_jk.ix[df_jk.sex_ratio>900,['name_of_city', 'sex_ratio','effective_literacy_rate_total','population_total']]

print(df_jktop)
df_hp = peices['HIMACHAL PRADESH']

df_hptop = df_hp.ix[df_hp.sex_ratio>700, ['name_of_city','sex_ratio', 'effective_literacy_total','population_total']]

print(df_hptop)

#Himachal Pradesh has only city i.e. capitalcity with worst sex_ratio
df_pun = peices['PUNJAB']

df_puntop = df_pun.ix[df_pun.sex_ratio>900, ['name_of_city','sex_ratio','effective_literacy_rate_total','population_total']]

print(df_puntop)
df_br = peices['BIHAR']



df_brtop = df_br.ix[df_br.sex_ratio>900, ['name_of_city','sex_ratio','effective_literacy_rate_total','population_total']]

print(df_brtop)

sns.swarmplot(y='name_of_city', x='effective_literacy_rate_total', data=df_brtop)

plt.show()
df_up = peices['UTTAR PRADESH']

df_uptop = df_up.ix[df_up.sex_ratio>900,['name_of_city','sex_ratio','effective_literacy_rate_total','population_total']]

sns.swarmplot(y='name_of_city',x='effective_literacy_rate_total',data=df_uptop)

plt.show()
df_hr = peices['HARYANA']

df_hrtop = df_hr.ix[df_hr.sex_ratio>900,['name_of_city','sex_ratio','effective_literacy_rate_total','population_total']]

print(df_hrtop)

#haryana has only one city with good sex ratio
df_mh = peices['MAHARASHTRA']

df_mhtop = df_mh.ix[df_mh.sex_ratio>900,['name_of_city','sex_ratio','effective_literacy_rate_total','population_total']]

sns.barplot(y='name_of_city', x='effective_literacy_rate_total', data=df_mhtop)

plt.show()

# maharashtra has good number of cities with healthy sex_ratio & effective_literacy_rate
df_tn = peices['TAMIL NADU']

df_tntop = df_tn.ix[df_tn.sex_ratio>900,['name_of_city','sex_ratio','effective_literacy_rate_total','population_total']]

sns.barplot(y='name_of_city',x='effective_literacy_rate_total',data=df_tntop)

plt.show()

df_as = peices['ASSAM']

df_astop = df_as.ix[df_as.sex_ratio>900,['name_of_city','sex_ratio','effective_literacy_rate_total','population_total']]

print(df_astop)
df_kl = peices['KERALA']

df_kltop = df_kl.ix[df_kl.sex_ratio>700,['name_of_city','sex_ratio','effective_literacy_rate_total','population_total']]

print("Kerala stands out with good sex_ratio and effective literacy rate among major INDIAN state")

print(df_kltop)