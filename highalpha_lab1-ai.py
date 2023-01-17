import pandas as pd

import matplotlib.pyplot as plt





large = 22; med = 16; small = 12

params = {'axes.titlesize': large,

          'legend.fontsize': med,

          'figure.figsize': (16, 10),

          'axes.labelsize': med,

          'axes.titlesize': med,

          'xtick.labelsize': med,

          'ytick.labelsize': med,

          'figure.titlesize': large}

plt.rcParams.update(params)

plt.style.use('seaborn-whitegrid')

file = pd.read_csv("/home/highalpha/Загрузки/1 Univer/cwurData.csv")



def ave (list):

    return round( float(sum(list)/len(list)) , 2)
copy_file = file

nan_data_interpolate = copy_file.interpolate(method='cubic')

nan_data_interpolate.head(20)

gdd = nan_data_interpolate.fillna(method='pad')

gd = gdd.sort_values(by=['world_rank'], ascending = True)
num = gd._get_numeric_data()

nl = list (num)

mini = []

maxi = []

mid = []

for name in nl:

    val = num[[name]].values

    mini.append(val.min())

    maxi.append(val.max())

    mid.append(ave(val))

print (mini)

print (maxi)

print (mid)

num.head(20)
plt.figure(figsize=(16, 10), dpi= 80, facecolor='w', edgecolor='k')
drs = gd

drk = gd



drs.query('country=="USA"')

us = drs[['world_rank','national_rank']].values

          

drk.query('country=="United Kingdom"')

uk = drk[['world_rank','national_rank']].values
plt.subplot(221)

plt.plot(us[:, 0], us[:, 1]) #сравнить вузы для каждой страны. ищем группу стран примерно с одинаковым кол-вом вузов

plt.subplot (222)

plt.plot(uk[:, 0], uk[:, 1])
nd2 = gd

nd3 = nd2.sort_values(by=['quality_of_education'], ascending = False)

data2 = nd3[['institution','quality_of_education']].values #сортировка по рейтингу



plt.barh(data2[:12, 0], (data2[:12, 1]-300))
data3 = gd[['quality_of_education','quality_of_faculty']].values

plt.scatter(data3[:, 0], data3[:, 1]) #сделать точками
nd2 = gd

nd3 = nd2.sort_values(by=['patents'], ascending = False)

data4 = nd3[['institution','patents']].values

plt.barh(data4[:10, 0], data4[:10, 1])
nd2 = gd

nd3 = nd2.sort_values(by=['publications'], ascending = False)

data5 = nd3[['institution','publications']].values

plt.barh(data5[:10, 0], data5[:10, 1]-1000) #по исксу рейтинг