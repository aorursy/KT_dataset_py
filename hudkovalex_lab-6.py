import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import bokeh



crime = pd.read_csv('../input/lab6.csv', index_col='year')

crime.head(5)
ibr = crime["tfr"] #3

fig, ax = plt.subplots()

ax.plot(ibr)

ax.set(xlabel='Year', ylabel='Total fertility rate')



# Динамика индекса рождаемости показывает себя очень хорошо в первой половине, во второй же видим сильный спад.
ibr = crime["tfr"] #4

fig, ax = plt.subplots()

ax.plot(ibr, label = ['Total fertility rate per 1000 women'], color = 'orange',linestyle = 'dashed', linewidth = 5, marker = 'o', markersize = 4, markerfacecolor = "blue")

ax.set_xlabel('Year')

ax.set_ylabel('Total fertility rate')

ax.legend(loc = 'lower right')



# См. в #3
m = crime["mtheft"][4:] #5

f = crime["ftheft"][4:]



fig, ax = plt.subplots()

ax.plot(m, label = "Men thieves")

ax.plot(f, label = "Women thieves")



ax.set_xlabel("Year")

ax.set_ylabel("Quantity")

ax.set_title("Dynamics of the thieves")

ax.legend()



plt.show()



# Оценивая динамику мужскимх и женских краж можно смело сказать, что мужчины ворую больше, но с некоторыми спадами и подъемами,

# женщин же примерно стабильное количество, хотя в конце на графике виден небольшой подъем.
degavg = np.mean(crime["degrees"]) #6

deg = crime["degrees"]

par = crime["partic"]



fig, ax1 = plt.subplots()





ax1.scatter(par, deg, color = "red")



plt.grid()





plt.show()



# На графике видно, что все больше и больше женщин которые ищет высшее образование - работают, тенденция соблюдается на всем графике.
x = crime["mconvict"] #7

y = crime["fconvict"]





fig, ax = plt.subplots()

ax.bar(crime.index, x, label = "Men convicted", color = "yellow")

ax.bar(crime.index, y, label = "Women convicted", color = "green")



ax.set_xlabel("Year", color = "pink")

ax.set_ylabel("Quantity", color = "blue")

ax.set_title("Dynamics of the convinctions", color = "red")

ax.legend()



plt.show()



# Количество осужденных мужчин гораздо выше, нежели количество женщин, тенденции двух типов сохраняются,

# колебаний в женской категории почти не было, но в мужских были около 250.
y1 = np.array(crime["fconvict"][4:]) #8

y2 = np.array(crime["ftheft"][4:])



fig, (ax_c, ax_t) = plt.subplots(2, 1, figsize = (10, 5), sharey = True)





ax_c.bar(crime.index[4:], y1, label = "Female indictable-offense conviction rate per 100,000.", color = "green")

ax_c.set_xlabel("Years", color = "lightblue", size = 15)

ax_c.set_ylabel("Quantity", color = "lightblue", size = 15)



ax_t.bar(crime.index[4:], y2, label = "Female theft conviction rate per 100,000.", color = "red")

ax_t.set_xlabel("Years", color = "lightblue", size = 15)

ax_t.set_ylabel("Quantity", color = "lightblue", size = 15)



ax_c.legend(loc = 'upper right')

ax_t.legend(loc = 'upper right')





fig.suptitle("Dynamics of changes")

plt.show()



# Число осужденных женщин в 1940 году имело наивысшую точку, после чего осуждения упали, но с 1960 года мы наблюдаем рост.

# Число краж среди женщин было стабильно низкое, до 1960 года, после это число начало рости.

# Можно наблюдать общую тенденцию роста краж и осуждений на графиках с 1960 года.
born = np.array(crime["tfr"][4:]) #9

theft = np.array(crime["mtheft"][4:])

convict = np.array(crime["mconvict"][4:])

crm = theft + convict

#print(born)





fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (23, 5), sharex = True)





ax1.scatter(born, convict, label = 'Born + male convicts')

ax2.scatter(born, theft, label = 'Born + male thefts')

ax3.scatter(born, crm, label = 'Born + all male')



ax1.set_ylabel('born', size = 15)

ax1.set_xlabel('Male crimes', size = 15)

ax2.set_ylabel('born', size = 15)

ax2.set_xlabel('Male crimes', size = 15)

ax3.set_ylabel('born', size = 15)

ax3.set_xlabel('Male crimes', size = 15)



ax1.legend(loc = "lower left")

ax2.legend()

ax3.legend(loc = "lower left")





plt.show()



#В общем и целом тенденциия мужской преступность начала падать, можно сказать чем меньше рождаемость, тем больше преступлений.
edu = np.array(crime["degrees"][4:]) #10

unemp = np.array(crime["partic"][4:])

fcrm = np.array(crime["fconvict"][4:]) + np.array(crime["ftheft"][4:])

#print(fcrm)





fig, ax = plt.subplots(1, 2, sharex = True)



sns.lineplot(fcrm, edu, data=crime, ax = ax[0])

sns.lineplot(fcrm, unemp, data=crime, ax = ax[1])





plt.show()



# Повышение уровня образования повлияло лучше, из за меньших колебаний и более высокого уровня на графике, хоть он и виден очень не сильно.

# На счет бехработицы можно сказать что присутствуют сильные колебания на графике.

# Так же если посмотреть на вертикальную шкалу, мы можем увидеть сильные различия.