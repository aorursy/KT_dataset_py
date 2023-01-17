from matplotlib import pyplot as plt

import pandas as pd

import seaborn as sns

import math
#Table 33 - Aliens Apprehended

df_app = pd.read_csv('../input/immigration-apprehended/immigration_apprehended.csv')

print(df_app)

#Table 39 - Aliens Removed or Returned

df_dep = pd.read_csv('../input/immigration-deported/immigration_deported.csv')

print(df_dep)
##Reverse list so the year is in acsending order

year = df_app.Year[::-1].tolist()

#print(year)



##Reverse list so the apprehended number is with the correct year that was originally organized to be in ascending order

num_app = df_app.Number[::-1].tolist()

#print(num_app)



##Set x-ticks for chart

x_tick_year = []

#ceil function rounds up

num_ticks = math.ceil(len(year)/10)

#print(num_ticks)

##Answer: 10

for x in range(num_ticks):

    x_tick_year.append(year[x*10])

#print(x_tick_year)
f, ax = plt.subplots(figsize=(15, 10))

ax.plot(year, num_app, color='darkblue')

ax.axis([1925, 2018, 0, 2000000])

ax.set_xticks(x_tick_year)

plt.title('Undocumented Immigrants Apprehended Within Each Fiscal Year')

plt.xlabel('Year')

plt.ylabel('Number of Undocumented Immigrants Apprehended')



plt.show()
##Reverse list so the year is in acsending order

year2 = df_dep.Year[::-1].tolist()

#print(year2)



##Reverse list so the removals  is with the correct year that was originally organized to be in ascending order

removals = df_dep.Removals[::-1].tolist()

#print(removals)



##Reverse list so the returns is with the correct year that was originally organized to be in ascending order

returns = df_dep.Returns[::-1].tolist()

#print(returns)



##Set x-ticks for chart

x_tick_year2 = []

#ceil function rounds up

num_ticks2 = math.ceil(len(year2)/10)

#print(num_ticks)

##Answer: 10

for x in range(num_ticks2):

    x_tick_year2.append(year2[x*10])

#print(x_tick_year2)
fig2, ax2 = plt.subplots(figsize=(15, 10))

ax2.bar(year2, removals, label='Removals')

ax2.bar(year2, returns, bottom= removals, label='Returns')

ax2.axis([1892, 2019, 0, 2000000])

ax2.set_xticks(x_tick_year2)

ax2.legend()

plt.title('Undocumented Immigrants Deported Within Each Fiscal Year')

plt.xlabel('Year')

plt.ylabel('Number of Undocumented Immigrants Deported')



plt.show()
fig3, ax3 = plt.subplots(figsize=(15, 10))

ax3 = sns.kdeplot(num_app, shade=True, color='darkblue')

sns.set_style("darkgrid")

plt.title('Probability Density of the Number of Undocumented Immigrants Apprehended')

plt.xlabel('Number of Undocumented Immigrants Apprehended')

plt.ylabel('Density')



plt.show()
fig4, ax4 = plt.subplots(figsize=(15, 10))

ax4 = sns.kdeplot(year, num_app, shade=True, cbar=True, color='darkblue')

plt.title('Probability Density of the Number of Undocumented Immigrants Apprehended with the Fiscal Year')

plt.xlabel('Year')

plt.ylabel('Number of Undocumented Immigrants Apprehended')



plt.show()
fig5, ax5 = plt.subplots(figsize=(15, 10))

ax5 = sns.kdeplot(removals, shade=True, label='Removals')

ax5 = sns.kdeplot(returns, shade=True, label='Returns')

sns.set_style('darkgrid')

plt.legend()

plt.title('Probability Density of the Number of Undocumented Immigrants Deported')

plt.xlabel('Number of Undocumented Immigrants Deported')

plt.ylabel('Density')



plt.show()
fig6, ax6 = plt.subplots(figsize=(15, 10))

ax6 = sns.kdeplot(year2, removals, cbar=True, shade=True, label='Removals')

ax6 = sns.kdeplot(year2, returns, cbar=True, shade=True, label='Returns')

sns.set_style('darkgrid')

plt.legend()

plt.title('Probability Density of the Number of Undocumented Immigrants Deported with the Fiscal Year')

plt.xlabel('Year')

plt.ylabel('Number of Undocumented Immigrants Deported')



plt.show()