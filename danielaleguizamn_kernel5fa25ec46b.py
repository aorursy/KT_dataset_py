import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

base = pd.read_csv('../input/suicide-rates-overview-1985-to-2016/master.csv')

base=pd.DataFrame(base)

base.head()


base1=base.drop(['HDI for year'], axis=1)

base1.head()
base1['suicides_no'].describe()
sns.heatmap(base1.corr(),cmap='coolwarm',annot=True)

plt.show()
corr2 = sns.PairGrid(base1)

corr2.map(plt.scatter)

plt.show()
plt.figure(figsize=(15,5))

graf1 = sns.barplot(x = 'sex', y = 'suicides_no', hue = 'age',data = base1,  palette='gnuplot2_r')

plt.xlabel=('Género')

graf1.set(ylabel="número de suicidios",xlabel="Género")
plt.figure(figsize=(15,5))

graf2 = sns.barplot(x = 'generation', y = 'suicides_no',data = base1, palette="RdYlBu_r")

plt.title('Número de suicidios por Generación')

graf2.set(ylabel="número de suicidios",xlabel="Generación")
plt.figure(figsize=(15,5))

graf3 = sns.lineplot(x = 'year', y = 'suicides/100k pop',data = base1)

plt.title('Suicidios/100k pop por año')

graf3.set(ylabel='Suicidios/100k',xlabel='Años')
sns.lmplot("suicides_no", "gdp_per_capita ($)", data=base1)

plt.show()
sns.kdeplot(base1['suicides_no'])

plt.show()
plt.figure(figsize=(10,20))

graf7 = sns.barplot(x = 'suicides/100k pop', y = 'country',data = base1,  palette='Accent_r')

plt.xlabel=('Género')

graf7.set(ylabel="País",xlabel='Suicidios por 100.000 pop')


