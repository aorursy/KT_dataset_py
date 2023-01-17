import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
# lineplot? scatterplot? barplot? boxplot? countplot?

x = np.linspace(-6, 6, 100)

y = 1 / (1 + np.exp(-x))

plt.plot(x, y)

plt.xlabel('x')

plt.ylabel('y')

plt.title(r'$y=\frac{1}{1+e^{-x}}$')

plt.show()
# цвет, тип линии, толщину линии, тип и размер маркера

x = np.linspace(-6, 6, 100)

y = 1 / (1 + np.exp(-x))

plt.plot(x, y, color='DarkSlateBlue', linestyle='--', linewidth=3, marker='v', markersize=10.5)

plt.xlabel('x')

plt.ylabel('y')

plt.title(r'$y=\frac{1}{1+e^{-x}}$')

plt.show()
x = np.linspace(-2.5, 2.5, 100)

y1 = 1 / (1 + np.exp(-x))

y2 = x / np.sqrt(1 + x**2)

plt.plot(x, y1, color='blue', label=r'$1/(1+e^{-x})$')

plt.plot(x, y2, color='green', linestyle='dotted', label=r'$x/\sqrt{1+x^2}$')



plt.xlabel('x')

plt.ylabel('y')

plt.legend(loc=4)

plt.show()
# по горизонтали

fig, ax = plt.subplots(1, 2, sharey=True, figsize=(10, 5))

ax[0].plot(x, y1, color='blue')

ax[1].plot(x, y2, color='green', linestyle='dotted')

ax[0].set_xlabel('x')

ax[0].set_ylabel('y')

ax[1].set_xlabel('x')

ax[1].set_ylabel('y')

ax[0].set_title(r'$y=1/(1+e^{-x})$')

ax[1].set_title(r'$y=x/\sqrt{1+x^2}$')

plt.show()
# по вертикали

fig, ax = plt.subplots(2, 1, sharex=True)

ax[0].plot(x, y1, color='blue')

ax[1].plot(x, y2, color='green', linestyle='dotted')

ax[0].set_xlabel('x')

ax[0].set_ylabel('y')

ax[1].set_xlabel('x')

ax[1].set_ylabel('y')

ax[0].set_title(r'$y=1/(1+e^{-x})$')

ax[1].set_title(r'$y=x/\sqrt{1+x^2}$')

plt.show()
x1 = np.linspace(-2*np.pi, 0, 100)

x2 = np.linspace(0, 5, 100)

y1 = np.sin(-2*x1)

y2 = x2**2 - x2

plt.plot(x1, y1, color='black')

plt.plot(x2, y2, color='black')

plt.xlabel('x')

plt.ylabel('y')

plt.show()
import os

os.listdir('../input')
df = pd.read_csv('../input/cardiovascular-disease-dataset/cardio_train.csv', sep=';')

df.head()
df.info()
df['age_years'] = np.floor(df['age'] / 365.25)

# lineplot? scatterplot? barplot? boxplot? countplot? hist? distplot?

sns.distplot(df['age_years'])

plt.show()
sns.boxplot(df['age_years'])

plt.show()
sns.boxplot(df['weight'])

plt.show()
sns.boxplot(y='weight', data=df, x='cardio')

plt.show()
sns.scatterplot(x='height', y='weight', data=df, alpha=0.5)

plt.show()
sns.scatterplot(x='height', y='weight', data=df, alpha=0.5, hue='cardio')

plt.show()
sns.relplot(x='height', y='weight', data=df, col='cardio', kind='scatter')

plt.show()
mean_weight_0 = df[ df['cardio']==0 ]['weight'].mean()

mean_weight_1 = df[ df['cardio']==1 ]['weight'].mean()

plt.bar(height=[mean_weight_0, mean_weight_1], x=[0, 1])

plt.xticks(ticks=[0, 1], labels=['cardio=0', 'cardio=1'])

plt.ylabel('Average weight')

plt.show()
sns.countplot(x="cholesterol", data=df, hue="cardio")

plt.show()