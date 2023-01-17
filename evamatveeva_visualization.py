import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
data = pd.read_csv('/kaggle/input/cardataset/data.csv')
data.shape
data.head()
data['highway l/km'] = round(235.214583 / data['highway MPG'], 2)

data['city l/km'] = round(235.214583 / data['city mpg'], 2)

data.drop(columns=['highway MPG', 'city mpg'], inplace=True)

data.head()
data[['Year', 'Engine Cylinders', 'Number of Doors']] = data[['Year', 'Engine Cylinders', 'Number of Doors']].astype('category')
data[['Make', 'Model', 'Engine Fuel Type', 'Transmission Type', 'Driven_Wheels', 'Market Category', 'Vehicle Size', 'Vehicle Style', 'Number of Doors', 'Engine Cylinders', 'Year']].describe()
sns.distplot(data['Engine HP'].dropna(), bins=73).set_title('Распредедление мощности автомобилей в л. с. ')
sns.distplot(data['MSRP'])
sns.distplot(data[data['MSRP']<150000]['MSRP']).set_title('Распредедление рекомендованной цены производителя')
sns.distplot(data[data['MSRP']<40000]['MSRP']).set_title('Распредедление рекомендованной цены производителя')
data['MSRP'].value_counts()
data[data['MSRP'] == 2000]['Year'].value_counts()
sns.distplot(data['Popularity'], bins=10).set_title('Распредедление популярности')
sns.distplot(data['city l/km'], label='В городе')

sns.distplot(data['highway l/km'], label='На трассе')

plt.legend()

plt.suptitle('Распределение расхода топлива на 100 км в городе и на трассе')
sns.set(style='whitegrid')

qualitative_colors = sns.cubehelix_palette(8)
sns.countplot(data['Engine Cylinders'], color=qualitative_colors[2])

plt.suptitle('Количество цилиндров')
sns.countplot(data['Number of Doors'], color=qualitative_colors[1])

plt.suptitle('Количество дверей')
fig, ax = plt.subplots(figsize=(10, 5))

sns.countplot(data['Transmission Type'], color=qualitative_colors[0], ax=ax)

plt.suptitle('Трансмиссия')
sns.set()

sns.set(font_scale=1.4)

fig, ax = plt.subplots(figsize=(38, 8))

sns.countplot(data['Vehicle Style'], color=qualitative_colors[3], ax=ax)

plt.suptitle('Типы')

sns.set()
sns.countplot(data['Vehicle Size'], color=qualitative_colors[4])

plt.suptitle('Размеры')
splot = sns.countplot(data['Driven_Wheels'], color=qualitative_colors[5])

plt.suptitle('Ведущие колёса')
data['Driven_Wheels'] = data['Driven_Wheels'].apply(lambda x: 'all wheel drive' if x == 'four wheel drive' else x).astype('category')
sns.countplot(data['Driven_Wheels'], color=qualitative_colors[5])

plt.suptitle('Ведущие колёса')
data['Engine Fuel Type'].value_counts()
data['Make'].value_counts()
sns.set()
sns.swarmplot(data['Engine Cylinders'], data['Engine HP'])
sns.barplot(data['Engine Cylinders'], data['Engine HP'])
len(data[(data['Engine Cylinders'] == 0) & (data['Engine Fuel Type'] != 'electric')])
sns.scatterplot(data['Engine HP'], data['MSRP'])

plt.suptitle('Зависимость MSRP от мощности')
data[data['MSRP'] > 1000000]
sns.scatterplot(data[data['MSRP'] < 900000]['Engine HP'], data[data['MSRP'] < 900000]['MSRP'])

plt.suptitle('Зависимость MSRP от мощности')
sns.barplot(data['Vehicle Size'], data['MSRP'])

plt.suptitle('Зависимость MSRP от размера')
sns.set()

a = data.groupby(['Make'])['MSRP'].mean().sort_values()

Maker_cheap = list(a[:3].index)

Maker_expensive = list(a[-3:].index)

sns.set(font_scale=1.5)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))

sns.swarmplot(data[data['Make'].isin(Maker_cheap)]['Make'], data[data['Make'].isin(Maker_cheap)]['MSRP'], ax=ax1)

sns.swarmplot(data[data['Make'].isin(Maker_expensive)]['Make'], data[data['Make'].isin(Maker_expensive)]['MSRP'], ax=ax2)

ax1.set_title('Зависимость MSRP от производителя\n(3 самых дешёвых в среднем)')

ax2.set_title('Зависимость MSRP от производителя\n(3 самых дорогих в среднем)')

sns.set()
sns.scatterplot(data['Engine HP'], data['city l/km'])

plt.suptitle('Зависимость расхода в городе от мощности')
sns.scatterplot(data['Engine HP'], data['highway l/km'])

plt.suptitle('Зависимость расхода на трассе от мощности')
sns.scatterplot(data['Engine HP'], data['city l/km'], label='В городе')

sns.scatterplot(data['Engine HP'], data['highway l/km'], label='На трассе')

plt.legend()

plt.suptitle('Зависимость расхода от мощности')