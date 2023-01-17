# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')
# Загружаем файл с описаниями колонок, выводим первые 20 строк

desc = pd.read_excel('/kaggle/input/vodafone-subset/column_descriptions.xlsx')

desc.head(20)
# Загружаем файл с данными, выбираем первые 115 колонок, очищаем от пропущенных значений

vodafone = pd.read_csv('/kaggle/input/vodafone-subset/vodafone-subset-1.csv')

vodafone = vodafone.iloc[:, :115].dropna()

print(vodafone.shape)
vodafone.head()
# Пример: срок использования мобильного телефона в месяцах (количественный признак)

sns.distplot(vodafone['how_long_same_model'])

plt.show()
sns.boxplot(vodafone['how_long_same_model'])

plt.show()
sns.violinplot(vodafone['how_long_same_model'])

plt.show()
# Пример: уровень дохода (порядковый признак)

sns.countplot(vodafone['SCORING'], order=['VERY LOW', 'LOW', 'MEDIUM', 'HIGH_MEDIUM', 'HIGH'])

plt.show()
# Пример: instagram_volume vs. intagram_count (два количественных признака)

sns.scatterplot(x="intagram_count", y="instagram_volume", data=vodafone) # intagram_count --- опечатка в исходных данных!

plt.show()
# Пример: instagram_volume vs. intagram_count по разным возрастным группам

sns.relplot(x="intagram_count", y="instagram_volume", data=vodafone, kind="scatter", col="target", col_wrap=3)

plt.show()
# Пример: car vs. gas_stations_sms (наличие машины и количество сообщений от заправок, категориальный и количественный)

sns.pointplot(x="car", y="gas_stations_sms", data=vodafone, estimator=np.mean)

plt.show()
# Пример: car vs. gas_stations_sms (наличие машины и количество сообщений от заправок, категориальный и количественный)

sns.barplot(x="car", y="gas_stations_sms", data=vodafone)

plt.show()
# Матрица корреляций + heatmap для признаков со 2 по 11 (количественные)

subset = vodafone.iloc[:, 2:12]

sns.heatmap(subset.corr())

plt.show()
# Пример: использование приложения facebook по разным возрастным группам (количественный и категориальный)

sns.pointplot(x="target", y="fb_count", data=vodafone, estimator=np.mean)

plt.show()
# Пример: уровень дохода и наличие авто (категориальный и категориальный)

sns.countplot(x='SCORING', data=vodafone, 

              hue="car", palette={0: 'red', 1:'green'},

              order=['VERY LOW', 'LOW', 'MEDIUM', 'HIGH_MEDIUM', 'HIGH'])

plt.show()
# Пример: доход vs. наличие машины vs. пол абонента (три категориальных признака)

sns.catplot(x='SCORING', data=vodafone, hue="car", palette={0: 'red', 1:'green'},

            order=['VERY LOW', 'LOW', 'MEDIUM', 'HIGH_MEDIUM', 'HIGH'],

            kind="count",

            col="gender")

plt.show()
# Пример: LAT_WORK vs. LON_WORK vs. SCORING (два количественных признака + категориальный)

fig, ax = plt.subplots(figsize=(10, 10))

order = ['VERY LOW', 'LOW', 'MEDIUM', 'HIGH_MEDIUM', 'HIGH']

sns.scatterplot(x="LAT_WORK", y="LON_WORK", data=vodafone[ vodafone.SCORING != '0' ],

                hue="SCORING", hue_order=order[::-1], palette="cubehelix", 

                size="SCORING", size_order=order[::-1])

ax.set_xlim(44, 53)

ax.set_ylim(20, 41)

plt.show()