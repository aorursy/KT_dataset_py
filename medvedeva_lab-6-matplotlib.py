# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import matplotlib as mpl

import matplotlib.pyplot as plt
df = np.genfromtxt('../input/Hartnagel.csv', delimiter=',', missing_values=None, skip_header=1)

print(df[:10, :])

print(df.shape)
fig, ax = plt.subplots(figsize=(22, 6))



x = df[:, 0]

y = df[:, 1]

plt.plot(x, y)



plt.xlabel('Year')

plt.ylabel('Total fertility rate')

plt.title(r'Динамика изменения индекса рождаемости')



plt.show()
fig, ax = plt.subplots(figsize=(22, 6))



x = df[:, 0]

y = df[:, 1]

plt.plot(x, y, label= 'Total fertility rate per 1000 women', linestyle='dashed', linewidth=5, color='orange', marker='o', markersize=40, markerfacecolor='blue')



plt.xlabel('Year')

plt.ylabel('Total fertility rate')

plt.title(r'Динамика изменения индекса рождаемости')



plt.legend(loc=4)



plt.show()
fig, ax = plt.subplots(figsize=(22, 6))







plt.plot(df[:, 0], df[:, -3], label='Женщины', linestyle='solid', linewidth=2, color='b', marker='o', markersize=12)

plt.plot(df[:, 0], df[:, -1], label='Мужчины', linestyle='dotted', linewidth=2, color='c', marker='s', markersize=6, markerfacecolor='white')







plt.xlabel('Year')

plt.ylabel('Quantity')

plt.title(r'Динамика числа мужчин и женщин, осуждённых за воровство с 1935 по 1968 год.')



plt.legend(loc=0)



plt.show()
fig, ax = plt.subplots(figsize=(22, 6))

plt.scatter(df[:, 2], df[:, 3])

plt.xlabel('Partic')

plt.ylabel('Degrees')



plt.grid()

fig, ax = plt.subplots(figsize=(22, 6))



width = 0.4



plt.bar(df[:, 0] - width/2, df[:, 6], width, label='Мужчины', color='green')

plt.bar(df[:, 0] + width/2, df[:, 4], width, label='Женщины', color='red')



plt.legend(loc=0);

fig, ax = plt.subplots(2, 1, figsize=(22, 6), sharex=True)



ax[0].plot(df[:, 0], df[:, 4], label='Fconvict', linestyle='solid', linewidth=2, color='b', marker='o', markersize=8)

ax[1].plot(df[:, 0], df[:, 5], label='Ftheft', linestyle='solid', linewidth=2, color='r', marker='s', markersize=8);



ax[0].legend(loc=4)

ax[1].legend(loc=4);



# ax[0].set_title(r'Динамика изменения показателей fconvict в течение периода 1935–1968.')

# ax[1].set_title(r'Динамика изменения показателей ftheft в течение периода 1935–1968.')
fig, ax = plt.subplots(figsize=(22, 6))

plt.scatter(df[:, 1], (df[:, -1] + df[:, -2]))



plt.xlabel('tfr')

plt.ylabel('mconvict и mtheft');

            
fig, ax = plt.subplots(figsize=(6, 12))



plt.subplot(2, 1, 1)

plt.scatter(df[:, 2], (df[:, 4] + df[:, 5]), label='уменьшение безработицы', marker='o')

plt.title(r'Влияние уменьшения безработицы на преступность у женщин')



plt.subplot(2, 1, 2)

plt.scatter(df[:, 3], (df[:, 4] + df[:, 5]), label='повышение уровня образования', marker='s')

plt.title(r'Влияние повышения уровня образования на преступность у женщин');

fig, ax = plt.subplots(figsize=(22, 6))



plt.plot(df[:, 0], (df[:, 4] + df[:, 5]), label='преступность у женщин', linestyle='solid', linewidth=2, color='b', marker='o', markersize=6)

plt.plot(df[:, 0], df[:, 2], label='уменьшение безработицы', linestyle='dotted', linewidth=2, color='c', marker='s', markersize=6)

plt.plot(df[:, 0], df[:, 3], label='повышение уровня образования', linestyle='dotted', linewidth=2, color='r', marker='v', markersize=6)



plt.legend(loc=0)



plt.show()