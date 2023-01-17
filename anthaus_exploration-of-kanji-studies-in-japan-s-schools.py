# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df_k = pd.read_csv('/kaggle/input/japanese-jy-kanji/joyo_kanji.csv', delimiter=',')

df_k.dataframeName = 'joyo_kanji.csv'

nRow, nCol = df_k.shape

print(f'There are {nRow} rows and {nCol} columns')
grade_s = df_k[df_k["grade"] == 'S']  # Kanji studied in secondary school

grade_e = df_k[df_k["grade"] != 'S']  # Kanji studied in elementary school

print("Part of kanji studied in elementary school: "+str(100*len(grade_e)/(len(df_k)))+" %")

print("Part of kanji studied in secondary school: "+str(100*len(grade_s)/(len(df_k)))+" %")
labels = sorted(set(df_k["grade"]))

sizes = []

for l in labels:

    sizes.append(len(df_k[df_k["grade"] == l]))



fig1, ax1 = plt.subplots()

plt.title("Repartition of the number of studied kanjis by year")

ax1.pie(sizes, labels=labels)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.



plt.show()
print("Average number of strokes in secondary school kanjis: "+str(grade_s['strokes'].mean()))

print("Average number of strokes in elementary school kanjis: "+str(grade_e['strokes'].mean()))
avg_strokes = []

for l in labels:

    avg_strokes.append(df_k[df_k['grade'] == l]['strokes'].mean())

fig2, ax2 = plt.subplots()

plt.title("Average number of strokes by year of study")

plt.ylabel("avg. nb. strokes")

plt.xlabel("Year of study")

ax2.plot(labels, avg_strokes)



strokes = []

for l in labels:

    strokes.append(df_k[df_k['grade'] == l]['strokes'])

figb, axb = plt.subplots()

axb.boxplot(strokes, labels)
rads = set(df_k['radical'])

rads_use = {}

for r in rads:

    rads_use[r] = len(df_k[df_k['radical'] == r])

top3 = sorted(rads_use.items(), key=lambda t: t[1], reverse=True)[:3]

print(top3)
nb_mizu = []

nb_hito = []

nb_te = []

for l in labels:

    m = 0

    h = 0

    t = 0

    for d in df_k[df_k['grade'] == l]['radical']:

        if d == '水':

            m += 1

        elif d == '人':

            h += 1

        elif d == '手':

            t += 1

    nb_mizu.append(m)

    nb_hito.append(h)

    nb_te.append(t)



fig, axs = plt.subplots(1, 3)

axs[0].pie(nb_mizu, labels=labels)

axs[0].set_title('Mizu')

axs[1].pie(nb_hito, labels=labels)

axs[1].set_title('Hito')

axs[2].pie(nb_te, labels=labels)

axs[2].set_title('Te')