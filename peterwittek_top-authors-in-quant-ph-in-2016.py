# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
records = pd.read_csv("../input/scirate_quant-ph.csv", dtype={"id": str})
all_authors = []

for authors in records[records["year"]==2016]["authors"]:

    authors = authors.split(";")

    for author in authors:

        all_authors.append(author)

frequencies = [[x, all_authors.count(x)] for x in set(all_authors)]

frequencies = sorted(frequencies, key=lambda x: -x[1])

top_authors = [f for f in frequencies if f[1] > 5]

top_authors = list(map(list, zip(*top_authors)))
width = 3

scale = 5

fig, ax = plt.subplots()

plt.bar(range(0, scale*len(top_authors[0]), scale), top_authors[1], width)

ax.set_xticks([scale*i+width/2 for i in range(scale*len(top_authors[0]))])

ax.set_xticklabels([author for author in top_authors[0]],  rotation='vertical')

ax.xaxis.grid(False)

plt.ylabel("No. of papers in quant-ph in 2016\nwith at least 10 Scites")

plt.ylim((0, 14))

plt.xlim((0, scale*len(top_authors[0])))

ax.set_aspect(4)

plt.show()
total_scites = [[x, sum(records[(records["year"] == 2016)&

                                 (records["authors"].str.contains(x))]["scites"])]

                        for x in set(all_authors)]

total_scites = sorted(total_scites, key=lambda x: -x[1])

total_scites = list(map(list, zip(*total_scites)))



width = 3

scale = 5

max_authors = 30

fig, ax = plt.subplots()

plt.bar(range(0, scale*max_authors, scale), total_scites[1][:max_authors], width)

ax.set_xticks([scale*i+width/2 for i in range(scale*max_authors)])

ax.set_xticklabels([author for author in total_scites[0][:max_authors]],  rotation='vertical')

ax.xaxis.grid(False)

plt.ylabel("Total no. of Scites")

# plt.ylim((0, 14))

plt.xlim((0, scale*max_authors))

ax.set_aspect(0.2)

plt.show()