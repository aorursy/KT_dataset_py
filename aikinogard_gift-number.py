# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
cnt = {}

with open("../input/gifts.csv", "r") as f:

    for i, line in enumerate(f):

        if i:

            cat = line.split("_")[0]

            cnt[cat] = cnt.get(cat, 0) + 1

print(cnt)
cnt_sort = sorted(list(cnt.items()), key=lambda x: x[1])

for k, v in cnt_sort:

    print(k, v)
def weight(toy, n):

    if toy == "horse":

        return np.maximum(0, np.random.normal(5, 2, n))

    elif toy == "ball":

        return np.maximum(0, 1 + np.random.normal(1, 0.3, n))

    elif toy == "bike":

        return np.maximum(0, np.random.normal(20, 10, n))

    elif toy == "train":

        return np.maximum(0, np.random.normal(10, 5, n))

    elif toy == "coal":

        return 47 * np.random.beta(0.5, 0.5, n)

    elif toy == "book":

        return np.random.chisquare(2, n)

    elif toy == "doll":

        return np.random.gamma(5, 1, n)

    elif toy == "blocks":

        return np.random.triangular(5, 10, 20, n)

    elif toy == "gloves":

        ws = np.random.rand(n)

        msk = ws < 0.3

        ws[msk] = 3.0 + np.random.rand(np.sum(msk))

        ws[~msk] = np.random.rand(n - np.sum(msk))

        return ws

    else:

        raise ValueError("%s is invalid toy" % toy)

        

TOYS = ["gloves", "ball", "doll", "horse", "book",

        "blocks", "train", "coal", "bike"]
mean_weight = {}

uplimit = {}

for toy in TOYS:

    m = np.mean(weight(toy, 100000000))

    mean_weight[toy] = m

    uplimit[toy] = int(50 / m)
mean_weight
uplimit