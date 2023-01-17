# Garima and Ghaida Notebook kaggle decal Fall 2016

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd 

# data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

import os

import matplotlib.pyplot as plt

from scipy.stats import gaussian_kde

import time

import seaborn as sns 

%matplotlib inline
df_train = pd.read_csv("../input/train.csv")

df_test = pd.read_csv("../input/test.csv")

df_train.head()



df_train_sample = df_train.sample(n=150000)

df_test_sample = df_test.sample(n=150000)
counts1, bins1 = np.histogram(df_train["accuracy"], bins=50)

binsc1 = bins1[:-1] + np.diff(bins1)/2.



counts2, bins2 = np.histogram(df_test["accuracy"], bins=50)

binsc2 = bins2[:-1] + np.diff(bins2)/2.



plt.figure(0, figsize=(14,4))



plt.subplot(121)

plt.bar(binsc1, counts1/(counts1.sum()*1.0), width=np.diff(bins1)[0])

plt.grid(True)

plt.xlabel("Accuracy")

plt.ylabel("Fraction")

plt.title("Train")



plt.subplot(122)

plt.bar(binsc2, counts2/(counts2.sum()*1.0), width=np.diff(bins2)[0])

plt.grid(True)

plt.xlabel("Accuracy")

plt.ylabel("Fraction")

plt.title("Test")



plt.show()
import datetime

from heapq import nlargest

from operator import itemgetter

import os

import time

import math

from collections import defaultdict





def prep_xy(x, y):

    range = 800

    ix = math.floor(range*x/10)

    if ix < 0:

        ix = 0

    if ix >= range:

        ix = range-1



    iy = math.floor(range*y/10)

    if iy < 0:

        iy = 0

    if iy >= range:

        iy = range-1



    return ix, iy





def run_solution():

    print('Preparing data...')

    f = open("../input/train.csv", "r")

    f.readline()

    total = 0



    grid = defaultdict(lambda: defaultdict(int))

    grid_sorted = dict()



    # Calc counts

    while 1:

        line = f.readline().strip()

        total += 1



        if line == '':

            break



        arr = line.split(",")

        row_id = arr[0]

        x = float(arr[1])

        y = float(arr[2])

        accuracy = arr[3]

        time = arr[4]

        place_id = arr[5]



        ix, iy = prep_xy(x, y)



        grid[(ix, iy)][place_id] += 1



    f.close()



    # Sort array

    for el in grid:

        grid_sorted[el] = nlargest(3, sorted(grid[el].items()), key=itemgetter(1))



    print('Generate submission...')

    sub_file = os.path.join('submission_' + str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")) + '.csv')

    out = open(sub_file, "w")

    f = open("../input/test.csv", "r")

    f.readline()

    total = 0

    out.write("row_id,place_id\n")



    while 1:

        line = f.readline().strip()

        total += 1



        if line == '':

            break



        arr = line.split(",")

        row_id = arr[0]

        x = float(arr[1])

        y = float(arr[2])

        accuracy = arr[3]

        time = arr[4]



        out.write(str(row_id) + ',')

        filled = []



        ix, iy = prep_xy(x, y)



        s1 = (ix, iy)

        if s1 in grid_sorted:

            topitems = grid_sorted[s1]

            for i in range(len(topitems)):

                if topitems[i][0] in filled:

                    continue

                if len(filled) == 3:

                    break

                out.write(' ' + topitems[i][0])

                filled.append(topitems[i][0])



        out.write("\n")



    out.close()

    f.close()





start_time = time.time()

run_solution()

print("Elapsed time overall: %s seconds" % (time.time() - start_time))
