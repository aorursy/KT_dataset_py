import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import random

import time

from IPython.display import display

from sklearn.metrics import cohen_kappa_score, confusion_matrix

from sklearn.model_selection import train_test_split
#From https://www.kaggle.com/cpmpml/ultra-fast-qwk-calc-method

def qwk3(a1, a2, max_rat=3):

    assert(len(a1) == len(a2))

    a1 = np.asarray(a1, dtype=int)

    a2 = np.asarray(a2, dtype=int)



    hist1 = np.zeros((max_rat + 1, ))

    hist2 = np.zeros((max_rat + 1, ))



    o = 0

    for k in range(a1.shape[0]):

        i, j = a1[k], a2[k]

        hist1[i] += 1

        hist2[j] += 1

        o +=  (i - j) * (i - j)



    e = 0

    for i in range(max_rat + 1):

        for j in range(max_rat + 1):

            e += hist1[i] * hist2[j] * (i - j) * (i - j)



    e = e / a1.shape[0]



    return 1 - o / e
data = pd.read_csv("../input/data-science-bowl-2019/train_labels.csv")
"""

    1. Start from perfect agreement, 

    2. Add mistakes one by one, watching how the public and private score change

    3. Do it until public and private scores are too low to be interesting

    4. Repeat the whole process 1000 times

"""

def test(public_size, private_size, min_score):

    size = public_size + private_size

    n_iter = 1000

    n_iter2 = 200



    stats = []

    for i in range(n_iter):

        data['submission'] = data['accuracy_group']



        while(True):

            for m in range(n_iter2):

                n = random.randrange(0, size)

                v = data.loc[n, 'submission']

                if v == 0:

                    data.loc[n,'submission'] = 1

                elif v == 3:

                    data.loc[n,'submission'] = 2

                else:

                    data.loc[n,'submission'] = v - 1 + 2 * random.randint(0, 1)

            public_set = data.iloc[private_size : size]

            public_kappa = qwk3(public_set['accuracy_group'], public_set['submission'])

            private_set = data.iloc[:private_size]

            private_kappa = qwk3(private_set['accuracy_group'], private_set['submission'])

            if public_kappa < min_score and private_kappa < min_score:

                break

            d = {

                'public_score': public_kappa,

                'private_score': private_kappa,

                'public_score_bin': int(public_kappa * 1000) / 1000,

                'private_score_bin': int(private_kappa * 1000) / 1000,

            }

            stats.append(d)



    return pd.DataFrame(stats)



stats = test(1000, round(86/14*1000), 0.300)   
def plot(df, cat, val, min_score, max_score, title):

    groups = df.where((df[cat] >= min_score) & (df[cat] <= max_score)).groupby([cat], as_index = False).agg({val: [

        ('5%', (lambda x: x.quantile(.05))),

        ('avg', 'mean'),

        ('95%', (lambda x: x.quantile(.95))),

        ('standard deviation', 'std')

    ]})

    

    mean = groups[val]['avg']

    error_low = mean - groups[val]['5%']

    error_high = groups[val]['95%'] - mean

    

    fig = plt.figure()

    fig.suptitle(title)

    plt.xlabel(cat)

    plt.ylabel(val)

    plt.errorbar(groups[cat], mean, yerr=[error_low, error_high], markersize=5, markeredgecolor='red', markerfacecolor='red', linestyle='')

    plt.grid()

    plt.show()



    return groups

groups1 = plot(stats, 'public_score_bin', 'private_score', 0.400, 0.600, 'Private score 95% confidence intervals assuming given public score')
pd.set_option('display.max_rows', 1000)

display(groups1)