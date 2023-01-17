from gensim.models import KeyedVectors as KV

from gensim.models import doc2vec

import warnings

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import sys

import os

import pymorphy2

from collections import namedtuple

import numpy as np

from sklearn import metrics

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import ExtraTreesClassifier



from nltk import tokenize



from sklearn.cluster import KMeans
train_file = open('./train.csv', 'r')

data = []

labels = []

test = []

ans = []

for line in train_file:

    id, days = line.split(', ')

    days = days.split(' ')

    days = [int(day) for day in days]

    days.sort()

    res = [0] * 7

    last = days[len(days) - 1]

    days.reverse()

    count = 0

    for day in days:

        if day >= 1099 - 365 * 1.5:

            res[(day - 1) % 7] += 1

            count += 1

    max = 0

    argmax = 0

    for i in range(7):

        if res[i] > max:

            max = res[i]

            argmax = i

    ans.append(argmax + 1)
out_f = open('./solution.csv', 'w')

out_f.write("id,nextvisit\n")

i = 1

for a in ans:

    out_f.write(str(i) + ", " + str(a) + "\n")

    i += 1