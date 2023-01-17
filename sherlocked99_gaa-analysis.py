# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.express as px

import plotly.graph_objects as go

import matplotlib.pyplot as plt
master = []

for i in range(1,5):

    p1 = []

    for i in range(1,6):

        p1.append(np.random.randint(0,2))

    master.append(p1)
def check_diff(master):

    dict  = {}

    for i in range(len(master)):

        temp = ""

        for j in range(len(master[i])):

            temp = temp + str(master[i][j])

        #print(temp)

        if temp not in dict.keys():

            dict[temp] = 1

        else:

            dict[temp] = dict[temp] + 1

    return dict
count = 4

points = []

points.append(count)

mapoflists = check_diff(master)

for i in range(1,2000):

    x = np.random.randint(1,len(master))

    y = -1

    while True:

        y = np.random.randint(1,len(master))

        if y!=x:

            break

    c1 = []

    c2 = []

    k = np.random.randint(0,5)

    for j in range(0,k):

        c1.append(master[x-1][j])

        c2.append(master[y-1][j])

    for j in range(k,5):

        c1.append(master[y-1][j])

        c2.append(master[x-1][j])

    master.append(c1)

    master.append(c2)

    #print(len(master))

    mapoflists = check_diff(master)

    #print(count)

    points.append(len(mapoflists.keys()))
fig = go.Figure(data=go.Scatter(x=np.arange(1,2000), y=points))

fig.show()