#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""

Created on Mon Sep 21 2020

@author: _drTom_ (Thomas Plocher)

"""



# This Python function generates a Pareto-Chart and returns some data characteristics.

# It can be used as utility fuction and as tutorial for applying Python and Matplotlib-Pyplot



import numpy as np # linear algebra

import matplotlib.pyplot as plt

%matplotlib inline



def paretoChart(dList, nameList=[], figSize=(10,6), barCol="C0", percCol="C1", percMark="D", 

                percLinestyle='--', truncate=False, treshold=80):

    '''

    Display Pareto-Chart and return bin-Indices of most frquented bins.

    

    Parameters

    ----------

    dList : list with bin values; e.g. (int)-values.

    nameList: list of x-axis labels to display instead of index number

    figSize : (width, height) of chart, optional. The default is (10,6).

    barCol : histogram bar color, optional. The default is "C0".

    percCol : percentage line color, optional. The default is "C1".

    percMark : percentage marker type, optional. The default is "D".

    percLinestyle: Default is dashed '--'

    truncate : boolean, optional

        truncate bins after treshold-percentage value has been reached. 

        The default is False.

    treshold : float, optional. Treshold value in percent. The default is 80 (%).



    Returns

    -------

    - (truncated) list of bin-numbers according to x-coordinates of Pareto-Chart

    - probability ]0,1] of visiting 'treshold-bin'

    - total count

    Author: Thomas Plocher; created 20200921

    '''

    from matplotlib.ticker import PercentFormatter

    n = len(dList)

    xs = np.sum(dList) # normalizing factor to get to total of 100%

    dat = np.zeros((n, 2)) # array holds [0] bin-values and [1] cumulative-percentage

    xCum = 0

    xCoords = [] # string-coordinates needed for bin-plot (x-values will be automatically sorted by PyPlot otherwise)

    n80 = None   # treshold index to first bin exceeding cumulative percentage treshold

    for i, j in enumerate(reversed(list(np.argsort(dList)))):

        xCum += 100*dList[j]/xs

        dat[i] = [dList[j], xCum]

        xCoords.append(str(j))

        if n80 == None and xCum >= treshold:

            n80 = i

            if truncate:

                break

    if truncate:

        dat = dat[:n80+1]

    dat = dat.transpose()

    fig, ax = plt.subplots(figsize=figSize)

    if len(nameList) != n:

        xAxisLabels = xCoords

    else:

        xAxisLabels = [nameList[int(i)] for i in xCoords]

    ax.bar(xAxisLabels, dat[0], color=barCol)

    ax2 = ax.twinx()

    ax2.plot(xAxisLabels, dat[1], ls=percLinestyle, color=percCol, marker=percMark, ms=7)

    ax2.yaxis.set_major_formatter(PercentFormatter())

    ax.tick_params(axis="y", colors=barCol)

    ax2.tick_params(axis="y", colors=percCol)

    plt.title(f"Pareto-Chart; cumulative percentage including element '{xCoords[n80]}' "+

              f"is {int(dat[1,n80])}%; count = {int(dat[0,n80])}")

    plt.show()

    return [int(s) for s in xCoords], dat[0,n80]/xs, int(xs)
nList = [2,6,88,20,10,8,60,5]

xLabel = ['one', 'two', '3', 'cat 4', '5', 'six', '7', '8']

paretoChart(nList, xLabel, barCol="C2", percCol="C3", percMark="o", percLinestyle='--')
# the for loop uses np.argsort and enumerate to generate a (reverse-)sorted sequence:

print('nList =', nList)

for i, j in reversed(list(enumerate(np.argsort(nList)))):

    print(i,j)
# The matplotlib.pyplot plot-command sorts x-values if they are numbers.

plt.plot(nList,nList)
# Therefore, the xCoord string-labels need to be generated for plotting the Pareto-chart above.

plt.plot(xLabel,nList)