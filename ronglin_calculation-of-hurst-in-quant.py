# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# -*- coding: utf-8 -*-

"""

Created on Sat Sep  2 20:13:30 2017



@author: dell

"""

import numpy as np

#计算hurst的值

def hurst(data):

    RS = list()

    ARS = []

    N = len(data)

    ranges = [2,4,8,16,32,64]

    L = N/np.array(ranges)

    for i in range(len(ranges)):

        for r in range(ranges[i]):

            Range = data[int(L[i]*r):int(L[i]*(r+1))]

            meanvalue = np.mean(Range)

            Deviation = np.subtract(Range,meanvalue)

            sigma = np.sqrt((sum(Deviation*Deviation))/(L[i]-1))

            Deviation = Deviation.cumsum()

            maxi = max(Deviation)

            mini = min(Deviation)

        RS.append((maxi-mini)/sigma)

        ARS.append(np.mean(RS))

    GAP = np.log(L)

    a = np.log(ARS)

    hurst_exponent = np.polyfit(GAP,a,1)[0]*2

    return(hurst_exponent)

#滚动窗口,N为窗口大小,

def rolling(close_data,N):

    hurst_value = close_data.rolling(window=N).apply(hurst)

    return(hurst_value)