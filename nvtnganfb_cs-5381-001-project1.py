!pip install dash.ly --upgrade 

!pip install dash_renderer

!pip install chart_studio
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import numpy

import math

import random as rand

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

web_ranking = pd.read_csv("../input/web_ranking.csv")



# add total ranking

weight = pd.DataFrame(pd.Series([1, 1, 1, 1, 1], index=list(web_ranking), name=0))

def calculateTotalRank(w):

    total_rank = web_ranking.dot(w).sum(axis=1)

    return web_ranking.join(pd.DataFrame({'total': total_rank}))

# data

web_ranking = calculateTotalRank(weight)

web_ranking
def sortWeb():

    return web_ranking.sort_values(by=['total']).drop(columns="total")



web_ranking = sortWeb()

web_ranking
data = web_ranking.source1.values

data
def simpleMethod(data):

    n = data.size

    inversion = 0

    for i in range(0,n):

        for j in range(i+1,n):

            if (data[j]<data[i]):

                inversion+=1

    return inversion

print(simpleMethod(data))
def mergeSort(A,inversion):

    n = A.size

    if (n==1):

        return A,inversion

    else:

        q = math.ceil(n/2)

        L = A[0:q]

        R = A[q:n]

        LS,inversion = mergeSort(L,inversion)

        RS,inversion = mergeSort(R,inversion)

        return merge(LS,RS,inversion)

        

def merge(L,R,inversion):

    i = 0

    j = 0

    n1 = L.size

    n2 = R.size

    B = np.empty([n1+n2])

    count = 0

    for k in range(n1+n2):

        if j>n2-1 or (i<n1 and L[i]<=R[j]):

            B[k] = L[i]

            i += 1

            inversion += count

        else:

            B[k] = R[j]

            count = count+1

            j += 1

    return B,inversion



def mergeSort_inversion(arr):

    inversion = 0

    sortedarr,inversion = mergeSort(arr,inversion)

    return inversion



mergeSort_inversion(data)
def quickSort(A,inversion):

    if (len(A)<=1):

        return inversion

    pivotIndex = rand.choice(range(len(A)))

    pivot = A[pivotIndex]

    center = np.array([])

    left = np.array([])

    right = np.array([])

    for index in range(len(A)):

        x = A[index]

        if (x<pivot):

            inversion += len(center) + len(right)

            left = np.concatenate([left,[x]])

        elif (x>pivot):

            right = np.concatenate([right,[x]])

        else:

            inversion += len(right)

            center = np.concatenate([center,[x]])

    inversion = quickSort(left,inversion)

    inversion = quickSort(right,inversion)

    return inversion

quickSort(data,0)
def binarySearch(A,x,l,r):

    if (l==r):

        if(x>A[l]): # l<-x

            return l+1;

        else:

            return l

    if (l>r):

        return l

    center = math.ceil((l+r)/2)

    if (x>A[center]):

        return binarySearch(A,x,center+1,r)

    elif (x<A[center]):

        return binarySearch(A,x,l,center-1)

    return center



def binarySort(A,inversion):

    for i in range(1,len(A)):

        x = A[i]

        newIndex = binarySearch(A,x,0,i-1)

        for j in range(newIndex,i):

            if (A[j]!=x):

                newIndex = j

                break

        inversion += (i - newIndex)

        A = np.concatenate([A[:newIndex],[x],A[newIndex:i],A[i+1:]])

    return inversion-1

binarySort(data,0)
import datetime



import dash

import plotly

import plotly.graph_objs as go

from dash.dependencies import Input, Output

   

def draw(t,v,fig):

    if len(t)==0:

        for y in range(len(v[0])):

            fig.add_trace(go.Scatter(x=t, y=v[:,y]))

        return fig

    else:

         for y in range(len(v[0])):

            fig.data[y].x = t

            fig.data[y].y = v[:,y]

            fig.data[y].name = 'source'+ str(y+1)

    return fig

fig = go.FigureWidget()

fig = draw([],np.array([[1.0, 1.0, 1.0, 1.0, 1.0]]),fig)

fig
def drawInversion(t,v,fig):

    if len(t)==0:

        for y in range(len(v[0])):

            fig.add_trace(go.Scatter(x=t, y=v[:,y]))

        return fig

    else:

         for y in range(len(v[0])):

            fig.data[y].x = t

            fig.data[y].y = v[:,y]

    return fig

fig2 = go.FigureWidget()

fig2 = draw([],np.array([[1.0]]),fig2)

fig2
web_ranking = pd.read_csv("../input/web_ranking.csv")

weight = pd.DataFrame(pd.Series([1.0, 1.0, 1.0, 1.0, 1.0], index=list(web_ranking), name=0))

weight_track = np.array([[]])

inversion_track = np.array([[]])

timeStep = 20

for t in range(timeStep):

    web_ranking = pd.read_csv("../input/web_ranking.csv")

    sourceList = list(web_ranking)

    web_ranking = calculateTotalRank(weight)

    web_ranking = sortWeb()

    data = web_ranking.values

    inversionValues = np.empty(len(data[0,:]))

#     print('weight=',weight)

    for s in range(len(data[0,:])):

        inversionValues[s] = quickSort(data[:,s],0)

    sumInversion = np.sum(inversionValues)

    if (len(inversion_track[0])==0):

        inversion_track = np.array([[sumInversion]])

    else:

        inversion_track = np.append(inversion_track, np.array([[sumInversion]]),axis=0)    

    print('t=',t,':')

    print(inversionValues,' = ',sumInversion)



    # recalculate weight

    for wn in range(len(sourceList)):

        weight[0][wn] = inversionValues[wn]*len(sourceList)/sumInversion

    if (len(weight_track[0])==0):

        weight_track = np.array(numpy.transpose(weight.values))

    else:

        weight_track = np.append(weight_track,numpy.transpose(weight.values),axis=0)

    print(weight)

    fig = draw(np.arange(t+1),weight_track,fig)

    fig2 = drawInversion(np.arange(t+1),inversion_track,fig2)
fig2.show()
fig.show()
import time

import resource



resource.getrusage(resource.RUSAGE_SELF).ru_maxrss



web_ranking = pd.read_csv("../input/web_ranking.csv")

weight = pd.DataFrame(pd.Series([1.0, 1.0, 1.0, 1.0, 1.0], index=list(web_ranking), name=0))

sourceList = list(web_ranking)

web_ranking = calculateTotalRank(weight)

web_ranking = sortWeb()

data = web_ranking.values

inversionValues = np.empty(len(data[0,:]))

runningTime = np.empty(len(data[0,:]))

#     print('weight=',weight)

for s in range(len(data[0,:])):

    time_start = time.clock()

    inversionValues[s] = quickSort(data[:,s],0)

    runningTime[s] = (time.clock() - time_start)

print('Quick sort: ')

print(inversionValues)

print(np.sum(runningTime)/5)





web_ranking = pd.read_csv("../input/web_ranking.csv")

weight = pd.DataFrame(pd.Series([1.0, 1.0, 1.0, 1.0, 1.0], index=list(web_ranking), name=0))

sourceList = list(web_ranking)

web_ranking = calculateTotalRank(weight)

web_ranking = sortWeb()

data = web_ranking.values

inversionValues = np.empty(len(data[0,:]))

runningTime = np.empty(len(data[0,:]))

#     print('weight=',weight)

for s in range(len(data[0,:])):

    time_start = time.clock()

    inversionValues[s] = mergeSort_inversion(data[:,s])

    runningTime[s] = (time.clock() - time_start)

print('Merge sort: ')

print(inversionValues)

print(np.sum(runningTime)/5)



web_ranking = pd.read_csv("../input/web_ranking.csv")

weight = pd.DataFrame(pd.Series([1.0, 1.0, 1.0, 1.0, 1.0], index=list(web_ranking), name=0))

sourceList = list(web_ranking)

web_ranking = calculateTotalRank(weight)

web_ranking = sortWeb()

data = web_ranking.values

inversionValues = np.empty(len(data[0,:]))

runningTime = np.empty(len(data[0,:]))

#     print('weight=',weight)

for s in range(len(data[0,:])):

    time_start = time.clock()

    inversionValues[s] = binarySort(data[:,s],0)

    runningTime[s] = (time.clock() - time_start)

print('Binary sort: ')

print(inversionValues)

print(np.sum(runningTime)/5)


web_ranking = pd.read_csv("../input/web_ranking.csv")

weight = pd.DataFrame(pd.Series([1.0, 1.0, 1.0, 1.0, 1.0], index=list(web_ranking), name=0))

sourceList = list(web_ranking)

web_ranking = calculateTotalRank(weight)

web_ranking = sortWeb()

data = web_ranking.values

inversionValues = np.empty(len(data[0,:]))

runningTime = np.empty(len(data[0,:]))

#     print('weight=',weight)

for s in range(len(data[0,:])):

    time_start = time.clock()

    inversionValues[s] = simpleMethod(data[:,s])

    runningTime[s] = (time.clock() - time_start)

print('Simple Method: ')

print(inversionValues)

print(np.sum(runningTime)/5)