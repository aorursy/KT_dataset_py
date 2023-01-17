# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import time
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random as rand
import math
weight = [1,4,1]
v_matrix = [[1  ,2 ,0.5],
            [2  ,1 ,0  ],
            [0.5,0 ,0.5]]
def create_qk(weight,v_matrix):
    qk = {'weight':[],'value':[]}
    for i in range(0,len(weight)):
        qk['weight'].append({'index':i,'val':weight[i]})
    qk['value'] = v_matrix
    return qk
qk = create_qk(weight,v_matrix)
qk
def calculate_k(input,c):
    sum_w = 0
    for wi in range(0,len(input['weight'])):
        w = input['weight'][wi]
        sum_w = sum_w+ w['val']
        if (sum_w>c):
            return wi-1
    return len(input['weight'])-1
def sum_pi(pi,i,e):
    sumpi = 0
    for j in range(0,i+1):
        sumpi = sumpi + math.ceil(pi[j]/e)
    return sumpi
def getM (M,i,v,w):
    try:
        return M[i][v]
    except KeyError:
        if (i==-1):
            return 0
        else:
            return w
def getI (I,i,v):
    try:
        return I[i][v]
    except KeyError:
        return [i]
def QKP_upperplane(input,c,e):
#     input : qk input
#     c: knapsack size
#     e: rounding factor
    
    n_item = len(input)+1
    theta = e*np.array(input['value']).max()/2/(n_item-1)
#     1. calculate upper plane 2
#     1.1 sort list of weight
    input['weight'].sort(key = lambda x: x['val'] )
    w = [wt['val'] for wt in input['weight']]
#     1.2 calculate k 
    k = calculate_k(input,c)
#     1.3 calculate pi2j
    pi={}
    for j_index in range(0,n_item):
        j = input['weight'][j_index]['index']
        pi[j] = 0
        for i in range(0,k+1):
            pi[j] = pi[j]+ input['value'][i][j]
#     2. dynamic programing for knapscak
    M = {}
    max_w = 0
    max_v = 0
    list_item =[]
    for i in range(-1,n_item):
        M[i]={}
        M[i][0] = 0
        for v in range(1,sum_pi(pi,i,theta)):
            M[i][v]=0
    for i in range(0,n_item):
        for v in range(1,sum_pi(pi,i,theta)+1):
            if (v>sum_pi(pi,i-1,e)):
                M[i][v] = w[i] + getM(M,i-1,v,w[i-1])
            else:
                i_1 = getM(M,i-1,v,w[i-1])
                add = w[i]+getM(M,i-1,max(0,v-math.ceil(pi[i]/theta)),w[i-1])
                M[i][v] = min(i_1,add)
#             if (M[i][v]<=c and max_v < v):
#                 max_w = M[i][v]
#                 max_v = v
    for v in M[n_item-1]:
        if (M[n_item-1][v]<=c and max_v < v):
                max_w = M[i][v]
                max_v = v
    return max_v*theta
QKP_upperplane(qk,5,0.5)        
