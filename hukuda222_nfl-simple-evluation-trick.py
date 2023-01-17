import os

import pandas as pd

from kaggle.competitions import nflrush

import random

import gc

import pickle

import tqdm

import matplotlib.pyplot as plt

import numpy as np

from scipy.stats import norm
plt.plot([i-99 for i in range(199)],[1 if i-99>=0 else 0 for i in range(199)],label="ans")

plt.plot([i-99 for i in range(199)],[0.5 for i in range(199)],label="pred")

plt.legend()
print("case0-1's score is ",sum([((1 if i-99>=0 else 0)-0.5)**2 for i in range(199)])/199)
plt.plot([i-99 for i in range(199)],[1 if i-99>=0 else 0 for i in range(199)],label="ans")

plt.plot([i-99 for i in range(199)],[i/199 for i in range(199)],label="pred")

plt.legend()
print("case0-2's score is ",sum([((1 if i-99>=0 else 0)-(i/199))**2 for i in range(199)])/199)
plt.plot([i-99 for i in range(199)],[1 if i-99>=0 else 0 for i in range(199)],label="ans")

plt.plot([i-99 for i in range(199)],[1 if i-99>=5 else 0 for i in range(199)],label="pred")

plt.legend()
print("case1's score is ",sum([((1 if i-99>=0 else 0)-(1 if i-99>=5 else 0))**2 for i in range(199)])/199)
plt.plot([i-99 for i in range(199)],[1 if i-99>=0 else 0 for i in range(199)],label="ans")

plt.plot([i-99 for i in range(199)],[1 if i-99>=5+10 else 0 if i-99<5-10 else ((i-99)-(5-10))/20 for i in range(199)],label="pred")

plt.legend()
print("case2's score is ",sum([((1 if i-99>=0 else 0)

                                - (1 if i-99>=5+10 else 0 if i-99<5-10 else ((i-99)-(5-10))/20))**2 

                               for i in range(199)])/199)
from scipy.stats import norm

x = np.arange(-10,10,0.01)

y = norm.pdf(x,0,3)

plt.plot(x,y)

plt.xlim(-10,10)
x = np.arange(-10,10,0.01)

y = norm.pdf(x,0,3)

plt.plot(x,np.cumsum(y))

plt.xlim(-10,10)
norm_cumsum = np.cumsum(norm.pdf(np.arange(-10,10,1),0,3))



plt.plot([i-99 for i in range(199)],[1 if i-99>=0 else 0 for i in range(199)],label="ans")

plt.plot([i-99 for i in range(199)],[1 if i-99>=5+10 else 0 if i-99<5-10 else norm_cumsum[(i-99)-(5-10)] for i in range(199)],label="pred")

plt.legend()
print("case3's score is ",sum([((1 if i-99>=0 else 0)

                                - (1 if i-99>=5+10 else 0 if i-99<5-10 else 

                                   norm_cumsum[(i-99)-(5-10)]))**2 

                               for i in range(199)])/199)
env = nflrush.make_env()

train_df = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv', low_memory=False)
print("the median value of train data is ",np.median(train_df["Yards"]))
y_train = np.array([train_df["Yards"][i] for i in range(0,509762,22)])

y_pred_case1 = np.zeros((509762//22,199))

y_pred_case2 = np.zeros((509762//22,199))

y_pred_case3 = np.zeros((509762//22,199))

y_ans = np.zeros((509762//22,199))

norm_cumsum = np.cumsum(norm.pdf(np.arange(-10,10,1),0,3))



p=3

w=10

for i in range(509762//22):

    for j in range(199):

        if j>=p+w:

            y_pred_case2[i][j]=1.0

            y_pred_case3[i][j]=1.0

        elif j>=p-w:

            y_pred_case2[i][j]=(j+w-p)/(2*w)

            y_pred_case3[i][j]=norm_cumsum[max(min(j+w-p,19),0)]

        if j>=p:

            y_pred_case1[i][j]=1.0



for i,p in enumerate(y_train):

    for j in range(199):

        if j>=p:

            y_ans[i][j]=1.0



print("validation score in case1:",np.sum(np.power(y_pred_case1-y_ans,2))/(199*(509762//22)))

print("validation score in case2:",np.sum(np.power(y_pred_case2-y_ans,2))/(199*(509762//22)))

print("validation score in case3:",np.sum(np.power(y_pred_case3-y_ans,2))/(199*(509762//22)))