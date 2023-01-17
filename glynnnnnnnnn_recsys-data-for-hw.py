# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/recsys_data_for_hw/recsys_data_for_hw"))



# Any results you write to the current directory are saved as output.
# data processing



import numpy as np



R = np.loadtxt('../input/recsys_data_for_hw/recsys_data_for_hw/user-shows.txt', dtype=np.float64)



# R = np.array([[1,0,1],[1,0,0],[1,1,1],[0,0,1]], dtype=np.float64)

P = np.diag(np.sum(R, axis=1))

Q = np.diag(np.sum(R, axis=0))



print(R[499,])
# uu

import math



P_1_2 = np.zeros((P.shape))

for i in range(P.shape[0]):

    for j in range(P.shape[1]):

        if P[i,j] != 0:

            P_1_2[i,j] = 1/math.sqrt(P[i][j])

            

Q_1_2 = np.zeros((Q.shape))

for i in range(Q.shape[0]):

    for j in range(Q.shape[1]):

        if Q[i,j] != 0:

            Q_1_2[i,j] = 1/math.sqrt(Q[i][j])



# uu

F_u = P_1_2 @ R @ R.T @ P_1_2 @ R



# ii

F_i = R @ Q_1_2 @ R.T @ R @ Q_1_2
F_u_100 = F_u[499,:100]

F_i_100 = F_i[499,:100]

F_u_top5 = np.argsort(-F_u_100, kind='mergesort')[:6]

F_i_top5 = np.argsort(-F_i_100, kind='mergesort')[:6]

print(F_u_top5)

print(F_u_100[F_u_top5])

print("++++++++")

print(F_i_top5)

print(F_i_100[F_i_top5])





shows = np.loadtxt('../input/recsys_data_for_hw/recsys_data_for_hw/shows.txt', dtype='str', delimiter='\n')

alex = np.loadtxt('../input/recsys_data_for_hw/recsys_data_for_hw/alex.txt')
print(alex[F_u_top5])

print('++++++')

print(alex[F_i_top5])
u, smat, vh = np.linalg.svd(R, full_matrices=False)

s = np.copy(smat)

s[320:] = 0

pp = np.sum(s**2)/np.sum(smat**2)

print('----variance----')

print(pp)



print(s.shape)

R_ = u @ np.diag(s) @ vh

print(R_)
R_100 = R_[499,:100]

R_top5 = np.argsort(-R_100, kind='mergesort')[:5]



print("----index----")

print(R_top5) # index

print("----ranks----")

print(R_100[R_top5]) # ranks

print("----show----")

print(shows[R_top5]) # show

print("----alex----")

print(alex[R_top5]) # alex