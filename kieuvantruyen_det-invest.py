# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from pprint import pprint

import numpy as np



#tinh detemine cua ma tran 3x3

def det3(m):

    a1 = m[0][0]*(m[1][1]*m[2][2]-m[1][2]*m[2][1])

    a2 = m[0][1]*(m[0][0]*m[2][2]-m[0][2]*m[2][0])

    a3 = m[0][2]*(m[1][0]*m[2][1]-m[1][1]*m[2][0])

    return a1 + (-1)*a2 + a3



#tinh detemine cua ma tran 2x2  

def det2(ma):

    return (ma[0]*ma[3] - ma[1]*ma[2])





#ham chuyen vị

def tranpos(m):

    n = len(m)

    for i in range(n):

        for j in range(n):

            if i>0 and j<n-1:

                if i != j:

                    c = m[i][j]

                    m[i][j] = m[j][i]

                    m[j][i] = c

    return m



#nhan man tran voi a = [[1,-1,1][-1,1,-1][1,-1,1]]

def changestatus(m):

    a = 1

    for i in range(len(m)):

        for j in range(len(m)):

            m[i][j]*=a

            a*=(-1)

    return m



#ham nay chac ai cung biet

def inverse(m):

    temM = []   

    ad = [[1,-1,1],[-1,1,-1],[1,-1,1]]

    for i in range(len(m)):

        temDet = []

        for j in range(len(m)):

            arr = []

            #xet ma tra khong cung dong/cot voi doi tuong dang chon

            for p in range(len(m)):

                for q in range(len(m)):

                    #them doi tuong thoa mang vao matran 2 chieu

                    if i != p and j != q:

                        arr.append(m[p][q])

#             print(arr)

            #tinh det cua ma tran hai chieu

#             print(det2(arr))

            #lay 3 det2 vao 1 dong

            temDet.append(det2(arr))

#         print("--------------------")

        #lay moi dong add vào matrix

        temM.append(temDet)

#     print(temM)

    

    #invest matra

    print(changestatus(temM))

    

    

    

m1 = [[1,4,1],[2,1,1],[5,3,3]]



print("matrix: ",m1)

if det3(m1) <= 0:

    print("can not inverst this matrix")

else:

    print("det of matrix: ",det3(m1))

    print("matrix was tranposed: ",tranpos(m1))

    print("matax is invested: ")

    inverse(m1)

    




