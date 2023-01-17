#coding=utf-8

import pandas as pd

from numpy import *



gray=pd.read_excel("../input/data-of-image/data2.xlsx")

#读取为df格式

gray=(gray - gray.min()) / (gray.max() - gray.min())

print(gray)

#标准化

std=gray.iloc[:,8]#为标准要素

ce=gray.iloc[:,0:8]#为比较要素



n=ce.shape[0]

m=ce.shape[1]#计算行列



#与标准要素比较，相减

a=zeros([m,n])

for i in range(m):

    for j in range(n):

        a[i,j]=abs(ce.iloc[j,i]-std[j])



#取出矩阵中最大值与最小值

print(a)

c=amax(a)

d=amin(a)

print(c,d)

#计算值

result=zeros([m,n])

for i in range(m):

    for j in range(n):

        result[i,j]=(d+0.5*c)/(a[i,j]+0.5*c)



#求均值，得到灰色关联值

result2=zeros(m)

for i in range(m):

        result2[i]=mean(result[i,:])

RT=pd.DataFrame(result2)

print(RT)

RT.to_csv("2.csv")