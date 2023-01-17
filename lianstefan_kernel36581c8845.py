import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import networkx as nx

plt.rcParams['axes.unicode_minus']=False
data=pd.read_csv('../input/cities.csv')

data.info()

data.head()
data.plot.scatter(x='X', y='Y', s=0.07, figsize=(15, 10))

north_pole = data[data.CityId==0]

#起点：南极标为红色

plt.scatter(north_pole.X, north_pole.Y, c='red', s=15)

plt.axis('off')

plt.show()
#判断是否是质数

def is_prime(num):

    if num > 1:

        for i in np.arange(2, np.sqrt(num+1)) :

            if num % i == 0:

                return False

        return True

    return False



data['prime'] = data['CityId'].apply(is_prime)

data.head()
#画出含质数和不含质数的图

fig, (ax0,ax1) = plt.subplots(ncols = 2, figsize = (15,7))



ax0.set_title('Cities chart. North Pole, non prime cities', fontsize = 16)

ax0.scatter(x = data.X[0], y = data.Y[0], c = 'r', s =12)

ax0.scatter(x = data[data.prime == False ].X, y = data[data.prime == False].Y, s = 1, c = 'green', alpha = 0.3)

ax0.annotate('North Pole', (data.X[0],data.Y[0]),fontsize =12)

ax0.set_axis_off()



ax1.scatter(x = data[data.prime].X, y = data[data.prime].Y, s = 1, c = 'purple', alpha = 0.3)

ax1.set_title('Cities chart. North Pole, prime cities', fontsize = 16)

ax1.annotate('North Pole', (data.X[0], data.Y[0]),fontsize =12)

ax1.set_axis_off()
#取出100行数据来计算

data=data.iloc[0:100,::]

data.info()
#计算距离

from scipy.spatial import distance



i=1

while(i<=100):

    all_dist = distance.cdist(data[['X','Y']][i-1:i],data[['X','Y']],metric = 'euclidean')

    data['%s'%(i-1)]= all_dist.T

    i+=1

data.head(10)
import copy

#以下为Dijkstra算法求解最短路径

def findCloestrout(inf,rout,S,U,cloest_rout):

    key_UtoS = {} #记录u的每个key到D点会通过哪个已经确定的最短路径，用于后面输出最短路线

    for key in U:

        for key2 in S:

            if key+key2 in rout:

                if rout[key+key2]+S[key2] <= U[key]: #保持存储最小值

                    U[key] = rout[key+key2]+S[key2]

                    key_UtoS[key] = key2

            elif key2+key in rout:

                if rout[key2+key]+S[key2] <= U[key]:

                    U[key] = rout[key2+key]+S[key2]

                    key_UtoS[key] = key2

            else:

                continue

    min_value = inf

    key_min = None

    for key in U: #找最小的路径

        if U[key]<min_value:

            min_value = U[key]

            key_min = key

    del U[key_min]#从未确定的最短路径集合删除可以确定的最短路径

    S[key_min] = min_value#添加已经确定的最短路径

    for num in range(len(cloest_rout)):

        if cloest_rout[num][-1] == key_UtoS[key_min]:

            temp_list = copy.deepcopy(cloest_rout[num])#这里一定要深拷贝，不然后一改全改

            temp_list.append(key_min)

            cloest_rout.append(temp_list)

 # 以下为绘制带权值无向图

def draw(rout,rout_list):

    G = nx.Graph()  # 创建一个空图

    G.add_weighted_edges_from(rout_list)  # 添加权值边

    weight_list = {}

    for it in rout_list:  # 制作一个全权值表

        weight_list[it[0], it[1]] = it[2]

    pos = nx.spring_layout(G)  # 设置点的布局

    nx.draw_networkx_edge_labels(G, pos, weight_list, font_size=10)  # 绘制权值

    nx.draw(G, pos, node_color='g', edge_color='r', with_labels=True, \

            font_color='b', font_size=20, node_size=800)  # 绘制权值边

    plt.show()  # 显示绘图
#用上面的得到的距离值简单测试一下



rout = {'AB':4468.691432,'AG':881.709252,'AF':2289.622603,'BF':2197.654267,'BC':2649.512213,'GF':1934.534016,'GE':515.285152,'CF':1696.232303,'EF':2320.858439,'CD':1239.367061,'CE':2478.737595,'ED':3689.688402}

rout_list = [('A','B',4468.691432),('A','G',881.709252),('A','F',2289.622603),('B','F',2197.654267),

             ('B','C',2649.512213),('G','F',1934.534016),('G','E',515.285152),('C','F',1696.232303),

             ('E','F',2320.858439),('C','D',1239.367061),('C','E',2478.737595),('E','D',3689.688402)]

 

inf = float('inf')#无穷大

cloest_rout = [['D']]

S = {'D':0}  #确定最短路径集合

U = {'A':inf,'B':inf,'C':inf,'E':inf,'F':inf,'G':inf} #未确定的最短路径集合

while(U != {}): #直到未确定最短路径为空，才最终确定完最短路径

    findCloestrout(inf,rout,S,U,cloest_rout)

    

for list1 in cloest_rout:

    print(list1[-1]+':',end='')

    print('->'.join(list1))

print('Total Path：',S)

draw(rout,rout_list)
# 最小生成树，动态规划解法

class Solution:

    def __init__(self,X,start_node):

        self.X = X

        self.start_node = start_node

        

    def prim(self):

        num = len(self.X)

        first_node = self.start_node

        last_node = self.start_node

        sets = [i for i in range(num)]

        sets.pop(first_node)

        first_set = [self.start_node]

        self.dtgh(first_set,sets)

        return first_set

    def dtgh(self,past_sets,sets):

        if len(sets) == 0:

            return

        d_i = [] 

        d_min = 10000

        # 遍历还未经过的节点

        for i in range(len(sets)):

            d_ij = [] # 储存已经过集合中所有节点（j）到新集合中i节点的距离

            for j in past_sets:

                d_ij.append(self.X[j][sets[i]])

            # 寻找最短的i节点到j节点的路径

            if min(d_ij)<d_min:

                # 最短路径中的（位于老集合中的节点）j节点

                j_min = d_ij.index(min(d_ij))

                # 最短路径中的（位于新集合中的节点）i节点

                i_d = i

            d_i.append(min(d_ij)) #求取所有i（新集合）节点与老集合的最小距离集合

            d_min = min(d_ij) #参与循环

        d_increase = min(d_i) # 当前最短路径（j，i）的最短距离

        print(past_sets[j_min], "---->", sets[i_d], "the distance", d_increase)

        past_sets.append(sets[i_d])

        sets.pop(i_d)

        self.dtgh(past_sets,sets)

sample=np.array([i for i in range(0,11)])

D = np.array(data.iloc[sample,sample+4])

start_node = 0

S = Solution(D,start_node)

S.prim()
import pandas as pd

cities = pd.read_csv("../input/cities.csv")