import numpy as np

import pandas as pd

from scipy.spatial import distance_matrix, distance

from scipy.stats import spearmanr
f=open('../input/nicn2/demo.csv').readlines()

f=[list(map(float, i[:-1].split(','))) for i in f]
X20_30=np.array(f[:20])

X20_30.shape
pd.DataFrame(distance_matrix(X20_30, X20_30))
X20_16=np.array(f[20:40])

X20_16.shape
pd.DataFrame(distance_matrix(X20_16, X20_16))
pd.DataFrame(distance_matrix(X20_30, X20_30)-distance_matrix(X20_16, X20_16)).round(7)
X20_1=np.array(f[40:60])

X20_1.shape
pd.DataFrame(distance_matrix(X20_1, X20_1))
pd.DataFrame(distance_matrix(X20_30, X20_30)-distance_matrix(X20_1, X20_1)).round(7)
pd.DataFrame(distance_matrix(X20_30, X20_30)).corrwith(pd.DataFrame(distance_matrix(X20_1, X20_1)), method='spearman')
counter=0

good=0

bad=0

for i in range(20):

    for j in range(20):

        if i > j:

            dst_i_j_30 = distance.euclidean(X20_30[i], X20_30[j])

            dst_i_j_1 = distance.euclidean(X20_1[i], X20_1[j])

            for k in range(20):

                for l in range(20):

                    if k > l:

                        dst_k_l_30 = distance.euclidean(X20_30[k], X20_30[l])

                        dst_k_l_1 = distance.euclidean(X20_1[k], X20_1[l])



                        if dst_i_j_30 < dst_k_l_30: 

                            if dst_i_j_1 < dst_k_l_1:

                                good+=1

                            else:

                                bad+=1



                        counter+=1

                

print(good, bad, good+bad)
counter=0

all_distances_30=[]

all_distances_1=[]

for i in range(20):

    for j in range(20):

        if i > j:

            all_distances_30.append((distance.euclidean(X20_30[i], X20_30[j]), counter))

            all_distances_1.append((distance.euclidean(X20_1[i], X20_1[j]), counter))

            counter+=1

print(counter)
spearmanr(list(zip(*all_distances_30))[0], list(zip(*all_distances_1))[0])
counter=0

all_distances_30_with_ranks=[[i,] for i,_ in all_distances_30]

all_distances_1_with_ranks=[[i,] for i,_ in all_distances_1]

for i in list(zip(*sorted(all_distances_30)))[1]:

    all_distances_30_with_ranks[i].append(counter)

    all_distances_1_with_ranks[i].append(counter)

    counter+=1

print(counter)
spearmanr(list(zip(*all_distances_30_with_ranks))[1], list(zip(*all_distances_1_with_ranks))[1])
spearmanr(list(zip(*all_distances_30_with_ranks))[0], list(zip(*all_distances_1_with_ranks))[1])
spearmanr(list(zip(*all_distances_30_with_ranks))[1], list(zip(*all_distances_1_with_ranks))[0])