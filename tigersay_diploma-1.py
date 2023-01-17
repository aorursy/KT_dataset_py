# # This Python 3 environment comes with many helpful analytics libraries installed

# # It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# # For example, here's several helpful packages to load in 



# import numpy as np # linear algebra

# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# # Input data files are available in the "../input/" directory.

# # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# # Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np



data = pd.read_excel('/kaggle/input/data-for-fpg/data2.xlsx')

data.drop_duplicates(inplace = True)

diftov = data['ФПД'].unique().tolist()

rantov = range(len(pd.unique(data['ФПД'])))

TS1 = pd.DataFrame({'num':rantov, 'goo':diftov})

data['ФПД'] = data['ФПД'].map(TS1.set_index('goo')['num'])

orders = data.set_index('ФПД')['Товар']

aprorders = data.groupby('ФПД')['Товар'].apply(list)

aprorders = aprorders.tolist()
from collections import defaultdict



def kick_rare(transactions, min_support):



    def delete_rare_in_transaction(transaction):

        transaction = filter(lambda v: v in items, transaction)

        transaction = sorted(transaction, key=lambda v: items[v], reverse=True)

        return transaction

    items = defaultdict(lambda: 0) 

    if min_support <= 1:

        min_support = int(min_support * len(transactions))

    Tree = []    

    for transaction in transactions:

        for item in transaction:

            items[item] += 1



    items = dict((item, support) for item, support in items.items() if support >= min_support)



    for trans in transactions:

        Tree.append(delete_rare_in_transaction(trans))

        

    return Tree
from sklearn.feature_extraction.text import CountVectorizer

usedaprords = kick_rare(aprorders, 15)
cv = CountVectorizer(tokenizer=lambda x: x, lowercase=False)

m = cv.fit_transform(usedaprords)



# To get the values correspond to each column

goods_name = cv.get_feature_names()



# If you need dummy columns, not count

m = (m > 0)

m = m * 1

m = m.todense()

df = pd.DataFrame(data=m, columns=goods_name)

df = df.drop_duplicates()
#check if empty columns

#(~df.any(axis=0)).any()

np.where(~df.any(axis=1))[0]

df1 = df.drop(df.index[1])

np.where(~df1.any(axis=0))[0]
df1 = df1.T
goods_list = list(df1.index.values)
# print(goods_list)
# from scipy.cluster.hierarchy import linkage, dendrogram

# from sklearn.metrics import pairwise_distances

# from sklearn.cluster import AgglomerativeClustering

# from matplotlib import pyplot as plt
df1_matr = df1.values
# def sim(x, y): 

#     # Subtracted from 1.0 (highest similarity), so now it represents distance

#     xy = np.equal(x, y)

#     ab = (x == 1)

#     a = len(np.where(xy & ab)[0])

#     b = len(np.where(xy & ~ab)[0])

#     c = len(np.where(~xy & ab)[0])

#     d = len(x) - a - b - c

#     return len(x)*(a*d-b*c)*(a*d-b*c)/((a+b)*(a+c)*(c+d)*(b+d))

#     #return 1.0 - np.sum(np.equal(np.array(x), np.array(y)))/len(x)



# # def sim_affinity(X):

# #     return pairwise_distances(X, metric=sim)



# # cluster = AgglomerativeClustering(n_clusters=None, affinity=sim_affinity, linkage='average', distance_threshold=5)

# # print('k')

# # cluster.fit_predict(df1)
# def plot_dendrogram(model, **kwargs):



#     # Children of hierarchical clustering

#     children = model.children_





#     # Distances between each pair of children

#     # Since we don't have this information, we can use a uniform one for plotting

#     distance = np.arange(children.shape[0])



#     # The number of observations contained in each cluster level

#     no_of_observations = np.arange(2, children.shape[0]+2)



#     # Create linkage matrix and then plot the dendrogram

#     linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(float)



#     # Plot the corresponding dendrogram

#     dendrogram(linkage_matrix, **kwargs)

    

# plot_dendrogram(cluster, labels=cluster.labels_)

# plt.show()
# Z = linkage(df[:100])

# from matplotlib import pyplot as plt

# fig = plt.figure(figsize=(25, 10))

# dn = dendrogram(Z)

# plt.show()
# clustering = AgglomerativeClustering().fit(df[:100])

# clustering.labels_
# tt = time()

# start_matrix = pairwise_distances(df1_matr[:500], metric=sim)

# ss = time()

# print(ss-tt)
# from scipy.spatial.distance import pdist

# tt = time()

# start_matrix = pdist(df1_matr, lambda u, v: sim(u,v))

# ss = time()

# print(ss-tt)
from numpy import genfromtxt

start_matrix_1 = genfromtxt("/kaggle/input/matrix-diploma-08/matrix_train.csv", delimiter=',')
# start_matrix_1 = np.zeros((df1_matr.shape[0], df1_matr.shape[0]))

# counter = 0

# for i in range(df1_matr.shape[0]):

#     for j in range(i+1, df1_matr.shape[0]):

#         start_matrix_1[i][j] = start_matrix[counter]

#         counter += 1

# for i in range(1, df1_matr.shape[0]):

#     for j in range(0, i):

#         start_matrix_1[i][j] = start_matrix_1[j][i]
# A = np.array([[0.,8.,8.,9.], [8.,0.,2.,1.], [8.,2.,0.,6.], [9.,1.,6.,0.]])

# #np.fill_diagonal(A, np.inf)

# #thres = 7.0

# x = list(range(4))

# i = 0

# matr_list=[]

# while i<len(x):

#     matr_list.append(x[i:i+1])

#     i+=1
A = start_matrix_1.copy()

np.fill_diagonal(A, 0.0)

x = list(range(df1_matr.shape[0]))

i = 0

matr_list=[]

while i<len(x):

    matr_list.append(x[i:i+1])

    i+=1

while True:

#     if np.max(A) < 30000:

#         break

    if A.shape[0] == 200:

        break

    f_con = np.argmax(A) // A.shape[0]

    s_con = np.argmax(A) % A.shape[0]

    new_str = []

    for i in range(A.shape[0]):

        if i == f_con:

            new_str.append(np.inf)

        else:

            mmm = min(A[f_con][i], A[s_con][i])

            kkk = (len(matr_list[f_con]) * A[f_con][i] + len(matr_list[s_con]) * A[s_con][i])/(len(matr_list[f_con]) + len(matr_list[s_con]))

            #nnn = (A[f_con][i] + A[s_con][i])/2

            new_str.append(0.5*(mmm)+0.5*(kkk))

    A[f_con] = np.array(new_str)

    A[:,f_con] = np.array(new_str)

    A = np.delete(A, (s_con), axis=0)

    A = np.delete(A, (s_con), axis=1)

    matr_list[f_con] = matr_list[f_con] + matr_list[s_con]

    del(matr_list[s_con])

    A[f_con][f_con] = 0.0
print(len(matr_list))

counter = 0

for i in matr_list:

    if len(i) > 1:

        print(len(i))

        counter += 1

print(counter)
sorted_clusters = sorted(matr_list, key=lambda l: (len(l), l))[::-1]
sorted_goods_in_clusters = sorted_clusters.copy()
for i in range(len(sorted_goods_in_clusters)):

    for j in range(len(sorted_goods_in_clusters[i])):

        sorted_goods_in_clusters[i][j] = goods_list[sorted_goods_in_clusters[i][j]]

#np.savetxt("goods_in_250_clusters.csv", sorted_goods_in_clusters, delimiter = ",")
sorted_goods_in_clusters
items = {}

cc = 0

for line in goods_list:

    key, value = line, cc

    items[key] = str(value)

    cc += 1
matr_list_1 = matr_list.copy()
sorted_nums_in_clusters = sorted_goods_in_clusters.copy()
for idx in range(len(sorted_nums_in_clusters)):

    for j in range(len(sorted_nums_in_clusters[idx])):

        sorted_nums_in_clusters[idx][j] = items[sorted_nums_in_clusters[idx][j]]

print(sorted_nums_in_clusters)
df2 = df1.T
# df2.iloc[[0]].squeeze().to_numpy().nonzero()[0]

# np.where(df2.iloc[[0]].squeeze().to_numpy() == 0)[0]
# df2 = df2.iloc[:, :500]
import random

data_for_forest = np.zeros((df2.shape[0], 4 * len(sorted_nums_in_clusters) + 1))

for i in range(df2.shape[0]):

    all_ones = df2.iloc[[i]].squeeze().to_numpy().nonzero()[0]

    all_zeros = np.where(df2.iloc[[0]].squeeze().to_numpy() == 0)[0]

    if len(all_ones) > 3:

        if random.uniform(0.0, 1.0) < 0.5:

            ones = random.sample(list(all_ones), 3) + random.sample(list(all_zeros), 1)

            target = 1

        else:

            ones = random.sample(list(all_ones), 4)

            target = 0

        outer_counter = 0

        for m in ones:

            for j in range(len(sorted_nums_in_clusters)):

                counter = 0

                dist = 0.0

                for k in sorted_nums_in_clusters[j]:

                    dist += start_matrix_1[int(m)][int(k)]

                    counter += 1

                data_for_forest[i][len(sorted_nums_in_clusters) * outer_counter + j] = dist / counter

            outer_counter += 1

        data_for_forest[i][4 * len(sorted_nums_in_clusters)] = target

#     if i % 100 == 0:

#         print(i)

data_for_forest = data_for_forest[~np.all(data_for_forest == 0, axis=1)]
np.savetxt("data_for_forest_trained_200_clusters.csv", data_for_forest, delimiter = ",")
X_train = data_for_forest[0:int(0.8*data_for_forest.shape[0]),:-1]

X_test = data_for_forest[int(0.8*data_for_forest.shape[0]):,:-1]

y_train = data_for_forest[0:int(0.8*data_for_forest.shape[0]),-1:].astype(int)

y_test = data_for_forest[int(0.8*data_for_forest.shape[0]):,-1:].astype(int)
from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import log_loss, roc_auc_score

X = data_for_forest[0:int(1.0*data_for_forest.shape[0]),:-1]

y = data_for_forest[0:int(1.0*data_for_forest.shape[0]),-1:].astype(int)

cclf = RandomForestClassifier()

print(cross_val_score(cclf, X, y, scoring='roc_auc', cv=5)) 
clf = RandomForestClassifier()

clf.fit(X_train, y_train.ravel())

clf_probs = clf.predict_proba(X_test)

lloss = log_loss(y_test.ravel(), clf_probs)

rocauc = roc_auc_score(y_test.ravel(), clf_probs[:,1])

print(lloss)

print(rocauc)
# from fast_fpgrowth import mine_frequent_itemsets, assocRule



# result = []

# res_for_rul = {}

# for itemset, support in mine_frequent_itemsets(aprorders,60, True):

#     res_for_rul[tuple(itemset)] = support

# rules = assocRule(res_for_rul, 0.01)

# sorted_rules_sup = sorted(rules, key=lambda k: k['supp'], reverse=True) 