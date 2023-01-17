import os

import time

import numpy as np

import pandas as pd

%matplotlib inline

import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib import rcParams

from tqdm import tqdm_notebook

from sklearn.tree import _tree, export_graphviz

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
def read_data(file_path):

    print('Loading datasets...')

    X_train = pd.read_csv(file_path + 'train.csv', sep=',')

    print('Datasets loaded')

    return X_train

PATH = '../input/cat-in-the-dat-ii/'

X_train = read_data(PATH)
X_train.head(25)
X_train.shape
X_train.columns
X_train = X_train.sample(n=50000, replace=False, random_state=42, axis=0)
y_train = X_train['target']

del X_train['target']

del X_train['id']
#Missing Value Approaches

print('Missing Value Approaches')

columns = X_train.columns

for cc in tqdm_notebook(columns):

    X_train[cc] = X_train[cc].fillna(X_train[cc].mode()[0])



#Label Encoding

print('Label Encoding')

for cc in tqdm_notebook(columns):

    le = LabelEncoder()

    le.fit(X_train[cc].values)

    X_train[cc] = le.transform(X_train[cc].values)



#OneHotEncoding

print('OneHotEncoding')

OHE = OneHotEncoder(dtype='uint16', handle_unknown="ignore")

OHE.fit(X_train)

X_train = OHE.transform(X_train)
X_train
X_train.shape
def leaf__depths(estimator, nodeid = 0):

     left__child = estimator.children_left[nodeid]

     right__child = estimator.children_right[nodeid]

     

     if left__child == _tree.TREE_LEAF:

         depths = np.array([0])

     else:

         left__depths = leaf__depths(estimator, left__child) + 1

         right__depths = leaf__depths(estimator, right__child) + 1

         depths = np.append(left__depths, right__depths)

 

     return depths



def leaf__samples(estimator, nodeid = 0):  

     left__child = estimator.children_left[nodeid]

     right__child = estimator.children_right[nodeid]



     if left__child == _tree.TREE_LEAF: 

         samples = np.array([estimator.n_node_samples[nodeid]])

     else:

         left__samples = leaf__samples(estimator, left__child)

         right__samples = leaf__samples(estimator, right__child)

         samples = np.append(left__samples, right__samples)



     return samples



def visualization__estimator(ensemble, tree_id=0):



     plt.figure(figsize=(20,20))

     plt.subplot(211)



     estimator = ensemble.estimators_[tree_id].tree_

     depths = leaf__depths(estimator)

     

     plt.hist(depths, histtype='step', color='blue', bins=range(min(depths), max(depths)+1))

     plt.grid(color='black', linestyle='dotted')

     plt.xlabel("Depth of leaf nodes (tree %s)" % tree_id)

     plt.show()



def visualization__forest(ensemble):



     plt.figure(figsize=(20,20))

     plt.subplot(211)



     depths__all = np.array([], dtype=int)



     for x in ensemble.estimators_:

         estimator = x.tree_

         depths = leaf__depths(estimator)

         depths__all = np.append(depths__all, depths)

         plt.hist(depths, histtype='step', color='blue', 

                  bins=range(min(depths), max(depths)+1))



     plt.hist(depths__all, histtype='step', color='blue', 

              bins=range(min(depths__all), max(depths__all)+1), 

              weights=np.ones(len(depths__all))/len(ensemble.estimators_), 

              linewidth=2)

     plt.grid(color='black', linestyle='dotted')

     plt.xlabel("Depth of leaf nodes")

    

     plt.show()
seed = 2020
start_time = time.clock()



model = RandomForestClassifier(n_jobs=-1,n_estimators=100, max_features=5, random_state=seed)

model.fit(X_train, y_train)



visualization__estimator(model)



end_time = time.clock()

print("")

print("Total Estimation Running Time:")

print(end_time - start_time, "Seconds")
start_time = time.clock()



model = RandomForestClassifier(n_jobs=-1,n_estimators=100, max_features=5, random_state=seed)

model.fit(X_train, y_train)



visualization__forest(model)



end_time = time.clock()

print("")

print("Total Estimation Running Time:")

print(end_time - start_time, "Seconds")
start_time = time.clock()



model = RandomForestClassifier(n_jobs=-1,n_estimators=100, max_features=5, max_depth=12, random_state=seed)

model.fit(X_train, y_train)



visualization__forest(model)



end_time = time.clock()

print("")

print("Total Estimation Running Time:")

print(end_time - start_time, "Seconds")
start_time = time.clock()



model = RandomForestClassifier(n_jobs=-1,n_estimators=100, max_features=5, min_samples_leaf=3, random_state=seed)

model.fit(X_train, y_train)



visualization__forest(model)



end_time = time.clock()

print("")

print("Total Estimation Running Time:")

print(end_time - start_time, "Seconds")
start_time = time.clock()



model = RandomForestClassifier(n_jobs=-1,n_estimators=100, max_features=3, min_samples_leaf=2, random_state=seed)

model.fit(X_train, y_train)



visualization__forest(model)



end_time = time.clock()

print("")

print("Total Estimation Running Time:")

print(end_time - start_time, "Seconds")
start_time = time.clock()



model = RandomForestClassifier(n_jobs=-1,n_estimators=100, max_features=5, bootstrap=False, random_state=seed)

model.fit(X_train, y_train)



visualization__forest(model)



end_time = time.clock()

print("")

print("Total Estimation Running Time:")

print(end_time - start_time, "Seconds")