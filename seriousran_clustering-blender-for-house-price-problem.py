import numpy as np

import pandas as pd

import seaborn as sns



# for hierarchical clusterization

from scipy.cluster.hierarchy import dendrogram, linkage  

from scipy.spatial.distance import  pdist

from matplotlib import pyplot as plt

from sklearn.preprocessing import MinMaxScaler



# system

from datetime import datetime

import os

sns.set()



print(os.listdir("../input/2019-2nd-ml-month-with-kakr"))

# inspired by [...]

s1=pd.read_csv('../input/kakr-outputs/A_Note_on_Using_a_Single_Model_XGBoost_103683.csv')['price']

s2=pd.read_csv('../input/kakr-outputs/House_Price_Prediction_EDA_108686.csv')['price']

s3=pd.read_csv('../input/kakr-outputs/lgb_xgb_v1_107382.csv')['price']

s4=pd.read_csv('../input/kakr-outputs/Default_EDA-Stacking_Introduction_104047.csv')['price']

s5=pd.read_csv('../input/kakr-outputs/ensemble_v2_r_102424.csv')['price']

s6=pd.read_csv('../input/kakr-outputs/XGBoost_102562.csv')['price']



submission = pd.read_csv('../input/2019-2nd-ml-month-with-kakr/sample_submission.csv')



solutions_set = pd.DataFrame({'s1': s1, 's2': s2, 's3': s3, 's4': s4, 's5': s5, 's6': s6})
# Scaling

scaler = MinMaxScaler()  

solutions_set_scaled = scaler.fit_transform(solutions_set)

solutions_set_scaled = pd.DataFrame(solutions_set_scaled, columns = solutions_set.columns)
# transpose and convert solutions set to numpy

np_solutions_set = solutions_set_scaled.T.values

# calculate the distances

solutions_set_dist = pdist(np_solutions_set)

# hierarchical clusterization

linked = linkage(solutions_set_dist, 'ward')



# dendrogram

fig = plt.figure(figsize=(8, 5))

dendrogram(linked, labels = solutions_set_scaled.columns)

plt.title('clusters')

plt.show()
sns.scatterplot(x = 's1', y = 's2', data = solutions_set_scaled)

plt.title('Submisions of s1 and s2')
sns.scatterplot(x = 's3', y = 's6', data = solutions_set_scaled)

plt.title('Submisions of s3 and s6')
# get scaled submissions s1, s2 ... s7

for s in solutions_set_scaled.columns:

    s = solutions_set_scaled[s]

    

cluster1 = 1/4 * (s1 + s2 + s3 + s4)

cluster2 = 1/2 * (s5 + s6)



submission['price'] = 0.3*cluster1 + 0.7*cluster2
submission.to_csv('submission.csv', index=False)