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
import numpy as np
import pandas as pd
import pylab as pl
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sqlalchemy import create_engine, MetaData
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn import metrics
from sklearn.metrics import pairwise_distances
# Any results you write to the current directory are saved as output.

#ps_csv = pd.read_csv("../input/pscsv2/ps_csv.csv")
ps_csv = pd.read_csv("../input/helloo3/ps_csv_2.csv")
df=ps_csv
df2=ps_csv
df.dropna(inplace=True)
ls=[19.1538, 15.3462, 2.65385, 1.34615, 1.34615, 1.03846, 1]
df=df.drop('orgid',axis=1)
df2=df
df=df*ls
from sklearn import preprocessing
#minmax_processed = preprocessing.MinMaxScaler().fit_transform(df)
df_numeric_scaled=df
#df_numeric_scaled = pd.DataFrame(minmax_processed, index=df.index, columns=df.columns)
kmeans = KMeans(n_clusters=3)
kmeans.fit(df_numeric_scaled)
df_numeric_scaled.head()
df['cluster'] = kmeans.labels_
cntr=kmeans.cluster_centers_
cntr=cntr/ls
#print(df)
#print(df2)
print(cntr)
print("new line")

df2=ps_csv
df2
#dataset_is_here
df2=df2.drop('orgid',axis=1)
from sklearn import metrics
X=df.drop('cluster',axis=1)

labels=df['cluster']
metrics.silhouette_score(df, labels)
#0.54344 siho

metrics.calinski_harabasz_score(X, labels)
#55.58210910688541 clas
metrics.davies_bouldin_score(X, labels)
#0.6215442639266918
print(labels)
df2['q7']

for i in range(0,20):
    if(df.loc[i,'cluster']==1):
        print( df2.iloc[i,2]  )
        #,",",df2.iloc[i,2]
df=ps_csv
df.dropna(inplace=True)
ls=[19.1538, 15.3462, 2.65385, 1.34615, 1.34615, 1.03846, 1]
df=df.drop('orgid',axis=1)
#df=df*ls
from sklearn import preprocessing
#minmax_processed = preprocessing.MinMaxScaler().fit_transform(df)
df_numeric_scaled=df
#df_numeric_scaled = pd.DataFrame(minmax_processed, index=df.index, columns=df.columns)
kmeans = KMeans(n_clusters=3)
kmeans.fit(df_numeric_scaled)
df_numeric_scaled.head()
df['cluster'] = kmeans.labels_
cntr=kmeans.cluster_centers_
#cntr=cntr/ls
print(cntr)
df

measure={}
measure['0']=metrics.silhouette_score(df_numeric_scaled,df['cluster'], metric='euclidean')

df['total'] =df['q1']+ df['q2']+df['q3']+ df['q4']+df['q5']
res=df.groupby(['cluster']).mean()
res=res.sort_values(by=['total'],ascending = False)
res['cluster']=res.index.values

for x in range(0,df.shape[0]):
    if df.iloc[x,5] == res.iloc[0,6]:
        df.iloc[x,5]=0
    elif df.iloc[x,5] == res.iloc[1,6]:
        df.iloc[x,5]=1
    elif df.iloc[x,5] == res.iloc[2,6]:
        df.iloc[x,5]=2 
print(res)

ls2=ls
ls2.append(1)
ls2.append(1)
res=res/ls2

print(res)
X = df.iloc[:,0:5].values
Y=df['cluster']
from sklearn.preprocessing import StandardScaler
X_std = StandardScaler().fit_transform(X)
from sklearn.decomposition import PCA as sklearnPCA
sklearn_pca = sklearnPCA(n_components=2)
Y_sklearn = sklearn_pca.fit_transform(X_std)
labelname = {0: 'Good',
              1: 'Mediocore',
              2: 'Corrupt'}
with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(9,6))
    for lab, col in zip((0, 1, 2),
                        ('blue', 'green', 'red')):
        plt.scatter(Y_sklearn[Y==lab, 0],
                    Y_sklearn[Y==lab, 1],
                    label=labelname[lab],
                    c=col)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(loc='upper right')
    plt.tight_layout()
    #plt.show()
Y_sklearn
print(res)
for x in range(0,df.shape[0]): 
    print(df.iloc[x,5])
clrs=['blue', 'green', 'red']
plt.figure(figsize=(12,7))
axis = sns.barplot(x=['Good', 'Mediocore', 'Bad'],y=df.groupby(['cluster']).count()['q1'].values,palette=clrs)
x=axis.set_xlabel("Cluster Class")
x=axis.set_ylabel("Number of Employees")

#plt.savefig('../input/foo.png')
dict['orgname']=[]
dict['address']=[]
dict['address'].append("A")
dict['address'].append("B")
dict['address'].append("C")
dict['orgname'].append("1")
dict['orgname'].append("2")
dict['orgname'].append("3")
for x in range(0,3):
  #  print(dict['address'][x])
    print(dict['orgid'][x])
#ques1[val.ins].append(val.id)
dict={}



dft=df
for x in dft:
    print(x)
dft['orgid']=df.index.values
dft
dft=dft.drop('q1',axis=1)
dft=dft.drop('q2',axis=1)
dft=dft.drop('q3',axis=1)
dft=dft.drop('q4',axis=1)
dft=dft.drop('q5',axis=1)
dft=dft.drop('total',axis=1)
dft
dict['orgid']=df['orgid']
dict['orgid'][1]

df2=df2.drop('orgid',axis=1)
df2
train=df2
from sklearn.model_selection import KFold
from sklearn import tree
cv = KFold(n_splits=10,shuffle= True)            # Desired number of Cross Validation folds
accuracies = list()
max_attributes = 7
depth_range = range(1, max_attributes + 1)

# Testing max_depths from 1 to max attributes
# Uncomment prints for details about each Cross Validation pass
for depth in depth_range:
    fold_accuracy = []
    tree_model = tree.DecisionTreeClassifier(max_depth = depth)
    # print("Current max depth: ", depth, "\n")
    for train_fold, valid_fold in cv.split(train):
        f_train = train.loc[train_fold] # Extract train data with cv indices
        f_valid = train.loc[valid_fold] # Extract valid data with cv indices

        model = tree_model.fit(X = f_train.drop(['cluster'], axis=1), 
                               y = f_train["cluster"]) # We fit the model with the fold train data
        valid_acc = model.score(X = f_valid.drop(['cluster'], axis=1), 
                                y = f_valid["cluster"])# We calculate accuracy with the fo
        
        fold_accuracy.append(valid_acc)

    avg = sum(fold_accuracy)/len(fold_accuracy)
    accuracies.append(avg)
    # print("Accuracy per fold: ", fold_accuracy, "\n")
    # print("Average accuracy: ", avg)
    # print("\n")
    
# Just to show results conveniently
df = pd.DataFrame({"Max Depth": depth_range, "Average Accuracy": accuracies})
df = df[["Max Depth", "Average Accuracy"]]
print(df.to_string(index=False))

tree_model = tree.DecisionTreeClassifier(max_depth = 7,criterion="entropy",splitter="random")
model = tree_model.fit(X = df2.drop(['cluster'], axis=1), 
                               y = df2["cluster"]) # We fit the model with the fold train data
# Create Decision Tree with max_depth = 3

tree.plot_tree(model) 
