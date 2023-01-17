# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

import numpy as np

import pandas as pd

from sklearn.metrics import accuracy_score

from sklearn.mixture import GaussianMixture

from sklearn.metrics import f1_score

from time import time

import matplotlib.pyplot as plt



def get_data(Path='../input/data.csv'):

    # load the breast_cancer dataset from csv

    BC_Data = pd.read_csv(Path, skiprows=1, header=None)

    # the second column is the types of Cancer (categorical--M/B)

    BC_Type = BC_Data[1].unique()

    # replace 'M' with 0, replace 'B' with 1

    for i, type in enumerate(BC_Type):

        # DataFrame.set_value(index, col, value, takeable=False)

        BC_Data.set_value(BC_Data[1] == type, 1, i)

    # split the features and labels

    # numpy.split(ary, indices_or_sections, axis=0)

    Y, X = np.split(BC_Data.values, (2,), axis=1)

    # set the features to float, set the labels to int

    X = X.astype(np.float)

    Y = Y.astype(np.int)

    # drop the 'id' column, since it is useless for analyzing

    Y=Y[:, 1]

    # return the features X and labels Y

    return X, Y



# replace 0 with 1, and replace 1 with 0, for comparing the true labels and the clustered labels in clustering algorithm

def data_reverve(pred):

    predict = []

    if 1.0 * np.sum(pred) / len(pred) < 0.5:

        for i, label in enumerate(pred):

            if pred[i] == 0:

                n = 1

            else:

                n = 0

            predict.append(n)

    predict = np.array(predict)

    return predict



# to visualize the scatter plot of the data in different color

def scatter_vis2(X1, X2, n):

    name=['Actual Groups after PCA', 'Predictive Clustering after PCA']

    plt.figure()

    plt.scatter(X1[:, 0], X1[:, 1], color='r', marker='^')    # red stands for 'M'

    plt.scatter(X2[:, 0], X2[:, 1], color='b', marker='s')    # blue stands for 'B'

    plt.xlabel('PC1')

    plt.ylabel('PC2')

    plt.title(name[n])

    plt.show()



# to visualize the comparison of the centers of clusters

def center_comp_vis(cluster_center):

    plt.bar(np.arange(1, len(cluster_center[0]) + 1), cluster_center[0], 0.5, color='b')

    plt.bar(np.arange(1.5, len(cluster_center[1]) + 1), cluster_center[1], 0.5, color='r')

    plt.xlabel('features')

    plt.ylabel('feature centers')

    plt.title('feature centers presentation')

    plt.show()

        

# perform unsupervised learning on data set(Gaussian Mixture)

if __name__=="__main__":



    time_results = {}  # for calculate the time efficiency



    X, Y = get_data()



    # capture the index of label 0 ('M') and label 1 ('B')

    cluster_M = []

    cluster_B = []

    for i, label in enumerate(Y):

        if label == 0:

            cluster_M.append(i)

        if label == 1:

            cluster_B.append(i)



    # visualize the actual data grouping

    scatter_vis2(X[cluster_M], X[cluster_B], 0)



    # data training

    start1 = time()

    EM = GaussianMixture(n_components=2, random_state=0).fit(X)    # use 2 clusters

    end1=time()

    time_results['training time'] = end1 - start1



    # make prediction

    start2=time()

    pred = EM.predict(X)

    end2=time()

    time_results['prediction time'] = end2 - start2



    predict=data_reverve(pred)    # make the labels easy to compare



    # capture the index of label predicted as 0 ('M') and label predicted as 1 ('B')

    cluster_M_pre=[]

    cluster_B_pre=[]

    for i, label in enumerate(predict):

        if label==0:

            cluster_M_pre.append(i)

        if label==1:

            cluster_B_pre.append(i)



    # visualize the result of clustering

    scatter_vis2(X[cluster_M_pre], X[cluster_B_pre], 1)



    # compute the matrics

    cluster_accuracy= accuracy_score(Y, predict)

    cluster_fscore=f1_score(Y, predict)



    cluster_center= EM.means_    # center of each cluster



    print (cluster_accuracy)

    print (cluster_fscore)

    print (time_results)



    # visualization

    center_comp_vis(cluster_center)