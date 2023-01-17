import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
# get input data from Data section of Kaggle
#mnist = fetch_openml('mnist_784')

train = pd.read_csv('../input/ucsc-cse142-project-4-mnist/training_x_1k.csv')
test = pd.read_csv('../input/ucsc-cse142-project-4-mnist/test_x.csv')
val = pd.read_csv('../input/ucsc-cse142-project-4-mnist/val_data.csv')

train = np.array(train)
test = np.array(test)
val = np.array(val)

# check shape (should be (9999, 784)) seems to be loading data correctly
print(train.shape)
val.shape
# removing label column from validation data
val_fixed = val[:,:-1]
val_fixed.shape

# making array of validation labels
valLabels = []
for x in val:
    print(x[784])
    lbl = x[784]
    valLabels.append(lbl)
    
    
# trying to plot
for x in val_fixed:
    print(x)
    # The columns are pixels
    #pixels = x
        
    # This array will be of 1D with length 784
    pixels = np.array(x)

    # Reshape the array into 28 x 28
    pixels = pixels.reshape((28, 28))

    # Plot
    plt.title('Label is {label}'.format(label="xxyyzz"))
    plt.imshow(pixels, cmap='gray')
    plt.show()

    break # test, just one
# Scale the data!
scaler = StandardScaler()

# Fit on training set
scaler.fit(train)

# transform both the training and test sets
train_scaled = scaler.transform(train)
test_scaled = scaler.transform(test)
val_scaled = scaler.transform(val_fixed)
# PCA (retains the minimum number of components to maintain 0.95Variance)
pca = PCA(0.98)

# fit PCA
pca.fit(train_scaled)

# transform train/test imgs
train_pca = pca.transform(train_scaled)
test_pca = pca.transform(test_scaled)
val_pca = pca.transform(val_scaled)

# check the reduced number of features in train set
print(pca.n_components_)
# KMeans
kmeans = KMeans(n_clusters=10)
# fit
kmeans.fit(train_pca)
# predict
y_pred = kmeans.predict(val_pca)

# code from Joseph to assess accuracy 

labels = kmeans.labels_
realLabels = valLabels

groups = [] #indices for each cluster
for i in range(10):
    groups.append([])
    
for i,l in enumerate(labels):
    groups[l].append(i)
    
groupLabels = []
for i in range(10):
    groupLabels.append([])
    
for g,group in enumerate(groups):
    for i in group:
        groupLabels[g].append(int(realLabels[i]))
        
distro = []
for i in range(10):
    distro.append([0,0,0,0,0,0,0,0,0,0])
    
for g,group in enumerate(groupLabels):
    #print(g)
    for l in group:
        distro[g][l] = distro[g][l]+1

print(distro)