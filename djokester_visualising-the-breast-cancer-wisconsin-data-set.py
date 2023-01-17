# Importing Statements

%matplotlib inline

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn import preprocessing

import numpy as np
# read the file

data = pd.read_csv('../input/data.csv')
# Checking the data set

data.head()
# Cleaning and modifying the data

data = data.drop('id',axis=1)

data = data.drop('Unnamed: 32',axis=1)

# Mapping Benign to 0 and Malignant to 1 

data['diagnosis'] = data['diagnosis'].map({'M':1,'B':0})

#Check the data stats

data.describe()
# Scaling the dataset

datas = pd.DataFrame(preprocessing.scale(data.iloc[:,1:32]))

datas.columns = list(data.iloc[:,1:32].columns)

datas['diagnosis'] = data['diagnosis']

datas.head()
#draw a heatmap between mean features and diagnosis

features_mean = ['radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean', 'compactness_mean', 'concavity_mean','concave points_mean', 'symmetry_mean', 'fractal_dimension_mean']

plt.figure(figsize=(15,15))

heat = sns.heatmap(datas[features_mean].corr(), vmax=1, square=True, annot=True)
# Splitting the dataset into malignant and benign.

dataMalignant=datas[datas['diagnosis'] ==1]

dataBenign=datas[datas['diagnosis'] ==0]



#Plotting these features as a histogram

fig, axes = plt.subplots(nrows=10, ncols=1, figsize=(15,60))

for idx,ax in enumerate(axes):

    ax.figure

    binwidth= (max(datas[features_mean[idx]]) - min(datas[features_mean[idx]]))/250

    ax.hist([dataMalignant[features_mean[idx]],dataBenign[features_mean[idx]]], bins=np.arange(min(datas[features_mean[idx]]), max(datas[features_mean[idx]]) + binwidth, binwidth) , alpha=0.5,stacked=True, normed = True, label=['M','B'],color=['r','g'])

    ax.legend(loc='upper right')

    ax.set_title(features_mean[idx])

plt.show()
# Seaborn's Stripplot 

data_drop = datas.drop('diagnosis',axis=1)

data_drop = data_drop[features_mean]

for index,columns in enumerate(data_drop):

    plt.figure(index)

    plt.figure(figsize=(15,15))

    sns.stripplot(x='diagnosis', y= columns, data= datas, jitter=True, palette = 'Set1');

    sns.plt.title('Diagnosis vs ' + str(columns))
# Seaborn's Swarmplot 

for index,columns in enumerate(data_drop):

    plt.figure(index)

    plt.figure(figsize=(15,15))

    sns.swarmplot(x='diagnosis', y= columns, data= datas, palette = 'Set1');

    sns.plt.title('Diagnosis vs ' + str(columns))
# Seaborn's jointplot

plt.close('all')

for index,columns in enumerate(data_drop):

    plt.figure(index)

    plt.figure(figsize=(15,15))

    sns.jointplot(x=data_drop[features_mean[len(features_mean)-index-1]], y= datas[columns],  stat_func=None, color="red", edgecolor = 'w', size = 6);
# Principal component analysis and it's Scatter Plot

from sklearn.decomposition import PCA 

X = data_drop.values

pca = PCA(n_components=2) #Binary Classifier

pca = pca.fit_transform(X)

plt.figure(figsize = (9,5))

plt.scatter(pca[:,0],pca[:,1], c = datas['diagnosis'], cmap = "RdBu_r", edgecolor = "Red", alpha=0.35)

plt.colorbar()

plt.title('PCA Scatter Plot')
# Principal component analysis and it's Scatter Plot

from sklearn.manifold import TSNE

tsne = TSNE(verbose=1, perplexity=40, n_iter= 4000)

tsne = tsne.fit_transform(X)

plt.scatter(tsne[:,0],tsne[:,1],  c = datas['diagnosis'], cmap = "winter", edgecolor = "None", alpha=0.35)

plt.title('t-SNE Scatter Plot')