# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Load the data

data = pd.read_csv(filepath_or_buffer="../input/HR_comma_sep.csv")

data.head(5)
# Let's explore the data: 

# 1. Correlation between data features

sns.set(style="white")

corr = data.corr()

f, ax = plt.subplots(figsize=(6, 5))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(10, 220, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, cmap=cmap, vmax=.5, square=True, xticklabels=True, 

            yticklabels=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
g = sns.FacetGrid(data, row="promotion_last_5years", col="left", size=3.5, aspect=1)

g.map(plt.hist, "satisfaction_level")
outcome = data["left"]

dataset = data.drop(["sales", "salary"], axis=1)



from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler



dataset_std = StandardScaler().fit_transform(dataset)

print(dataset.columns)

n_comp=[]

for n in [i+1 for i in range(8)]:

    pca = PCA(n_components=n)

    pca.fit(dataset_std)

    n_comp.append(sum(pca.explained_variance_ratio_))



# Obtendo a base que melhor descreve "left"

pca = PCA(n_components=8)

pca.fit(dataset_std)

print(pca.components_[6])
from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.cross_validation import train_test_split

from sklearn.metrics import accuracy_score

dataframe_all = data.ix[:,1:8]

x = datamod.ix[:,:-1].values

standard_scaler = StandardScaler()

x_std = standard_scaler.fit_transform(x)



# step 4: get class labels y and then encode it into number 

# get class label data

y = dataframe_all.ix[:,-1].values

# encode the class label

class_labels = np.unique(y)

label_encoder = LabelEncoder()

y = label_encoder.fit_transform(y)



# step 5: split the data into training set and test set

test_percentage = 0.1

x_train, x_test, y_train, y_test = train_test_split(x_std, y, test_size = test_percentage, random_state = 0)



# t-distributed Stochastic Neighbor Embedding (t-SNE) visualization

from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, random_state=0)

x_test_2d = tsne.fit_transform(x_test)



# scatter plot the sample points among 5 classes

markers=('s', 'd', 'o', '^', 'v')

color_map = {0:'red', 1:'blue', 2:'lightgreen', 3:'purple', 4:'cyan'}

plt.figure()

for idx, cl in enumerate(np.unique(y_test)):

    plt.scatter(x=x_test_2d[y_test==cl,0], y=x_test_2d[y_test==cl,1], c=color_map[idx], marker=markers[idx], label=cl)

plt.xlabel('X in t-SNE')

plt.ylabel('Y in t-SNE')

plt.legend(loc='upper left')

plt.title('t-SNE visualization of test data')

plt.show()
