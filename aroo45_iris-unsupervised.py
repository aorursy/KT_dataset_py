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



# Any results you write to the current directory are saved as output.
# Load the data as a pandas dataframe

df = pd.read_csv("/kaggle/input/iris/Iris.csv")



# Drop the "Species" column, this will be an unsupervised excercise. Also drop Id, as this is useless information, just an index representing row position.

features = df.drop(["Species","Id"], axis = 1)



# Print and verify Species, Id are dropped.

print(features)
# Summarize each feature column.

df.describe()
# You can plot a histogram of any column like below:

df["SepalLengthCm"].hist()
# Import plotting libraries, PCA

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

pca = PCA(n_components = 3)
# Fit the PCA model

pca.fit(features)
# Transform the features using the fit PCA model (will end up with 3 pc's)

features_red = pca.transform(features)
# Features now contain 3 columns instead of 4

print(features_red[:5])
# Put the first two principal components in an X and Y list for plotting

xvals=[]

yvals=[]

for entry in features_red:

    xvals.append(entry[0])

    yvals.append(entry[1])

    

plt.plot(xvals,yvals,'o')
# Now lets apply kmeans clustering to the dataset.

from sklearn.cluster import KMeans

model = KMeans(n_clusters=3)



#Fit the model and predict labels for each row of data.

model.fit(features_red)

labels = model.labels_

# Labels are the output of the clustering model. Numeric integer is the cluster category.

print(labels)
# Create a color map for plotting. 

cmap = {}

cmap[0]="ro"

cmap[1]="bo"

cmap[2]="go"

cmap[3]="co"
# Create a colors list, where each index corresponds to the cluster color of that data point

colors = [cmap[x] for x in labels]
#Our clustering result plotted in 2D

for x,y,c in zip(xvals,yvals,colors):

    plt.plot(x,y,c)

plt.show()
# Let's compare our result to the result using the labels we dropped as clusters instead.

true_labels = df["Species"]

print(set(list(true_labels)))

cmap["Iris-setosa"]="ro"

cmap["Iris-virginica"]="bo"

cmap["Iris-versicolor"]="go"

true_colors = [cmap[x] for x in true_labels]

for x,y,c in zip(xvals,yvals,true_colors):

    plt.plot(x,y,c)

plt.show()
