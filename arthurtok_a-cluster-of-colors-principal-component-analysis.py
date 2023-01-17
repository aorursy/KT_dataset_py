# Importing the usual libraries

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt



# Read in the mushroom data into a dataframe called "data" - what a creative name

data = pd.read_csv("../input/mushrooms.csv")

data.head()
# I use a list "color_features" to store the color column names. 

# Not really sure if there is an easier way to do this. Do let me know if there is

color_features = []

for i in data.columns:

    if 'color' in i:

        color_features.append(i)

# create our color dataframe and inspect first 5 rows with head()

data_color = data[color_features]

data_color.head()
from sklearn.preprocessing import LabelEncoder

# List to store all unique categories

ListToEncode = pd.Series(data_color.values.ravel()).unique()

# Use sklearn Labelencoder for transformation

EncodedList = LabelEncoder().fit_transform(ListToEncode)



# Define a dictionary "encodedict" to store our encoding

encodedict = {}

for i in range(0, len(EncodedList)):

    encodedict.update({ListToEncode[i]:EncodedList[i]})



# Finally use dictionary to generate encoded dataframe

for i in range(len(data_color.columns)):

    for j in range(len(data_color['cap-color'].values)):

        data_color.values[j][i] =  encodedict[data_color.values[j][i]]

data_color.head()       

  
data_color.plot(y= 'cap-color', x ='stalk-color-below-ring',kind='hexbin',gridsize=45, sharex=False, colormap='spectral', title='Hexbin of cap-color and stalk-color-below-ring')
data_color.plot(y= 'cap-color', x ='stalk-color-above-ring',kind='hexbin',gridsize=35, sharex=False, colormap='gnuplot', title='Hexbin of cap-color and stalk-color-above-ring')
# correlation matrix using the corr() method

data_corr = data_color.astype(float).corr()  # used the astype() or else I get empty results

data_corr
# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(7, 7))

plt.title('Pearson Correlation of Mushroom Features')

# Draw the heatmap using seaborn

sns.heatmap(data_color.astype(float).corr(),linewidths=0.5,vmax=1.0, square=True, annot=True)
# import the relevant modules

from sklearn.decomposition import PCA

from sklearn.cluster import KMeans
X = data_color.values

# calling sklearn PCA 

pca = PCA(n_components=3)

# fit X and apply the reduction to X 

x_3d = pca.fit_transform(X)



# Let's see how it looks like in 2D - could do a 3D plot as well

plt.figure(figsize = (7,7))

plt.scatter(x_3d[:,0],x_3d[:,1], alpha=0.1)

plt.show()
# Set a 3 KMeans clustering

kmeans = KMeans(n_clusters=3, random_state=0)

# Compute cluster centers and predict cluster indices

X_clustered = kmeans.fit_predict(x_3d)
LABEL_COLOR_MAP = {0 : 'r',

                   1 : 'g',

                   2 : 'b'}



label_color = [LABEL_COLOR_MAP[l] for l in X_clustered]

plt.figure(figsize = (7,7))

plt.scatter(x_3d[:,0],x_3d[:,1], c= label_color, alpha=0.1)

plt.show()