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
import pandas as pd

import seaborn as sns

import numpy as np

import matplotlib.pyplot as plt

df = pd.read_csv("/kaggle/input/fashionmnist/fashion-mnist_train.csv")

df
df.shape
# STORING the label column in l variable

l = df["label"]

l
d = df.drop("label",axis=1)

d
plt.figure(figsize=(10,10))

idx = 102

grid_shape = d.iloc[idx].as_matrix().reshape(28,28)

plt.imshow(grid_shape,cmap="gray")

plt.show()



print(l[idx])
df1 = pd.read_csv("../input/fashionmnist/fashion-mnist_test.csv")

df1
df1.shape
# Storing the label column in variable l

l = df1["label"]

l

# Storing the pixcel in d 

d = df1.drop("label",axis=1)

d
print(d.shape)

print(l.shape)
# Defining the size in which we want the image to be displayed

plt.figure(figsize=(10,10))

#idx=int(input("Enter the index : "))

idx=100

# Storing the pixcel into grid_shape where first we select the row of the index provided by user thenconverting it to a matrix and reshaping it to 28*28 

grid_shape = d.iloc[idx].as_matrix().reshape(28,28)

plt.imshow(grid_shape,cmap="RdBu")

plt.show()



print(l[idx])
# Defining the size in which we want the image to be displayed

plt.figure(figsize=(10,10))

#idx=int(input("Enter the index : "))

idx=1000

# Storing the pixcel into grid_shape where first we select the row of the index provided by user thenconverting it to a matrix and reshaping it to 28*28 

grid_shape = d.iloc[idx].as_matrix().reshape(28,28)

plt.imshow(grid_shape,cmap="rocket")

plt.show()



print(l[idx])
# Defining the size in which we want the image to be displayed

plt.figure(figsize=(10,10));

#idx=int(input("Enter the index : "));

idx=20

# Storing the pixcel into grid_shape where first we select the row of the index provided by user thenconverting it to a matrix and reshaping it to 28*28 

grid_shape = d.iloc[idx].as_matrix().reshape(28,28);

plt.imshow(grid_shape,cmap="afmhot");

plt.show();



print(l[idx])
# Defining the size in which we want the image to be displayed

plt.figure(figsize=(10,10));

#idx=int(input("Enter the index : "));

idx=1010

# Storing the pixcel into grid_shape where first we select the row of the index provided by user thenconverting it to a matrix and reshaping it to 28*28 

grid_shape = d.iloc[idx].as_matrix().reshape(28,28);

plt.imshow(grid_shape,cmap="coolwarm");

plt.show();



print(l[idx])
# Defining the size in which we want the image to be displayed

plt.figure(figsize=(10,10));

#idx=int(input("Enter the index : "));

idx=333

# Storing the pixcel into grid_shape where first we select the row of the index provided by user thenconverting it to a matrix and reshaping it to 28*28 

grid_shape = d.iloc[idx].as_matrix().reshape(28,28);

plt.imshow(grid_shape,cmap="Blues");

plt.show();



print(l[idx])
# Defining the size in which we want the image to be displayed

plt.figure(figsize=(10,10));

#idx=int(input("Enter the index : "));

idx = 5001

# Storing the pixcel into grid_shape where first we select the row of the index provided by user thenconverting it to a matrix and reshaping it to 28*28 

grid_shape = d.iloc[idx].as_matrix().reshape(28,28);

plt.imshow(grid_shape,cmap="bone");

plt.show();



print(l[idx])
# Defining the size in which we want the image to be displayed

plt.figure(figsize=(10,10));

#idx=int(input("Enter the index : "));

idx=932

# Storing the pixcel into grid_shape where first we select the row of the index provided by user thenconverting it to a matrix and reshaping it to 28*28 

grid_shape = d.iloc[idx].as_matrix().reshape(28,28);

plt.imshow(grid_shape,cmap="icefire");

plt.show();



print(l[idx])


# Defining the size in which we want the image to be displayed

plt.figure(figsize=(10,10));

#idx=int(input("Enter the index : "));

idx=900

# Storing the pixcel into grid_shape where first we select the row of the index provided by user thenconverting it to a matrix and reshaping it to 28*28 

grid_shape = d.iloc[idx].as_matrix().reshape(28,28);

plt.imshow(grid_shape,cmap="Oranges");

plt.show();



print(l[idx])


# Defining the size in which we want the image to be displayed

plt.figure(figsize=(10,10));

#idx=int(input("Enter the index : "));

idx=123

# Storing the pixcel into grid_shape where first we select the row of the index provided by user thenconverting it to a matrix and reshaping it to 28*28 

grid_shape = d.iloc[idx].as_matrix().reshape(28,28);

plt.imshow(grid_shape,cmap="gist_ncar");

plt.show();



print(l[idx])


# Defining the size in which we want the image to be displayed

plt.figure(figsize=(10,10));

#idx=int(input("Enter the index : "));

idx=455

# Storing the pixcel into grid_shape where first we select the row of the index provided by user thenconverting it to a matrix and reshaping it to 28*28 

grid_shape = d.iloc[idx].as_matrix().reshape(28,28);

plt.imshow(grid_shape,cmap="viridis");

plt.show();



print(l[idx])
### PCA for data visualization

# To use pca we require dataset to be standardized



df
data = df.drop("label",axis=1)

data
label = df["label"]

label
# Standardizing the data

from sklearn.preprocessing import StandardScaler

standardized_data = StandardScaler().fit_transform(data)

standardized_data
# PCA 

from sklearn.decomposition import PCA

pca = PCA()

pca.n_components = 2

pca_data = pca.fit_transform(standardized_data)

pca_data
# Converting label series to a dataframe

a = label.to_frame(name="label")

a
b = pd.DataFrame(pca_data,columns=("1st principal","2nd principal"))

b
df1 = pd.concat([a,b],axis=1)

df1
# Visualizing the dataframe made using seaborn 

sns.FacetGrid(df1,hue="label",size=10).map(plt.scatter,"1st principal","2nd principal").add_legend();
# Visualization using TSNE 

# As in tsne computes neighbourhood of each point with other and then creates clusters so doing this computation takes a lot of time 

# therefore we will work only on first 5000 points

standardized_data_5000 = standardized_data[:5000,:]

len(standardized_data_5000)
label = label[:5000]

label
#import tsne

from sklearn.manifold import TSNE

model = TSNE(random_state=0)

model.n_component = 2

### Configuring the parameters

# the number of components = 2

# the default perplexity= 30

# default learning rate = 200

# default maximum number of iterations for optimization = 1000

tsne_data = model.fit_transform(standardized_data_5000)
tsne_data
### Converting label to dataframe

a = pd.DataFrame(tsne_data,columns=("1st dim","2nd dim"))

a

b = label.to_frame(name="label")

b
df1 = pd.concat([a,b],axis=1)

df1
plt.figure(figsize=(5,5))

# Visualizing the dataframe made using seaborn 

sns.FacetGrid(df1,hue="label",size=10).map(plt.scatter,"1st dim","2nd dim").add_legend();

model = TSNE(random_state=0,perplexity=100,n_iter=5000)

model.n_component = 2

### Configuring the parameters

# the number of components = 2

# the default perplexity= 30

# default learning rate = 200

# default maximum number of iterations for optimization = 1000

tsne_data = model.fit_transform(standardized_data_5000)
tsne_data
### Converting label to dataframe

a = pd.DataFrame(tsne_data,columns=("1st dim","2nd dim"))

a

b = label.to_frame(name="label")

b
df1 = pd.concat([a,b],axis=1)

df1
plt.figure(figsize=(5,5))

# Visualizing the dataframe made using seaborn 

sns.FacetGrid(df1,hue="label",size=10).map(plt.scatter,"1st dim","2nd dim").add_legend();

model = TSNE(random_state=0,perplexity=50,n_iter=5000)

model.n_component = 2

### Configuring the parameters

# the number of components = 2

# the default perplexity= 30

# default learning rate = 200

# default maximum number of iterations for optimization = 1000

tsne_data = model.fit_transform(standardized_data_5000)
tsne_data
### Converting label to dataframe

a = pd.DataFrame(tsne_data,columns=("1st dim","2nd dim"))

a

### Converting label to dataframe

a = pd.DataFrame(tsne_data,columns=("1st dim","2nd dim"))

a

df1 = pd.concat([a,b],axis=1)

df1
plt.figure(figsize=(5,5))

# Visualizing the dataframe made using seaborn 

sns.FacetGrid(df1,hue="label",size=10).map(plt.scatter,"1st dim","2nd dim").add_legend();
