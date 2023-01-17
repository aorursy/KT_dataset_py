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
# We will be reading train dataset to perform PCA.

# For reading datframe 'train' , we will be using pandas library and function 'read_csv' as our file is in csv format

train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

print('data train is available')
# shape of the data - this give us number of rows and columns in the dataframe

print('shape of the data is {} rows and {} columns'.format(train.shape[0],train.shape[1]))
# we should take a look at complete dataset... so as our dataset have large number of rows and columns. we will be looking at only top 5 rows

# for looking only top 5 rows, we will be using train.head()

print(train.head())
# getting more info about dataframe..

print(train['label'].value_counts()) # value_counts give us count of different labels in particular column, here we use it on our 'label' column of train dataset

# it always better to see things in graphs and plot...

# for plot, we will be using seaborn package

import seaborn as sns

import matplotlib.pyplot as plt

label_count = train['label'].value_counts()

plt.figure(figsize=(10,5))

sns.barplot(label_count.index, label_count.values, alpha=0.8)

plt.title('Count of different labels')

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('Digits', fontsize=12)

plt.show()
# 1. droping labels from the train dataset and but we need to save them to use later

label = train['label']

train.drop('label',axis=1,inplace=True)

print(train.shape) 
# 2. We should standardize the datset 'train', we will use StandardScaler from sklearn.preprocessing 



from sklearn.preprocessing import StandardScaler # So what standardscaler do? Ans - Details given below

standardized_train =  StandardScaler().fit_transform(train)

print(standardized_train.shape)
# Co-variance matrix of A is A*A^T i.e A is multiplied by transpose of A

sample_data = standardized_train



# matrix multiplication using numpy

covar_matrix = np.matmul(sample_data.T,sample_data) 



print('the shape of co-variance matrix is', covar_matrix.shape)
from scipy.linalg import eigh



# the parameter 'eighvals' is defined (low to high value)

# eigh function will return the eigen value in the ascending order

# this code will generate only top two (782,783) eigen value



values,vectors = eigh(covar_matrix,eigvals= (782,783))



print('shape of eigen vector', vectors.shape)



# coverting eigen vector into (2,d) shape



vectors = vectors.T



print('shape of updated eigen vector', vectors.shape)
# projecting the original data frame on the plane formed by two principal eigen vectors by vector-vector multiplication\



new_coordinates = np.matmul(vectors,sample_data.T)

print('resultant matrix by multiplication of matrix vector having shape of ',vectors.shape,' and co-variance matrix having shape ', sample_data.shape ,' is new_coordinates matrix having shape ',new_coordinates.shape )
# appending labels to 2D projected data

new_coordinates = np.vstack((new_coordinates,label)).T



# creating the new dataframe for ploting the labeled points.

dataframe = pd.DataFrame(data=new_coordinates,columns = ('1stPrincipal','2ndPrincipal','label'))

print(dataframe.head())

# Ploting the 2D data point with seaborn



sns.FacetGrid(dataframe, hue="label",height=6).map(plt.scatter,'1stPrincipal','2ndPrincipal').add_legend()

plt.show()
# initailization of PCA

from sklearn import decomposition

pca = decomposition.PCA()
# configuring the parameters

pca.n_components = 2

pca_data = pca.fit_transform(sample_data)



# pca_data will contain th 2-d projection of sample_data

print('shape of pca_data is ', pca_data.shape)
# attaching the labels with pca_data



pca_data = np.vstack((pca_data.T,label)).T



# creating the new dataframe for ploting the labeled points.

dataframe2 = pd.DataFrame(data=pca_data,columns = ('1stPrincipal','2ndPrincipal','label'))

# Ploting the 2D data point with seaborn



sns.FacetGrid(dataframe2, hue="label",height=6).map(plt.scatter,'1stPrincipal','2ndPrincipal').add_legend()

plt.show()