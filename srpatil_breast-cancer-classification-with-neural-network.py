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
import tensorflow as tf

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns # data visualization library 
from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import train_test_split



breast_cancer = load_breast_cancer()

class_names = list(breast_cancer.target_names)

label_dict ={0:"malignant", 1:"benign"}  # labels 0 and 1 correspond to class names malignant and benign



X, y = breast_cancer.data, breast_cancer.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
target_values, counts = np.unique(y_train, return_counts=True) # returns target values 0 and 1

diagnosis_labels=map(lambda x:label_dict[x],target_values)# map target value to diagnosis labels



plt.bar(list(diagnosis_labels),counts)  

plt.show()
''' 

Attribute Information:



1) ID number 

2) Diagnosis (M = malignant, B = benign) 



3-32) Ten real-valued features are computed for each cell nucleus: 



a) radius (mean of distances from center to points on the perimeter) 

b) texture (standard deviation of gray-scale values) 

c) perimeter 

d) area 

e) smoothness (local variation in radius lengths) 

f) compactness (perimeter^2 / area - 1.0) 

g) concavity (severity of concave portions of the contour) 

h) concave points (number of concave portions of the contour) 

i) symmetry 

j) fractal dimension ("coastline approximation" - 1)



The mean, standard error and "worst" or largest (mean of the three largest values) of these features were computed 

for each image, resulting in 30 features. For instance, field 3 is Mean Radius, field 13 is Radius SE, field 23 is 

Worst Radius.

'''



import pandas as pd



print(breast_cancer.keys())

print(breast_cancer.feature_names)



# String convertor to convert scientific numbers to strings to read and understand the data better



example_datapoint=pd.Series(X_train[60]).apply(lambda x: '%.3f' % x)

print(example_datapoint)
''' 

In the data-set there are high possibilities for some features to be correlated. PCA is essentially a method that 

reduces the dimension of the feature space in such a way that new variables are orthogonal to each other (i.e. 

they are independent or not correlated). 



From the cancer data-set we see that it has 30 features, so let’s reduce it to only 3 principal features and then 

we can visualize the scatter plot of these new independent variables. Before applying PCA, we scale our data such 

that each feature has unit variance. This is necessary because fitting algorithms highly depend on the scaling of 

the features.

'''

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA



scaler = StandardScaler() #Instantiate the scaler



scaler.fit(X_train) # Compute the mean and standard which will be used in the next command

X_train_scaled=scaler.transform(X_train)# fit and transform can be applied together and I leave that for simple exercise



scaler.fit(X_test)

X_test_scaled=scaler.transform(X_test)



# we can check the minimum and maximum of the scaled features which we expect to be 0 and 1

print("After scaling minimum", X_train_scaled.min(axis=0))



pca=PCA(n_components=3)

pca.fit(X_train_scaled)



X_train_pca=pca.transform(X_train_scaled)

X_test_pca=pca.transform(X_test_scaled) 



#let's check the shape of X_pca array

print("shape of X_train_pca", X_train_pca.shape)

print("shape of X_test_pca", X_test_pca.shape)



ex_variance=np.var(X_train_pca,axis=0)

ex_variance_ratio = ex_variance/np.sum(ex_variance)

# Here we see that first 2 components contributes to 86% of the total variance

print(ex_variance_ratio)
'''

Now, since the PCA components are orthogonal to each other and they are not correlated, we can expect to see 

malignant and benign classes as distinct. We plot the malignant and benign classes based on the 

first two principal components.

'''

Xax=X_train_pca[:,0]

Yax=X_train_pca[:,1]

labels=y_train

cdict={0:'red',1:'green'}

labl={0:'Malignant',1:'Benign'}

marker={0:'*',1:'o'}

alpha={0:.3, 1:.5}

fig,ax=plt.subplots(figsize=(7,5))

fig.patch.set_facecolor('white')

for l in np.unique(labels):

    ix=np.where(labels==l)

    ax.scatter(Xax[ix],Yax[ix],c=cdict[l],s=40,

           label=labl[l],marker=marker[l],alpha=alpha[l])

# for loop ends

plt.xlabel("First Principal Component",fontsize=14)

plt.ylabel("Second Principal Component",fontsize=14)

plt.legend()

plt.show()

# please check the scatter plot of the remaining component and you will understand the difference
model = tf.keras.Sequential([

    tf.keras.layers.Dense(units = 5, activation = 'relu', input_dim=3),

    tf.keras.layers.Dense(units = 3, activation = 'relu'),

    tf.keras.layers.Dense(units = 1, activation = 'sigmoid')

])



model.compile(optimizer='adam', 

              loss='mean_squared_error', 

              metrics=['accuracy'])



model.fit(X_train_pca,y_train, batch_size = 10, epochs = 100)
'''

With a simple randomization of target values, we get a test accuracy of 46% 

With a 3-Layered Neural Network, we get a test accuracy of 67%.

With the same Neural Network after Dimensionality Reduction using PCA we get a test accuracy of 92%

'''



import random

from sklearn.metrics import accuracy_score



def Rand(start, end, num): 

    res = [] 

    for j in range(num): 

        res.append(random.randint(start, end)) 

    return res 



my_randoms = Rand(0,1,len(y_test))

print(accuracy_score(y_test, my_randoms))

results = model.evaluate(X_test_pca, y_test)

print('test loss, test acc:', results)