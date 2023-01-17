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
# import necessary modules
import matplotlib.pyplot as plt

# Load datasets for Maths and portuguese classes
maths = pd.read_csv("/kaggle/input/student-alcohol-consumption/student-mat.csv")
portuguese = pd.read_csv("/kaggle/input/student-alcohol-consumption/student-por.csv")

# Get some first insights of the data
print(maths.columns)
print(maths.describe())
# Take a look at possible correlations between each attribute and the final grades

# Graphical representation
plt.figure(figsize=(20,20))
n = len(maths.columns[:-3])
for i in range(n):
    plt.subplot(n,1,i+1)
    plt.scatter(maths[maths.columns[i]],maths['G3'])
plt.show()

# Matrix of correlations
maths_corr = np.corrcoef(np.transpose(maths))
portuguese_corr = np.corrcoef(np.transpose(portuguese))

print(maths_corr[-1])
print(portuguese_corr[-1])
# Take a closer look at some of the correlations
plt.figure()
plt.scatter(maths['failures'],maths['G3'])
plt.scatter(portuguese['failures'],portuguese['G3'])
plt.legend(['Mathematics','Portuguese'])
plt.xlabel("Failures")
plt.ylabel("Grade")
plt.show()

# This plot shows that students having faced failures in past classes have a tendency not to obtain good grades,
# and this goes stronger with the number of failures experienced

plt.figure()
plt.scatter(maths['Dalc'],maths['G3'])
plt.scatter(portuguese['Dalc'],portuguese['G3'])
plt.legend(['Mathematics','Portuguese'])
plt.xlabel("Weekday drinking score")
plt.ylabel("Grade")
plt.show()

# From this comparison of grades given the weekdays alcohol consumption indicator, one can observe that there is 
# a tendency of grades being lowered by a higher consumption. At least, students consuming a lot during weekdays 
# do not achieve excellent grades

plt.figure()
plt.scatter(maths['absences'],maths['G3'])
plt.scatter(portuguese['absences'],portuguese['G3'])
plt.legend(['Mathematics','Portuguese'])
plt.xlabel("Absences")
plt.ylabel("Grade")
plt.show()

# A robust pattern cannot easily withdrawn from this analysis of grades given the number of absences.
# It looks like absences do not have a real impact on grades, but for more than 30 absences, there are too little
# examples to conclude
# Convert all data to numeric in order to be able to perform PCA analysis
# Do PCA so that one can see which variables influence the most on the final grades
# Build a grade predictor based on other factors
# Think of preprocessing normalize and scale 

# Prepare storage for memory of mapping between categories and category codes
mappings = []

# Convert all non-numeric values to numerics
for dataset in [maths,portuguese]:
    for attribute in dataset.columns:
        if(dataset[attribute].dtype == 'object'):
            # Convert column data type to categorical
            dataset[attribute] = dataset[attribute].astype('category')
            
            # Store mapping in mappings dictionnary
            mappings.append(dict(enumerate(dataset[attribute].cat.categories)))
            
            # Replace categorical values by numerical codes
            dataset[attribute] = dataset[attribute].cat.codes
# Compute PCA anlaysis on the datasets
from sklearn.decomposition import PCA

# Center and scale data
from sklearn.preprocessing import StandardScaler

std_scale_maths = StandardScaler().fit(maths[maths.columns[:-3]])
maths_scaled = std_scale_maths.transform(maths[maths.columns[:-3]])

std_scale_portuguese = StandardScaler().fit(portuguese[portuguese.columns[:-3]])
portuguese_scaled = std_scale_portuguese.transform(portuguese[portuguese.columns[:-3]])

# Treat targets separately
targets_maths = maths[maths.columns[-3:]]/20
targets_portuguese = portuguese[portuguese.columns[-3:]]/20

# Do PCA analysis
pca_maths = PCA()
pca_maths.fit(maths_scaled)
print(pca_maths.explained_variance_ratio_)
print(pca_maths.singular_values_)

pca_portuguese = PCA()
pca_portuguese.fit(portuguese_scaled)
print(pca_portuguese.explained_variance_ratio_)
print(pca_portuguese.singular_values_)

# One can withdraw from this simple analysis that there is no clear direction where information is concentrate
# Also, most of the variables have similar influence on the grades.
# Grade prediction
# Let's build a deep learning model that predicts final grade based on all the given attributes

# Create train and test sets
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(maths_scaled,np.array(targets_maths[targets_maths.columns[-1]]),test_size=0.2)

# Import deep learning library
from keras.models import Sequential
from keras.layers.core import Activation, Dense
from keras.optimizers import SGD, Adam, RMSprop

# Build our model
model = Sequential()
 
# Declare the layers
layers = [Dense(units=128, input_dim=30), Activation('relu'),
          Dense(units=64, input_dim=30), Activation('relu'),
          Dense(units=32, input_dim=30), Activation('relu'),
         Dense(units=1, input_dim=30), Activation('sigmoid')]
 
# Add the layers to the model
for layer in layers:
    model.add(layer)

# Configure an optimizer used to minimize the loss function
sgd = SGD(learning_rate=0.1)

# Compile our model
model.compile(loss='mean_squared_error', optimizer=sgd)
 
# Fit the model
history = model.fit(x_train, y_train, validation_split=0.2, epochs=50)
# try some prediction
print(x_test[0])
print(y_test[0])
print(model.predict(x_test)[0])