# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
dataset = pd.read_csv('../input/HeartDisease.csv')


# Any results you write to the current directory are saved xas output.
#import all the libraries
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

#Step1: EDA Exploratory Data Analysis

#Check columns and rows in the data
print(dataset.shape)
#Check data type of data
print(dataset.info())

#See top 5 rows
print(dataset.head(15))
#Check bottom 5 rows
print(dataset.tail())
#Check sample of any 5 rows
print(dataset.sample(5))

#Print the main matrices for the data
print(dataset.describe())

# Get the number of missing data points per column. This will show up in variable explorer
missing_values_count = dataset.isnull().sum()
print(missing_values_count)

#Print unique values of all the columns
print(dataset.Age.unique())

#Plot only the values of num- the value to be predicted/Label
dataset["num"].value_counts().sort_index().plot.bar()

#Heat map to see the coreelation between variables, use annot if you want to see the values in the heatmap
plt.subplots(figsize=(12,8))
sns.heatmap(dataset.corr(),robust=True,annot=True)
#CONCLUSION: Ignoring ID since it was manually added
#Positive correlation: num vs cp, exang vs cp,num vs exang, old peak vs exang,
#Negative orelation:Age vs thalach,cp vs thalach,exang vs thalach,num vs thalach

#custom correlogram
#sns.pairplot(dataset, hue="Age")

#Histogram for all features
dataset.hist(figsize=(15,12),bins = 20, color="#007959AA")
plt.title("Features Distribution")
plt.show()

#Detect outliers
plt.subplots(figsize=(15,6))
dataset.boxplot(patch_artist=True, sym="k.")
#plt.xticks(rotation=90)
#Step 2: Defining X,y, train and test 
#Defining X and y
X = dataset.iloc[:, 1:11].values
y = dataset.iloc[:, -2].values

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#Taking care of nans
from sklearn.preprocessing import Imputer
imputer= Imputer(strategy='mean')
imputer = imputer.fit(X[:,4:11])
X[:,4:11]= imputer.transform(X[:,4:11])


# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense

# Step 3: Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 10))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)