# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Read the csv file 



vehicle_csv=pd.read_csv("/kaggle/input/st4035-2019-assignment-1/sampleSubmisssionFile.csv")

vehicle_test=pd.read_csv("/kaggle/input/st4035-2019-assignment-1/vehicle_test.csv")

vehicle_train=pd.read_csv("/kaggle/input/st4035-2019-assignment-1/vehicle_train.csv")

vehicle_labels=pd.read_csv("/kaggle/input/st4035-2019-assignment-1/vehicle_training_labels.csv")

#Shape of the dataset

print(vehicle_csv.shape)

print(vehicle_labels.shape)

print(vehicle_test.shape)

print(vehicle_train.shape)
#Display the training dataset



vehicle_train.head(5)
#Displaying the columns



vehicle_train.columns
#Data type of the attributes



vehicle_csv.dtypes
#Five point summary of the dataset



vehicle_train.describe().T
vehicle_train.dtypes
#Boxplot to understand spread and outliers

vehicle_train.plot(kind='box', figsize=(25,15))
vehicle_train.hist(figsize=(15,15))
vehicle_train.isnull().sum()
vehicle_train.info()
#Identify outliers and replace them by median

for col_name in vehicle_train.columns[:-1]:

    q1 = vehicle_train[col_name].quantile(0.25)

    q3 = vehicle_train[col_name].quantile(0.75)

    iqr = q3 - q1

    

    low = q1-1.5*iqr

    high = q3+1.5*iqr

    

    vehicle_train.loc[(vehicle_train[col_name] < low) | (vehicle_train[col_name] > high), col_name] = vehicle_train[col_name].median()
vehicle_train.plot(kind='box', figsize=(20,10))
from pandas.plotting import scatter_matrix

scatter_matrix(vehicle_train, alpha=0.2, figsize=(20, 20),diagonal='kde')



#spd = pd.scatter_matrix(vehicle_train, figsize = (20,20), diagonal='kde')
#Create and view the correlation matrix

vehicle_train.corr()
## There are several variables that are highly correlated with each other
#Run PCA and plot to visualise the ideal number of components

from sklearn.decomposition import PCA



pca = PCA().fit(vehicle_train)



plt.plot(np.cumsum(pca.explained_variance_ratio_))





#Based on the plot, we will select 10 components

pca = PCA(n_components=10)

pca.fit(vehicle_train)



#Assign the components to the X variable

vehicle_train = pca.transform(vehicle_train)
## We will use the Naive Bayes & Support Vector Classifiers
from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC
#Grid search to tune model parameters for SVC

from sklearn.model_selection import GridSearchCV



model = SVC()



params = {'C': [0.01, 0.1, 0.5, 1], 'kernel': ['linear', 'rbf']}



model1 = GridSearchCV(model, param_grid=params, verbose=5)



model1.fit(vehicle_train, vehicle_labels)



print("Best Hyper Parameters:\n", model1.best_params_)
#Build the model with the best hyper parameters

from sklearn.cross_validation import cross_val_score



model = SVC(C=0.5, kernel="linear")



scores = cross_val_score(model, X, Y, cv=10)



print(scores)
#Use the Naive Bayes CLassifier with k fold cross validation

model = GaussianNB()



scores = cross_val_score(model, X, Y, cv=10)



print(scores)