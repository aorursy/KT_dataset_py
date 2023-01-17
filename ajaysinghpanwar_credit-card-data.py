

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

dataset = pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')
# Let's check out the data

dataset.head()
dataset.info()
# Create independent and dependent features

X = dataset.drop('Class', axis = 1)

y = dataset['Class']
print(X.shape)

print(y.shape)
# Let's check for null values

dataset.isnull().values.any()
# Plot the count of imbalanced classes 

class_counts = pd.value_counts(dataset['Class'])



class_counts.plot.bar(rot = 0)   ## This rot = 0 stops the rotation of the xticks



plt.title("Transaction Class Distribution")

plt.xticks(range(2), ['Normal','Fraud'])

plt.xlabel("Class")

plt.ylabel("Frequency")
# Get the fraud and normal data

fraud = dataset[dataset['Class'] == 0]

normal = dataset[dataset['Class'] == 1]
print(fraud.shape)

print(normal.shape)
# Import the library to do undersampling

from imblearn.under_sampling import NearMiss 
# Implementing undersampling for handling imbalanced classes

nm = NearMiss()

X_res,y_res=nm.fit_sample(X,y)
print(X_res.shape)

print(y_res.shape)
from collections import Counter

print(f"Original data shape : {Counter(y)}")

print(f"Resampled data shape : {Counter(y_res)} ")
from imblearn.combine import SMOTETomek
# Implementing oversampling to handle imbalanced classes

smk = SMOTETomek(random_state = 42)

X_res,y_res=smk.fit_sample(X,y)
print(X_res.shape)

print(y_res.shape)
print(f"Original data shape : {Counter(y)}")

print(f"Resampled data shape : {Counter(y_res)} ")