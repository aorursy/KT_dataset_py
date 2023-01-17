# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# The following notebook uses Support Vector Machines on the famous Iris dataset.
# This dataset was introduced by the British statistician and biologist Sir Ronald Fisher 
# in his 1936 paper The use of multiple measurements in taxonomic problems

# This dataset is openly available at UCI Machine Learning Repository
#The iris dataset contains measurements for 150 iris flowers from three different species.

#The three classes in the Iris dataset:

#    Iris-setosa (n=50)
#    Iris-versicolor (n=50)
#    Iris-virginica (n=50)

# The four features of the Iris dataset:

#    sepal length in cm
#    sepal width in cm
#    petal length in cm
#    petal width in cm

## Get the data

#**Use seaborn to get the iris data by using: iris = sns.load_dataset('iris') **
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
# Import the dataset using Seaborn library
iris=pd.read_csv('../input/IRIS.csv')
# Checking the dataset
iris.head()
# Creating a pairplot to visualize the similarities and especially difference between the species
sns.pairplot(data=iris, hue='species', palette='Set2')
from sklearn.model_selection import train_test_split
# Separating the independent variables from dependent variables
x=iris.iloc[:,:-1]
y=iris.iloc[:,4]
x_train,x_test, y_train, y_test=train_test_split(x,y,test_size=0.30)
from sklearn.svm import SVC
model=SVC()
model.fit(x_train, y_train)
pred=model.predict(x_test)
# Importing the classification report and confusion matrix
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,pred))
print(classification_report(y_test, pred))