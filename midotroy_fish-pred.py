# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#for visualization 
import matplotlib.pyplot as plt
import seaborn as sns 

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
fish=pd.read_csv('../input/fish-market/Fish.csv')
fish.shape
fish.head()
#to get first 5 row ,if you want last 5 row use (tail())
#if you want to get any number of row write that number in head() Like: head(13) 
fish.info()
#it help to now data type of column and if we hane missing data 
fish.describe()
#get more information about column 
fish['Species'].value_counts()
#count the species of fish we have in data 
#visualization 
plt.figure(figsize=(12,8))
sns.countplot(fish['Species'])
plt.show()
# We can look at an individual feature in Seaborn through a 

plt.figure(figsize=(12,8))
sns.boxplot(x="Species", y="Weight", data=fish)
plt.show()
#we will split our data to dependent and independent
#first dependent data 
X=fish.iloc[:,1:]
X.head(2)
#second independent
# we add more [] to make it 2d array
y=fish[["Species"]]
y.head(2)
#split our data to train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
#there are to many model to train it but we will use two model 
from sklearn.linear_model import LogisticRegression
logistic_classifier = LogisticRegression(random_state =42)
logistic_classifier.fit(X_train,y_train)
logistic_classifier.score(X_test,y_test)
# Support Vector Machine (SVM)
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 42)
classifier.fit(X_train, y_train)
classifier.score(X_test,y_test)