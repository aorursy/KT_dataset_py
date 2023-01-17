# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#importing data set

from sklearn import datasets
#Load dataset

cancer = datasets.load_breast_cancer()
cancer
print("Features: ", cancer.feature_names)
print("Labels: ", cancer.target_names)
print(cancer.target)
# Import train_test_split function

from sklearn.model_selection import train_test_split



# Split dataset into training set and test set

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.3,random_state=1)


from sklearn import svm



#Create a svm Classifier

clf = svm.SVC(kernel='linear') # Linear Kernel



#Train the model using the training sets

clf.fit(X_train, y_train)



#Predict the response for test dataset

y_pred = clf.predict(X_test)
from sklearn import metrics



# Model Accuracy: how often is the classifier correct?

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))