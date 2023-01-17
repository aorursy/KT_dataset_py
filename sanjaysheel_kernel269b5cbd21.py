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
# Load libraries

import pandas as pd



from sklearn.tree import DecisionTreeClassifier 





# Import Decision Tree Classifier



from sklearn.model_selection import train_test_split 





# Import train_test_split function



from sklearn import metrics 





#Import scikit-learn metrics module for accuracy calculation





#import matplotlib

import matplotlib.pyplot as plt





#import seaborn 

import seaborn as sns
df=pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')
df.head()
df.columns
feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI', 'DiabetesPedigreeFunction', 'Age']

x = df[feature_cols] # Features

y = df.Outcome # Target variable
x
y
# Split dataset into training set and test set

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0) # 70% training and 30% test

# Create Decision Tree classifer object

clf = DecisionTreeClassifier()# Train Decision Tree Classifer

clf = clf.fit(X_train,y_train)#Predict the response for test dataset

y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

#  Accuracy: 0.6753246753246753