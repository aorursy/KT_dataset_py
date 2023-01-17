import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
from sklearn.model_selection import train_test_split # No test data is available to evaluate. so we are splitting data into train and test dataset.

from sklearn.linear_model import LogisticRegression # classification model
from sklearn.naive_bayes import CategoricalNB # classification model
from sklearn.svm import LinearSVC # classification model
from sklearn.tree import DecisionTreeClassifier # classification model

import xgboost as xgb # classification model

from sklearn.metrics import accuracy_score,classification_report # classification metrics
df=pd.read_csv("../input/ann-churn-modelling/Churn_Modelling.csv",index_col=['CustomerId']).drop('RowNumber',axis=1)
df.head()
df.shape
df.info()
df.describe()
df['Exited'].value_counts(normalize=True)