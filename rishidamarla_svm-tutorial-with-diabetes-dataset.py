import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import sklearn

import sklearn.preprocessing

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from sklearn.metrics import f1_score

from sklearn.model_selection import train_test_split

import warnings

warnings.filterwarnings('ignore')

from sklearn import svm
df = pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')
df.head()
df.describe()
# Checking for missing values.

df.isnull().values.any()
zero_not_allowed = ["Glucose","BloodPressure","SkinThickness"]



for column in zero_not_allowed:

    df[column] = df[column].replace(0, np.NaN)

    mean = int(df[column].mean(skipna = True))

    df[column] = df[column].replace(np.NaN, mean)
# Splitting the dataset into training and testing sets.

x = df.iloc[:, :-2]

y = df.iloc[:, -1]

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 0, test_size = 0.2)
# Creating the SVM model.

clf = svm.SVC(kernel='rbf')

clf.fit(x_train,y_train)

y_pred = clf.predict(x_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_pred)