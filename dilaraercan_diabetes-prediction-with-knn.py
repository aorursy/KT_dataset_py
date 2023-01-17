

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
diabetes = pd.read_csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv')
print(diabetes.head())
diabetes.info()
dcorr = diabetes.corr()

print(dcorr)
sns.heatmap(dcorr, xticklabels = dcorr.columns, yticklabels = dcorr.columns)
print(diabetes['Outcome'].value_counts())
sns.catplot(x="Outcome", y="Age", hue="Outcome",

            kind="bar", data=diabetes[['Outcome', 'Age']])
sns.catplot(x="Outcome", y="BMI", hue="Outcome",

            kind="bar", data=diabetes[['Outcome', 'BMI']])
from sklearn.model_selection import train_test_split



X_train, x_test, y_train, y_test = train_test_split(diabetes[['Pregnancies', 'Glucose', 'BMI', 'Age', 'BloodPressure', 'SkinThickness', 'Insulin', 'DiabetesPedigreeFunction']], diabetes['Outcome'], random_state = 0)

print("X_train shape: ", X_train.shape)

print("y_test shape: ", y_test.shape)
diabetes_scatter = pd.DataFrame(X_train)

pd.plotting.scatter_matrix(diabetes_scatter, c = y_train)

#I don't know how to make this prettier (yet)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, y_train) 
X_new = np.array([[6, 150, 15, 30, 70, 30, 100, 2]])

print(knn.predict(X_new))
y_pred = knn.predict(x_test)

occurrences = np.count_nonzero(y_pred == 1)

volume = len(y_pred)

print('Test set predictions: \n', str(y_pred), '\n The model predicted ' + str(occurrences) + ' occurences of diabetes patients out of ' +str(volume)+ '. \n Ratio: '+ str(occurrences/volume))
score_knn = knn.score(x_test, y_test)

print("Accuracy of our model: "+ str(score_knn))