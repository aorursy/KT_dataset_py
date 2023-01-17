#Importing all the necessary libraries and data set

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline





from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#display the data

data = pd.read_csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv')

data.head()
#checking 'NaN' values in the data

data.isnull().sum()
#describing data to view various statstical variables

data.describe()
#Plot a count plot

sns.countplot(data = data, x = "Outcome",hue = "Outcome")

plt.title("WomenDiabetesData")
#Plot Box plot

fig, ax = plt.subplots(figsize = (5,5))

sns.boxplot(data = data, y = "Pregnancies", x = "Outcome", hue = "Outcome")

plt.title("Pregnancies")
#Plot a line plot

sns.lineplot(data = data, x = 'Age', y = 'BloodPressure', hue = 'Age')

plt.title("Age and BP")
#Plotting histograms for all features in the data set

for i in data.columns:

    plt.figsize=(5,5)

    plt.hist(data[i])

    plt.title(i)

    plt.show()
#Plot scatter plot for all features in the data set

sns.pairplot(data = data, hue = "Outcome")
#Plot heatmap and show case correlation of each features

corr_mat = data.corr()

plt.figure(figsize=(12,10))

sns.heatmap(corr_mat, annot = True, cmap = "coolwarm")

plt.show()
#Assigning X and Y

X = data[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']]

Y = data.Outcome
#Assigning training and test data

from sklearn.model_selection import train_test_split

X_train_orig, X_test, Y_train,Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
#Feature Scaling the data

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

sc.fit(X_train_orig)

X_train = sc.transform(X_train_orig)

X_test = sc.transform(X_test)
from sklearn.linear_model import LogisticRegression

lr_classifier = LogisticRegression()

lr_classifier.fit(X_train, Y_train)

Y_pred = lr_classifier.predict(X_test)

print("Accuracy of the model:",accuracy_score(Y_test, Y_pred))

print(classification_report(Y_test, Y_pred))
from sklearn.naive_bayes import GaussianNB

gnb_classifier = GaussianNB()

gnb_classifier.fit(X_train, Y_train)

Y_pred = gnb_classifier.predict(X_test)

print("Accuracy of the model:",accuracy_score(Y_test, Y_pred))

print(classification_report(Y_test, Y_pred))
from sklearn.ensemble import RandomForestClassifier

rf_classifier = RandomForestClassifier()

rf_classifier.fit(X_train, Y_train)

Y_pred = rf_classifier.predict(X_test)

print("Accuracy of the model:",accuracy_score(Y_test, Y_pred))

print(classification_report(Y_test, Y_pred))
from sklearn.svm import SVC

svm_classifier = SVC()

svm_classifier.fit(X_train, Y_train)

Y_pred = svm_classifier.predict(X_test)

print("Accuracy of the model:",accuracy_score(Y_test, Y_pred))

print(classification_report(Y_test, Y_pred))
import tensorflow as tf 

from keras.models import Sequential

from keras.layers import Dense



#define keras model

model = Sequential()

model.add(Dense(12, input_dim = 8, activation = 'relu'))

model.add(Dense(8, activation = 'relu'))

model.add(Dense(1, activation = 'sigmoid'))



#compile the keras model

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])





#fit the model

model.summary()

model.fit(X_train, Y_train, epochs = 150, batch_size = 10)
#Predict the model

_, accuracy = model.evaluate(X_train, Y_train, verbose = 0)

print("Accuracy of the model:", accuracy)