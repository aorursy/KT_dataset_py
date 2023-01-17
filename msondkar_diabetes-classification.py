import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, classification_report
data = pd.read_csv("../input/pima indian diabetes.csv", 

                 names=["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",

                       "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"])
data.head()
data.info()
features = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",

                       "BMI", "DiabetesPedigreeFunction", "Age"]



for feature in features:

    data[feature] = pd.to_numeric(data[feature], errors='coerce')



# print the data summary again

data.info()
data.isnull().sum()
for feature in features:

    data[feature].fillna(data[feature].mean(), inplace=True)



# check if there are still any missing values

data.isnull().sum()
plt.subplots(figsize=(20,20))



for i, j in zip(features, range(len(features))):

    plt.subplot(4, 2, j+1)

    sns.distplot(data[i])
sns.scatterplot(data['BMI'], data['SkinThickness'])
sns.scatterplot(data['Insulin'],data['Glucose'])
sns.countplot(data['Outcome'])
# Input Features

x = data[features].values

# Output

y = data['Outcome'].values 
min_max_scaler = preprocessing.MinMaxScaler()

x_scaled = min_max_scaler.fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)



print('Training data samples : {}'.format(x_train.shape[0]))

print('Testing data samples  : {}'.format(x_test.shape[0]))
# Knn with 5 neighbors

knn = KNeighborsClassifier(n_neighbors=5)

# Fit Knn on training dataset

knn.fit(x_train, y_train)

# Perform classification on test dataset

y_pred = knn.predict(x_test)

# Print Claasification Report

target_names = ['Class 0', 'Class 1']

print(classification_report(y_test, y_pred, target_names=target_names))
# Gaussian Naive Bayes 

gnb = GaussianNB()

# Fit Gaussian Naive Bayes on training dataset

gnb.fit(x_train, y_train)

# Perform classification on test dataset

y_pred = gnb.predict(x_test)

# Print classification report

print(classification_report(y_test, y_pred, target_names=target_names))