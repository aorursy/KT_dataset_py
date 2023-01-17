# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# load Pima data 
df = pd.read_csv('../input/diabetes.csv')
# rows by columns
df.shape
# Get top 5
df.head(5)
# Check for null values
df.isnull().values.any()
# Correlation
corr = df.corr()
fig, ax = plt.subplots(figsize=(11,11))
ax.matshow(corr)
plt.xticks(range(len(corr.columns)), corr.columns)
plt.yticks(range(len(corr.columns)), corr.columns)
df.corr()
df.head(5)
# Check data types - mold data
diabetes_map = {True:1, False:0}
df['Outcome'] = df['Outcome'].map(diabetes_map)
df.head(5)
# Check true/false ratio
num_true = len(df.loc[df['Outcome'] == True])
num_false = len(df.loc[df['Outcome'] == False])
print ("number of True cases: {0} ({1:2.2f}%)".format(num_true, (num_true/(num_true + num_false)) * 100))
print ("number of False cases: {0} ({1:2.2f}%)".format(num_false, (num_false/(num_true + num_false)) * 100))
# Split the data
from sklearn.model_selection import train_test_split
feature_col_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction','Age', 'Outcome']
predicted_class_names = ['Outcome']
# predictor feature columns
x = df[feature_col_names].values
# predictor class (1=true 0=false)
y = df[predicted_class_names].values


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)

print('{0:0.2f}% in training set'.format((len(x_train)/len(df.index))*100))
print('{0:0.2f}% in test set'.format((len(x_test)/len(df.index))*100))

# Verify predicted value was split correctly
print("Original True  : {0} ({1:0.2f}%)".format(len(df.loc[df['Outcome'] == 1]), (len(df.loc[df['Outcome'] == 1])/len(df.index)) * 100.0))
print("Original False : {0} ({1:0.2f}%)".format(len(df.loc[df['Outcome'] == 0]), (len(df.loc[df['Outcome'] == 0])/len(df.index)) * 100.0))
print("")
print("Training True  : {0} ({1:0.2f}%)".format(len(y_train[y_train[:] == 1]), (len(y_train[y_train[:] == 1])/len(y_train) * 100.0)))
print("Training False : {0} ({1:0.2f}%)".format(len(y_train[y_train[:] == 0]), (len(y_train[y_train[:] == 0])/len(y_train) * 100.0)))
print("")
print("Test True      : {0} ({1:0.2f}%)".format(len(y_test[y_test[:] == 1]), (len(y_test[y_test[:] == 1])/len(y_test) * 100.0)))
print("Test False     : {0} ({1:0.2f}%)".format(len(y_test[y_test[:] == 0]), (len(y_test[y_test[:] == 0])/len(y_test) * 100.0)))
df.head(2)
# post split data preparation 
# Hidden missing values
print("# rows in dataframe {0}".format(len(df)))
print("# rows missing Pregnancies: {0}".format(len(df.loc[df['Pregnancies'] == 0])))
print("# rows missing Glucose: {0}".format(len(df.loc[df['Glucose'] == 0])))
print("# rows missing BloodPressure: {0}".format(len(df.loc[df['BloodPressure'] == 0])))
print("# rows missing SkinThickness: {0}".format(len(df.loc[df['SkinThickness'] == 0])))
print("# rows missing Insulin: {0}".format(len(df.loc[df['Insulin'] == 0])))
print("# rows missing BMI: {0}".format(len(df.loc[df['BMI'] == 0])))
print("# rows missing DiabetesPedigreeFunction: {0}".format(len(df.loc[df['DiabetesPedigreeFunction'] == 0])))
print("# rows missing Age: {0}".format(len(df.loc[df['Age'] == 0])))
# Use the mean imputing
from sklearn.impute import SimpleImputer
#Impute with mean all 0 readings
imp = SimpleImputer(missing_values=0, strategy='mean')

x_train = imp.fit_transform(x_train)
x_test = imp.fit_transform(x_test)

from sklearn.naive_bayes import GaussianNB

# create Gaussian Naive Bayes model object and train it with the data
nb_model = GaussianNB()

nb_model.fit(x_train, y_train.ravel())
# predict values using the training data
nb_predict_train = nb_model.predict(x_train)

# import the performance metrics library
from sklearn import metrics

# Accuracy
print("Accuracy: {0:.4f}".format(metrics.accuracy_score(y_train, nb_predict_train)))
print()
# predict values using the testing data
nb_predict_test = nb_model.predict(x_test)

from sklearn import metrics

# test metrics
print("Accuracy: {0:.4f}".format(metrics.accuracy_score(y_test, nb_predict_test)))
print("Confusion Matrix")
print("{0}".format(metrics.confusion_matrix(y_test, nb_predict_test)))
print("")


print("Classification Report")
print(metrics.classification_report(y_test, nb_predict_test))