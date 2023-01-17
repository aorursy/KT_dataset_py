#Import the file

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #for visualization

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
data = pd.read_csv('../input/diabetes/diabetes.csv')
data.describe
data.head(20)
data.isnull().sum()
# Replace all zero values as mean (in reality there is no zero values for these features: Glocuse, Blood Pressure, Skin Thickness, Insulin, BMI)
columns_nozero_values = ['Glucose','BloodPressure', 'SkinThickness','Insulin','BMI']

for n in columns_nozero_values:
    data[n] = data[n].replace(0,np.NaN)
    mean = int(data[n].mean())
    data[n] = data[n].replace(np.NaN,mean)
data.head()
# Visualizing the data
import seaborn as sns
import matplotlib.pyplot as plt
sns.countplot(x='Outcome',data=data)
data.hist(figsize=(12,12))
plt.show()
# Correlation matrix
corrmat = data.corr()
fig = plt.figure(figsize = (12, 12))

sns.heatmap(corrmat, vmax = 1, square = True,annot=True,vmin=-1)
plt.show()
# Assign the independent and dependent variables
X=data.iloc[:,0:8]
Y=data.iloc[:,8]
# Split the dataset into train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test= train_test_split(X, Y, test_size=0.3, random_state=0)
# Apply feature scaling in the dataset
from sklearn.preprocessing import StandardScaler
standard_scaler = StandardScaler()
X_train_standard = X_train.copy()
X_test_standard = X_test.copy()
X_train_standard = standard_scaler.fit_transform(X_train_standard)
X_test_standard = standard_scaler.fit_transform(X_test_standard)
print('X_train size: {}, X_test size: {}'.format(X_train.shape, X_test.shape))
print('X_train size: {}, X_test size: {}'.format(Y_train.shape, Y_test.shape))
print('X_train_standard size: {}, X_test_standard size: {}'.format(X_train_standard.shape, X_test_standard.shape))
# Fit the data in the Logistic Regression Model
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=100,random_state=0, solver='lbfgs')
lr.fit(X_train_standard, Y_train)
Y_predict = lr.predict(X_test_standard)
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)

# To show the confusion Matrix
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(Y_test,Y_predict)

import seaborn as sns
sns.heatmap(confusion, annot = True)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Value')
plt.ylabel('Actual Value')
print(confusion)

tp = confusion[1, 1]
tn = confusion[0, 0]
fp = confusion[0, 1]
fn = confusion[1, 0]
models = []
models.append(("Logistic Regression", lr))
classification_reports = []

for name, model in models:

    # Classification Accuracy
    classification_accuracy = accuracy_score(Y_test, Y_predict)
    print("{} Classification Accuracy: {:.4f}".format(
        name, classification_accuracy))
    # Classification Error
    classificaton_error = 1 - classification_accuracy
    print("{} Classification Error: {:.4f}".format(name, classificaton_error))
    # Sensitivity or Recall Score or True Positive Rate
    sensitivity = recall_score(Y_test, Y_predict)
    print("{} Sensitivity: {:.4f}".format(name, sensitivity))
    # Specificity or True Negative Rate
    specificity = tn / (tn + fp)
    print("{} Specificity: {:.4f}".format(name, specificity))
    # False Positive Rate
    fpr = 1 - specificity
    print("{} False Positive Rate: {:.4f}".format(name, fpr))
    # Precision or False Negative Rate
    precision = precision_score(Y_test, Y_predict)
    print("{} Precision: {:.4f}".format(name, precision))
    # F1 Score
    classification_f1 = f1_score(Y_test, Y_predict)
    print("{} Classification F1: {:.4f}".format(name, classification_f1))
    # Classification Report
    classificationReport = classification_report(Y_test, Y_predict)
    classification_reports.append(classificationReport)
    print("\n")
    print("{} Classification Report: \n{}".format(name, classificationReport))
    print("\n\n")

def predict_diabetes(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age):
    preg = int(Pregnancies)
    glucose = float(Glucose)
    bp = float(BloodPressure)
    st = float(SkinThickness)
    insulin = float(Insulin)
    bmi = float(BMI)
    dpf = float(DiabetesPedigreeFunction)
    age = int(Age)

    inputs = [[preg, glucose, bp, st, insulin, bmi, dpf, age]]
    inputs = standard_scaler.transform(inputs)

    return lr.predict(inputs)    
prediction = predict_diabetes(7,195,70,33,145,25.1,0.163,55)[0]
if prediction:
  print('Diabetes Test Result: Positive')
else:
  print('Diabetes Test Result: Negative')
prediction = predict_diabetes(0,120.0,98.0,55.0,170.0,31.6,0.127,22)[0]
if prediction:
  print('Diabetes Test Result: Positive')
else:
  print('Diabetes Test Result: Negative')
lr.predict(X_test_standard[:10])
Y_test[:10]
lr.predict_proba(X_test_standard[:10])
