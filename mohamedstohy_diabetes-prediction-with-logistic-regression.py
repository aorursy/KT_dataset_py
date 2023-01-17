#Diabetes Prediction with Logistic Regression (direct Machine learning algorithm implementation without data analysis)



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
dia = pd.read_csv('/kaggle/input/diabetes-dataset/diabetes2.csv')

 #Reading dataset
dia.info() #information about dataset
dia.head() #head_items
#Splitting_data



from sklearn.model_selection import train_test_split

#defining_X_Y_before splitting

X=dia[['Pregnancies','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']]

y=dia['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
from sklearn.linear_model import LogisticRegression #importing_linear_regression_model
logmodel=LogisticRegression()
logmodel.fit(X_train,y_train)  #fitting_trainig_data
predictions=logmodel.predict(X_test) #predicting_upon_xtest
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions)) #classifying_data
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test,predictions)
