import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import os
#Here we use os to see what the csv is titled. Then add the ../input/ to the path
print(os.listdir("../input"))
df = pd.read_csv('../input/diabetes.csv')
df.dtypes
df.isnull().sum()
df.describe()
df['Outcome'].unique()
#Grab the logistic regression model
from sklearn.linear_model import LogisticRegression
#We import test train split so we can test our model 
from sklearn.model_selection import train_test_split
model = LogisticRegression()
X = df.drop('Outcome', axis =1)
y= df['Outcome']
#Now we split the data into test data and training data.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
#Train our data using .fit
model.fit(X_train, y_train)
#Use the .predict on our test data
predictions = model.predict(X_test)
#Import classification report and confusion matrix so we can see how we did
from sklearn.metrics import classification_report, confusion_matrix
#Use 'print' with both to make them look pretty.
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test,predictions))