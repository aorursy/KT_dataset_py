#We will begin by importing all the libraries that are required
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #visualisation
import os
#import the dataset with pandas
dataset = pd.read_csv("../input/diabetes.csv")
dataset.head() #sample data

dataset.describe() #abstract analysis
correlation = dataset.corr()
correlation.iloc[:,8].sort_values(ascending=False)
X = dataset.iloc[:, :-1].values #independent variable
y = dataset.iloc[:, 8].values #dependent variable

from sklearn.preprocessing import Imputer #import Scikit library
imputer = Imputer(missing_values=0, strategy="mean", axis=0)
imputer = imputer.fit(X[:,1:-1]) #except the 1st column
X[:,1:-1] = imputer.transform(X[:,1:-1])
X[0:8,2:4].view()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
X[0:8,2:4].view()
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
plt.scatter(X_test[0:50,1], y_test[0:50], color="red") #Plotting only 50 sample values of the Actual test set
plt.scatter(X_test[0:50,1],y_pred[0:50], color="green", s=8) #Plotting only 50 sample values of the predictions
plt.title('Actual Outcome over Predicted Outcome (Test set)')
plt.xlabel('No. of samples')
plt.ylabel('Binary Outcome')
import matplotlib
matplotlib.rc('figure', figsize=(15, 4))
plt.show()