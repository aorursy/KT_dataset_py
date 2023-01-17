#Numpy is used so that we can deal with array's, which are necessary for any linear algebra
# that takes place "under-the-hood" for any of these algorithms.
import numpy as np

#Pandas is used so that we can create dataframes, which is particularly useful when
# reading or writing from a CSV.
import pandas as pd

#Matplotlib is used to generate graphs in just a few lines of code.
import matplotlib.pyplot as plt

#Import the classes we need to test linear, ridge, and lasso to compare
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LassoCV

#Need these for selecting the best model
from sklearn.model_selection import cross_val_score

#These will be our main evaluation metrics 
from sklearn.metrics import confusion_matrix

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

# Will use this to encode our categorical data.
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

#import xgboost
from xgboost import XGBClassifier
#read the data from csv
dataset = pd.read_csv('../input/churn-modeling/Churn_Modelling.csv')

#take a look at our dataset.  head() gives the first 5 lines. 
dataset.head()
#Grab X, ignoring row number, customer ID, and surname
X = dataset.iloc[:, 3:13].values

#grab the output variable
y = dataset.iloc[:, 13].values

#take a look
X[0:10]
# Transform the Geograpy Column.
ct = ColumnTransformer([("Geography", OneHotEncoder(), [2])], remainder = 'passthrough')
X = ct.fit_transform(X)
#These were tacked onto the front, so we need to exclude one to avoid the "Dummy Variable Trap"
X = X[:, 1:]

#repeat this for Gender
ct = ColumnTransformer([("Gender", OneHotEncoder(), [2])], remainder = 'passthrough')
X = ct.fit_transform(X)
X = X[:, 1:]

#take a look
X[0,:]
#split the datasets, leaving 20% for testing.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
#create an object of the XGBoost Class
classifier = XGBClassifier()
#fit it to the data.
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
cm
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("MEAN ACCURACY: {:.4f}".format(accuracies.mean()))
print("ACCURACY STD: {:.2f}".format(accuracies.std()))
