import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Import Data
data = pd.read_csv("../input/zoo.csv")
data.head(6)
data.info() # https://stackoverflow.com/questions/27637281/what-are-python-pandas-equivalents-for-r-functions-like-str-summary-and-he
# Define target variables, predictor 
y = data.class_type 
X = data.select_dtypes(include=["int64"]).drop("class_type",1)
# Train-test split 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# Load SVM model
from sklearn import svm

# Define model Support Vector Classification
model_svm = svm.SVC()

# Fit model
model_svm.fit(X_train, y_train)
# Load decision tree model
from sklearn.tree import DecisionTreeRegressor

# Define model
model_dtree = DecisionTreeRegressor()

# Fit model
model_dtree.fit(X_train, y_train)
# Load random forest model
from sklearn.ensemble import RandomForestRegressor

# Define model
model_rf = RandomForestRegressor()

# Fit model
model_rf.fit(X_train, y_train)
# Build Pipeline for Cross-Validation 
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer
pipeline = make_pipeline(Imputer(), RandomForestRegressor())
# Cross-Validation Score 
from sklearn.model_selection import cross_val_score
scores = cross_val_score(pipeline, X, y, scoring='neg_mean_absolute_error')
print(scores)