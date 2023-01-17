#Imports
import pandas as pd
import numpy as np
# Load the data
data_clf=pd.read_csv('../input/iris/Iris.csv') # for classification problem
data_reg=pd.read_csv('../input/50-startups/50_Startups.csv') # for regression problem
# Check first five datapoints by using head() method
print(data_clf.head(2))
print(data_reg.head(2))
# Check numerical statistics using info() method
data_clf.info(), data_reg.info()
# Create feature and target variable for Classification problem
X_clf=data_clf.iloc[:,1:5] # features: SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'
y_clf=data_clf.iloc[:,5] # Target variable: Iris species
# Create feature and target variable for Regression problem
X_reg=data_reg.iloc[:,0:3] # features: R&D Spend, Administration, Marketing Spend
# I have not considered 'State' in feature set. You can use it after label encoding.
y_reg=data_reg.iloc[:,4] # Target variable: Profit
# Import SelectKBest, chi2(score function for classification), f_regression (score function for regression)
from sklearn.feature_selection import SelectKBest, chi2, f_regression
# Create the object for SelectKBest and fit and transform the classification data
# k is the number of features you want to select [here it's 2]
X_clf_new=SelectKBest(score_func=chi2,k=2).fit_transform(X_clf,y_clf)
# Check the newly created variable for top two best features
print(X_clf_new[:5])
# Compare the newly created values with feature set values to know the selected features
print(X_clf.head())
# Create the object for SelectKBest and fit and transform the regression data
X_reg_new=SelectKBest(score_func=f_regression, k=2).fit_transform(X_reg,y_reg)
# Check the newly created variable for top two best features
print(X_reg_new[:5])
# Compare the newly created values with feature set values to know the selected features
print(X_reg.head())