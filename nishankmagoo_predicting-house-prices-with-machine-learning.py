import numpy as np 
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

#Acquire Data
features = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
#sample = pd.read_csv('../input/sample_submission.csv')
#sample.head()
#features.head()
#features.describe()
#features.shape

#Find Correlation matrix
plt.subplots(figsize=(20,15))
corr = features.corr()
sns.heatmap(corr)

#Doubt No 1- why 79 features are not displayed ? 

#1 Basic Analysis
#2 Fill Missing data
#-> a) By Imputation - process of replacing missing data with substituted values.
#Mean
#Median
#Mode
#-> b) By new Missing value feature
#-> c) By Model based imputation
#3 Remove Outlier

#4 Corelation matrix
#Firstly find the Correlation matrix 
#Find the top prioirty vaiables which are srongly corelated with the "predicted feature"
#Find the variables which are strongly corelated with each other and merge them

#5 ScatterPlot
#Take the top features from Correlation Matrix and draw a scatter plot of them.

#4 Test
#Normality
#Homoscedasticity
#Linearity
#Absence of correlated errors 

#action based on test results 
#If data is not normal
#check for peakedness and skewness(positive or negative)
#Apply log transformation to skewness 
k = 10 #number of variables for heatmap
cols = corr.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(features[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
#scatterplot
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(features[cols], height = 2.5)
plt.show();
