#Problem---Remove the features with low variance(likely containing little information) from Numerical dataset

#Solution---We will use varianceThreshold of scikit learn library



#importing libraries

from sklearn import datasets

from sklearn.feature_selection import VarianceThreshold



#importing data

iris =  datasets.load_iris()



#creating features and target

features= iris.data

target = iris.target



#creating Thresholder

thresholder = VarianceThreshold(threshold=.5)



#creating high variance feature matrix

features_high_variance = thresholder.fit_transform(features)



#Viewing high variance feature matrix upto 3X3

features_high_variance[0:3]
#viewing variances

thresholder.fit(features).variances_
#loading library

from sklearn.preprocessing import StandardScaler



#standardize Feature matrix

scaler = StandardScaler()

features_std = scaler.fit_transform(features)



#calculate variance of each feature

selector = VarianceThreshold()

selector.fit(features_std).variances_
#Problem---Remove the features with low variance(likely containing little information) from Binary dataset

#Solution---We will use varianceThreshold of scikit learn library



#loading library

from sklearn.feature_selection import VarianceThreshold



#creating feature matrix with:

#feature 0: 80% class 0

#feature 1: 80% class 1

#feature 2: 60% class 0, 40% class 1

features= [[0,1,0],

           [0,1,1],

           [0, 1, 0],

           [0, 1, 1],

           [1, 0, 0]]



#run threshold by variance

thresholder = VarianceThreshold(threshold=(.75*(1-0.75)))

thresholder.fit_transform(features)
#Problem---Feature matrix have some features that are higly correlated

#Solution--We will use correlation matrix to find higly correlated feaures.If higly correlated features exist consider dropping one of the correlated features



#importing libraries

import pandas as pd

import numpy as np



#creating feature matrix with two highly correlated features

features = np.array([[1, 1, 1],

                    [2, 2, 0],

                    [3, 3, 1],

                    [4, 4, 0],

                    [5, 5, 1],

                    [6, 6, 0],

                    [7, 7, 1],

                    [8, 7, 0],

                    [9, 7, 1]])



#converting feature matrix into DataFrame

dataframe=pd.DataFrame(features)



#creating correlation matrix

corr_matrix = dataframe.corr().abs()



#Selecting upper triangle of correlation matrix

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))



#finding index of feature columns with correlation greater than 0.95

to_drop = [column for column in upper.columns if any(upper[column]>0.95)]



#drop features

dataframe.drop(dataframe.columns[to_drop], axis =1).head(3)

#In our solution, first create a correlation matrix of all features:

#correlation matrix

dataframe.corr()
#Secondly, we look at the upper triangle of the correlation matrix to identify pairs of higly correlated features:

#upper triangle of correlated features

upper
#Problem--- We have a categorical target vector and want to remove uniformative features.

#Solution---If the features are categorical, calculate a chi-square(X^2) statistic between each feature and the target vector



# importing libraries

from sklearn.datasets import load_iris

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2, f_classif





#loading data

iris = load_iris()

features = iris.data

target = iris.target



#convert to categorical data by converting data to integers

features =  features.astype(int)



#selecting two features with highest chi-squared statistics

chi2_selector = SelectKBest(chi2, k=2)

features_kbest = chi2_selector.fit_transform(features, target)



#Showing results

print("Original number of features:", features.shape[1])

print("Reduced number of features:", features_kbest.shape[1])
#importing library

from sklearn.feature_selection import SelectPercentile



#Select top 75% of features with highest F-values

fvalue_selector = SelectPercentile(f_classif, percentile =75)

features_kbest = fvalue_selector.fit_transform(features,target)



#showing results

print("Original number of features:", features.shape[1])

print("Reduced number of features:", features_kbest.shape[1])
#PRoblem---Select automatically the best features to keep.

#Solution---We will use scikit learns's RFECV to conduct recursive feature elimination(RFE) using cross-validation(CV). That is, repeatedly train a model, each time removing a feature until model performance (eg, accuracy) becomes worse.The remaining features are the best:



#importing libraries

import warnings

from sklearn.datasets import make_regression

from sklearn.feature_selection import RFECV

from sklearn import datasets, linear_model



#Suppress an annoying but harmless warning

warnings.filterwarnings(action="ignore", module="scipy",message="^internal gelsd")



#generating features matrix, target vector, and the true coefficients

features, target = make_regression(n_samples = 10000,

                                   n_features = 100,

                                   n_informative = 2,

                                  random_state= 1)



#creating a linear regression

ols = linear_model.LinearRegression()



#Recursively eliminating features

rfecv = RFECV(estimator=ols, step=1, scoring="neg_mean_squared_error")

rfecv.fit(features, target)

rfecv.transform(features)
#number of best features

rfecv.n_features_
#We can also see which of those features we should keep:

#which categories are best

rfecv.support_
#We can even view the ranking of the features:

#rnk features best(1) to worst

rfecv.ranking_