#Load Packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
%matplotlib inline 
# Linear Regression-Ordinary Least Squares
from sklearn.linear_model import LinearRegression
# Ridge Regression
from sklearn.linear_model import Ridge
# functions to cross validation to optimize hyperparameters
from sklearn.model_selection import GridSearchCV
# Lasso Regression-Least Absolute Shrinkage and Selection Operator
from sklearn.linear_model import Lasso
# ElasticNet Regression
from sklearn.linear_model import ElasticNet
#LARS Regression Model- Least Angle Regression model
from sklearn import linear_model

# using statsmodels
import statsmodels.formula.api as smf # smf -- statsmodels formula
url = "http://archive.ics.uci.edu/ml/datasets/Communities+and+Crime"
crime = pd.read_csv(url, sep=",")