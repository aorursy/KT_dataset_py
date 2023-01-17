%matplotlib inline 



import numpy as np 

import scipy as sp 

import matplotlib as mpl

import matplotlib.cm as cm 

import matplotlib.pyplot as plt

import pandas as pd 

#from pandas.tools.plotting import scatter_matrix

pd.set_option('display.width', 500)

pd.set_option('display.max_columns', 100)

pd.set_option('display.notebook_repr_html', True)

import seaborn as sns

sns.set(style="whitegrid")

import warnings

warnings.filterwarnings('ignore')

import string

import math

import sys

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder

import sklearn

from IPython.core.interactiveshell import InteractiveShell



InteractiveShell.ast_node_interactivity = "all"







#importing from sklearn

from sklearn.pipeline import make_pipeline

from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_selection import SelectKBest

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import learning_curve,GridSearchCV

from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import LinearRegression

import statsmodels.formula.api as sm

from sklearn.model_selection import train_test_split

from sklearn.svm import SVC, LinearSVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.feature_selection import SelectFromModel

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.metrics import confusion_matrix

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

#import xgboost as xgb

from sklearn.metrics import roc_curve, auc

import scikitplot as skplt #conda install -c conda-forge scikit-plot

from sklearn.metrics import accuracy_score 

from sklearn.metrics import mean_absolute_error, accuracy_score



from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV

from sklearn.model_selection import cross_val_score



train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")





train.loc[:, ['GrLivArea', 'SalePrice']]
# let our predictor be LotArea in square feet, basically it your house's area above ground level, non americans can treat

# this predictor as house's area since most of the world don't have basements -.-

X = train.loc[:,['GrLivArea']]



# let our target be the sale price, be careful that y must be a series of values extracted this way



y = train['SalePrice']







lr = LinearRegression()

lr.fit(X,y)





  



figure, lotarea = plt.subplots(figsize=(20,10), dpi = 100)

lotarea.scatter(X,y,color='red',s = 5)

lotarea.plot(X, lr.predict(X), color = 'blue')

lotarea.set_title('Fitted Line for House Area vs Sale Price', 

        fontsize = 20)

lotarea.set_xlabel('GrLivArea/House Area (sq feet)', fontsize = 15)

lotarea.set_ylabel('SalePrice ($))', fontsize = 15)
from IPython.display import display, Math



print('The intercept term \N{GREEK SMALL LETTER BETA}\N{SUBSCRIPT ZERO}\u0302 is:', lr.intercept_)

print('The intercept term \N{GREEK SMALL LETTER BETA}\N{SUBSCRIPT ONE}\u0302 is:', lr.coef_)



ols_parameters = [lr.intercept_, lr.coef_[0]]
from scipy import stats



m = X.shape[0]      # number of training samples

n = X.shape[1]      # number of features

X_with_intercept = np.c_[np.ones(X.shape[0]), X]

np.linalg.inv(np.dot(X_with_intercept.T, X_with_intercept)).shape



# RSE = sqrt(RSS/(m-n))

# thus, sigma square estimate is RSS/(m-n)



sigma_square_hat = np.linalg.norm(y - lr.predict(X)) ** 2 / (m-(n+1)) 

var_beta_hat = sigma_square_hat * np.linalg.inv(np.dot(X_with_intercept.T, X_with_intercept))



for i in range(n+1):

    standard_error = var_beta_hat[i,i] ** 0.5   # standard error for beta_0 and beta_1

    print(f"Standard Error of (beta_hat[{i}]): {standard_error}")

    



    

    t_values = ols_parameters[i]/standard_error

    print(f"t_value of (beta_hat[{i}]): {t_values}")

    

        

    print("━"*60)

    print("━"*60)

from scipy import stats



m = X.shape[0]      # number of training samples

n = X.shape[1]      # number of features

X_with_intercept = np.c_[np.ones(X.shape[0]), X]





# RSE = sqrt(RSS/(m-n))

# thus, sigma square estimate is RSS/(m-n)











sigma_square_hat = np.linalg.norm(y - lr.predict(X)) ** 2 / (m-(n+1))

var_beta_hat = sigma_square_hat * np.linalg.inv(np.dot(X_with_intercept.T, X_with_intercept))



for i in range(n+1):

    standard_error = var_beta_hat[i,i] ** 0.5   # standard error for beta_0 and beta_1

    print(f"Standard Error of (beta_hat[{i}]): {standard_error}")

    



    

    t_values = ols_parameters[i]/standard_error

    print(f"t_value of (beta_hat[{i}]): {t_values}")

    



    

    p_values = 1 - stats.t.cdf(abs(t_values), df= X.shape[0] -(X.shape[1] + 1))

    print(f"p_value of (beta_hat[{i}]): {p_values}")   

    

    print("━"*60)

    print("━"*60)





import numpy as np

import pandas as pd

import statsmodels.api as sm

import math

from statsmodels.api import OLS



X_with_intercept = np.c_[np.ones(X.shape[0]), X]

OLS(y,X_with_intercept).fit().summary()