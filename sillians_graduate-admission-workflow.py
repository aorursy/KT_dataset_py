# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

import pandas as pd  # data processing
import numpy as np   # linear algebra
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Ignore useless warnings (see SciPy issue #5998)
import warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd")
dataset = pd.read_csv("/kaggle/input/graduate-admissions/Admission_Predict_Ver1.1.csv")
dataset.head()
dataset.columns
dataset.rename(columns={'GRE Score':'Gre_Score','TOEFL Score':'TOEFL_Score','University Rating':'University_Rating',
                                       'LOR ':'LOR',  'Chance of Admit ':'Chance_of_Admit'}, inplace = True)
# shape of the dataset (Rows and Columns)
print (f"dataset has {dataset.shape[0]} rows and {dataset.shape[1]} columns")
# check for duplicates 
assert dataset.duplicated().any() == False
# Gives use the count of different types of objects.
dataset.dtypes.value_counts()
# Check for info about the dataset, for missing values and data type
dataset.info()
# Descriptive analysis of the data
dataset.describe().T
# A graphical view/representation of columns(features) with missing data using missingno
# Missingno library offers a very nice way to visualize the distribution of NaN values.
import missingno as msno
msno.matrix(dataset)
plt.show()
dataset = dataset.drop(['Serial No.'], axis=1)
dataset.head()
pearson_corr = dataset.corr(method='pearson')
pearson_corr
with sns.axes_style("whitegrid"):
    f, ax = plt.subplots(figsize=(15, 10))
    ax = sns.heatmap(dataset.corr(), annot=True)
    plt.title('Correlation chart of all the features')
from pandas_profiling import ProfileReport
profile = ProfileReport(dataset, title='Pandas Profiling Report', html={'style':{'full_width':True}})
# The HTML report can be included in a Juyter notebook
profile.to_notebook_iframe()
# Generating reports interactively through widgets
profile.to_widgets()
# profile
# Data distribution
%matplotlib inline
import matplotlib.pyplot as plt
dataset.hist(bins=20, figsize=(20,15))
# save_fig("attribute_histogram_plots")
plt.show()
top_prob = dataset[(dataset['Gre_Score']>=330) & (dataset['TOEFL_Score']>=115) & \
                   (dataset['CGPA']>=9.5)].sort_values(by=['Chance_of_Admit'],ascending=False)
top_prob
# plotting a pair plot to see the correlations

plt.rcParams['figure.figsize'] = (20,21)
plt.style.use('ggplot')

sns.pairplot(dataset, palette="crest")
# sns.color_palette("flare", as_cmap=True)
plt.show()
# density plot
dataset.plot(kind='density', subplots=True, layout=(4,4), sharex=False, fontsize=1, figsize=(15,15))
plt.show()
# box and whisker plots
sns.set_style("whitegrid")
dataset.plot(kind='box', subplots=True, layout=(4,4), sharex=False, sharey=False, figsize=(15, 15))
plt.show()
# University Ratings vs TOEFL Score
plt.rcParams['figure.figsize'] = (15, 7)
plt.style.use('ggplot')

sns.boxenplot(dataset['University_Rating'], dataset['TOEFL_Score'], palette='husl')
plt.title('University Ratings vs TOEFL Score', fontsize = 20)
plt.show()
# University Rating vs CGPA
plt.rcParams['figure.figsize'] = (15, 7)
plt.style.use('ggplot')

sns.swarmplot(dataset['University_Rating'], dataset['CGPA'], palette='twilight')
plt.title('University Ratings vs CGPA', fontsize = 20)
plt.show()
# University Ratings vs Chance of Admission
plt.rcParams['figure.figsize'] = (15, 7)
plt.style.use('ggplot')

sns.violinplot(dataset['University_Rating'], dataset['Chance_of_Admit'], palette="Dark2")
plt.title('University Ratings vs Chance of Admission', fontsize = 20)
plt.show()
fig=sns.lmplot(x='Gre_Score',y='CGPA',data=dataset)
plt.title("CGPA VS GRE_SCORE")
plt.show()
fig = sns.lmplot(x="Gre_Score", y="TOEFL_Score", data=dataset)
plt.title("GRE Score vs TOEFL Score")
plt.show()
fig = sns.lmplot(x="SOP", y="LOR", data=dataset, hue="Research", markers=["x", "o"])
plt.title("SOP vs LOR")
plt.show()
fig = sns.lmplot(x="CGPA", y="LOR", data=dataset, hue="Research", markers=["x", "o"])
plt.title("GRE Score vs CGPA")
plt.show()
fig = sns.lmplot(x="Gre_Score", y="LOR", data=dataset, hue="Research", markers=["x", "o"])
plt.title("GRE Score vs CGPA")
plt.show()
fig = sns.lmplot(x="CGPA", y="SOP", data=dataset, hue='Research', markers=["x", "o"])
plt.title("GRE Score vs CGPA")
plt.show()
fig = sns.lmplot(x="TOEFL_Score", y="CGPA", data=dataset, hue='Research', markers=["x", "o"])
plt.title("TOEFL Score vs CGPA")
plt.show()
fig = sns.lmplot(x="TOEFL_Score", y="SOP", data=dataset)
plt.title("GRE Score vs CGPA")
plt.show()
fig = sns.lmplot(x="Gre_Score", y="Chance_of_Admit", data=dataset)
plt.title("GRE Score vs CGPA")
plt.show()
fig=sns.jointplot(x='TOEFL_Score',y='Chance_of_Admit',data=dataset,kind='kde')
plt.show()
fig=sns.jointplot(x='CGPA',y='Chance_of_Admit',data=dataset,kind='kde')
plt.show()
corr = dataset.corr()
fig, ax = plt.subplots(figsize=(8, 8))
colormap = sns.diverging_palette(220, 10, as_cmap=True)
dropSelf = np.zeros_like(corr)
dropSelf[np.triu_indices_from(dropSelf)] = True
colormap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, cmap=colormap, linewidths=.5, annot=True, fmt=".2f", mask=dropSelf)
plt.show()
# Import library for VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor

def calc_vif(X):

    # Calculating VIF
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    return(vif)
X = dataset.iloc[:,:-1]
calc_vif(X)
X = dataset.drop(['Gre_Score','TOEFL_Score', 'Chance_of_Admit'],axis=1)
calc_vif(X)
X = dataset.drop(['SOP','Gre_Score','TOEFL_Score', 'Chance_of_Admit'],axis=1)
calc_vif(X)
dataset = dataset.drop(['SOP', 'Gre_Score', 'TOEFL_Score'], axis=1)
dataset.head()
corr = dataset.corr()
fig, ax = plt.subplots(figsize=(8, 8))
colormap = sns.diverging_palette(220, 10, as_cmap=True)
dropSelf = np.zeros_like(corr)
dropSelf[np.triu_indices_from(dropSelf)] = True
colormap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, cmap=colormap, linewidths=.5, annot=True, fmt=".2f", mask=dropSelf)
plt.show()
# plot both together to compare
fig, axs = plt.subplots(ncols=5, figsize=(20, 7))

sns.distplot(dataset['LOR'], ax=axs[0])
axs[0].set_title("LOR")

sns.distplot(dataset['CGPA'], ax=axs[1])
axs[1].set_title("CGPA")

sns.distplot(dataset['Research'], ax=axs[2])
axs[2].set_title("Research")

sns.distplot(dataset['University_Rating'], ax=axs[3])
axs[3].set_title("University_Rating")

sns.distplot(dataset['Chance_of_Admit'], ax=axs[4])
axs[4].set_title("Chance_of_Admit")
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Lasso,Ridge,BayesianRidge,ElasticNet,SGDRegressor
# from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
X = dataset.drop(['Chance_of_Admit'], axis=1)  # predictor
y = dataset['Chance_of_Admit'] # target(label)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42) # For reproducibilty

print('Size of X_test data is: ', X_test.shape)
print('Size of X_train data is: ', X_train.shape)
print('Size of y_test data is: ', y_test.shape)
print('Size of y_train data is: ', y_train.shape)
X_train.head()
y_train.head()
X_test.head()
# Create a copy of the train and test test so we don't harm the original dataset
X_train_copy = X_train.copy()
y_train_copy = y_train.copy()
X_test_copy = X_test.copy()
y_test_copy = y_test.copy()
# Instantiate you PREDICTOR
linear_model = LinearRegression()

#fit the model to the training dataset
linear_model.fit(X_train_copy, y_train_copy)

#obtain predictions
y_pred = linear_model.predict(X_test_copy)

print('Training Accuracy: %.3f' % linear_model.score(X_train_copy, y_train_copy))
print('Linear Model is intercept (beta_0) :', linear_model.intercept_)
print('Coefficient of features (other betas) :', linear_model.coef_)
# The mean squared error or mean squared deviation of an estimator measures the average of the squares of the errors, that is, 
# the average squared difference between the estimated values and the actual value.

# Mean Squared Error
from sklearn.metrics import mean_squared_error
mse = (mean_squared_error(y_test_copy, y_pred))
print('Mean Squared Error : ', round(mse, 3)) 
# MAE is easy and intuitive such that it calculates the sum of the average of the absolute error between the predicted
# values and the true values. Since the absolute difference is taken, this metric does not consider direction. 
# However, because the absolute difference is obtained, it is unable to give information about the model overshooting or 
# undershooting. The smaller the MAE is, the better the model. Therefore, if the MAE is 0, the model is perfect and accurately 
# predicts results which is almost impossible. The mean absolute error is more robust to outliers.

# Mean Absolute Error
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test_copy, y_pred)
print('Mean Absolute Error : ', round(mae, 3))
# Also known as the sum of squared residuals (SSR), this metric explains the variance in the representation of the dataset 
# by the model; it measures how well the model approximates the data. A residual is the estimated error made by a model. 
# In simpler terms, it is the difference between the nth true value and the nth predicted value by the model. RSS is the sum 
# of the square of errors between the residuals in a model.The lower the RSS, the better the model’s estimations and vice versa.

# Residual Sum of Squares (RSS)
rss = np.sum(np.square(y_test_copy - y_pred))
print('Residual Sum of Squares : ', round(rss, 3))
# Root Mean Squared Error
# When the RMSE is low, it means that the error made by the model has a small deviation from the true values.

rmse = np.sqrt(mean_squared_error(y_test_copy, y_pred))
print('Root Mean Squared Error : ', round(rmse, 3))
# coefficient of determination, r-squared is a metric used in regression to determine the goodness of fit of the model. 
# With values ranging from 0 to 1, It gives information on the percentage of the response variable explained by the model. 
# Mostly, the higher the value, the better the model 

# R2 Score
from sklearn.metrics import r2_score
r2_score = r2_score(y_test_copy, y_pred)
print('Coefficient of Determination : ', round(r2_score, 2))
# Instantiate you PREDICTOR
tree_reg = DecisionTreeRegressor(random_state=0, max_depth=3)

#fit the model to the training dataset
tree_reg.fit(X_train_copy, y_train_copy)

#obtain predictions
tree_y_pred = tree_reg.predict(X_test_copy)

print('Training Accuracy: %.3f' % tree_reg.score(X_train_copy, y_train_copy))
# Root Mean Squared Error

rmse = np.sqrt(mean_squared_error(y_test_copy, tree_y_pred))
print('Root Mean Squared Error : ', round(rmse, 3))
# R2 Score
from sklearn.metrics import r2_score
r2_score = r2_score(y_test_copy, tree_y_pred)
print('Coefficient of Determination : ', round(r2_score, 2))
# Instantiate you PREDICTOR
reg_model = RandomForestRegressor()

#fit the model to the training dataset
reg_model.fit(X_train_copy, y_train_copy)

#obtain predictions
reg_y_pred = reg_model.predict(X_test_copy)

# Model Accuracy
print('Training Accuracy: %.3f' % reg_model.score(X_train_copy, y_train_copy))
# Root Mean Squared Error

rmse = np.sqrt(mean_squared_error(y_test_copy, reg_y_pred))
print('Root Mean Squared Error : ', round(rmse, 3))
# R2 Score
from sklearn.metrics import r2_score
r2_score = r2_score(y_test_copy, reg_y_pred)
print('Coefficient of Determination : ', round(r2_score, 2))
#Instantiate you PREDICTOR
gbb_model = GradientBoostingRegressor()

#fit the model to the training dataset
gbb_model.fit(X_train_copy, y_train_copy)

#obtain predictions
gbb_y_pred = gbb_model.predict(X_test_copy)

# Model Accuracy
print('Training Accuracy: %.3f' % gbb_model.score(X_train_copy, y_train_copy))
# Root Mean Squared Error

rmse = np.sqrt(mean_squared_error(y_test_copy, gbb_y_pred))
print('Root Mean Squared Error : ', round(rmse, 3))
# Import 'r2_score'
from sklearn.metrics import r2_score

def performance_metric(y_true, y_predict):
    """ Calculates and returns the performance score between 
        true and predicted values based on the metric chosen. """
    
    # TODO: Calculate the performance score between 'y_true' and 'y_predict'
    score = r2_score(y_true, y_predict)
    
    # Return the score
    return score
# Create a copy of the train and test test so we don't harm the original dataset
X_train_cv = X_train.copy()
y_train_cv = y_train.copy()
X_test_cv = X_test.copy()
y_test_cv = y_test.copy()
# R2 Score
from sklearn.metrics import r2_score
r2_score = r2_score(y_test_copy, gbb_y_pred)
print('Coefficient of Determination : ', round(r2_score, 2))
# Import 'make_scorer', 'DecisionTreeRegressor', and 'GridSearchCV'
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit

def fit_model(X, y):
    """ Performs grid search over the 'max_depth' parameter for a 
        decision tree regressor trained on the input data [X, y]. """
    
    # Create cross-validation sets from the training data
    # sklearn version 0.23: ShuffleSplit(n_splits=10, *, test_size=None, train_size=None, random_state=None)
    cv_sets = ShuffleSplit(n_splits = 10, test_size = 0.20, random_state = 0)

    # Create a decision tree regressor object
    regressor = DecisionTreeRegressor()

    # Create a dictionary for the parameter 'max_depth' with a range from 1 to 10
    params = {'max_depth': range(1,11)}

    # Transform 'performance_metric' into a scoring function using 'make_scorer' 
    scoring_fnc = make_scorer(performance_metric)

    # Create the grid search object
    grid = GridSearchCV(regressor, params, scoring = scoring_fnc, cv=cv_sets)

    # Fit the grid search object to the data to compute the optimal model
    grid = grid.fit(X, y)
    
    # Return the optimal model after fitting the data
    return grid.best_estimator_
# Fit the training data to the model using grid search
reg = fit_model(X_train_cv, y_train_cv)

# obtain predictions
reg_pred = reg.predict(X_test_cv)
    
# Produce the value for 'max_depth'
print ("Parameter 'max_depth' is {} for the optimal model.".format(reg.get_params()['max_depth']))
# R2 Score
print('Coefficient of Determination : ', round(performance_metric(y_test_cv, reg_pred), 2))
# Import 'make_scorer', 'DecisionTreeRegressor', and 'GridSearchCV'
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit

def fit_model(X, y):
    """ Performs grid search over the 'max_depth' parameter for a 
        decision tree regressor trained on the input data [X, y]. """
    
    # Create cross-validation sets from the training data
    # sklearn version 0.23: ShuffleSplit(n_splits=10, *, test_size=None, train_size=None, random_state=None)
    cv_sets = ShuffleSplit(n_splits = 10, test_size = 0.20, random_state = 0)

    # Create a decision tree regressor object
    regressor = RandomForestRegressor()

    # Create a dictionary for the parameter 'max_depth' with a range from 1 to 10
    params = {'max_depth': range(1,11)}

    # Transform 'performance_metric' into a scoring function using 'make_scorer' 
    scoring_fnc = make_scorer(performance_metric)

    # Create the grid search object
    grid = GridSearchCV(regressor, params, scoring = scoring_fnc, cv=cv_sets)

    # Fit the grid search object to the data to compute the optimal model
    grid = grid.fit(X, y)
    
    # Return the optimal model after fitting the data
    return grid.best_estimator_
# Fit the training data to the model using grid search
reg = fit_model(X_train, y_train)

# obtain predictions
reg_pred = reg.predict(X_test)
    
# Produce the value for 'max_depth'
print ("Parameter 'max_depth' is {} for the optimal model.".format(reg.get_params()['max_depth']))
# R2 Score
print('Coefficient of Determination : ', round(performance_metric(y_test, reg_pred), 3))
