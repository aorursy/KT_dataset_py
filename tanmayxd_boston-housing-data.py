#Importing Sklearn 

import sklearn 



#Common imports

import numpy as np

import pandas as pd



#To plot data 

%matplotlib inline

import matplotlib as mpl

import matplotlib.pyplot as plt 

mpl.rc('axes', labelsize=14)

mpl.rc('xtick', labelsize=12)

mpl.rc('ytick', labelsize=12)
#Defining data path 

DATA_PATH = '../input/boston-house-prices/housing.csv'
#Defining a function to load data from the given path

def loading_data(data_path=DATA_PATH):

    return pd.read_csv(data_path)
#Creating a Dataframe which contains data from housing.csv

housing = loading_data()
#Previewing our dataframe 

housing.head()
#defining column names 

column_names = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']



#Creating a dataframe which contains data form housing.csv and includes column_names 

housing_with_columns = pd.read_csv(DATA_PATH,delim_whitespace=True,names = column_names)
#We copy housing dataframe with columns into housing dataframe

housing_df = housing_with_columns.copy()
#Taking a look at our dataframe 

housing_df.head()
#Getting info about our dataset 

housing_df.info()
#Describing dataframe 

housing_df.describe()
#Plotting the data 

housing_df.hist(bins = 50, figsize=(20,15))

plt.savefig('attribute_histogram_plots')

plt.show()
#Setting a random seed 

np.random.seed(42)
#Using train_test_spilt to spilt data into train and test set 

from sklearn.model_selection import train_test_split



train_set, test_set = train_test_split(housing_df, test_size=0.2, random_state=42)
#Finding correlation in data 

corr_matrix = housing_df.corr()
corr_matrix['MEDV'].sort_values(ascending=False)
housing = train_set.drop("MEDV",axis = 1) # drop label for training set 

housing_labels = train_set['MEDV'].copy()
#Finding incomplete or null rows 

incomplete_rows = housing[housing.isnull().any(axis=1)]

incomplete_rows.head()
#Using standard scaler using pipeline 

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline 



num_pipeline = Pipeline([

    ('std_scaler',StandardScaler()),

])



housing = num_pipeline.fit_transform(housing)
housing 
#Using linear regression 

from sklearn.linear_model import LinearRegression



lin_reg = LinearRegression()

lin_reg.fit(housing, housing_labels)
#Using mean squared error

from sklearn.metrics import mean_squared_error 



housing_predictions = lin_reg.predict(housing)

lin_mse = mean_squared_error(housing_labels,housing_predictions)

lin_rmse = np.sqrt(lin_mse)

lin_rmse 
#Using mean absolute error

from sklearn.metrics import mean_absolute_error 



lin_mae = mean_absolute_error(housing_labels,housing_predictions)

lin_mae 
#Using Decision Tree Regressor 

from sklearn.tree import DecisionTreeRegressor 



tree_reg = DecisionTreeRegressor(random_state=42)

tree_reg.fit(housing,housing_labels)
housing_predictions = tree_reg.predict(housing)

tree_mse = mean_squared_error(housing_labels, housing_predictions)

tree_rmse = np.sqrt(tree_mse)

tree_rmse
#Finding Cross validation score

from sklearn.model_selection import cross_val_score 



#Finding cross validation score for tree regression 

scores = cross_val_score(tree_reg, housing, housing_labels, scoring='neg_mean_squared_error',cv = 10)

tree_scores = np.sqrt(-scores)



#Finding cross validation score for linear regression

scores = cross_val_score(lin_reg,housing,housing_labels,scoring="neg_mean_squared_error",cv=10)

lin_scores = np.sqrt(-scores)
#Display scores

def display_scores(scores):

    print("Scores: ",scores)

    print("Mean: ", scores.mean())

    print("Standard deviation: ", scores.std())
#Displaying scores 

print("Score for decision tree regression")

print(display_scores(tree_scores))

print("Score for linear regression")

print(display_scores(lin_scores))
#Using Random forest regression 

from sklearn.ensemble import RandomForestRegressor



forest_reg = RandomForestRegressor(n_estimators=100,random_state=42)

forest_reg.fit(housing,housing_labels)
scores = cross_val_score(forest_reg, housing, housing_labels,

                                scoring="neg_mean_squared_error", cv=10)

forest_scores = np.sqrt(-scores)

display_scores(forest_scores)
#Using support vector regression 

from sklearn.svm import SVR



svm_reg = SVR(kernel="linear")

svm_reg.fit(housing,housing_labels)

housing_predictions = svm_reg.predict(housing)

svm_mse = mean_squared_error(housing_labels,housing_predictions)

svm_rmse = np.sqrt(svm_mse)

svm_rmse
#Using Grid Search CV

from sklearn.model_selection import GridSearchCV



param_grid = [

    {'n_estimators':[3,10,30], 'max_features': [2,4,6,8]},

    {'bootstrap':[False],'n_estimators':[3,10],'max_features':[2,3,4]},

]



forest_reg = RandomForestRegressor(random_state = 42)



# train across 5 folds, that's a total of (12+6)*5=90 rounds of training 

grid_search = GridSearchCV(forest_reg, param_grid, cv=5,

                           scoring='neg_mean_squared_error',

                           return_train_score=True)

grid_search.fit(housing, housing_labels)
#Best Hyperparameter combination found:

grid_search.best_params_
grid_search.best_estimator_

#Let's look at the score of each hyperparameter combination tested during the grid search:



cvres = grid_search.cv_results_

for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):

    print(np.sqrt(-mean_score), params)
pd.DataFrame(grid_search.cv_results_)

#Using randomized search cv



from sklearn.model_selection import RandomizedSearchCV

from scipy.stats import randint



param_distribs = {

        'n_estimators': randint(low=1, high=200),

        'max_features': randint(low=1, high=8),

    }



forest_reg = RandomForestRegressor(random_state=42)

rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs,

                                n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)

rnd_search.fit(housing, housing_labels)
cvres = rnd_search.cv_results_

for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):

    print(np.sqrt(-mean_score), params)
feature_importances = grid_search.best_estimator_.feature_importances_

feature_importances
#Using final model

final_model = grid_search.best_estimator_



#Preparing test set 

X_test = test_set.drop("MEDV",axis = 1)

y_test = test_set["MEDV"].copy()



X_test = num_pipeline.transform(X_test)

final_predictions = final_model.predict(X_test)



#Finding rsme erroer for our final prediction 

final_mse = mean_squared_error(y_test, final_predictions)

final_rmse = np.sqrt(final_mse)
final_rmse