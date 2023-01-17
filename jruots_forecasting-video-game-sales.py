import numpy as np

import pandas as pd

import sklearn

import seaborn as sn

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats

from scipy.stats import norm, skew 

%matplotlib inline



import warnings

def ignore_warn(*args, **kwargs):

    pass

warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)

data = pd.read_csv('../input/Video_Games_Sales_as_at_22_Dec_2016.csv')

data.head()
print(data.shape)
fig, ax = plt.subplots()

ax.scatter(x = data['Critic_Score'], y = data['Global_Sales'])

plt.ylabel('Global_Sales', fontsize=13)

plt.xlabel('Critic_Score', fontsize=13)

plt.show()
data = data.drop(data[(data['Critic_Score']>60) & (data['Global_Sales']>60)].index)
fig, ax = plt.subplots()

ax.scatter(x = data['Critic_Score'], y = data['Global_Sales'])

plt.ylabel('Global_Sales', fontsize=13)

plt.xlabel('Critic_Score', fontsize=13)

plt.show()
sns.distplot(data['Global_Sales'] , fit=norm);



# Get the fitted parameters used by the function

(mu, sigma) = norm.fit(data['Global_Sales'])

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))



#Now plot the distribution

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('Global_Sales distribution')



#Get also the QQ-plot

fig = plt.figure()

res = stats.probplot(data['Global_Sales'], plot=plt)

plt.show()
str_list = [] # empty list to contain columns with strings (words)

for colname, colvalue in data.iteritems():

    if type(colvalue[2]) == str:

         str_list.append(colname)

# Get to the numeric columns by inversion            

num_list = data.columns.difference(str_list) 

# Create Dataframe containing only numerical features

data_num = data[num_list]

f, ax = plt.subplots(figsize=(14, 11))

plt.title('Pearson Correlation of Video Game Numerical Features')

# Draw the heatmap using seaborn

sns.heatmap(data_num.astype(float).corr(),linewidths=0.25,vmax=1.0, 

            square=True, cmap="cubehelix_r", linecolor='k', annot=True)
data_na = (data.isnull().sum() / len(data)) * 100

data_na = data_na.drop(data_na[data_na == 0].index).sort_values(ascending=False)[:30]

missing_data = pd.DataFrame({'Missing Ratio' :data_na})

missing_data.head(16)
print(pd.value_counts(data["Platform"]))
data = data[(data['Platform'] == 'PS3') | (data['Platform'] == 'PS4') | (data['Platform'] == 'X360') | (data['Platform'] == 'XOne') | (data['Platform'] == 'Wii') | (data['Platform'] == 'WiiU') | (data['Platform'] == 'PC')]



#Let's double check the value counts to be sure

print(pd.value_counts(data["Platform"]))



#Let's see the shape of the data again

print(data.shape)



#Lets see the missing ratios again

data_na = (data.isnull().sum() / len(data)) * 100

data_na = data_na.drop(data_na[data_na == 0].index).sort_values(ascending=False)[:30]

missing_data = pd.DataFrame({'Missing Ratio' :data_na})

missing_data.head(16)
data = data.dropna(subset=['Critic_Score'])



#Let's see the shape of the data again

print(data.shape)



#Lets see the missing ratios again

data_na = (data.isnull().sum() / len(data)) * 100

data_na = data_na.drop(data_na[data_na == 0].index).sort_values(ascending=False)[:30]

missing_data = pd.DataFrame({'Missing Ratio' :data_na})

missing_data.head(16)
data['Publisher'] = data['Publisher'].fillna(data['Publisher'].mode()[0])
data['Developer'] = data['Developer'].fillna(data['Developer'].mode()[0])
data['Rating'] = data['Rating'].fillna(data['Rating'].mode()[0])
data['Year_of_Release'] = data['Year_of_Release'].fillna(data['Year_of_Release'].median())
#There's "tbd" values in the mix here which we need to handle first

data['User_Score'] = data['User_Score'].replace('tbd', None)



#Now we can handle the N/A's appropriately

data['User_Score'] = data['User_Score'].fillna(data['User_Score'].median())
data['User_Count'] = data['User_Count'].fillna(data['User_Count'].median())
#Lets see the missing ratios again

data_na = (data.isnull().sum() / len(data)) * 100

data_na = data_na.drop(data_na[data_na == 0].index).sort_values(ascending=False)[:30]

missing_data = pd.DataFrame({'Missing Ratio' :data_na})

missing_data.head(16)
print(data.shape) #pre-dummies shape

data = pd.get_dummies(data=data, columns=['Platform', 'Genre', 'Rating'])

print(data.shape) #post-dummies shape

data.head #Check to verify that dummies are ok
data = data.drop(['Name', 'Publisher', 'Developer', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales'], axis=1)
print(data.columns) #easy to copy-paste the values to rearrange from here



X = data[['Year_of_Release', 'Critic_Score', 'Critic_Count',

       'User_Score', 'User_Count', 'Platform_PC', 'Platform_PS3',

       'Platform_PS4', 'Platform_Wii', 'Platform_WiiU', 'Platform_X360',

       'Platform_XOne', 'Genre_Action', 'Genre_Adventure', 'Genre_Fighting',

       'Genre_Misc', 'Genre_Platform', 'Genre_Puzzle', 'Genre_Racing',

       'Genre_Role-Playing', 'Genre_Shooter', 'Genre_Simulation',

       'Genre_Sports', 'Genre_Strategy', 'Rating_E', 'Rating_E10+', 'Rating_M',

       'Rating_RP', 'Rating_T']]



Y = data[['Global_Sales']]



#Double checking the shape

print(X.shape)

print(Y.shape)
from sklearn.model_selection import train_test_split



X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42)



#Let's check the shape of the split data as a precaution

print("X_train shape: {}".format(X_train.shape))

print("Y_train shape: {}".format(Y_train.shape))



print("X_test shape: {}".format(X_test.shape))

print("Y_test shape: {}".format(Y_test.shape))
#We use the numpy fuction log1p which  applies log(1+x) to all elements of the column

Y_train = np.log1p(Y_train)

Y_test = np.log1p(Y_test)
#Check the new distribution 

Y_log_transformed = np.log1p(data['Global_Sales']) #For comparison to earlier, here's the whole Y transformed

sns.distplot(Y_log_transformed , fit=norm);



# Get the fitted parameters used by the function

(mu, sigma) = norm.fit(Y_log_transformed)

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))



#Now plot the distribution

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('Global_Sales distribution')



#Get also the QQ-plot

fig = plt.figure()

res = stats.probplot(Y_log_transformed, plot=plt)

plt.show()

from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler()

scaler.fit(X_train)



X_train_scaled = scaler.transform(X_train) 

X_test_scaled = scaler.transform(X_test)
#No grid to define for vanilla linear regression

param_grid_lr = [

    {}

]



#Parameter grid for lasso

param_grid_lasso = [

    {'alpha': [10, 1, 0.1, 0.01, 0.001, 0.0001], 'max_iter': [1000000, 100000, 10000, 1000]}

]



#Parameter grid for Ridge Regression

param_grid_rr = [

    {'alpha': [100, 10, 1, 0.1, 0.01, 0.001]}

]



#Parameter grid for Support Vector Regressor

param_grid_svr = [

    {'C': [0.01, 0.1, 1, 10], 'gamma': [0.0001, 0.001, 0.01, 0.1, 1],

     'kernel': ['rbf']}

]



#Parameter grid for Random Forest

param_grid_rf = [

    {'n_estimators': [3, 10, 30, 50, 70], 'max_features': [2,4,6,8,10,12], 'max_depth': [2, 3, 5, 7, 9]}

]



#Parameter grid for Gradient Boosting Regressor

param_grid_gbr = [

    {'n_estimators': [200, 225, 250, 275], 'max_features': [6, 8, 10, 12], 'max_depth': [5, 7, 9]}

]



#Parameter grid for MLPRegressor. 

#Current set of hyperparameters are the result of grid search that took forever.

param_grid_mlpr = [

    {'hidden_layer_sizes': [(10,5)], 'solver': ['lbfgs'], 'batch_size': [200],

     'learning_rate': ['adaptive'], 'max_iter': [800], 'verbose': [True], 

     'nesterovs_momentum': [True], 'early_stopping': [True], 'validation_fraction': [0.12],

     'random_state': [100], 'alpha': [0.1], 'activation': ['logistic']}

]
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import GridSearchCV



grid_search_lr = GridSearchCV(LinearRegression(), param_grid_lr, scoring='neg_mean_squared_error',  cv=5)

grid_search_lr.fit(X_train, Y_train)

print("Best parameters: {}".format(grid_search_lr.best_params_))

lr_best_cross_val_score = (np.sqrt(-grid_search_lr.best_score_))

print("Best cross-validation score: {:.2f}".format(np.expm1(lr_best_cross_val_score)))

lr_score = np.sqrt(-grid_search_lr.score(X_test, Y_test))

print("Test set score: {:.2f}".format(np.expm1(lr_score)))
from sklearn.linear_model import Lasso



grid_search_lasso = GridSearchCV(Lasso(), param_grid_lasso, cv=5, scoring='neg_mean_squared_error')

grid_search_lasso.fit(X_train, Y_train)

print("Best parameters: {}".format(grid_search_lasso.best_params_))

lasso_best_cross_val_score = (np.sqrt(-grid_search_lasso.best_score_))

print("Best cross-validation score: {:.2f}".format(np.expm1(lasso_best_cross_val_score)))

lasso_score = np.sqrt(-grid_search_lasso.score(X_test, Y_test))

print("Test set score: {:.2f}".format(np.expm1(lasso_score)))
from sklearn.linear_model import Ridge



grid_search_rr = GridSearchCV(Ridge(), param_grid_rr, cv=5, scoring='neg_mean_squared_error')

grid_search_rr.fit(X_train, Y_train)

print("Best parameters: {}".format(grid_search_rr.best_params_))

rr_best_cross_val_score = (np.sqrt(-grid_search_rr.best_score_))

print("Best cross-validation score: {:.2f}".format(np.expm1(rr_best_cross_val_score)))

rr_score = np.sqrt(-grid_search_rr.score(X_test, Y_test))

print("Test set score: {:.2f}".format(np.expm1(rr_score)))
from sklearn.svm import SVR



grid_search_svr = GridSearchCV(SVR(), param_grid_svr, cv=5, scoring='neg_mean_squared_error')

grid_search_svr.fit(X_train, Y_train)

print("Best parameters: {}".format(grid_search_svr.best_params_))

svr_best_cross_val_score = (np.sqrt(-grid_search_svr.best_score_))

print("Best cross-validation score: {:.2f}".format(np.expm1(svr_best_cross_val_score)))

svr_score = np.sqrt(-grid_search_svr.score(X_test, Y_test))

print("Test set score: {:.2f}".format(np.expm1(svr_score)))
from sklearn.ensemble import RandomForestRegressor



grid_search_rf = GridSearchCV(RandomForestRegressor(), param_grid_rf, cv=5, scoring='neg_mean_squared_error')

grid_search_rf.fit(X_train, Y_train)

print("Best parameters: {}".format(grid_search_rf.best_params_))

rf_best_cross_val_score = (np.sqrt(-grid_search_rf.best_score_))

print("Best cross-validation score: {:.2f}".format(np.expm1(rf_best_cross_val_score)))

rf_score = np.sqrt(-grid_search_rf.score(X_test, Y_test))

print("Test set score: {:.2f}".format(np.expm1(rf_score)))
from sklearn.ensemble import GradientBoostingRegressor



grid_search_gbr = GridSearchCV(GradientBoostingRegressor(), param_grid_gbr, cv=5, scoring='neg_mean_squared_error')

grid_search_gbr.fit(X_train, Y_train)

print("Best parameters: {}".format(grid_search_gbr.best_params_))

gbr_best_cross_val_score = (np.sqrt(-grid_search_gbr.best_score_))

print("Best cross-validation score: {:.2f}".format(np.expm1(gbr_best_cross_val_score)))

gbr_score = np.sqrt(-grid_search_gbr.score(X_test, Y_test))

print("Test set score: {:.2f}".format(np.expm1(gbr_score)))
from sklearn.neural_network import MLPRegressor



grid_search_mlpr = GridSearchCV(MLPRegressor(), param_grid_mlpr, cv=5, scoring='neg_mean_squared_error')

grid_search_mlpr.fit(X_train, Y_train)

print("Best parameters: {}".format(grid_search_mlpr.best_params_))

mlpr_best_cross_val_score = (np.sqrt(-grid_search_mlpr.best_score_))

print("Best cross-validation score: {:.2f}".format(np.expm1(mlpr_best_cross_val_score)))

mlpr_score = np.sqrt(-grid_search_mlpr.score(X_test, Y_test))

print("Test set score: {:.2f}".format(np.expm1(mlpr_score)))
# Plot feature importance

feature_importance = grid_search_gbr.best_estimator_.feature_importances_

# make importances relative to max importance

feature_importance = 100.0 * (feature_importance / feature_importance.max())

sorted_idx = np.argsort(feature_importance)

pos = np.arange(sorted_idx.shape[0]) + .5

plt.figure(figsize=(20,10))

plt.subplot(1, 2, 2)

plt.barh(pos, feature_importance[sorted_idx], align='center')

plt.yticks(pos, X_train.columns.values[sorted_idx]) #Not 100 % sure the feature names match the importances correctly...

plt.xlabel('Relative Importance')

plt.title('Variable Importance')

plt.show()