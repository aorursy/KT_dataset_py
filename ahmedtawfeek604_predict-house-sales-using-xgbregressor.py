# Importing the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

import statsmodels.formula.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

import warnings
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
# Upload data
data = pd.read_csv('../input/kc_house_data.csv')
# Check if there are missing observations
data.info()
# Check some of the observations
data.head()
# 1- Break down (date) into (year, month & day)
data['year'] = data['date'].apply(lambda x: x[:4]).astype(int)
data['month'] = data['date'].apply(lambda x: x[4:6]).astype(int)
data['day'] = data['date'].apply(lambda x: x[6:8]).astype(int)

# Dropp (id and date)
data.drop(['id', 'date'], axis=1, inplace=True)


# 2- Round the number of bathrooms
data['bathrooms'] = data['bathrooms'].apply(lambda x: round(x, 0))
data['bathrooms'] = data['bathrooms'].astype(int)

# Round the number of floors
data['floors'] = data['floors'].apply(lambda x: round(x, 0))
data['floors'] = data['floors'].astype(int)


# 3- Create X & y (independent and dependent variables) vectors to be used in charts and models
X = data.drop("price",axis=1).values
y = data["price"].values
# Create a list with the algorithms that will be used to check features importance
combine = [RandomForestRegressor(random_state=5), XGBRegressor(random_state=5)]

# Make a list of features' names to label the importance bars with it
columns = data.drop("price",axis=1).columns

# Plot features importance charts when using Random Forest and XGBRegressor algorithms
for classifier in combine:
    classifier.fit(X, y)
    f, axes = plt.subplots(1, 1, figsize=(12, 4))
    (pd.Series(classifier.feature_importances_, index=columns)
       .nlargest(len(classifier.feature_importances_))
       .plot(kind='barh'))
    if classifier == combine[0]:
        plt.title('Random Forest')
    else:
        plt.title('XGB Regressor')
    plt.show()
    accuracy = cross_val_score(estimator = classifier, X = X, y = y, cv = 10, n_jobs = -1)
    print('Prices prediction accuracy using this model is: ', str(round((accuracy.mean() * 100), 2)))
    print('The highest 5 features in terms of importance represent %.2f percent\n\n' % (pd.Series(classifier.feature_importances_, index=columns).sort_values(ascending=False)[0:5].sum() * 100))
# Set chart size
plt.figure(figsize = (17,8))

# Create scatter plot to check the relationship between (lat & price)
ax1 = plt.subplot(221)
ax1 = sns.regplot('lat', 'price', data=data, fit_reg=False, ax=ax1)

# Create scatter plot to check the relationship between (long & price)
ax2 = plt.subplot(222)
ax2 = sns.regplot('long', 'price', data=data, fit_reg=False, ax=ax2)
plt.show()
# Set chart size
plt.figure(figsize = (17,8))

# Create scatter plot to check the relationship between (sqft_living & price)
ax1 = plt.subplot(221)
ax1 = sns.regplot('sqft_living', 'price', data=data, fit_reg=False, ax=ax1)

# Create scatter plot to check the relationship between (sqft_living15 & price)
ax2 = plt.subplot(222)
ax2 = sns.regplot('sqft_living15', 'price', data=data, fit_reg=False, ax=ax2)
plt.show()
# Set chart size
plt.figure(figsize = (25,8))

# Create scatter plot to check the relationship between (grade & price)
ax1 = plt.subplot(221)
ax1 = sns.regplot('grade', 'price', data=data, fit_reg=False, ax=ax1)
plt.show()
# Create a new dataframe
copy_data = data.copy()

# Group 'yr_built' values
for number in range(1890, 2021, 10):
    copy_data.loc[(copy_data['yr_built'] > number) & (copy_data['yr_built'] <= (number + 10)), 'yr_built'] = number + 10

# Group 'yr_renovated' values
for number in range(1890, 2021, 10):
    copy_data.loc[(copy_data['yr_renovated'] > number) & (copy_data['yr_renovated'] <= (number + 10)), 'yr_renovated'] = number + 10

# Group 'sqft_lot' values
for number in range(0, 1700001, 100000):
    copy_data.loc[(copy_data['sqft_lot'] > number) & (copy_data['sqft_lot'] <= (number + 100000)), 'sqft_lot'] = number + 100000
    
# Group 'sqft_basement' values
for number in range(0, 5001, 500):
    copy_data.loc[(copy_data['sqft_basement'] > number) & (copy_data['sqft_basement'] <= (number + 500)), 'sqft_basement'] = number + 500
    
# Group 'sqft_above' values
for number in range(0, 10001, 1000):
    copy_data.loc[(copy_data['sqft_above'] > number) & (copy_data['sqft_above'] <= (number + 1000)), 'sqft_above'] = number + 1000

# Create a list of all features that we will checked against grade
parameters = ['yr_built', 'yr_renovated', 'sqft_lot', 'sqft_basement', 'sqft_above', 'floors', 'month', 'bedrooms', 
            'condition', 'waterfront', 'view', 'grade']
cm = sns.light_palette("green", as_cmap=True)

# Display a table for each feature against grade
for number in range(0, len(parameters) - 1):
    display(pd.crosstab(copy_data[parameters[number]], copy_data[parameters[len(parameters) - 1]]).style.background_gradient(cmap = cm))
plt.figure(figsize=(15,12))
plt.title('Correlation of Features', fontsize=20)
sns.heatmap(copy_data.corr().astype(float).corr(),vmax=1.0, annot=True)
plt.show()
# Create function to check the best hyperparameters for each model
def params_checker (algo, parameters, x, y):
    grid_search = GridSearchCV(estimator = algo, param_grid = parameters, scoring = 'neg_mean_absolute_error', cv = 10, n_jobs = -1)
    grid_search = grid_search.fit(x, y)
    
    # Print the mean absolute error for best parameters reached
    print("- mean absolute error: %.2f" % ((round(grid_search.best_score_, 2))))
# Create a dataframe that will hold each model's prediction accuracy calculated using cross validation.
accuracy_dataframe = pd.DataFrame(columns=['Model', 'CV_Score'])
# Aggregate all independent features and add a column of ones at the beginning (since statsmodels 'sm' library doesn't take in consideration the constant coefficient (b0) at the multiple linear regression equation)
X_multi = np.append(arr = np.ones((len(data), 1)).astype(int), values = X, axis = 1)

# Create an object with the significant features (after performing backward elimination, i removed feature number 20 "month" since it had a (P) value above 5% "its (P) value was 12.8%")
X_opt = X_multi [:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21]]

# Fitting the data and checking the R-squared
classifier_OLS = sm.OLS(endog = y, exog = X_opt).fit()
classifier_OLS.summary()
# Create parameters dictionary to use it with the 'params_checker' function
params = dict(normalize = [False])

# Use grid search through 'params_checker' function and specify the algorithm, parameters, x & y to get the best parameters that generate the highest prediction accuracy
params_checker(LinearRegression(), params, X, y)

# Create object without 'month' feature (since it's not a significant feature according to it's (P) value)
X_opt = data.drop([data.columns[0], data.columns[len(data.columns)-2]], axis=1).values

# Calculate the accuracy of the model using 10 fold cross validation
accuracies = cross_val_score(estimator = LinearRegression(), X = X_opt, y = y, cv = 10, n_jobs = -1)
print('- Accuracy using 10 folds is:', round((accuracies.mean()) * 100, 3), '%')

# Add the name of the model and the accuracy result to accuracy_dataframe
accuracy_dataframe.loc[len(accuracy_dataframe)] = 'Multiple Linear', (str(round((accuracies.mean()) * 100, 3)) + ' %')
# Create parameters dictionary to use it with the 'params_checker' function
params = dict(n_estimators=[165], min_samples_split=[3], random_state=[0])

# Use grid search through 'params_checker' function and specify the algorithm, parameters, x & y to get the best parameters that generate the highest prediction accuracy
params_checker(RandomForestRegressor(), params, X, y)

# Create object with the best parameters to use it in the cross validation
algo = RandomForestRegressor(n_estimators=165, min_samples_split=3, random_state=0)

# Calculate the accuracy of the model using 10 fold cross validation
accuracies = cross_val_score(estimator = algo, X = X, y = y, cv = 10, n_jobs = -1)
print('- Accuracy using 10 folds is:', round((accuracies.mean()) * 100, 3), '%')

# Add the name of the model and the accuracy result to accuracy_dataframe
accuracy_dataframe.loc[len(accuracy_dataframe)] = 'Random Forest', (str(round((accuracies.mean()) * 100, 3)) + ' %')
# Create parameters dictionary to use it with the 'params_checker' function
params = dict(max_depth=[7], learning_rate=[0.1], n_estimators=[350], gamma=[0.00001], min_child_weight=[3], colsample_bytree=[0.7])

# Use grid search through 'params_checker' function and specify the algorithm, parameters, x & y to get the best parameters that generate the highest prediction accuracy
params_checker(XGBRegressor(), params, X, y)

# Create object with the best parameters to use it in the cross validation
algo = XGBRegressor(max_depth=7, learning_rate=0.1, n_estimators=350, gamma=0.00001, min_child_weight=3, colsample_bytree=0.7)

# Calculate the accuracy of the model using 10 fold cross validation
accuracies = cross_val_score(estimator = algo, X = X, y = y, cv = 10, n_jobs = -1)
print('- Accuracy using 10 folds is:', round((accuracies.mean()) * 100, 3), '%')

# Add the name of the model and the accuracy result to accuracy_dataframe
accuracy_dataframe.loc[len(accuracy_dataframe)] = 'XGBRegressor', (str(round((accuracies.mean()) * 100, 3)) + ' %')
accuracy_dataframe = accuracy_dataframe.sort_values(['CV_Score'], ascending=False)
accuracy_dataframe.reset_index(drop=True)