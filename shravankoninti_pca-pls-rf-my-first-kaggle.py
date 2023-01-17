# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import scale 
from sklearn import model_selection
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression, PLSSVD
from sklearn.metrics import mean_squared_error


import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib notebook


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Read train and test files
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
print("Train rows and columns : ", train.shape)
print("Test rows and columns : ", test.shape)
# Train rows and columns :  (4459, 4993) - the number of rows or instances are more compared to features
# Test rows and columns :  (49342, 4992) - the number of rows or instances are more compared  to features and surprisingly we have more test instances than train instances
### Below are the steps involved to understand, clean and prepare your data for building your predictive model:
##### Variable Identification
##### Univariate Analysis
##### Bi-variate Analysis
##### Missing values treatment
##### Outlier treatment
##### Variable transformation
##### Variable creation
train.describe()
# target  - is a Target variable
# Other Columns - Are predictors
train.dtypes
dtype_df = train.dtypes.reset_index()
dtype_df.columns = ["Count", "Column Type"]
dtype_df.groupby("Column Type").aggregate('count').reset_index()
# Majority of the columns have integer , followed by float and ID (category / string)
### we will check for scatter plot of the target variable to see for any otuliers which are visible clearly
plt.figure(figsize=(8,6))
plt.scatter(range(train.shape[0]), np.sort(train['target'].values))
plt.xlabel('index', fontsize=12)
plt.ylabel('Target', fontsize=12)
plt.title("Target Distribution", fontsize=14)
plt.show()
## the above distribution shows that there are no outliers but the values are increasing from low range to very high range
# As there are lot of variables we cannot see all the distributions at one place. But we can definitely see the distribution of Target variable
plt.figure(figsize=(8,6))
sns.distplot(train['target']);
plt.title('target histogram.');
### The above distributionis right skewed one and hence it needs a transformation -let us try with Log transformation and see the plot
log_train_target = np.log(train.target)
plt.figure(figsize=(8,6))
sns.distplot(log_train_target);
plt.title('Logarithm transformed target histogram.');
# The above distribution looks better when compared to the original variable (target)
train.nunique()
# Checking the columns which has constant values and remove them from our analysis
unique_list = train.nunique().reset_index()
unique_list.columns = ["col_name", "unique_count"]
constant_list = unique_list[unique_list["unique_count"]==1]
constant_list.shape


## Check for test also
unique_list_test = test.nunique().reset_index()
unique_list_test.columns = ["col_name", "unique_count"]
constant_list_test = unique_list_test[unique_list_test["unique_count"]==1]
constant_list_test.shape
# Yes, there are 256 columns with same constant values , it is better to remove them from our analysis. Let us see what are those columns
# Dropping id variables
print("The Original train dataset has:", train.shape[1], 'Columns')
train = train.drop(constant_list.col_name.tolist(), axis=1)
print("The Final train dataset after removing the constant columns list has :", train.shape[1], 'Columns')
# The test dataset also needs to be removed - all the constant columns
test = test.drop(constant_list.col_name.tolist(), axis=1)
print("The Final test dataset after removing the constant columns list has :", test.shape[1], 'Columns')
# Now let us look at the correlation of all other remaining variables w.r.t target variables
#### Identify Highly Correlated Features

# Create correlation matrix
corr_matrix = train.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))


upper_pd = pd.DataFrame(upper) 
upper_first_row = upper_pd.head(1)
data = upper_first_row.iloc[0]
columns = ['target']
names = upper_first_row.columns.values[1:]
df = pd.DataFrame(data,  index = names, columns=columns)
# identify the maximum correlation value 
s=df.max()
s
print('The maximum correlation value:', s)
# Hence filter out those attributes with > 0.2 correlation value and consider them as important features

## Correlation heatmap
## To further reduce the variables - we can put a cut off of correlation values to be 0.25 becuase we have only maximum correlation - 0.27
df1 = df[df['target'] > 0.25]
print('the features with greater than 0.25 correlation value with target column are :', df1.shape[0])
# Let us print out all the variables with > 0.25 Correlation value
pkmn_type_colors = ['#78C850',  # Grass
                    '#F08030',  # Fire
                    '#6890F0',  # Water
                    '#A8B820',  # Bug
                    '#A8A878',  # Normal
                    '#A040A0',  # Poison
                    '#F8D030',  # Electric
                    '#E0C068',  # Ground
                    '#EE99AC',  # Fairy
                    '#C03028',  # Fighting
                    '#F85888',  # Psychic
                    '#B8A038',  # Rock
                    '#705898',  # Ghost
                    '#98D8D8',  # Ice
                    '#7038F8',  # Dragon
                   ]
plt.figure(figsize=(15,12))
# Count Plot (a.k.a. Bar Plot)
sns.barplot( x = df1.target,  y = df1.index, data=df1, palette=pkmn_type_colors)

plt.title(" Top features with their Correlation Values", fontsize=15)
# Rotate x-labels
plt.xticks(rotation=-45)
plt.show()
# The features - 555f18bd3 and 9fd594eec are having better correlation values with target variable
# Visualizing the top features based on the correlation values with all other features
imp_corr = train[df1.index].corr()
 # Heatmap
plt.figure(figsize=(15,12))
sns.heatmap(imp_corr, annot = True, vmax=.8, square=True, cmap="BuPu")
plt.title("Important features with Correlation Map", fontsize=15)
plt.show()
#Checking for train
missing_df = train.isnull().sum(axis=0).reset_index()
missing_df.columns = ['column_name', 'missing_count']
missing_df = missing_df[missing_df['missing_count']>0]
missing_df = missing_df.sort_values(by='missing_count')
missing_df
print("The number of missing columns in train set are :",missing_df.shape[0])
#Checking for Test
missing_df = test.isnull().sum(axis=0).reset_index()
missing_df.columns = ['column_name', 'missing_count']
missing_df = missing_df[missing_df['missing_count']>0]
missing_df = missing_df.sort_values(by='missing_count')
missing_df
print("The number of missing columns in test set are :",missing_df.shape[0])
### There are no missing values.
plt.figure(figsize=(20,15))
# fig, axs = plt.subplots()
sns.boxplot(data=train[df1.index].iloc[:],orient='h',palette="Set2")
plt.show()
## Let us check the summary of these variables
train[df1.index].describe()
# All most all the variables are looking like outliers
# Let us not remove any of these variables outliers, let us develop a baseline model and then we will come into this.
# Too much extreme values are found with f190486d6 and 58e2e02e6
# we have already checked the target variable distribution and we have already transformed into Log 
# I do not see any variables new variables to be created (Already lot of variables )
## For any dimensionality reduction problem , first thing that comes to our mind is doing a Principal Component Analysis - Let us see how we can implement here!
%matplotlib inline

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import scale 
from sklearn import model_selection
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression, PLSSVD
from sklearn.metrics import mean_squared_error

# To implement PCA in python, simply import PCA from sklearn library. 
# categorical variables have to be converted into numeric.  Here, we have only numeric features

#convert it to numpy arrays

X1=train.drop(['ID', 'target'], axis=1).values

#Scaling the values
# X1 = scale(X1)
# pca = PCA(n_components = 800)
pca = PCA(n_components = 15)
X_reduced = pca.fit_transform(scale(X1))
#Cumulative Variance explains
var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
print(pca.explained_variance_ratio_.sum())
# 3-fold CV, with shuffle
n = len(X_reduced)
kf_3 = model_selection.KFold( n_splits=3, shuffle=True, random_state=1)

regr = LinearRegression()
mse = []

y = log_train_target

# Calculate MSE with only the intercept (no principal components in regression)
score = -1*model_selection.cross_val_score(regr, np.ones((n,1)), y.ravel(), cv=kf_3, scoring='neg_mean_squared_error').mean()    
mse.append(score)

# Calculate MSE using CV for the 200 principle components, adding one component at the time.
for i in np.arange(1, 15):
    score = -1*model_selection.cross_val_score(regr, X_reduced[:,:i], y.ravel(), cv=kf_3, scoring='neg_mean_squared_error').mean()
    mse.append(score)
 
plt.figure(figsize=(15,12))
# Plot results    
plt.plot(mse, '-v')
plt.xlabel('Number of principal components in regression')
plt.ylabel('MSE')
plt.title('Target')
plt.xlim(xmin=-1);

from sklearn.metrics import mean_squared_error
from math import sqrt
X=train.drop(['ID', 'target'], axis=1).values

y = log_train_target


# As there are > 4000 attributes, I have tested above to how many components this is reaching with 85% variance ( 1000 components contribute) 
pca2 = PCA(n_components = 15)

# Split into training and test sets
X_train, X_test , y_train, y_test = model_selection.train_test_split(X, y, test_size=0.5, random_state=1)

# Scale the data
X_reduced_train = pca2.fit_transform(scale(X_train))
n = len(X_reduced_train)



# 3-fold CV, with shuffle
kf_3 = model_selection.KFold( n_splits=3, shuffle=True, random_state=1)

mse = []

# Calculate MSE with only the intercept (no principal components in regression)
score = -1*model_selection.cross_val_score(regr, np.ones((n,1)), y_train.ravel(), cv=kf_3, scoring='neg_mean_squared_error').mean()    
mse.append(score)

# Calculate MSE using CV for the 1000 principle components, adding one component at the time.
for i in np.arange(1, 15):
    score = -1*model_selection.cross_val_score(regr, X_reduced_train[:,:i], y_train.ravel(), cv=kf_3, scoring='neg_mean_squared_error').mean()
    mse.append(score)


plt.figure(figsize=(15,12))
plt.plot(np.array(mse), '-v')
plt.xlabel('Number of principal components in regression')
plt.ylabel('MSE')
plt.title('Target')
plt.xlim(xmin=-1);


# # Scale the data(Validation test data)
X_reduced_test = pca2.transform(scale(X_test))

# Train regression model on training data 
regr = LinearRegression()
regr.fit(X_reduced_train[:,:15], y_train)

# Prediction with validation_test data
# Scale the data(Validation test data)

pred = regr.predict(X_reduced_test)
rmse = sqrt(mean_squared_error(y_test, pred))
# rmse = 1.692 - Leaderboad score gave 1.73
## prepare this - predictions on the final test dataset
test_X = test.drop( ["ID"], axis=1)
final_X_reduced_test = pca2.transform(scale(test_X))[:,:15]
pred_final_test = regr.predict(final_X_reduced_test)
pred_final_test = np.expm1(pred_final_test)
# n = len(X_train)

# # 10-fold CV, with shuffle
# kf_10 = model_selection.KFold(n_splits=10, shuffle=True, random_state=1)

# mse = []

# for i in np.arange(1, 100):
#     pls = PLSRegression(n_components=i)
#     score = model_selection.cross_val_score(pls, scale(X_train), y_train, cv=kf_10, scoring='neg_mean_squared_error').mean()
#     mse.append(-score)

# # Plot results
# plt.plot(np.arange(1, 100), np.array(mse), '-v')
# plt.xlabel('Number of principal components in regression')
# plt.ylabel('MSE')
# plt.title('Target')
# plt.xlim(xmin=-1)
from sklearn.metrics import mean_squared_error
from math import sqrt
X=train.drop(['ID', 'target'], axis=1).values

y = log_train_target
# Split into training and test sets
X_train, X_test , y_train, y_test = model_selection.train_test_split(X, y, test_size=0.5, random_state=1)
pls = PLSRegression(n_components=24)
pls.fit(scale(X_train), y_train)

sqrt(mean_squared_error(y_test, pls.predict(scale(X_test))))
## 2.68 - not a good score
#Import Library
from sklearn.ensemble import RandomForestRegressor #use RandomForestRegressor for regression problem
model= RandomForestRegressor(n_estimators=1000,  n_jobs = -1,random_state =50, max_features = "auto",
                                 min_samples_leaf = 1)

# Train the model using the training sets and check score
from sklearn.metrics import mean_squared_error
from math import sqrt
X=train.drop(['ID', 'target'], axis=1).values

y = log_train_target

# Split into training and test sets
X_train, X_test , y_train, y_test = model_selection.train_test_split(X, y, test_size=0.5, random_state=1)

model.fit(X_train, y_train)

 #Predict Output
print("RMSE : ", sqrt(mean_squared_error(y_test,  model.predict(X_test))))  



#Predict Output (test)
test1=test.drop(['ID'], axis=1).values
pred_final_test= model.predict(test1) 
pred_final_test = np.expm1(pred_final_test)

# Making a submission file #
sub_df = pd.DataFrame({"ID":test["ID"].values})
sub_df["target"] = pred_final_test
sub_df.to_csv("baseline_RandomForest.csv", index=False)

# RMSE :  1.4634184706155577




