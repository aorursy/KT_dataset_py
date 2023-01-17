# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# Common imports
import numpy as np
import os
import pandas as pd
import seaborn as sns

# to make this notebook's output stable across runs
np.random.seed(42)

# To plot pretty figures
%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt

# Import some statics
from scipy.stats import skew
from scipy.stats.stats import pearsonr


# set the folder path in systems' child directory
# os.getcwd()
os.chdir("../input")        
# view the data set under the defined path
os.listdir("../input")
# read the data frame
train_data = pd.read_csv('train.csv')

# Print out some data samples
print(train_data.head(1))
print("---"*28)
#save index and view the data type
index_array = train_data['index']
# print(len(index_array))

if  'index' in train_data.columns :
    train_data = train_data.drop('index',axis=1)
print(train_data.columns)
train_data.info()
train_non_object = train_data.select_dtypes(include =['float64','int64'])
# print(train_non_object.head())
train_data.describe()
matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)
train_data['resale_price'].hist()
#saleprice correlation matrix
corrmat = train_data.corr()
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'resale_price')['resale_price'].index
cm = np.corrcoef(train_data[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
#scatter plot floor area/re-saleprice
var = 'floor_area_sqm'
train_data.plot(kind='scatter', x = var, y = 'resale_price')
# Separate Features and Target Variables

X_train = train_data.drop('resale_price',axis=1)
y_train = train_data['resale_price']
X_train.columns
if 'postal_code' in X_train.columns and 'street_name' in X_train.columns and 'block' in X_train.columns:
    X_train = pd.get_dummies(X_train.drop(['postal_code','street_name','block'],axis =1))

# Visualize
print(len(X_train.columns))
# print(X_train.head())

#filling NA's with the mean of the column:
X_train = X_train.fillna(X_train.mean())
matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)
prices = pd.DataFrame({"price":train_data["resale_price"], "log(price)":np.log1p(train_data["resale_price"])})
prices.hist()
y_train = np.log1p(y_train)
print(X_train.info())
X_train.describe()
# Handle other skewed features.  
numerical_feats = X_train.dtypes[X_train.dtypes != 'object'].index
# compute skewness of the numerical features
skewed_feats = X_train[numerical_feats].apply(lambda x: skew(x.dropna()))
skewed_feats.describe()
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index
# print(skewed_feats)
X_train[skewed_feats] = np.log1p(X_train[skewed_feats])
# But if you use scikit learn, cross-validation is built-in method
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val, train_id, val_id = train_test_split(X_train, y_train, index_array,test_size = 0.2, random_state = 42)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
model_linear_reg = LinearRegression()

# Train your Linear Regression Model on Training Set
model_linear_reg.fit(X_train, y_train)
# Yeah, you get your first Machine Learning Model
# Print out some values 

data_sample = X_train.iloc[100:105]
y_sample = y_train.iloc[100:105]
print("Predictions: \t", model_linear_reg.predict(data_sample))
print("True Labels: \t", np.array(y_sample))

# compute the mean-square error on training set
y_pred_linear_reg = model_linear_reg.predict(X_train)
mse_model_linear_reg = mean_squared_error(y_train, y_pred_linear_reg)
print("The Mean-Square-Error of the linear regression model is:", mse_model_linear_reg)
# Explore the learned parameters we get

print("The value of w is:", model_linear_reg.coef_)
from sklearn.linear_model import Lasso
model_lasso = Lasso(alpha = 0.01, max_iter= 4000)

# choose the value of your hyperparameter alpha
# Train your model
model_lasso.fit(X_train, y_train)

# Print out some results to see the performance
data_sample = X_train.iloc[100:105]
y_sample = y_train.iloc[100:105]
print("Predictions: \t", model_lasso.predict(data_sample))
print("True Labels: \t", np.array(y_sample))

y_pred_lasso = model_lasso.predict(X_train)
mse_model_lasso = mean_squared_error(y_train, y_pred_lasso)
print("The Mean-Square-Error of the Lasso model is:", mse_model_lasso)
model_lasso_2 = Lasso(alpha= 0.0002, max_iter= 4000)
model_lasso_2.fit(X_train, y_train)

# print out something
data_sample = X_train.iloc[100:105]
y_sample = y_train.iloc[100:105]
print("Predictions: \t", model_lasso_2.predict(data_sample))
print("True Labels: \t", np.array(y_sample))

y_pred_lasso_2 = model_lasso_2.predict(X_train)
mse_model_lasso_2 = mean_squared_error(y_train, y_pred_lasso_2)
print("The Mean-Square-Error of the new lasso model is:", mse_model_lasso_2)
from sklearn.model_selection import cross_val_score
def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, X_train, y_train, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)
alphas = [0.01, 0.0002]
cv_lasso = [rmse_cv(Lasso(alpha = alpha, max_iter= 4000)).mean() 
            for alpha in alphas]
cv_lasso = pd.Series(cv_lasso, index = alphas)
print("Cross Validation Error for Lasso:", cv_lasso)
#cv_lasso.plot(title = "Validation Error")
#plt.xlabel("alpha")
#plt.ylabel("rmse")

cv_linear_reg = rmse_cv(model_linear_reg).mean()
print("Cross Validation Error for Linear Regression:", cv_linear_reg)
cv_lasso.plot(title = "Validation Error")
plt.xlabel("alpha")
plt.ylabel("rmse")
print("The smallest validation error is:",(cv_lasso.min()))
model_lasso_better = Lasso(alpha= 0.0002, max_iter= 4000)
model_lasso_better.fit(X_train, y_train)
coef = pd.Series(model_lasso_better.coef_, index = X_train.columns)
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
imp_coef = pd.concat([coef.sort_values().head(10),
                     coef.sort_values().tail(10)])

# Visualize them.
matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Important Coefficients in the Lasso Model")
from sklearn.model_selection import GridSearchCV

param_grid = [
    {'alpha':np.arange(0.0001,0.01,0.0002)}
    
]

model_lasso_cv = Lasso(max_iter=4000)

grid_search = GridSearchCV(model_lasso_cv, param_grid, cv = 5, scoring = 'neg_mean_squared_error')
grid_search.fit(X_train, y_train)
## Find the best parameters
grid_search.best_params_
### Use the best One 
model_lasso_best = grid_search.best_estimator_
print("Cross_Validation rmse is:", rmse_cv(model_lasso_best).mean())
###load the test dataset
test_data = pd.read_csv('test.csv')
if  'index' in test_data.columns :
    test_data = test_data.drop('index',axis=1)
test_data.info()
'''
test_data = pd.get_dummies(test_data.drop(['postal_code','street_name','block'],axis =1))
col = X_train.columns
if len(test_data.columns)!=len(X_train.columns):
    test_data = test_data.astype('category',categories=col)
test_data.info()
'''

y_pred_linear_reg = model_linear_reg.predict(X_val)
y_pred_lasso = model_lasso_best.predict(X_val)
print("The Mean-Square-Error of the linear regression model is:", mean_squared_error(y_val, y_pred_linear_reg))
print("The Mean-Square-Error of the  lasso model is:", mean_squared_error(y_val, y_pred_lasso))
solution = pd.DataFrame({"id":val_id, "SalePrice":y_pred_lasso})
solution.to_csv("lasso_sol.csv", index = False)

