# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import statsmodels.api as sm
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVR
from sklearn.svm import SVR
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import xgboost

sns.set(color_codes=False)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
def display_scores(scores_train):
    print("###### MODEL EVALUATION WITH CROSS VALIDATION######") 
    print("Scores :", scores_train)
    print("Mean :", scores_train.mean())   
    print("Standard Deviation :", scores_train.std())

import pandas as pd
Submission = pd.read_csv("../input/big-mart-sales-prediction/Submission.csv")
test = pd.read_csv("../input/big-mart-sales-prediction/Test.csv")
train = pd.read_csv("../input/big-mart-sales-prediction/Train.csv")
train.head()
train.describe().transpose()
train.info()
test.info()
train.loc[:,['Outlet_Size','Outlet_Type','Outlet_Location_Type']].drop_duplicates()
train.loc[:,['Outlet_Size','Outlet_Type']].drop_duplicates()
train.loc[:,['Outlet_Size','Outlet_Location_Type']].drop_duplicates()
test.loc[:,['Outlet_Size','Outlet_Type','Outlet_Location_Type']].drop_duplicates()
def data_treatment(train):
    # Standardize the different variations of Low fat and regular.
    train['Item_Fat_Content'] = train['Item_Fat_Content'].str.replace('LF', 'Low Fat')
    train['Item_Fat_Content'] = train['Item_Fat_Content'].str.replace('reg', 'Regular')
    train['Item_Fat_Content'] = train['Item_Fat_Content'].str.title()

    # NA Treatment: Outlet size and item weight have null values.
    # For Outlet size, we can refer to our data exploration below to see the best suitable replacement based on location type and outlet type
    #if train['Outlet_Size'] == NA ]

    for i in range(train.shape[0]):
        if((train.loc[i, 'Outlet_Location_Type'] == 'Tier 2')):
            train.loc[i, 'Outlet_Size'] = 'Small'

        elif((train.loc[i, 'Outlet_Type'] == 'Grocery Store')): 
            train.loc[i, 'Outlet_Size'] = 'Small'

    # For Item type, lets use the median value by item type and item fat content

    WeightLookup = train.groupby(['Item_Fat_Content','Item_Type'])['Item_Weight'].agg(np.mean).reset_index()
    train = pd.merge(train, WeightLookup, how = 'inner', on = ['Item_Fat_Content','Item_Type'])
    train['Item_Wt'] = np.where(train['Item_Weight_x'].isnull(), train['Item_Weight_y'], train['Item_Weight_x'])
    train.drop(['Item_Weight_x','Item_Weight_y'], axis=1, inplace=True)
    return train
train = data_treatment(train)
test = data_treatment(test)
sns.distplot(train['Item_Weight'],hist=True, kde=True, rug=False)
sns.boxplot(y=train['Item_Weight'])
sns.relplot(x='Item_Weight', y='Item_Outlet_Sales', data=train)
sns.lmplot( x="Item_Weight", y="Item_Outlet_Sales", data=train, fit_reg=False, hue='Outlet_Location_Type', legend=False, scatter_kws={"alpha":0.3,"s":20})
sns.distplot(train['Item_Visibility'], hist=True, kde=False, rug=False)
sns.boxplot(y=train['Item_Visibility'])
sns.relplot(x='Item_Visibility', y='Item_Outlet_Sales', data=train)
# Store sales by visibility and store type
sns.relplot(x='Item_Visibility', y='Item_Outlet_Sales', col = 'Outlet_Type', data=train)
# Store sales by visibility and store loc type
sns.relplot(x='Item_Visibility', y='Item_Outlet_Sales', col = 'Outlet_Location_Type', data=train)
# Store sales by visibility and store size
sns.relplot(x='Item_Visibility', y='Item_Outlet_Sales', col = 'Outlet_Size' ,data=train)
train.columns
sns.distplot(train['Item_MRP'])
# Sales by Item_MRP
sns.relplot(x="Item_MRP", y="Item_Outlet_Sales", data=train)
train["Item_Fat_Content"].value_counts(normalize=True)
#scatter plot
sns.catplot(x="Item_Fat_Content", y="Item_Outlet_Sales", data=train )
# Swarm Plot Useful for only small datasets
sns.catplot(x="Item_Fat_Content", y="Item_Outlet_Sales", kind="swarm", data=train )
# Box plot
sns.catplot(x="Item_Fat_Content", y="Item_Outlet_Sales", kind="box", data=train )
sns.catplot(x="Item_Fat_Content", y="Item_Outlet_Sales", hue = "Outlet_Location_Type", kind="box", data=train )
# violinplot a boxplot with the kernel density estimation procedure
sns.catplot(x="Item_Fat_Content", y="Item_Outlet_Sales", hue = "Outlet_Location_Type", kind="violin", data=train);
sns.barplot(x="Item_Fat_Content", y="Item_Outlet_Sales", data=train)
# Variation is sales with Item Fat Content by store location
sns.catplot(x="Item_Fat_Content", y="Item_Outlet_Sales", hue="Outlet_Location_Type", kind="bar", data=train)
sns.catplot(x="Item_Fat_Content", y="Item_Outlet_Sales", hue="Outlet_Location_Type", kind="point", data=train)
train["Item_Type"].value_counts(normalize=True)
sns.catplot(x="Item_Type", y="Item_Outlet_Sales", kind='bar', data=train, height=5, aspect=3)
train["Outlet_Location_Type"].value_counts(normalize=True)
sns.barplot(x="Outlet_Location_Type", y="Item_Outlet_Sales", data=train)
sns.catplot(x="Outlet_Location_Type", y="Item_Outlet_Sales", kind="box", data=train)
sns.catplot(x="Outlet_Location_Type", y="Item_Outlet_Sales", kind="box", col='Outlet_Size',data=train)
sns.catplot(y="Outlet_Type", x="Item_Outlet_Sales", kind="bar", data=train, height=5, aspect=2)
sns.catplot(y="Outlet_Type", x="Item_Outlet_Sales", kind="box", data=train, height=5, aspect=2)
sns.catplot(x="Outlet_Type", y="Item_Outlet_Sales", kind="violin", data=train, height=5, aspect=2)
sns.catplot(y="Outlet_Type", x="Item_Outlet_Sales", row="Outlet_Location_Type", kind="box", data=train, height=5, aspect=2)
sns.catplot(y="Outlet_Type", x="Item_Outlet_Sales", row="Outlet_Size", kind="box", data=train, height=5, aspect=2)
train.columns
sns.catplot(x="Outlet_Size", y="Item_Outlet_Sales", kind="bar", data=train)
sns.catplot(x="Outlet_Size", y="Item_Outlet_Sales", kind="box", data=train)
sns.catplot(x="Outlet_Size", y="Item_Outlet_Sales", kind="violin", data=train)
sns.catplot(x="Outlet_Size", y="Item_Outlet_Sales", hue ="Outlet_Location_Type" ,kind="violin", data=train)
sns.catplot(x="Outlet_Size", y="Item_Outlet_Sales", col ="Outlet_Type" ,kind="violin", data=train)
train["Outlet_Establishment_Year"].value_counts(normalize=True)
sns.catplot(x="Outlet_Establishment_Year", y="Item_Outlet_Sales", kind="bar", data=train)
sns.catplot(y="Item_Outlet_Sales", x="Outlet_Establishment_Year", kind="box", data=train, height=5, aspect=2)
# OneHot encoding
train_ohe = pd.get_dummies(train, columns=['Item_Fat_Content','Item_Type','Outlet_Establishment_Year','Outlet_Size','Outlet_Location_Type','Outlet_Type'], drop_first=True)
X = train_ohe[ ['Item_Visibility', 'Item_MRP',
       'Item_Wt',
       'Item_Fat_Content_Regular', 'Item_Type_Breads', 'Item_Type_Breakfast',
       'Item_Type_Canned', 'Item_Type_Dairy', 'Item_Type_Frozen Foods',
       'Item_Type_Fruits and Vegetables', 'Item_Type_Hard Drinks',
       'Item_Type_Health and Hygiene', 'Item_Type_Household', 'Item_Type_Meat',
       'Item_Type_Others', 'Item_Type_Seafood', 'Item_Type_Snack Foods',
       'Item_Type_Soft Drinks', 'Item_Type_Starchy Foods',
       'Outlet_Establishment_Year_1987', 'Outlet_Establishment_Year_1997',
       'Outlet_Establishment_Year_1998', 'Outlet_Establishment_Year_1999',
       'Outlet_Establishment_Year_2002', 'Outlet_Establishment_Year_2004',
       'Outlet_Establishment_Year_2007', 'Outlet_Establishment_Year_2009',
       'Outlet_Size_Medium', 'Outlet_Size_Small',
       'Outlet_Location_Type_Tier 2', 'Outlet_Location_Type_Tier 3',
        'Outlet_Type_Supermarket Type1', 'Outlet_Type_Supermarket Type2',
       'Outlet_Type_Supermarket Type3']]

Y = train_ohe['Item_Outlet_Sales']
#OneHot Encoding Test
test_ohe = pd.get_dummies(test, columns=['Item_Fat_Content','Item_Type','Outlet_Establishment_Year','Outlet_Size','Outlet_Location_Type','Outlet_Type'], drop_first=True)
X_final_test = test_ohe[ ['Item_Visibility', 'Item_MRP',
       'Item_Wt',
       'Item_Fat_Content_Regular', 'Item_Type_Breads', 'Item_Type_Breakfast',
       'Item_Type_Canned', 'Item_Type_Dairy', 'Item_Type_Frozen Foods',
       'Item_Type_Fruits and Vegetables', 'Item_Type_Hard Drinks',
       'Item_Type_Health and Hygiene', 'Item_Type_Household', 'Item_Type_Meat',
       'Item_Type_Others', 'Item_Type_Seafood', 'Item_Type_Snack Foods',
       'Item_Type_Soft Drinks', 'Item_Type_Starchy Foods',
       'Outlet_Establishment_Year_1987', 'Outlet_Establishment_Year_1997',
       'Outlet_Establishment_Year_1998', 'Outlet_Establishment_Year_1999',
       'Outlet_Establishment_Year_2002', 'Outlet_Establishment_Year_2004',
       'Outlet_Establishment_Year_2007', 'Outlet_Establishment_Year_2009',
       'Outlet_Size_Medium', 'Outlet_Size_Small',
       'Outlet_Location_Type_Tier 2', 'Outlet_Location_Type_Tier 3',
        'Outlet_Type_Supermarket Type1', 'Outlet_Type_Supermarket Type2',
       'Outlet_Type_Supermarket Type3']]

features = ['Item_Visibility', 'Item_MRP',
       'Item_Wt',
       'Item_Fat_Content_Regular', 'Item_Type_Breads', 'Item_Type_Breakfast',
       'Item_Type_Canned', 'Item_Type_Dairy', 'Item_Type_Frozen Foods',
       'Item_Type_Fruits and Vegetables', 'Item_Type_Hard Drinks',
       'Item_Type_Health and Hygiene', 'Item_Type_Household', 'Item_Type_Meat',
       'Item_Type_Others', 'Item_Type_Seafood', 'Item_Type_Snack Foods',
       'Item_Type_Soft Drinks', 'Item_Type_Starchy Foods',
       'Outlet_Establishment_Year_1987', 'Outlet_Establishment_Year_1997',
       'Outlet_Establishment_Year_1998', 'Outlet_Establishment_Year_1999',
       'Outlet_Establishment_Year_2002', 'Outlet_Establishment_Year_2004',
       'Outlet_Establishment_Year_2007', 'Outlet_Establishment_Year_2009',
       'Outlet_Size_Medium', 'Outlet_Size_Small',
       'Outlet_Location_Type_Tier 2', 'Outlet_Location_Type_Tier 3',
        'Outlet_Type_Supermarket Type1', 'Outlet_Type_Supermarket Type2',
       'Outlet_Type_Supermarket Type3']
# Train Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.2, random_state=42)
mod = sm.OLS(y_train, X_train)
res = mod.fit()
print(res.summary())
# Linear Regression
lm = LinearRegression().fit(X_train, y_train)
y_predict = lm.predict(X_train)
y_test_predict = lm.predict(X_test)

print('Linear model, coefficients: ', lm.coef_)
print('Root Mean squared error(Train): {:.2f}'.format(np.sqrt(mean_squared_error(y_train , y_predict))))
print('Root Mean squared error(Test): {:.2f}'.format(np.sqrt(mean_squared_error(y_test , y_test_predict))))
print('r2_score (linear model): {:.2f}'.format(r2_score(y_train, y_predict)))
#checking the magnitude of coefficients

predictors = X_train.columns
coef = pd.Series(lm.coef_,predictors).sort_values()
coef.plot(kind='bar', title='Modal Coefficients')
# Model Evaluation using Cross Validation
scores_train = cross_val_score(linreg, X_train, y_train, scoring="neg_mean_squared_error", cv=5)
linreg_rmse_scores = np.sqrt(-scores)
display_scores(linreg_rmse_scores)
# Export Predicitons
y_test_predict = linridge.predict(X_test_final_scaled)
y_test_predict = np.where(y_test_predict < 0 , 0, y_test_predict )
sum(y_test_predict < 0)
test['Item_Outlet_Sales'] = y_test_predict
submission = test[['Item_Identifier', 'Outlet_Identifier','Item_Outlet_Sales']]
submission.to_csv('submission6.csv',mode = 'w', index=False)
features = ['Item_Visibility', 'Item_MRP', 'Item_Fat_Content_Regular',
            'Outlet_Location_Type_Tier 2', 'Outlet_Location_Type_Tier 3',
            'Outlet_Type_Supermarket Type1', 'Outlet_Type_Supermarket Type2',
            'Outlet_Type_Supermarket Type3']

train_sizes = [1, 100, 500, 2000, 4000, 5113]
# Learning Curve
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import learning_curve
train_sizes, train_scores, validation_scores = learning_curve(
estimator = LinearRegression(),
X = X_train[features],
y = y_train, 
train_sizes = train_sizes, 
cv = 5,
scoring = 'neg_mean_squared_error')
print('Training scores:\n\n', train_scores)
print('\n', '-' * 70) # separator to make the output easy to read
print('\nValidation scores:\n\n', validation_scores)
train_scores_mean = -train_scores.mean(axis = 1)
validation_scores_mean = -validation_scores.mean(axis = 1)
print('Mean training scores\n\n', pd.Series(train_scores_mean, index = train_sizes))
print('\n', '-' * 20) # separator
print('\nMean validation scores\n\n',pd.Series(validation_scores_mean, index = train_sizes))
import matplotlib.pyplot as plt

plt.style.use('seaborn')
plt.plot(train_sizes, train_scores_mean, label = 'Training error')
plt.plot(train_sizes, validation_scores_mean, label = 'Validation error')
plt.ylabel('MSE', fontsize = 14)
plt.xlabel('Training set size', fontsize = 14)
plt.title('Learning curves for a linear regression model', fontsize = 18, y = 1.03)
plt.legend()
plt.ylim(0,2000000)
#for alpha in [0.01,0.05,0.1,0.5, 1, 2, 3, 5, 10, 20, 50]:
linridge = Ridge(alpha=50).fit(X_train, y_train)
y_predict_train = linridge.predict(X_train)
y_predict_test = linridge.predict(X_test)
print('Model results for alpha = {}'.format(alpha) )
print('ridge regression linear model intercept: {}'.format(linridge.intercept_))
#print('ridge regression linear model coeff:\n{}'.format(linridge.coef_))
print('R-squared score (training): {:.3f}'.format(linridge.score(X_train, y_train)))
print('R-squared score (test): {:.3f}'.format(linridge.score(X_test, y_test)))
print('RMSE TRAIN: {:.2f}'.format(np.sqrt(mean_squared_error(y_train, y_predict_train))))
print('RMSE TEST: {:.2f}'.format(np.sqrt(mean_squared_error(y_test, y_predict_test))))
print('Number of non-zero features: {}'.format(np.sum(linridge.coef_ != 0)))
print('#########################################################################')
#checking the magnitude of coefficients

predictors = X_train.columns
coef = pd.Series(linridge.coef_,predictors).sort_values()
coef.plot(kind='bar', title='Modal Coefficients')
# Ridge regression with feature normalization

scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_test_final_scaled = scaler.transform(X_final_test)

linridge = Ridge(alpha=0.01).fit(X_train_scaled, y_train)
y_predict_train = linridge.predict(X_train_scaled) 
y_predict_test = linridge.predict(X_test_scaled)

#print('Model results for alpha = {}'.format(alpha) )
print('ridge regression linear model intercept: {}'.format(linridge.intercept_))
print('ridge regression linear model coeff:\n{}'.format(linridge.coef_))
print('R-squared score (training): {:.3f}'.format(linridge.score(X_train_scaled, y_train)))
print('R-squared score (test): {:.3f}'.format(linridge.score(X_test_scaled, y_test)))
print('Number of non-zero features: {}'
.format(np.sum(linridge.coef_ != 0)))
print('RMSE TRAIN: {:.2f}'.format(np.sqrt(mean_squared_error(y_train, y_predict_train))))
print('RMSE TEST: {:.2f}'.format(np.sqrt(mean_squared_error(y_test, y_predict_test))))
print('#########################################################################')
predictors = X_train.columns
coef = pd.Series(linridge.coef_,predictors).sort_values()
coef.plot(kind='bar', title='Modal Coefficients')
from sklearn.linear_model import Lasso
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#for alpha in [0.01, 0.1, 0.5, 1, 2, 3, 5, 10, 20, 50]:
linlasso = Lasso(alpha=0.05, max_iter = 10000).fit(X_train_scaled, y_train)
y_predict = linlasso.predict(X_test_scaled)
print('ridge regression linear model intercept: {}'.format(linlasso.intercept_))
print('ridge regression linear model coeff:\n{}'.format(linlasso.coef_))
print('R-squared score (training): {:.3f}'.format(linlasso.score(X_train_scaled, y_train)))
print('R-squared score (test): {:.3f}'.format(linlasso.score(X_test_scaled, y_test)))
print('Number of non-zero features: {}'.format(np.sum(linlasso.coef_ != 0)))
print('RMSE TRAIN: {:.2f}'.format(np.sqrt(mean_squared_error(y_train, y_predict_train))))
print('RMSE TEST: {:.2f}'.format(np.sqrt(mean_squared_error(y_test, y_predict_test))))
print('#########################################################################')
predictors = X_train.columns
coef = pd.Series(linlasso.coef_,predictors).sort_values()
coef.plot(kind='bar', title='Modal Coefficients')
for epsilon in [1,3,5,7,9]:
    svm_reg = LinearSVR(epsilon=epsilon)
    svm_reg.fit(X_train,y_train)
    y_predict_train = svm_reg.predict(X_train)
    y_predict_test = svm_reg.predict(X_test)
    print('Model results for epsilon = {}'.format(epsilon) )
    print('ridge regression linear model intercept: {}'.format(svm_reg.intercept_))
    #print('ridge regression linear model coeff:\n{}'.format(linridge.coef_))
    print('R-squared score (training): {:.3f}'.format(svm_reg.score(X_train, y_train)))
    print('R-squared score (test): {:.3f}'.format(svm_reg.score(X_test, y_test)))
    print('RMSE TRAIN: {:.2f}'.format(np.sqrt(mean_squared_error(y_train, y_predict_train))))
    print('RMSE TEST: {:.2f}'.format(np.sqrt(mean_squared_error(y_test, y_predict_test))))
    print('#################################################################################')
svm_poly_reg = SVR(kernel="poly", degree=2, C=100, epsilon=0.1)
svm_poly_reg.fit(X_train,y_train)
y_predict_train = svm_poly_reg.predict(X_train)
y_predict_test = svm_poly_reg.predict(X_test)
print('Model results for epsilon = {}'.format(epsilon) )
print('ridge regression linear model intercept: {}'.format(svm_poly_reg.intercept_))
#print('ridge regression linear model coeff:\n{}'.format(linridge.coef_))
print('R-squared score (training): {:.3f}'.format(svm_poly_reg.score(X_train, y_train)))
print('R-squared score (test): {:.3f}'.format(svm_poly_reg.score(X_test, y_test)))
print('RMSE TRAIN: {:.2f}'.format(np.sqrt(mean_squared_error(y_train, y_predict_train))))
print('RMSE TEST: {:.2f}'.format(np.sqrt(mean_squared_error(y_test, y_predict_test))))
print('#################################################################################')
from sklearn.tree import DecisionTreeRegressor

for depth in [2,3,4,5,6,7,8,9]:
    tree_reg = DecisionTreeRegressor(max_depth=depth)
    tree_reg.fit(X_train, y_train)

    y_predict_train = tree_reg.predict(X_train)
    y_predict_test = tree_reg.predict(X_test)
    
    print('Depth equals: {:.1f}'.format(depth))
    #print('ridge regression linear model intercept: {}'.format(tree_reg.intercept_))
    #print('ridge regression linear model coeff:\n{}'.format(tree_reg.coef_))
    print('R-squared score (training): {:.3f}'.format(tree_reg.score(X_train, y_train)))   
    print('RMSE Train: {:.2f}'.format(np.sqrt(mean_squared_error(y_train, y_predict_train))))
    print('RMSE Test: {:.2f}'.format(np.sqrt(mean_squared_error(y_test, y_predict_test))))
    #print('Number of non-zero features: {}'.format(np.sum(tree_reg.coef_ != 0)))
    print('#########################################################################')
# Model Evaluation with Cross Validation
scores = cross_val_score(tree_reg, X_train, y_train, scoring="neg_mean_squared_error", cv=5)
tree_reg_rmse_scores = np.sqrt(-scores)
display_scores(tree_reg_rmse_scores)
# Export Model Predictions
y_test_predict = tree_reg.predict(X_final_test)
y_test_predict = np.where(y_test_predict < 0 , 0, y_test_predict )
test['Item_Outlet_Sales'] = y_test_predict
submission = test[['Item_Identifier', 'Outlet_Identifier','Item_Outlet_Sales']]
submission.to_csv('submission1.csv',mode = 'w', index=False)
# Random Forest Model
#for n in [3,6,9,12,15,18,21,24]:
forest_reg = RandomForestRegressor(n_estimators= 500 , random_state=0)
forest_reg.fit(X_train, y_train)
y_predict_train = forest_reg.predict(X_train)
y_predict_test = forest_reg.predict(X_test)
#print('n_estimators: {:.1f}'.format(n))
#print('ridge regression linear model intercept: {}'.format(tree_reg.intercept_))
#print('ridge regression linear model coeff:\n{}'.format(tree_reg.coef_))
print('R-squared score (training): {:.3f}'.format(forest_reg.score(X_train, y_train)))
print('R-squared score (test): {:.3f}'.format(forest_reg.score(X_test, y_test)))
print('RMSE Train: {:.2f}'.format(np.sqrt(mean_squared_error(y_train, y_predict_train))))
print('RMSE Test: {:.2f}'.format(np.sqrt(mean_squared_error(y_test, y_predict_test))))
#print('Number of non-zero features: {}'.format(np.sum(tree_reg.coef_ != 0)))
print('#########################################################################')
# Feature importances
for name, score in zip(train_ohe.columns, forest_reg.feature_importances_):
    print(name, score)
feature_imp = pd.DataFrame({'features':features, 'scores':forest_reg.feature_importances_})
feature_imp
# Model Evaluation with Cross Validation
scores = cross_val_score(forest_reg, X_train, y_train, scoring="neg_mean_squared_error", cv=5)
forest_reg_rmse_scores = np.sqrt(-scores)
display_scores(forest_reg_rmse_scores)
# Hyperparameter Tuning

param_grid = [
    {'n_estimators':[3,10,30,50], 'max_features':[4,8,12,16,20,24,28,32]},
    {'bootstrap':[False],'n_estimators':[3,10],'max_features': [4,8,12,16,20]}
];

forest_regr = RandomForestRegressor()
grid_search = GridSearchCV(forest_regr, param_grid, cv=5, scoring = 'neg_mean_squared_error', return_train_score=True)
grid_search.fit(X_train, y_train)     
     
# Best params
print(grid_search.best_params_)

#Best Estimator
print(grid_search.best_estimator_)
xgb_reg = xgboost.XGBRegressor()
xgb_reg.fit(X_train, y_train)
y_predict = xgb_reg.predict(X_test)
#print('n_estimators: {:.1f}'.format(n))
#print('ridge regression linear model intercept: {}'.format(tree_reg.intercept_))
#print('ridge regression linear model coeff:\n{}'.format(tree_reg.coef_))
print('R-squared score (training): {:.3f}'.format(xgb_reg.score(X_train, y_train)))
print('R-squared score (test): {:.3f}'.format(xgb_reg.score(X_test, y_test)))
print('Root Mean squared error (linear model): {:.2f}'.format(np.sqrt(mean_squared_error(y_test, y_predict))))
#print('Number of non-zero features: {}'.format(np.sum(tree_reg.coef_ != 0)))
print('#########################################################################')
y_test_predict = xgb_reg.predict(X_final_test)
y_test_predict = np.where(y_test_predict < 0 , 0, y_test_predict )
test['Item_Outlet_Sales'] = y_test_predict
submission = test[['Item_Identifier', 'Outlet_Identifier','Item_Outlet_Sales']]
submission.to_csv('submission8.csv',mode = 'w', index=False)
