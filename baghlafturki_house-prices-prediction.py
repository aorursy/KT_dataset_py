# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

import numpy as np



#for visualization

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns







from sklearn.model_selection import train_test_split, cross_val_score, KFold

from sklearn.metrics import r2_score





from sklearn.tree import DecisionTreeRegressor 

from sklearn.linear_model import Lasso, Ridge, ElasticNet, LinearRegression

from sklearn.model_selection import GridSearchCV,cross_validate

from sklearn.metrics import mean_squared_error





from scipy.stats import normaltest, norm, skew



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

df1 = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

# this is not to skip some columns when showing the dataframe

pd.set_option('display.max_columns', df.shape[-1])
print("The shape of train is {}\nThe shape of test is {}".format(df.shape,df1.shape))
df.describe()
df1.describe()
# For train.csv

plt.figure(figsize=(15, 4))

sns.heatmap(df.isna(),cbar=False)
#For test.csv

plt.figure(figsize=(15, 4))

sns.heatmap(df1.isna(),cbar=False)
# getting the correlation table

corr = df.corr()

# selecting only correlations that have a strength above 0.5 both ways

top_corr = corr.index[abs(corr['SalePrice'])>0.5]



# plot it in a heatmap

plt.figure(figsize = (20,20))

sns.heatmap(df[top_corr].corr(),cbar=True, annot=True)
#WARNING: This will take very long to execute

sns.set()

sns.pairplot(df[top_corr], height = 2.5)
fig, ax = plt.subplots()

ax.scatter(x = df['GrLivArea'], y = df['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('GrLivArea', fontsize=13)

plt.show()
# removing some outliers

df = df.drop(df[(df['GrLivArea']>4000) & (df['SalePrice']>100000)].index)
fig, ax = plt.subplots()

ax.scatter(x = df['GrLivArea'], y = df['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('GrLivArea', fontsize=13)

plt.show()
fig, ax = plt.subplots()

ax.scatter(x = df['GrLivArea'], y = df['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('GrLivArea', fontsize=13)

plt.show()
df = df.drop(df[(df['1stFlrSF']>2700) & (df['SalePrice']>100000)].index)
fig, ax = plt.subplots()

ax.scatter(x = df['1stFlrSF'], y = df['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('1stFlrSF', fontsize=13)

plt.show()
fig, ax = plt.subplots()

ax.scatter(x = df['GarageArea'], y = df['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('GarageArea', fontsize=13)

plt.show()
df = df.drop(df[(df['GarageArea']>1240) & (df['SalePrice']<300000)].index)
fig, ax = plt.subplots()

ax.scatter(x = df['GarageArea'], y = df['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('GarageArea', fontsize=13)

plt.show()
fig, ax = plt.subplots()

ax.scatter(x = df['TotalBsmtSF'], y = df['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('TotalBsmtSF', fontsize=13)

plt.show()
df = df.drop(df[(df['TotalBsmtSF']>3000) & (df['SalePrice']>200000)].index)
fig, ax = plt.subplots()

ax.scatter(x = df['TotalBsmtSF'], y = df['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('TotalBsmtSF', fontsize=13)

plt.show()
fig, ax = plt.subplots()

ax.scatter(x = df['YearRemodAdd'], y = df['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('YearRemoAdd', fontsize=13)

plt.show()



df = df.drop(df[(df['YearRemodAdd']<1970) & (df['SalePrice']>300000)].index)

df = df.drop(df[(df['YearRemodAdd']<2000) & (df['SalePrice']>600000)].index)

fig, ax = plt.subplots()

ax.scatter(x = df['YearRemodAdd'], y = df['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('YearRemoAdd', fontsize=13)

plt.show()
fig, ax = plt.subplots()

ax.scatter(x = df['YearBuilt'], y = df['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('YearBulit', fontsize=13)

plt.show()

df = df.drop(df[(df['YearBuilt']<1900) & (df['SalePrice']>400000)].index)


fig, ax = plt.subplots()

ax.scatter(x = df['YearBuilt'], y = df['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('YearBulit', fontsize=13)

plt.show()
target = df.pop('SalePrice')
#concatinating the two dataframes

whole_df = pd.concat([df,df1],keys=[0,1])

# Sanity check to ensure the total length matches 

whole_df.shape[0] == df.shape[0]+df1.shape[0]
whole_IDs = whole_df.pop('Id')
non_num_cols = whole_df.select_dtypes(include=object).columns.tolist()

num_cols = whole_df.select_dtypes(include=np.number).columns.tolist()

#checking column discription and adding each column containing NA as a value to the list

check_list = ['Alley','BsmtQual',"BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2","FireplaceQu","GarageType",'GarageFinish','GarageQual',"GarageCond",'PoolQC','Fence','MiscFeature']
def unique_values(df,cols,n_unique_vals=None, return_cols=False):

    '''

    Given the dataframe and feature names, this function will print out or return

    all the unique values in each column.

    

    In case n_unique_vals is set to an integer, any column which has the number

    of unique values over the value passed as an argument will only be printed

    '''

    output = []  #to accumulate the column names (only usefull when return_cols is True)

    

    for col in cols:

        # if user didnt specify a threshold

        if n_unique_vals == None: 

            print("{}: {}".format(col,df[col].unique()))

            output.append(col)

        # if number of unique values in the feature is more than specified by user

        elif n_unique_vals >= len(df[col].unique()):

            print("{}: {}".format(col,df[col].unique()))

            output.append(col)

    # if user wants the name of the columns to be returned

    if return_cols:

        return output

    

    

def make_category_numeric(df,col):

    '''

    This function helps convert non numeric categorical features to numeric categorical features

    it is used to find the distribution

    '''

    # count how many unique values in the column

    # for each unique value, mape all similar values to a number starting from 0

    return df[col].dropna().map({value:idx for idx,value in enumerate(df[col].unique())})









def missing_percentage(df, cols, verbose = 1, threshold = 0.00, return_output=False):

    '''

    This function calculates the percentage of missing enteries in each column in cols

    Threshold is a percentage of which the feature will only be shown if the percentage

    of missing data in the feature is more than the percentage specified

    '''

    result = []

    for col in cols:

        

        # calculate the percentage of missing data in the column

        percentage = df[col].isna().sum() / df[col].shape[0]

        

        # user wants to see the percentage of every column

        if verbose == 2:

            print("Missing data in {} is {:.2%}".format(col,percentage))

            

        # user only wants to see for columns which actually has missing data

        elif verbose == 1 and percentage > float(threshold):

            print("Missing data in {} is {:.2%}".format(col,percentage))

            

        # this is used with return output to return columns which the missing

        # data percentage is above the specified threshold

        if percentage > float(threshold):

            result.append(col)

            

    if return_output:

        return result







def classify_distribution(df,columns,numeric=True):

    '''This function runs a normality test using the p-value. The threshold is set to 0.05'''

    result = []

    for col in columns:

        num_cat_col = make_category_numeric(df,col) if not numeric else df[col].dropna()



            # run normality test >> reference:#http://mathforum.org/library/drmath/view/72065.html

        if normaltest(num_cat_col)[-1] > 0.05: 

            # if test succeeds it means it is normal

            result.append((col,0))

        else:

            # if not normal then return the skew

            result.append((col,num_cat_col.skew()))



    return result







# show only categorical columns that have missing data        

missing_percentage(whole_df,non_num_cols,verbose=1)
unique_values(whole_df,check_list)
whole_df[check_list] = whole_df[check_list].fillna(value='NA',axis=1)
# rechecking the unique values after filling nun with NA

unique_values(whole_df,check_list)
# checking if they still have a nan values

missing_percentage(whole_df,check_list,verbose=1)

#nothing is returned which means no nan values, feel free to change verbose to 2
missing_categorical = missing_percentage(whole_df,non_num_cols,verbose=1,return_output=True)
# getting all the columns that have less than 25 unique values in them

numeric_category = unique_values(whole_df,num_cols,25, return_cols=True)
# getting the names of those columns

numeric_cats = unique_values(whole_df,num_cols,25, return_cols=True)
#change it to type string to dummify it later

whole_df[numeric_cats] = whole_df[numeric_cats].astype(str)

whole_df[numeric_cats].info()
missing_categorical = missing_categorical+numeric_cats



# plotting a heatmap to see whether there are any missing values

plt.figure(figsize=(15, 4))

sns.heatmap(whole_df[missing_categorical].isna(),cbar=False)
whole_df['MasVnrType'].fillna('None',inplace=True)
whole_df['MasVnrArea'].fillna(0,inplace=True)
plt.figure(figsize=(15, 4))

sns.heatmap(whole_df[missing_categorical].isna(),cbar=False)
# this is because some columns in num_cols is categorical is seen above

actual_numeric = [col for col in num_cols if col not in numeric_cats]
# checking how where the nan is in our continuos features

plt.figure(figsize=(15, 4))

sns.heatmap(whole_df[actual_numeric].isna(),cbar=False)
whole_df['GarageYrBlt'].fillna(whole_df['YearBuilt'], inplace=True)
# checking for changes

plt.figure(figsize=(15, 4))

sns.heatmap(whole_df[actual_numeric].isna(),cbar=False)
# grouping per neighborhood

# replacing nan values with the median of the neighborhood group



whole_df["LotFrontage"] = whole_df.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
#checking for changes

plt.figure(figsize=(15, 4))

sns.heatmap(whole_df[num_cols].isna(),cbar=False)
#checking for missing enteries

missing_percentage(whole_df,whole_df.columns, verbose = 1)
# All of these will be filled with 0 as the nan values are present when there is no basement or no garage or NA somewhere

whole_df.fillna(0,inplace=True)
#checking again

missing_percentage(whole_df,whole_df.columns, verbose = 1)
# Check the skew of all numerical features

numeric_skew = classify_distribution(whole_df,actual_numeric)

col_names = [col for col, val in numeric_skew]

skewness = [val for col, val in numeric_skew]



sns.barplot(x=skewness, y=col_names).set_title('Skewness of features')
# this will contain all the numerical features

to_log = actual_numeric.copy()



# this contains some features that we do not want to transform

# transforming them will increase their skewness in the other direction

to_remove = ['GarageYrBlt','TotalBsmtSF','GarageArea','BsmtUnfSF']



# removing the columns in to_remove from to_log

for x in to_remove:

    to_log.remove(x)



# Apply log transformation to the remaining column sin to_log

whole_df[to_log] = np.log1p(whole_df[to_log])





# This is for plotting-----------------------------------

numeric_skew = classify_distribution(whole_df,actual_numeric)

col_names = [col for col, val in numeric_skew]

skewness = [val for col, val in numeric_skew]



sns.barplot(x=skewness, y=col_names).set_xlim((-2,23))
# non_num_cols contains all categorical columns from the start

# numeric_cats contains numeric columns that we converted to string

all_category_cols = non_num_cols + numeric_cats





for col in all_category_cols:

    # get the dummies of that column

    dummy = pd.get_dummies(whole_df[col],prefix=col)

    # concatenate it with the dataframe 

    whole_df = pd.concat([whole_df,dummy], axis= 1)



    # remove the old categorical columns

whole_df.drop(labels=all_category_cols, axis=1,inplace=True)



# log transforming the target 

target = np.log1p(target)



# readding the IDs back

whole_df['Id'] = whole_IDs



# separating the two dataframes (train.csv and test.csv)

df,df1 = whole_df.xs(0),whole_df.xs(1)



# readding the target to train.csv

df['SalePrice'] = target
# Making X and y data to train the model



X = df.drop(['SalePrice','Id'],axis=1)

y = df['SalePrice']



# this is used for experimentation only

x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
# making a linear regression

lnr = LinearRegression()

# train it on train data from test_train split

lnr.fit(x_train, y_train)

y_hat = lnr.predict(x_test)

# measure the RMSE

linear_regression_rmse = np.sqrt(mean_squared_error(y_test,y_hat))

linear_regression_rmse


# making folds to be 5 (it will be used for all the CV below)

kf = KFold(n_splits=5, shuffle=True)





lnr_cv = LinearRegression()



# getting a score after training and validating the model on 5 folds

cv_results = cross_val_score(lnr_cv, X, y, cv=kf,scoring='neg_root_mean_squared_error')

np.mean(cv_results)

# doing the same process but for lasso



lasso = Lasso(random_state=100)

lasso.fit(x_train,y_train)



y_hat = lasso.predict(x_test)

lasso_rmse = np.sqrt(mean_squared_error(y_test,y_hat))

lasso_rmse
# Using cross-validation with lasso



lasso_cv = Lasso(random_state=100)

cv_results = cross_val_score(lasso_cv, X, y, cv=kf,scoring='neg_root_mean_squared_error')

lasso_cv_rmse = np.mean(cv_results)

lasso_cv_rmse
rigde = Ridge(random_state=100)

rigde.fit(x_train,y_train)



y_hat = rigde.predict(x_test)

rigde_rmse = np.sqrt(mean_squared_error(y_test,y_hat))

rigde_rmse
ridge_cv = Ridge(random_state=100)

cv_results = cross_val_score(ridge_cv, X, y, cv=kf,scoring='neg_root_mean_squared_error')

ridge_cv_rmse = np.mean(cv_results)

ridge_cv_rmse
elastic = ElasticNet(random_state=100)

elastic.fit(x_train,y_train)



y_hat = elastic.predict(x_test)

elastic_rmse = np.sqrt(mean_squared_error(y_test,y_hat))

elastic_rmse
dct = DecisionTreeRegressor(random_state=100)

dct.fit(x_train,y_train)



y_hat = dct.predict(x_test)

dct_rmse = np.sqrt(mean_squared_error(y_test,y_hat))

dct_rmse
dct_cv = DecisionTreeRegressor(random_state=100)



cv_results = cross_val_score(dct_cv, X, y, cv=kf,scoring='neg_root_mean_squared_error')

dct_cv_rmse = np.mean(cv_results)

dct_cv_rmse
lasso_gc = Lasso(random_state=100)

params = {'alpha':[0.0008,0.0007,0.0009,0.001,0.002,0.003,0.6,0.7,0.8],'max_iter':[10000,1000]}

lasso_gc = GridSearchCV(lasso_gc, params,cv=kf,scoring='neg_root_mean_squared_error')



lasso_gc.fit(X,y)

lasso_gc.best_score_
lasso_gc.best_params_
ridge_gc = Ridge(random_state=100)

params = {'alpha':[0.8,0.9,1,1.1],'max_iter':[10000,1000]}

ridge_gc = GridSearchCV(ridge_gc, params,cv=kf,scoring='neg_root_mean_squared_error')



ridge_gc.fit(X,y)

ridge_gc.best_score_
ridge_gc.best_params_
elastic_gc = ElasticNet()

params = {'l1_ratio':[0.9,0.8,0.7,0.1,0.2,0.3,0.4],'max_iter':[10000,1000]}

elastic_gc = GridSearchCV(elastic_gc, params,cv=kf,scoring='neg_root_mean_squared_error')



elastic_gc.fit(X,y)

elastic_gc.best_score_
elastic_gc.best_params_
dct_select = DecisionTreeRegressor(random_state=24)



params = {'max_depth':[2,5,8,10,12,15],'min_samples_split':[20,25,30,40,50,60],'min_samples_leaf':[1,4,10,20,25,30],"max_features":['auto']}

dct_select_gc = GridSearchCV(dct_select, params,cv=kf,scoring='neg_root_mean_squared_error')



dct_select_gc.fit(X,y)

dct_select_gc.best_score_
dct_select_gc.best_params_
lasso_model = Lasso(alpha=0.0007, max_iter=10000)

lasso_model.fit(X,y)

coefs = [idx for idx, col in enumerate(lasso_model.coef_) if abs(col) > 0.00005]

print("{} features selected".format(len(coefs)))



selected_cols = [X.columns[col_index] for col_index in coefs]
select_lnr = LinearRegression()

cv_results = cross_val_score(select_lnr, X[selected_cols], y, cv=kf,scoring='neg_root_mean_squared_error')

np.mean(cv_results)

# improved 
ridge_cv_select = Ridge(alpha= 9.5, max_iter= 10000)

cv_results = cross_val_score(ridge_cv_select, X[selected_cols], y, cv=kf,scoring='neg_root_mean_squared_error')

np.mean(cv_results)

elastic_select = ElasticNet(l1_ratio=0.1)

cv_results = cross_val_score(elastic_select, X[selected_cols], y, cv=kf,scoring='neg_root_mean_squared_error')

np.mean(cv_results)
dct_select = DecisionTreeRegressor(max_depth= 12,max_features= 'auto',min_samples_leaf= 10,min_samples_split=20)

cv_results = cross_val_score(dct_select, X[selected_cols], y, cv=kf,scoring='neg_root_mean_squared_error')

np.mean(cv_results)
# ridge on feature selection

ridge_cv = Ridge(alpha=0.99, max_iter=10000)

cv_results = cross_val_score(ridge_cv, X[selected_cols], y, cv=kf,scoring='neg_root_mean_squared_error')

ridge_cv_rmse = np.mean(cv_results)

ridge_cv_rmse
# ridge without feature selection

ridge_cv = Ridge(alpha=0.99, max_iter=10000)

cv_results = cross_val_score(ridge_cv, X, y, cv=kf,scoring='neg_root_mean_squared_error')

ridge_cv_rmse = np.mean(cv_results)

ridge_cv_rmse
# linear reg with feature selection

l = LinearRegression()

cv_results = cross_val_score(l, X[selected_cols], y, cv=kf,scoring='neg_root_mean_squared_error')

l_cv_rmse = np.mean(cv_results)

l_cv_rmse
# linear reg without feature selection

l = LinearRegression()

cv_results = cross_val_score(l, X, y, cv=kf,scoring='neg_root_mean_squared_error')

l_cv_rmse = np.mean(cv_results)

l_cv_rmse
ridge = Ridge(alpha=0.99, max_iter=10000)

ridge.fit(X[selected_cols],y)
# IDs is not used in prediction

ids = df1.pop('Id')
X_test = df1



# using the model to predict the house prices

yhat = ridge.predict(X_test[selected_cols])
# inverse the log transforem and store it in the dataframe

outdf = pd.DataFrame({'Id':ids,'SalePrice':np.expm1(yhat)})

#save it as a csv file

outdf.to_csv('/kaggle/working/submissionv6.csv', index=False)