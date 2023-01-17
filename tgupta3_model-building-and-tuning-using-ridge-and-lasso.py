'''
    importing the required libraries here
'''

import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from statsmodels.stats.outliers_influence import variance_inflation_factor
'''
    Importing the library for PCA
'''
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')
path =r'../input/train.csv'

df = pd.read_csv(path)

df.info()
'''
    this method will find the null percentage of all the columns in the dataframe passed to it
'''
def find_null_per(df):
    return round((df.isnull().sum()/len(df))*100,2).sort_values(ascending=False)
'''
    Checking for the null percentage in the dataframe df
'''
find_null_per(df)
'''
    This function will replace all the blank values for the passed dataframe to NA values. which will be acted as a string.
'''
def replace_val(df):
    for i in df.columns:
        df[i] = df[i].fillna('NA')
    return df

'''
    na_col_val variable has list of all those columns, which has null values, and all those null values should be considered
    as a value of the column and not the nan value.
'''

na_col_val = ['Alley','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','FireplaceQu','GarageType',
 'GarageFinish','GarageQual','GarageCond','PoolQC','Fence','MiscFeature']
df[na_col_val] = replace_val(df[na_col_val])
df_copy = df.copy(deep=True)
'''
    now we will check for all the columns where the data is skewed and then will remove all those columns, where the 
    one of the column value is greater than 80
'''

def feature_removal(df,col,skew_val):
    lst=[]
    for i in col:
        if round(df[i].value_counts()/len(df)*100,2)[0] > skew_val:
            df = df.drop(i, axis =1)
            lst.append(i)        
    print("removed columns are {val}".format(val = lst))
    return df       
df = feature_removal(df,na_col_val,skew_val=80)
'''
    list of removed columns are -:
    removed columns are ['Alley', 'BsmtCond', 'BsmtFinType2', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature']
    
    Here we are doing EDA on few columns. We will be checking if the columns which have been removed above were causing
    biasness in the data or not.
    
    And from the below graph it is evident that the columns were biased towards one of the categories in the categorical 
    column.
'''

#in order to visualise a categorical variable we should use a box plot
plt.figure(figsize=(5, 5))

lst = ['Alley', 'BsmtCond', 'BsmtFinType2', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature']

l = len(lst)/2
for i in lst:
    sns.countplot(x = i, data = df_copy)
    plt.show()

#boxplot boundaries represents - 25%, median, 75 %
df.head()
find_null_per(df)
'''
    Imputing the left over NA values, and checking what to do with the NA value. Whether to use mean, median or mode. 
    ]And how to decide that.
'''
df['LotFrontage'].describe(percentiles = [0.25,0.50,0.75,0.90,0.95,0.99])
'''
    It seems, median is a good option to deal with this numeric column for now.
'''

df['LotFrontage'] = df['LotFrontage'].fillna(value = df['LotFrontage'].median())
df['LotFrontage'].describe(percentiles = [0.25,0.50,0.75,0.90,0.95,0.99])
'''
    Imputing the left over NA values, and checking what to do with the NA value.
    Whether to use mean, median or mode. And how to decide that.
'''

df['GarageYrBlt'].describe(percentiles = [0.25,0.50,0.75,0.90,0.95,0.99])
'''
    It seems, median is a good option to deal with this numeric year column also.
    we are imputing this first, because with this we can change the year column easily
'''

df['GarageYrBlt'] = df['GarageYrBlt'].fillna(value = df['GarageYrBlt'].median())
df['GarageYrBlt'].describe(percentiles = [0.25,0.50,0.75,0.90,0.95,0.99])
'''
    converting GarageYrBlt and YrSold to the age columns and will drop this columnn at the end.
'''

df['GarageYrBlt_age'] = max(df['GarageYrBlt']) - df['GarageYrBlt']
df['YrSold_age'] = max(df['YrSold']) - df['YrSold']
df['YearBuilt_age'] = max(df['YearBuilt']) - df['YearBuilt']
df['YearRemodAdd_age'] = max(df['YearRemodAdd']) - df['YearRemodAdd']

df = df.drop('GarageYrBlt', axis =1)
df = df.drop('YrSold', axis =1)
df = df.drop('YearBuilt', axis =1)
df = df.drop('YearRemodAdd', axis =1)
sns.pairplot(df)
'''
    checking for the percentage of null values.
'''
find_null_per(df)
df.head()
df = df.drop('Id', axis = 1)
'''
    Dropping the left over Null values as they are very less and would not impact much with the model creation.
'''

df = df.dropna()
find_null_per(df)
'''
    here i am checking if my target variable is normally distributed or not. And it seems the target variable is 
    skewed towards right. So, i will be using the logarithmic function to remove the skewness from the target
    variable
'''

# Plot the histogram of the target variable
fig = plt.figure()
sns.distplot(df['SalePrice'], bins = 20)
fig.suptitle('Sale Price', fontsize = 20)                  # Plot heading 
plt.xlabel('Sales', fontsize = 18)                         # X-label
'''
    using the np.log function to normalize the target variable
'''
log_func = np.log(df.SalePrice)
df['SalePrice'] = log_func
df.describe()
'''
    Here, we can see that after using the logarithmic function, we have received the target variable as a normally
    distributed variable
'''
# Plot the histogram of the target variable
fig = plt.figure()
sns.distplot(df['SalePrice'], bins = 20)
fig.suptitle('Sale Price', fontsize = 20)                  # Plot heading 
plt.xlabel('Sales', fontsize = 18)                         # X-label
'''
    we will start the analysis with the heatmap which will tell us the correlations of the 
    columnms
'''
plt.figure(figsize = (20,16))
sns.heatmap(df.corr(), annot=True)
'''
    Here we are creating two variables df_num which will have numerical variables and df_Rest which will have all the 
    variables except for the target variable.
    
    The df_temp created here will be used to find the multi collinearity of the numerical columns
'''
df_temp = df.drop(['SalePrice'], axis = 1)
df_num = np.array(df_temp.select_dtypes(include=[np.number]).columns.values)
df_rest = set(df.columns) - set(df_num)
df_temp[df_num].head()
df_temp.columns
'''
    I have created this method to automate the VIF process of feature removal. In this method, I am passing
    a dataframe and the set of numerical columns to prcoeed with the removal of collinear columns.
    For this assignment, the value for accepted coefficient is less than 5.
    
    Also, this method will print the order in which we are deleting the columns from the data frame
'''
def calculate_vif(df,col):
    vif = pd.DataFrame()
    X = df[col]
    vif['Features'] = X.columns
    vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif['VIF'] = round(vif['VIF'], 2)
    vif = vif.sort_values(by = "VIF", ascending = False)
    if vif.iloc[0][1] < 5 :
        return X
    else:
        Y = X.drop(vif.iloc[0][0], axis =1)
        print("column dropped as part of VIF is  {val}".format(val = vif.iloc[0][0]))
        return(calculate_vif(X,Y.columns))
'''
    calling calculate_vif which will iteratively remove the columns with more than 5 coefficient value. This deletes
    one coefficient at a time.
'''
df_vif = calculate_vif(df_temp,df_num)
'''
    plotting the heatmap after removing most of the multicollinear columns
'''
plt.figure(figsize = (20,16))
sns.heatmap(df_vif.corr(), annot=True)
plt.figure(figsize=(16,8))

plt.subplot(2,3,1)
sns.boxplot(x = 'LotArea', data = df_vif) 

plt.subplot(2,3,2)
sns.boxplot(x = 'MasVnrArea', data = df_vif)  

plt.subplot(2,3,3)
sns.boxplot(x = 'BsmtFinSF1', data = df_vif)

plt.subplot(2,3,4)
sns.boxplot(x = 'BsmtFinSF2', data = df_vif)

plt.subplot(2,3,5)
sns.boxplot(x = 'BsmtUnfSF', data = df_vif)

plt.subplot(2,3,6)
sns.boxplot(x = '2ndFlrSF', data = df_vif)

plt.show()
plt.figure(figsize=(16,8))

plt.subplot(2,3,1)
sns.boxplot(x = 'LowQualFinSF', data = df_vif) 

plt.subplot(2,3,2)
sns.boxplot(x = 'BsmtFullBath', data = df_vif)  

plt.subplot(2,3,3)
sns.boxplot(x = 'BsmtHalfBath', data = df_vif)

plt.subplot(2,3,4)
sns.boxplot(x = 'HalfBath', data = df_vif)

plt.subplot(2,3,5)
sns.boxplot(x = 'Fireplaces', data = df_vif)

plt.subplot(2,3,6)
sns.boxplot(x = 'WoodDeckSF', data = df_vif)

plt.show()
plt.figure(figsize=(16,8))

plt.subplot(2,3,1)
sns.boxplot(x = 'OpenPorchSF', data = df_vif) 

plt.subplot(2,3,2)
sns.boxplot(x = 'EnclosedPorch', data = df_vif)  

plt.subplot(2,3,3)
sns.boxplot(x = '3SsnPorch', data = df_vif)

plt.subplot(2,3,4)
sns.boxplot(x = 'ScreenPorch', data = df_vif)

plt.subplot(2,3,5)
sns.boxplot(x = 'PoolArea', data = df_vif)

plt.subplot(2,3,6)
sns.boxplot(x = 'MiscVal', data = df_vif)

plt.show()
plt.figure(figsize=(16,8))

plt.subplot(2,3,1)
sns.boxplot(x = 'GarageYrBlt_age', data = df_vif) 

plt.subplot(2,3,2)
sns.boxplot(x = 'YrSold_age', data = df_vif)  

plt.subplot(2,3,3)
sns.boxplot(x = 'YearRemodAdd_age', data = df_vif)

df_vif.shape
'''
    This function will help us in removing the outliers from a list of columns and a dataframe passed to this
'''
def remove_outliers(df,col):
    for i in col:
        Q1 = df[i].quantile(0.05)
        Q3 = df[i].quantile(0.97)
        df = df[(df[i] >=Q1) &(df[i] <=Q3)]
    return df
df_no_outlier = remove_outliers(df_vif, list(df_vif.columns))
df_no_outlier.shape
'''
    concatenating the data recieved after removing the multicollinearity from the raw data.
'''

df1 = df[df_vif.columns]
df3 = df[df_rest]

df_final = pd.concat([df1,df3] , axis = 1)
df_final.shape
'''
    there is one variable which is part of df_num showing numeric columns, where as it is a categorical column.
    Hence changing the type to object.
'''
df_final['MSSubClass'] = df_final['MSSubClass'].astype(object)
'''
    getting a list of columns which have data type as objects, to get the list of categorical columns.
'''

others = ['object']
df_cat = np.array(df_final.select_dtypes(exclude=[np.number]).columns.values)
df_cat
'''
    generating the dummy variables from a list of category variables df_cat.
'''

df_dummies = pd.get_dummies(df_final[df_cat], drop_first = True)
df_dummies.shape

'''
    printing the dummy columns, to verify if the dummy variables are generated fine or not.
'''

df_dummies.columns
'''
    Here I am concatenating the original data frame with the dummy variables which were created. This is done to get the 
    entire dataframe in 1's and 0's for a Linear Regression.
'''

df_final_1 = pd.concat([df_final, df_dummies], axis = 1)
df_final_1.shape
'''
    since the dummy variables are created and concatenated to the actual dataframe hence, i am removing the original columns
    so that we donot have redundancy in the columns.
'''

df_final_1 = df_final_1.drop(df_cat, axis = 1)
df_final_1.shape
df_final_1.columns
df_copy.columns
'''
    Here i will be taking two columns at a time to do the EDA. TO understand the relation  of one column 
    with the target column.
'''
sns.pairplot(df_copy[['SalePrice','LotArea']])
plt.show()
sns.pairplot(df_copy[['MasVnrArea','SalePrice']])
plt.show()
sns.pairplot(df_copy[['BsmtFinSF1','SalePrice']])
plt.show()
sns.pairplot(df_copy[['BsmtFinSF2','SalePrice']])
plt.show()
#sns.pairplot(df_copy[['MSSubClass','SalePrice']])
sns.catplot(x="MSSubClass", y="SalePrice", data=df_copy);
plt.show()
sns.catplot(x="SaleType", y="SalePrice", data=df_copy)
plt.show()
sns.catplot( x="LotShape", y="SalePrice", data=df_copy)
plt.show()
sns.catplot(x="SaleCondition", y="SalePrice", data=df_copy)
plt.show()
sns.catplot(x="RoofStyle", y="SalePrice", data=df_copy)
plt.show()
sns.catplot(x="GarageType", y="SalePrice", data=df_copy)
plt.show()
'''
    Splitting the data into train and test data to start with the model creation.
'''

df_train, df_test = train_test_split(df_final_1, train_size = 0.80, random_state = 100)
df_train.shape
df_test.shape
df_train.head()
df_num = np.array(df_final.select_dtypes(include=[np.number]).columns.values)
'''
    Scaling the test and the train data, using the Min Max scaler. So that everything is in the range 0 to 1
'''
scaler = MinMaxScaler()
df_train[df_num] = scaler.fit_transform(df_train[df_num])
df_train[df_num].describe()
df_test[df_num] = scaler.transform(df_test[df_num])
'''
    For creating the equation, we have separated the independent variable with the rest of the dependent variables.
'''
y_train = df_train.pop('SalePrice')
X_train = df_train

y_test = df_test.pop('SalePrice')
X_test = df_test
X_train.head()
y_train.head()
df_test.describe()
'''
    I have created this function, so as to reduce the line of code 
    - for trying multiple paramter values of alpha
    - create multiple models for both the ridge and the lasso regression
    - evaluate the models graphically.
    
    Here i am passing 6 paramters which have the -:
    - type of regression ridge/lasso
    - param - containing the alpha paramter value
    - scr - 'neg_mean_absolute_error' or the scoring term to be used in ridge regression.
    - fold - no of folds, for now we are using 5
    - X_train/y_train to train the models.
'''

def lasso_ridge_model(estimator, param,scr,fold,X_train,y_train):
    folds = fold
    if estimator.lower() == "ridge":
        estimator = Ridge()
        model_cv = GridSearchCV(estimator = estimator,
                            param_grid = param,
                            scoring = 'neg_mean_absolute_error',
                            cv = folds,
                            return_train_score = True,
                            verbose = 1                            
                           )
    else:
        estimator = Lasso()
        model_cv = GridSearchCV(estimator = estimator,
                              param_grid = params,
                              cv = fold,
                              return_train_score = True,
                              verbose =1)

    model_cv.fit(X_train, y_train) 
    print("best alpha paramater value is {val}".format(val = model_cv.best_params_))    
    # plotting
    cv_results = pd.DataFrame(model_cv.cv_results_)
    plt.plot(cv_results['param_alpha'], cv_results['mean_train_score'])
    plt.plot(cv_results['param_alpha'], cv_results['mean_test_score'])
    plt.xlabel('alpha')
    plt.ylabel('Negative Mean Absolute Error')
    plt.title("Negative Mean Absolute Error and alpha")
    plt.legend(['train score', 'test score'], loc='upper left')
    plt.show()
    return model_cv
'''
    Here we have created the alpha parameter and then used the lasso_ridge_model function
'''
params ={'alpha':[0.0001, 0.001, 0.01, 0.05, 0.1, 
 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 
 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 20, 50, 100, 500, 1000]}
model_ridge_1 = lasso_ridge_model("ridge", params,'neg_mean_absolute_error',5,X_train,y_train)
'''
    Here we have tried to tune the alpha parameter and then used the lasso_ridge_model function
'''
params={'alpha':np.arange(0.2,1,0.01)}
model_ridge_2 = lasso_ridge_model("ridge",params,'neg_mean_absolute_error',5,X_train,y_train)
'''
    using the best alpha value to fit the ridge model
'''
alpha = 0.22
ridge = Ridge(alpha=alpha)

ridge.fit(X_train, y_train)
len(ridge.coef_)
'''
    R2 score obtained for the ridge regression model with alpha = 0.22, even though Ridge does not zero out the coefficients
    but still there are 4 coefficicnets whose values are so small that they are considered as 0

'''
y_train_pred_ridge = ridge.predict(X_train)
y_test_pred_ridge = ridge.predict(X_test)
print("R2 score  Train {}".format(r2_score(y_true=y_train,y_pred=y_train_pred_ridge)))
print("R2 score  Test {}".format(r2_score(y_true=y_test,y_pred=y_test_pred_ridge)))
print("Number of non-zero coefficents {}".format(np.sum(ridge.coef_!=0)))
print("RMSE train {}".format(np.sqrt(mean_squared_error(y_train,y_train_pred_ridge))))
print("RMSE test {}".format(np.sqrt(mean_squared_error(y_test,y_test_pred_ridge))))
'''
    Here we have created the alpha parameter and then used the lasso_ridge_model function
'''
params ={'alpha':[0.0001, 0.001, 0.01, 0.05, 0.1, 
 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 
 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 20, 50, 100, 500, 1000]}
model_lasso_1 = lasso_ridge_model("lasso", params,'',5,X_train,y_train)
'''
    Here we have tried to tune the alpha parameter and then used the lasso_ridge_model function
'''
params = {'alpha' : np.arange(0,0.001,0.0001)}
model_lasso_2 = lasso_ridge_model("lasso", params,'',5,X_train,y_train)
'''
    Here we have tried again to further tune the alpha parameter and then used the lasso_ridge_model function
'''
params = {'alpha' : np.arange(0.0002,0.0005,0.00001)}
model_lasso_3= lasso_ridge_model("lasso", params,'',5,X_train,y_train)
cv_results_lasso = pd.DataFrame(model_lasso_3.cv_results_)
cv_results_lasso.head()
'''
    using the best alpha value to fit the lasso model
'''
alpha = 0.00023
lasso = Lasso(alpha=alpha)

lasso.fit(X_train, y_train)
len(lasso.coef_)
'''
    Value of score obtained for the lasso regression and the non zero coefficicnet are 111
'''
y_train_pred_lasso = lasso.predict(X_train)
y_test_pred_lasso = lasso.predict(X_test)

print("R2 score  Train {}".format(r2_score(y_true=y_train,y_pred=y_train_pred_lasso)))
print("R2 score  Test {}".format(r2_score(y_true=y_test,y_pred=y_test_pred_lasso)))
print("Number of non-zero coefficents {}".format(np.sum(lasso.coef_!=0)))
print("RMSE train {}".format(np.sqrt(mean_squared_error(y_train,y_train_pred_lasso))))
print("RMSE test {}".format(np.sqrt(mean_squared_error(y_test,y_test_pred_lasso))))
'''
    List of coefficients obtained after using the ridge regression
'''
model_parameter_ridge = list(ridge.coef_)
model_parameter_ridge.insert(0,ridge.intercept_)
cols = X_train.columns
cols.insert(0,'constant')
ridge_coef = pd.DataFrame(list(zip(cols,model_parameter_ridge)))
ridge_coef.columns = ['Feaure','Coef']
ridge_coef.sort_values(by='Coef', ascending=False).head(10)
'''
    List of coefficients obtained after using the lasso regression
'''
model_parameter_lasso = list(lasso.coef_)
model_parameter_lasso.insert(0,lasso.intercept_)
cols = X_train.columns
cols.insert(0,'constant')
lasso_coef = pd.DataFrame(list(zip(cols,model_parameter_lasso)))
lasso_coef.columns = ['Feaure','Coef']
lasso_coef.sort_values(by='Coef',ascending=False).head(10)
'''
    Splitting the data into train and test data to start with the model creation.
'''

df_train, df_test = train_test_split(df_final_1, train_size = 0.70, random_state = 100)
df_train.shape
df_test.shape
df_train.head()
df_num = np.array(df_final.select_dtypes(include=[np.number]).columns.values)
df_num
'''
    Scaling the test and the train data, using the Min Max scaler. So that everything is in the range 0 to 1
'''
scaler = MinMaxScaler()
df_train[df_num] = scaler.fit_transform(df_train[df_num])
df_train[df_num].describe()
df_test[df_num] = scaler.transform(df_test[df_num])
'''
    For creating the equation, we have separated the independent variable with the rest of the dependent variables.
'''
y_train = df_train.pop('SalePrice')
X_train = df_train

y_test = df_test.pop('SalePrice')
X_test = df_test
'''
    Here we have created the alpha parameter and then used the lasso_ridge_model function
'''
params ={'alpha':[0.0001, 0.001, 0.01, 0.05, 0.1, 
 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 
 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 20, 50, 100, 500, 1000]}
model_ridge_1 = lasso_ridge_model("ridge", params,'neg_mean_absolute_error',5,X_train,y_train)
'''
    Here we have tried to tune the alpha parameter and then used the lasso_ridge_model function
'''
params={'alpha':np.arange(0.2,1,0.01)}
model_ridge_2 = lasso_ridge_model("ridge",params,'neg_mean_absolute_error',5,X_train,y_train)
'''
    using the best alpha value to fit the ridge model
'''
alpha = 0.2
ridge = Ridge(alpha=alpha)

ridge.fit(X_train, y_train)
len(ridge.coef_)
'''
    R2 score obtained for the ridge regression model with alpha = 0.22, even though Ridge does not zero out the coefficients
    but still there are 4 coefficicnets whose values are so small that they are considered as 0

'''
y_train_pred_ridge = ridge.predict(X_train)
y_test_pred_ridge = ridge.predict(X_test)
print("R2 score  Train {}".format(r2_score(y_true=y_train,y_pred=y_train_pred_ridge)))
print("R2 score  Test {}".format(r2_score(y_true=y_test,y_pred=y_test_pred_ridge)))
print("Number of non-zero coefficents {}".format(np.sum(ridge.coef_!=0)))
print("RMSE train {}".format(np.sqrt(mean_squared_error(y_train,y_train_pred_ridge))))
print("RMSE test {}".format(np.sqrt(mean_squared_error(y_test,y_test_pred_ridge))))
'''
    Here we have created the alpha parameter and then used the lasso_ridge_model function
'''
params ={'alpha':[0.0001, 0.001, 0.01, 0.05, 0.1, 
 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 
 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 20, 50, 100, 500, 1000]}
model_lasso_1 = lasso_ridge_model("lasso", params,'',5,X_train,y_train)
'''
    Here we have tried to tune the alpha parameter and then used the lasso_ridge_model function
'''
params = {'alpha' : np.arange(0,0.001,0.0001)}
model_lasso_2 = lasso_ridge_model("lasso", params,'',5,X_train,y_train)
'''
    Here we have tried again to further tune the alpha parameter and then used the lasso_ridge_model function
'''
params = {'alpha' : np.arange(0.0002,0.0005,0.00001)}
model_lasso_3= lasso_ridge_model("lasso", params,'',5,X_train,y_train)
cv_results_lasso = pd.DataFrame(model_lasso_3.cv_results_)
cv_results_lasso.head()
'''
    using the best alpha value to fit the lasso model
'''
alpha = 0.00023
lasso = Lasso(alpha=alpha)

lasso.fit(X_train, y_train)
len(lasso.coef_)
'''
    Value of score obtained for the lasso regression and the non zero coefficicnet are 111
'''
y_train_pred_lasso = lasso.predict(X_train)
y_test_pred_lasso = lasso.predict(X_test)

print("R2 score  Train {}".format(r2_score(y_true=y_train,y_pred=y_train_pred_lasso)))
print("R2 score  Test {}".format(r2_score(y_true=y_test,y_pred=y_test_pred_lasso)))
print("Number of non-zero coefficents {}".format(np.sum(lasso.coef_!=0)))
print("RMSE train {}".format(np.sqrt(mean_squared_error(y_train,y_train_pred_lasso))))
print("RMSE test {}".format(np.sqrt(mean_squared_error(y_test,y_test_pred_lasso))))
'''
    List of coefficients obtained after using the ridge regression
'''
model_parameter_ridge = list(ridge.coef_)
model_parameter_ridge.insert(0,ridge.intercept_)
cols = X_train.columns
cols.insert(0,'constant')
ridge_coef = pd.DataFrame(list(zip(cols,model_parameter_ridge)))
ridge_coef.columns = ['Feaure','Coef']
val1 = ridge_coef.sort_values(by='Coef', ascending=False).head(5)
val2 = ridge_coef.sort_values(by='Coef', ascending=True).head(5)
# top 5 features from lasso
val1
# bottom 5 features from lasso
val2
'''
    List of coefficients obtained after using the lasso regression
'''
model_parameter_lasso = list(lasso.coef_)
model_parameter_lasso.insert(0,lasso.intercept_)
cols = X_train.columns
cols.insert(0,'constant')
lasso_coef = pd.DataFrame(list(zip(cols,model_parameter_lasso)))
lasso_coef.columns = ['Feaure','Coef']
val1 = lasso_coef.sort_values(by='Coef',ascending=False).head(5)
val2 = lasso_coef.sort_values(by='Coef',ascending=True).head(5)
# top 5 features from lasso
val1
'''
    bottom 5 features from lasso
'''
val2
