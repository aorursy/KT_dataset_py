import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline
#Loading the provided training and test data

train = pd.read_csv('../input/train.csv' )

test = pd.read_csv('../input/test.csv' )
train.head()
plt.scatter(x=train['GrLivArea'], y=train['SalePrice'])

plt.xlabel('GrLivArea')

plt.ylabel('SalePrice')
#Remove indicated outliers from data set

train = train[train['GrLivArea'] <= 4000]
train[['SalePrice']].describe()
sns.distplot(train['SalePrice'])

plt.xlabel('SalePrice\n' + 'Skew: ' + str(round(train['SalePrice'].skew(), 3)))
from scipy.stats import shapiro



w, p = shapiro(train['SalePrice'])

print('p-value on Shapiro–Wilk test of normality:', p)
#training set columns with missing values

train_missing = train.isnull().sum() 

train_missing = train_missing[train_missing > 0].dropna(how='all', axis=0) 



#test set columns with missing values

test_missing = test.isnull().sum() 

test_missing = test_missing[test_missing > 0].dropna(how='all', axis=0) 



#set difference between test and training columns with missing values

diff = (np.setdiff1d(test_missing.index, train_missing.index))

print('Columns with missing values in the test set, that do not have missing values in the training set:\n', diff)
train.info()
#Convert to object dtype

train['MSSubClass'] = train['MSSubClass'].astype(object)

test['MSSubClass'] = test['MSSubClass'].astype(object)

train['Id'] = train['Id'].astype(object)

test['Id'] = test['Id'].astype(object)



#Calculate percent missing values

train_missing = train.isnull().sum().sort_values(ascending=False)

percent = (train.isnull().sum()/len(train)*100).sort_values(ascending=False)

percent_missing = pd.concat([train_missing, percent], axis=1, keys=['Number_missing', 'Percent'])

percent_missing.head(10)
#Fill missing values

train['Electrical'] = train['Electrical'].fillna(train['Electrical'].mode()[0])

train['LotFrontage'] = train.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))



qualitative = []

quantitative = []

for i in list(train.columns):

    if train[i].dtype == 'object':

        qualitative.append(i)

    else:

        quantitative.append(i)

              

train.update(train[qualitative].fillna('None'))

train.update(train[quantitative].fillna(0))
#Combine test set to training set

all_X = pd.concat([train.drop('SalePrice', axis=1), test]).reset_index(drop=True)



#Extract columns with missing values imparted by test set

test_missing = all_X.isnull().sum()

print('Test set columns with missing values:\n', test_missing[test_missing > 0].dropna(how='all', axis=0).index)
#Fill missing values

all_X['MSZoning'] = all_X.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))



for col in ['Utilities', 'Exterior1st', 'Exterior2nd', 'KitchenQual', 'Functional', 'SaleType']:

    all_X[col] = all_X[col].fillna(all_X[col].mode()[0])

    

all_X['LotFrontage'] = all_X.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))

 

qualitative = []

quantitative = []

for i in list(all_X.columns):

    if all_X[i].dtype == 'object':

        qualitative.append(i)

    else:

        quantitative.append(i)  

  

all_X.update(all_X[qualitative].fillna('None'))

all_X.update(all_X[quantitative].fillna(0))

#Separate training and test sets 

train_dropped = all_X.iloc[:len(train), :].copy()

test_dropped = all_X.iloc[len(train):, :].copy()
#Replace dependent varaible to training set

train_dropped['SalePrice'] = train['SalePrice'].values



#Calculate and plot training set correlations

corr = np.abs(train_dropped.corr())

plt.subplots(figsize=(12,9))

sns.heatmap(corr, vmin=0, vmax=0.8)

plt.title('Absolute Pearson correlation coefficients between quantitative variables.')
#Drop correlated variable

train_dropped = train_dropped.drop(['GarageArea'], axis=1)

test_dropped = test_dropped.drop(['GarageArea'], axis=1)
columns = list(train_dropped.columns)

columns.remove('SalePrice')

columns.remove('Id')



plt.subplots(figsize=(22, 100))

for i in range(len(columns)):

    j = i+1

    plt.subplot(22,4,j)

    if train_dropped[columns[i]].dtype == 'int64' or train_dropped[columns[i]].dtype == 'float64':

        sns.scatterplot(data=train_dropped, x=columns[i], y='SalePrice')

    else:

        sns.boxplot(data=train_dropped, x=columns[i], y='SalePrice')

#Plot variable of interest

sns.swarmplot(data=train_dropped, x='Utilities', y='SalePrice')
#Drop columns from training and test sets

train_dropped = train_dropped.drop(['Id', 'Utilities'], axis=1)

test_dropped = test_dropped.drop(['Id', 'Utilities'], axis=1)
#Save copies training and test sets in current states

train_cp1 = train_dropped.copy()

test_cp1 = test_dropped.copy() 
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.linear_model import LinearRegression, Lasso

from sklearn import metrics



from scipy.stats import pearsonr
#Define model building function

def build_model(X, y, model, scaler=None, plot=False):

    """Split data into training and test sets, scale according to specified scaler,

    fit the specified model, calculate metrics and residuals, and plot true vs. predicted 

    values and residuals of both train and test data.

    

    Args:

        X (array or DataFrame): independent variables

        y (array or DataFrame): dependent variable

        model (model): sklearn model and parameters

        scaler (scaler): sklearn scaler and parameters, default None

        plot (bool): True/False, default False



    Returns:

        rmse_train (float): root mean square error on the training split

        rmse_test (float): root mean square error on the test split

        r_squared (float): r_squared on the test split

        residuals_train (array): residuals on the training split 

        residuals_test (array): residuals on the test split

    """

    

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=11)

    

    if scaler != None:

        scaler.fit(X_train)

        X_train = scaler.transform(X_train)

        X_test = scaler.transform(X_test)     

    

    model.fit(X_train, y_train)



    X_train_pred = model.predict(X_train)

    X_test_pred = model.predict(X_test)

  

    rmse_train = np.sqrt(metrics.mean_squared_error(y_train, X_train_pred))

    rmse_test = np.sqrt(metrics.mean_squared_error(y_test, X_test_pred))

    r_squared= metrics.r2_score(y_test, X_test_pred)



    residuals_train = y_train - X_train_pred

    residuals_test = y_test - X_test_pred

    

    if plot == True:

      

        plt.subplots(figsize=(22, 5))



        plt.subplot(1,4,1)

        plt.scatter(x=y_train, y=X_train_pred)

        plt.xlabel('Training set true values')

        plt.ylabel('Training set predicted values')

        plt.title('True vs. predicted train set values')



        plt.subplot(1,4,2)

        sns.distplot(residuals_train)

        plt.xlabel('Residuals\nSkew:' + str(round(float(residuals_train.skew()), 3)))

        plt.title('Distribution of training residuals')



        plt.subplot(1,4,3)

        plt.scatter(x=y_test, y=X_test_pred)

        plt.xlabel('Test set true values')

        plt.ylabel('Test set predicted values')

        plt.title('True vs. predicted test set values')



        plt.subplot(1,4,4)

        sns.distplot(residuals_test)

        plt.xlabel('Residuals\nSkew:' + str(round(float(residuals_test.skew()), 3)))

        plt.title('Distribution of testing residuals')

    

    return rmse_train, rmse_test, r_squared, residuals_train, residuals_test
#Define non-numeric ordinal variable encoding function

def encode(all_X):

    """Encodes non-numeric ordinal variables using LableEncoder from sklearn.preprocessing.

    

    Args:

        all_X (array or DataFrame): independent variables



    Returns:

        all_X (array): encoded independent varaibles

    """    

    

    all_X = all_X.copy()

    le = LabelEncoder()



    for col in list(all_X.columns):

        if all_X[col].dtype == 'object':

            if col in ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']:

                le.fit(['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'])

                all_X[col] = le.transform(all_X[col])

            elif col == 'BsmtExposure':

                le.fit(['None', 'No', 'Mn', 'Av', 'Gd'])

                all_X[col] = le.transform(all_X[col])

    

    return all_X

                             
#Define function to indicate variables with significant correlation with 'SalePrice'

def significant(all_vars, level=0.05):

    """Extract the variables with a p-value for their correlation below the indicated level.

    

    Args:

        all_vars (DataFrame): independent and dependent ('SalePrice') variables

        level (float): p-value cut-off, default 0.05



    Returns:

        sig_vars (list): column/variable names

    """  

    

    all_vars = all_vars.copy()

    corr = all_vars.corr()

    pval = np.empty([all_vars.shape[1], all_vars.shape[1]])



    for i in range(all_vars.shape[1]): 

        for j in range(all_vars.shape[1]):

            r, p = pearsonr(all_vars.values[:,i], all_vars.values[:,j])

            pval[i,j] = p

        

    pval = pd.DataFrame(pval, index=corr.columns, columns=corr.columns)



    pval = pval[['SalePrice']].sort_values('SalePrice').drop(['SalePrice'], axis=0)

    pval = pval[pval <= level].dropna(axis=0)

    sig_vars = list(pval.index)

    

    return sig_vars
train_simple = train_dropped.copy()



#Extract the 2 variables with the highest correlation to 'SalePrice', and their respective correlations 

correlation = np.abs(train_simple.corr())

price_corr = correlation[['SalePrice']].sort_values('SalePrice', ascending=False).drop(['SalePrice'], axis=0)

price_corr = price_corr.nlargest(2, 'SalePrice')



#Visualise the extracted correlations

cols = list(price_corr.index)

values = price_corr.values

plt.subplots(figsize=(22, 10))

for i in range(len(cols)):

    j = i+1

    plt.subplot(2,3,j)

    plt.scatter(x=train_simple[cols[i]], y=train_simple['SalePrice'])

    plt.xlabel(cols[i] + '\nCorrelation: ' + str(round(float(values[i]), 3)))

    plt.ylabel('SalePrice')
#Fit and test a linear regression using the 2 extracted variables

train_rmse, test_rmse, test_rr, train_res, test_res = build_model(train_simple[cols], train_simple['SalePrice'], LinearRegression(), plot=True)



print('RMSE train:', train_rmse)

print('RMSE test:', test_rmse)

print('R_squared test:', test_rr)
#Encode non-numeric ordinal variables and convert nominal variables into dummies

train_encode = train_dropped.copy()

train_encode = encode(train_encode)

train_encode = pd.get_dummies(train_encode, drop_first=True).reset_index(drop=True)



#Extract variables with significant correlation to 'SalePrice'

train_encode = train_encode[significant(train_encode)]



#Fit and test a linear regression using the extracted variables

train_rmse, test_rmse, test_rr, train_res, test_res = build_model(train_encode, train_dropped['SalePrice'], LinearRegression(), plot=True)



print('Number of variables:', train_encode.shape[1])

print('RMSE train:', train_rmse)

print('RMSE test:', test_rmse)

print('R_squared test:', test_rr)
#Convert non-numeric variables into dummies

train_dum = train_dropped.copy()

train_dum = pd.get_dummies(train_dum, drop_first=True).reset_index(drop=True)



#Extract variables with significant correlation to 'SalePrice' 

train_dum = train_dum[significant(train_dum)]



#Fit and test a linear regression using the extracted variables

train_rmse, test_rmse, test_rr, train_res, test_res = build_model(train_dum, train_dropped['SalePrice'], LinearRegression(), plot=True)



print('Number of variables:', train_dum.shape[1])

print('RMSE train:', train_rmse)

print('RMSE test:', test_rmse)

print('R_squared test:', test_rr)
from scipy.stats import zscore



#Calculate the z-scores of the values of all numeric variables and filter records with z-score > 3 for any variable.

num_col = []

for col in train_dropped.columns:

    if train_dropped[col].dtype == 'int64' or train_dropped[col].dtype == 'float64':

        num_col.append(col)



num_col.remove('SalePrice')

scores = train_dropped[num_col].copy()

scores = scores[(np.abs(zscore(scores)) > 3).any(axis=1)]



print('Percent records with an outlier variable:', round(len(scores)/len(train_dropped)*100, 1))
skew_col = []



plt.subplots(figsize=(22, 100))



#Collate a list of names of variables with skew distributions

for col in num_col:

    skewness = round(train_dropped[col].skew(), 1)

    if skewness >= 0.5 or skewness <= -0.5:

        skew_col.append(col)

            

        plt.subplot(22,5,len(skew_col))

        sns.distplot(train_dropped[col])

        plt.xlabel(col + '; Skew:' + str(round(skewness, 3)))



print('Percent numeric variables with skewed distributions:', round(len(skew_col)/len(num_col), 2))
#Employ Shapiro–Wilk test to statistically attain that above variable distributions are not normal

p_val = []

for col in skew_col:

    w, p = shapiro(train_dropped[col])

    if p <= 0.05:

        p_val.append(col)



print('All skew distributions significant:', p_val == skew_col)    
#Create lists to store variable names according to the variable distribution

high_pos = []

mod_pos = []

high_neg = []

mod_neg = []



#Transform variables according to their distributions

for col in skew_col:

    skewness = round(train_dropped[col].skew(), 1)

    

    if skewness <= -1:

        high_neg.append(col)

        const = all_X[col].max() + 1

        train_dropped[col] = np.log(const - train_dropped[col].values)

        test_dropped[col] = np.log(const - test_dropped[col].values)

        

    elif skewness > -1 and skewness <= -0.5:

        mod_neg.append(col)

        const = all_X[col].max() + 1

        train_dropped[col] = np.sqrt(const - train_dropped[col].values)

        test_dropped[col] = np.sqrt(const - test_dropped[col].values)

        

    elif skewness >= 0.5 and skewness < 1:

        mod_pos.append(col)

        train_dropped[col] = np.sqrt(train_dropped[col].values + 1)

        test_dropped[col] = np.sqrt(test_dropped[col].values + 1)

        

    elif skewness >= 1:

        high_pos.append(col)

        train_dropped[col] = np.log(train_dropped[col].values + 1)

        test_dropped[col] = np.log(test_dropped[col].values + 1)

  

print('High positive skew:', high_pos)

print('Moderate positive skew:', mod_pos)

print('High negative skew:', high_neg)

print('Moderate negative skew:', mod_neg)

#Transform the dependent varaible

train_dropped['SalePrice'] = np.log(train['SalePrice'].values + 1)



sns.distplot(train_dropped['SalePrice'])

plt.xlabel('SalePrice; skew:' + str(round(train_dropped['SalePrice'].skew(), 3)))

plt.title('Distribution of SalePrice')
#Convert non-numeric variables into dummies

train_trans_dum = train_dropped.copy()

train_trans_dum = pd.get_dummies(train_trans_dum, drop_first=True).reset_index(drop=True)



#Extract variables with significant correlation to 'SalePrice'

train_trans_dum = train_trans_dum[significant(train_trans_dum)]



#Fit and test a linear regression using the extracted variables

train_rmse, test_rmse, test_rr, train_res, test_res = build_model(train_trans_dum, train_dropped['SalePrice'], LinearRegression(), plot=True)



print('Number of variables:', train_trans_dum.shape[1])

print('RMSE train:', train_rmse)

print('RMSE test:', test_rmse)

print('R_squared test:', test_rr)
import statsmodels.api as sm



#Define backward elimination function

def backwardelimination(x, SL, cols):

    """Sequentially fit a linear regression model and remove the variable with the highest 

    coefficient p-value after each fit, provided that the p-value is above the indicated cut-off.

    

    Args:

        x (array): independent variables

        SL (float): p-value cut-off

        cols (list): independent variable column numbers



    Returns:

        x (array): remaining independent variables

        columns (list): remaining column numbers including a constant id applicable (denoted by 'ones')

        reg_OLS.params (array): variable coefficients

    """

    numVars = len(x[0])

    columns = ['ones'] + cols

    

    for i in range(numVars):

        reg_OLS = sm.OLS(y, x).fit()

        pvalues = reg_OLS.pvalues

        maxP = max(pvalues)

        pval_n = np.argmax(pvalues)

       

        if maxP > SL:

            x = np.delete(x, pval_n, 1)

            col_num = columns[pval_n]

            columns.remove(col_num)



    return x, columns, reg_OLS.params

#Convert non-numeric variables into dummies

train_elim = train_dropped.copy()

train_elim = pd.get_dummies(train_elim, drop_first=True).reset_index(drop=True)

train_elim = train_elim.drop(['SalePrice'], axis=1)



#Split data into training and test sets, keeping the random_state constant as before

X_train, X_test, y_train, y_test = train_test_split(train_elim, train_dropped['SalePrice'], random_state=11)



#Scale the training set

sc = StandardScaler()

sc.fit(X_train)

X_train_scaled = sc.transform(X_train)



#Obtain the required arrays and list for backwards elimination

X = X_train_scaled

y = y_train.values

cols = list(np.arange(len(X[0])))

X = sm.add_constant(X)



#Perform backwards elimination

X_modeled, X_columns, coeff = backwardelimination(X, 0.05, cols)



#Remove the intercept from the remaining columns if it is present

if 'ones' in X_columns:

    X_columns.remove('ones')



print('Number of remaining variables: ', len(X_columns))
if len(coeff) == len(X_columns)+1:

    coeff = coeff[1:]



cols = list(X_train.iloc[:, X_columns].columns)

print('Remaining variables and coefficients:')

for i in range(len(X_columns)):

    print(X_columns[i], cols[i], coeff[i])
#Extract the list of remaining variables from all the training variables

X_train_shrunk = train_elim.values[:, X_columns]



#Fit and test a linear regression using the extracted variables

train_rmse, test_rmse, test_rr, train_res, test_res = build_model(X_train_shrunk, train_dropped['SalePrice'], LinearRegression(), scaler=StandardScaler(), plot=True)



print('RMSE train:', train_rmse)

print('RMSE test:', test_rmse)

print('R_squared test:', test_rr)
#Convert non-numeric variables into dummies

train_dummies = train_dropped.copy()

train_dummies = pd.get_dummies(train_dummies, drop_first=True).reset_index(drop=True)

train_dummies = train_dummies.drop(['SalePrice'], axis=1)



#Construct the list of regularisation levels to investigate

alphas = [0.0001, 0.001, 0.01, 0.1, 1]



#Create lists to store training and test residuals, as well as metrics for each model

residuals_train = []

residuals_test = []

rmse_train = []

rmse_test = []

r_squared = []



#Fit and test a Lasso regression model at each regularisation level

for i in alphas:

    train_rmse, test_rmse, test_rr, train_res, test_res = build_model(train_dummies, train_dropped['SalePrice'], Lasso(alpha=i), scaler=StandardScaler(), plot=False)

    rmse_train.append(train_rmse)

    rmse_test.append(test_rmse)

    r_squared.append(test_rr)

    residuals_train.append(train_res)

    residuals_test.append(test_res)

    

#Extract the best model's regularisation level and metrics       

best_alpha = np.argmin(rmse_test)

print('Optimal alpha:', alphas[best_alpha])

print('RMSE train:', rmse_train[best_alpha])

print('RMSE test:', rmse_test[best_alpha])

print('R^2 test:', r_squared[best_alpha])



#Fit a Lasso regression model according to the indicated optimal regularisation in order to visualise the residuals

train_rmse, test_rmse, test_rr, train_res, test_res = build_model(train_dummies, train_dropped['SalePrice'], Lasso(alpha=alphas[best_alpha]), scaler=StandardScaler(), plot=True)

#Lasso on all training data

#Concatenate training and test sets and convert non-numeric variables into dummies

X_all = pd.concat([train_dropped.drop(['SalePrice'], axis=1), test_dropped]).reset_index(drop=True)

y_all = train_dropped[['SalePrice']]

X_all_dummies = pd.get_dummies(X_all, drop_first=True).reset_index(drop=True)



#Separate training and test sets

X_train_all = X_all_dummies.iloc[:len(y_all), :]

X_blind = X_all_dummies.iloc[len(y_all):, :]



#Scale the training data and transform the test data accordingly

sc = StandardScaler()

sc.fit(X_train_all)

X_train_scaled = sc.transform(X_train_all)

X_blind_scaled = sc.transform(X_blind)



#Define and fit a Lasso regressor

linLasso = Lasso(alpha=alphas[best_alpha])

linLasso.fit(X_train_scaled, y_all)



#Predict logarithm of sale prices for both the training and test data

X_train_pred = linLasso.predict(X_train_scaled)

X_blind_pred = linLasso.predict(X_blind_scaled)

  

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_all, X_train_pred)))



#Convert the test data predictions back to sale prices and write it to a .csv file

results = test[['Id']]

results['SalePrice'] = (np.exp(X_blind_pred)) - 1

results.to_csv('Lasso_results.csv', index=False)

#Write the Lasso regressor variables and coefficiets to a DataFrame and eliminate those with coefficients of 0

coeffs = pd.DataFrame(linLasso.coef_, index=X_train_all.columns)

coeffs.columns = ['Coefficient']

coeffs = coeffs[coeffs > 0].sort_values('Coefficient', ascending=False).dropna()

print('Number of remaining variables: ', len(coeffs))

#Compare the selected variables from backward elimination and Lasso regression

sorted(list(coeffs.index)) == sorted(cols)
print(coeffs.head(48))

print(coeffs.tail(48))
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=20, max_features=0.5, max_depth=20, n_jobs=-1, random_state=11)



train_cp_dum = train_cp1.drop(['SalePrice'], axis=1).copy()

train_cp_dum = pd.get_dummies(train_cp_dum, drop_first=True).reset_index(drop=True)



train_rmse, test_rmse, test_rr, train_res, test_res = build_model(train_cp_dum, train_cp1['SalePrice'], model, plot=False)



print('RMSE train:', train_rmse)

print('RMSE test:', test_rmse)

print('R_squared test:', test_rr)
from sklearn.linear_model import Ridge
#Convert non-numeric variables into dummies

train_dummies = train_dropped.copy()

train_dummies = pd.get_dummies(train_dummies, drop_first=True).reset_index(drop=True)

train_dummies = train_dummies.drop(['SalePrice'], axis=1)



#Construct the list of regularisation levels to investigate

alphas = [0.1, 1, 10, 100, 1000]



#Create lists to store training and test residuals, as well as metrics for each model

residuals_train = []

residuals_test = []

rmse_train = []

rmse_test = []

r_squared = []



#Fit and test a Ridge regression model at each regularisation level

for i in alphas:

    train_rmse, test_rmse, test_rr, train_res, test_res = build_model(train_dummies, train_dropped['SalePrice'], Ridge(alpha=i), scaler=StandardScaler(), plot=False)

    rmse_train.append(train_rmse)

    rmse_test.append(test_rmse)

    r_squared.append(test_rr)

    residuals_train.append(train_res)

    residuals_test.append(test_res)

    

#Extract the best model's regularisation level and metrics       

best_alpha = np.argmin(rmse_test)

print('Optimal alpha:', alphas[best_alpha])

print('RMSE train:', rmse_train[best_alpha])

print('RMSE test:', rmse_test[best_alpha])

print('R^2 test:', r_squared[best_alpha])



#Fit a Ridge regression model according to the indicated optimal regularisation in order to visualise the residuals

train_rmse, test_rmse, test_rr, train_res, test_res = build_model(train_dummies, train_dropped['SalePrice'], Ridge(alpha=alphas[best_alpha]), scaler=StandardScaler(), plot=True)
