import pandas as pd
housing_train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
housing_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
housing_train.head()
housing_train.info()
nhousing_train = housing_train.shape[0]
nhousing_test = housing_test.shape[0]
y_train = housing_train['SalePrice'].values
test_ID = housing_test['Id'].values
all_data = pd.concat((housing_train,housing_test),axis=0).reset_index(drop=True)
all_data.drop(['Id','SalePrice'],axis=1,inplace=True)
print('all_data size is: {}'.format(all_data.shape))
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
%matplotlib inline

cols = all_data.columns
colours = ['#000099', '#ffff00']
f, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(all_data[cols].isnull(), cmap=sns.color_palette(colours))
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na_values = all_data.isnull().sum()
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
all_data_na_values = all_data_na_values.drop(all_data_na_values[all_data_na_values == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Nº Missing Values':all_data_na_values,'Missing Ratio' :all_data_na})
print(missing_data)
f, ax = plt.subplots(figsize=(8, 6))
plt.xticks(rotation='90')
sns.barplot(x=all_data_na.index, y=all_data_na)
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)
missing_values_obs = pd.DataFrame(all_data.isnull().sum(axis=1).value_counts())
missing_values_obs.reset_index(inplace=True)
missing_values_obs.rename(columns={'index':'Nº missing values',0:'Nº observations'},inplace=True)
missing_values_obs.sort_values('Nº missing values',axis=0,ascending=True,inplace=True)
missing_values_obs

f, ax = plt.subplots(figsize=(8, 6))
sns.barplot(x=missing_values_obs['Nº missing values'], y=missing_values_obs['Nº observations'])
plt.xlabel('Nº missing values', fontsize=15)
plt.ylabel('Nº observations', fontsize=15)
plt.title('Nº of observations with missing values', fontsize=15)
all_data[all_data.duplicated(keep=False)]
import numpy as np

all_data.drop([193,829],axis=0,inplace=True)
y_train = np.delete(y_train,193)
y_train = np.delete(y_train,829)
all_data.reset_index(inplace=True)
all_data["PoolQC"] = all_data["PoolQC"].fillna("None")
all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")
all_data["Alley"] = all_data["Alley"].fillna("None")
all_data["Fence"] = all_data["Fence"].fillna("None")
all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")
#Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    all_data[col] = all_data[col].fillna('None')
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    all_data[col] = all_data[col].fillna(0)
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    all_data[col] = all_data[col].fillna(0)
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    all_data[col] = all_data[col].fillna('NoBsmt')
all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)
all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])
all_data = all_data.drop(['Utilities'], axis=1)
all_data["Functional"] = all_data["Functional"].fillna("Typ")
all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])
all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])
all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])
all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na_values = all_data.isnull().sum()
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
all_data_na_values = all_data_na_values.drop(all_data_na_values[all_data_na_values == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Nº Missing Values':all_data_na_values,'Missing Ratio' :all_data_na})
print(missing_data)
nhousing_train = all_data.shape[0]-nhousing_test #Remember we removed two observations from the train set, so we need to update the length
housing_train = all_data[:nhousing_train]
housing_test = all_data[nhousing_train:]
housing_train['SalePrice'] = y_train #We have to add the SalePrice column removed before
from scipy import stats
from scipy.stats import norm, skew #for some statistics

f, ax = plt.subplots(figsize=(8, 6))
sns.distplot(housing_train['SalePrice'],fit=norm)

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(housing_train['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')
f, ax = plt.subplots(figsize=(8,6))
sns.boxplot(housing_train['OverallQual'],housing_train['SalePrice'])
f, ax = plt.subplots(figsize=(15,7))
sns.boxplot(housing_train['YearBuilt'],housing_train['SalePrice'])
plt.xticks(rotation=90);
f, ax = plt.subplots(figsize=(8,6))
sns.regplot(housing_train['GrLivArea'],housing_train['SalePrice'],housing_train)
f, ax = plt.subplots(figsize=(8,6))
sns.regplot(housing_train['TotalBsmtSF'],housing_train['SalePrice'],housing_train)
housing_train[housing_train['TotalBsmtSF']>6000]['GrLivArea']
# Calculate first and third quartile
first_quartile = housing_train['GrLivArea'].describe()['25%']
third_quartile = housing_train['GrLivArea'].describe()['75%']

# Interquartile range
iqr = third_quartile - first_quartile

# Remove outliers
housing_train = housing_train[(housing_train['GrLivArea'] > (first_quartile - 3 * iqr)) &
            (housing_train['GrLivArea'] < (third_quartile + 3 * iqr))]
f, ax = plt.subplots(figsize=(8,6))
sns.regplot(housing_train['GrLivArea'],housing_train['SalePrice'],housing_train)
f, ax = plt.subplots(figsize=(8,6))
sns.regplot(housing_train['TotalBsmtSF'],housing_train['SalePrice'],housing_train)
f, ax = plt.subplots(figsize=(8, 6))
sns.distplot(housing_train['SalePrice'],fit=norm)

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(housing_train['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')
#Correlation map to see how features are correlated with SalePrice
corrmat = housing_train.iloc[:,1:].corr()
plt.subplots(figsize=(22,15))
sns.heatmap(corrmat, square=True,annot=True)
num_feat = housing_train.select_dtypes(include='number').columns
for i in num_feat:
    if abs(housing_train[i].corr(housing_train['SalePrice'])) > 0.2:
        print(i, '-', 'SalesPrice:', housing_train[i].corr(housing_train['SalePrice']))
# Select the numeric columns
numeric_subset = housing_train.select_dtypes('number')

# Create columns with square root and log of numeric columns
for col in numeric_subset.columns:
    # Skip the SalePrice column
    if col == 'SalePrice':
        next
    else:
        numeric_subset['sqrt_' + col] = np.sqrt(numeric_subset[col])
        numeric_subset['log_' + col] = np.log(numeric_subset[col]+1)

# Select the categorical columns
categorical_subset = housing_train.select_dtypes(exclude='number')

# One hot encode
categorical_subset = pd.get_dummies(categorical_subset)

# Join the two dataframes using concat
# Make sure to use axis = 1 to perform a column bind
features = pd.concat([numeric_subset, categorical_subset], axis = 1)

# Find correlations with the score 
correlations = features.corr()['SalePrice'].sort_values()
# Display most negative correlations
correlations.head(15)
# Display most positive correlations
correlations.tail(15)
nhousing_train = housing_train.shape[0]
nhousing_test = housing_test.shape[0]
y_train = housing_train['SalePrice'].values
all_data = pd.concat((housing_train,housing_test),axis=0).reset_index(drop=True)
all_data.drop(['SalePrice'],axis=1,inplace=True)
all_data = all_data.replace({"MSSubClass" : {20 : "SC20", 30 : "SC30", 40 : "SC40", 45 : "SC45", 
                                       50 : "SC50", 60 : "SC60", 70 : "SC70", 75 : "SC75", 
                                       80 : "SC80", 85 : "SC85", 90 : "SC90", 120 : "SC120", 
                                       150 : "SC150", 160 : "SC160", 180 : "SC180", 190 : "SC190"},
                       "MoSold" : {1 : "Jan", 2 : "Feb", 3 : "Mar", 4 : "Apr", 5 : "May", 6 : "Jun",
                                   7 : "Jul", 8 : "Aug", 9 : "Sep", 10 : "Oct", 11 : "Nov", 12 : "Dec"},
                             'YrSold' : {2006 : '2006', 2007 : '2007', 2008 : '2008', 2009 : '2009', 2010 : '2010'}
                      })
all_data = all_data.replace({"Alley" : {'None' : 0, "Grvl" : 1, "Pave" : 2},
                       "BsmtCond" : {"NoBsmt" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "BsmtExposure" : {'NoBsmt' : 0, "No" : 1, "Mn" : 2, "Av": 3, "Gd" : 4},
                       "BsmtFinType1" : {"NoBsmt" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4, 
                                         "ALQ" : 5, "GLQ" : 6},
                       "BsmtFinType2" : {"NoBsmt" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4, 
                                         "ALQ" : 5, "GLQ" : 6},
                       "BsmtQual" : {"NoBsmt" : 0, "Po" : 1, "Fa" : 2, "TA": 3, "Gd" : 4, "Ex" : 5},
                       "ExterCond" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},
                       "ExterQual" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},
                       "FireplaceQu" : {"None" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "Functional" : {"Sal" : 1, "Sev" : 2, "Maj2" : 3, "Maj1" : 4, "Mod": 5, 
                                       "Min2" : 6, "Min1" : 7, "Typ" : 8},
                       "GarageCond" : {"None" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "GarageQual" : {"None" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "HeatingQC" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "KitchenQual" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "LandSlope" : {"Sev" : 1, "Mod" : 2, "Gtl" : 3},
                       "LotShape" : {"IR3" : 1, "IR2" : 2, "IR1" : 3, "Reg" : 4},
                       "PavedDrive" : {"N" : 0, "P" : 1, "Y" : 2},
                       "PoolQC" : {"None" : 0, "Fa" : 1, "TA" : 2, "Gd" : 3, "Ex" : 4},
                       "Street" : {"Grvl" : 1, "Pave" : 2}})
all_data['BsmtCond'] = pd.to_numeric(all_data['BsmtCond'])
# Overall quality of the house
all_data["OverallGrade"] = all_data["OverallQual"] * all_data["OverallCond"]
# Overall quality of the garage
all_data["GarageGrade"] = all_data["GarageQual"] * all_data["GarageCond"]
# Overall quality of the exterior
all_data["ExterGrade"] = all_data["ExterQual"] * all_data["ExterCond"]
# Overall kitchen score
all_data["KitchenScore"] = all_data["KitchenAbvGr"] * all_data["KitchenQual"]
# Overall fireplace score
all_data["FireplaceScore"] = all_data["Fireplaces"] * all_data["FireplaceQu"]
# Overall garage score
all_data["GarageScore"] = all_data["GarageArea"] * all_data["GarageQual"]
# Overall pool score
all_data["PoolScore"] = all_data["PoolArea"] * all_data["PoolQC"]

# Total number of bathrooms
all_data["TotalBath"] = all_data["BsmtFullBath"] + (0.5 * all_data["BsmtHalfBath"]) + all_data["FullBath"] + (0.5 * all_data["HalfBath"])
# Total SF for house (incl. basement)
all_data["AllSF"] = all_data["GrLivArea"] + all_data["TotalBsmtSF"]
# Total SF for 1st + 2nd floors
all_data["AllFlrsSF"] = all_data["1stFlrSF"] + all_data["2ndFlrSF"]
# Total SF for porch
all_data["AllPorchSF"] = all_data["OpenPorchSF"] + all_data["EnclosedPorch"] + all_data["3SsnPorch"] + all_data["ScreenPorch"]
# Has masonry veneer or not
all_data["HasMasVnr"] = all_data.MasVnrType.replace({"BrkCmn" : 1, "BrkFace" : 1, "CBlock" : 1, 
                                               "Stone" : 1, "None" : 0})
# House completed before sale or not
all_data["BoughtOffPlan"] = all_data.SaleCondition.replace({"Abnorml" : 0, "Alloca" : 0, "AdjLand" : 0, 
                                                      "Family" : 0, "Normal" : 0, "Partial" : 1})
numerical_features = all_data.select_dtypes('number')
categorical_features = all_data.select_dtypes(exclude='number')
numerical_features.drop(['index'],axis=1,inplace=True)
for col in numerical_features.columns:
    numerical_features['sqrt_'+col] = np.sqrt(numerical_features[col])
    numerical_features['log_'+col] = np.log(numerical_features[col]+1)
    numerical_features['s2_'+col] = numerical_features[col]**2
    numerical_features['s3_'+col] = numerical_features[col]**3

all_data = pd.concat([categorical_features,numerical_features],axis=1)
categorical_features = all_data.select_dtypes(exclude='number').columns
all_data_cat = all_data[categorical_features]
all_data_cat_dumm = pd.get_dummies(all_data_cat)
all_data_cat_dumm.shape
all_data = pd.concat([all_data,all_data_cat_dumm],axis=1)
all_data.drop(categorical_features,axis=1,inplace=True)
all_data.head()
housing_train = all_data[:nhousing_train]
housing_test = all_data[nhousing_train:]
housing_train['SalePrice'] = y_train #We have to add the SalePrice column removed before
housing_train.info()
num_feat = housing_train.select_dtypes(include='number').columns
drop_cols = []
for i in num_feat:
    if abs(housing_train[i].corr(housing_train['SalePrice'])) < 0.3:
        drop_cols.append(i)
        print(i, '-', 'SalesPrice:', housing_train[i].corr(housing_train['SalePrice']))

housing_train = housing_train.drop(columns = drop_cols)
housing_test = housing_test.drop(columns = drop_cols)
housing_test.shape
corr_matrix = housing_train.corr()
item = corr_matrix.iloc[3:4, 9:10]
item.columns[0]
def remove_collinear_features(x, threshold):
    '''
    Objective:
        Remove collinear features in a dataframe with a correlation coefficient
        greater than the threshold. Removing collinear features can help a model
        to generalize and improves the interpretability of the model.
        
    Inputs: 
        threshold: any features with correlations greater than this value are removed
    
    Output: 
        dataframe that contains only the non-highly-collinear features
    '''
    
    # Dont want to remove correlations between SalePrice
    y = x['SalePrice']
    x = x.drop(columns = ['SalePrice'])
    
    # Calculate the correlation matrix
    corr_matrix = x.corr()
    iters = range(len(corr_matrix.columns)-1)
    drop_cols = []

    # Iterate through the correlation matrix and compare correlations
    for i in iters:
        for j in range(i):
            item = corr_matrix.iloc[j:(j+1), (i+1):(i+2)]
            col = item.columns
            row = item.index
            val = abs(item.values)
            
            # If correlation exceeds the threshold
            if val >= threshold:
                # Print the correlated features and the correlation value
                print(col.values[0], "|", row.values[0], "|", round(val[0][0], 2))
                drop_cols.append(col.values[0])   
               

    # Drop one of each pair of correlated columns
    drops = set(drop_cols)
    x = x.drop(columns = drops)
    
    # Add the SalePrice back in to the data
    x['SalePrice'] = y
               
    return drops, x
# Remove the collinear features above a specified correlation coefficient
drops, housing_train = remove_collinear_features(housing_train, 0.6);
housing_train.shape
drops
housing_train.corr()['SalePrice'].sort_values()
housing_test = housing_test.drop(columns = drops)
X_train = housing_train.drop('SalePrice',axis=1)
y_train = np.log1p(housing_train['SalePrice'])
X_test = housing_test
# Scaling values
from sklearn.preprocessing import MinMaxScaler

# Save de column names of the training set
columns = X_train.columns

# Create the scaler object with a range of 0-1
scaler = MinMaxScaler(feature_range=(0, 1))

# Fit on the training data
scaler.fit(X_train)

# Transform both the training and testing data
X_train = scaler.transform(X_train)
X_train = pd.DataFrame(X_train)
X_train.columns = columns
X_test = scaler.transform(X_test)
X_test = pd.DataFrame(X_test)
X_test.columns = columns
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV

#Validation function
n_folds = 10

def rmse_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42)
    rmse= np.sqrt(-cross_val_score(model, X_train, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse.mean())
# Create the model to use for hyperparameter tuning
model = Ridge(random_state=42)

# Regularization strength; must be a positive float. Larger values specify stronger regularization.
alpha = [0.0001, 0.0003, 0.0005, 0.001, 0.1, 0.3, 0.75, 1, 1.5]

# Define the grid of hyperparameters to search
hyperparameter_grid = {'alpha': alpha}

# Set up the random search with 10-fold cross validation
random_cv = RandomizedSearchCV(estimator=model,
                               param_distributions=hyperparameter_grid,
                               cv=10, n_iter=9, 
                               scoring = 'neg_mean_squared_error',
                               n_jobs = -1, verbose = 1, 
                               return_train_score = True,
                               random_state=42)

# Fit on the training data
random_cv.fit(X_train, y_train)

# Best Estimators
best_estimators = random_cv.best_estimator_
print(best_estimators)
model_Ridge = Ridge(alpha=0.3, random_state=42)

rmse_cv_Ridge = rmse_cv(model_Ridge)
print('Ridge Regressor RMSE is: {}'.format(rmse_cv_Ridge))
# Create the model to use for hyperparameter tuning
model = Lasso(random_state=42)

# Constant that multiplies the L1 term.
alpha = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.001, 0.1, 0.3, 0.75, 1, 1.5]

# Define the grid of hyperparameters to search
hyperparameter_grid = {'alpha': alpha}

# Set up the random search with 10-fold cross validation
random_cv = RandomizedSearchCV(estimator=model,
                               param_distributions=hyperparameter_grid,
                               cv=10, n_iter=11, 
                               scoring = 'neg_mean_squared_error',
                               n_jobs = -1, verbose = 1, 
                               return_train_score = True,
                               random_state=42)

# Fit on the training data
random_cv.fit(X_train, y_train)

# Best Estimators
best_estimators = random_cv.best_estimator_
print(best_estimators)
model_Lasso = Lasso(alpha=0.0002, random_state=42)

rmse_cv_Lasso = rmse_cv(model_Lasso)
print('Lasso Regressor RMSE is: {}'.format(rmse_cv_Lasso))
# Create the model to use for hyperparameter tuning
model = ElasticNet(random_state=42)

# Constant that multiplies the penalty terms
alpha = [0.0001, 0.0003, 0.0005, 0.001, 0.1, 0.3, 0.75, 1, 1.5]

# Define the grid of hyperparameters to search
hyperparameter_grid = {'alpha': alpha}

# Set up the random search with 10-fold cross validation
random_cv = RandomizedSearchCV(estimator=model,
                               param_distributions=hyperparameter_grid,
                               cv=10, n_iter=9, 
                               scoring = 'neg_mean_squared_error',
                               n_jobs = -1, verbose = 1, 
                               return_train_score = True,
                               random_state=42)

# Fit on the training data
random_cv.fit(X_train, y_train)

# Best Estimators
best_estimators = random_cv.best_estimator_
print(best_estimators)
model_ElasticNet = ElasticNet(alpha=0.0003, random_state=42)

rmse_cv_ElasticNet = rmse_cv(model_ElasticNet)
print('ElasticNet Regressor RMSE is: {}'.format(rmse_cv_ElasticNet))
# The strategy used to choose the split at each node
splitter = ['best', 'random']

# Maximum depth of the tree
max_depth = [2, 3, 5, 10, 15, None]

# Minimum number of samples to split a node
min_samples_split = [2, 4, 6, 10]

# Minimum number of samples per leaf
min_samples_leaf = [1, 2, 4, 6, 8]

# Maximum number of features to consider for making splits
max_features = ['auto', 'sqrt', 'log2', None]

# Define the grid of hyperparameters to search
hyperparameter_grid = {'splitter': splitter,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf,
                       'max_features': max_features}

# Create the model to use for hyperparameter tuning
model = DecisionTreeRegressor(random_state = 42)

# Set up the random search with 10-fold cross validation
random_cv = RandomizedSearchCV(estimator=model,
                               param_distributions=hyperparameter_grid,
                               cv=10, n_iter=40, 
                               scoring = 'neg_mean_squared_error',
                               n_jobs = -1, verbose = 1, 
                               return_train_score = True,
                               random_state=42)

# Fit on the training data
random_cv.fit(X_train, y_train)

# Best Estimators
best_estimators = random_cv.best_estimator_
print(best_estimators)
model_DecisionTree = DecisionTreeRegressor(splitter='best', max_depth=None, min_samples_split=6, min_samples_leaf=8, 
                                           max_features='auto', random_state=42)

rmse_cv_DecisionTree = rmse_cv(model_DecisionTree)
print('Decision Tree Regressor RMSE is: {}'.format(rmse_cv_DecisionTree))
# The number of trees in the forest
n_estimators = [100, 500, 900, 1100, 1500]

# Maximum depth of the tree
max_depth = [2, 3, 5, 10, 15, None]

# Minimum number of samples to split a node
min_samples_split = [2, 4, 6, 10]

# Minimum number of samples per leaf
min_samples_leaf = [1, 2, 4, 6, 8]

# Maximum number of features to consider for making splits
max_features = ['auto', 'sqrt', 'log2', None]

# Define the grid of hyperparameters to search
hyperparameter_grid = {'n_estimators': n_estimators,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf,
                       'max_features': max_features}

# Create the model to use for hyperparameter tuning
model = RandomForestRegressor(random_state = 42)

# Set up the random search with 10-fold cross validation
random_cv = RandomizedSearchCV(estimator=model,
                               param_distributions=hyperparameter_grid,
                               cv=10, n_iter=40, 
                               scoring = 'neg_mean_squared_error',
                               n_jobs = -1, verbose = 1, 
                               return_train_score = True,
                               random_state=42)

# Fit on the training data
random_cv.fit(X_train, y_train)

# Best Estimators
best_estimators = random_cv.best_estimator_
print(best_estimators)
model_RandomForest = RandomForestRegressor(n_estimators=1500, max_depth=15, min_samples_split=4, min_samples_leaf=1, 
                                           max_features='sqrt', random_state=42)

rmse_cv_RandomForest = rmse_cv(model_RandomForest)
print('Random Forest Regressor RMSE is: {}'.format(rmse_cv_RandomForest))
# Loss function to be optimized
loss = ['ls', 'lad', 'huber']

# Number of trees used in the boosting process
n_estimators = [100, 500, 900, 1100, 1500]

# Maximum depth of each tree
max_depth = [2, 3, 5, 10, 15]

# Minimum number of samples per leaf
min_samples_leaf = [1, 2, 4, 6, 8]

# Minimum number of samples to split a node
min_samples_split = [2, 4, 6, 10]

# Maximum number of features to consider for making splits
max_features = ['auto', 'sqrt', 'log2', None]

# Define the grid of hyperparameters to search
hyperparameter_grid = {'loss': loss,
                       'n_estimators': n_estimators,
                       'max_depth': max_depth,
                       'min_samples_leaf': min_samples_leaf,
                       'min_samples_split': min_samples_split,
                       'max_features': max_features}

# Create the model to use for hyperparameter tuning
model = GradientBoostingRegressor(random_state = 42)

# Set up the random search with 10-fold cross validation
random_cv = RandomizedSearchCV(estimator=model,
                               param_distributions=hyperparameter_grid,
                               cv=10, n_iter=40, 
                               scoring = 'neg_mean_squared_error',
                               n_jobs = -1, verbose = 1, 
                               return_train_score = True,
                               random_state=42)

# Fit on the training data
random_cv.fit(X_train, y_train)

# Best Estimators
best_estimators = random_cv.best_estimator_
print(best_estimators)
model_GB = GradientBoostingRegressor(loss='huber', max_depth=3, max_features='sqrt', min_samples_leaf=4,
                                     min_samples_split=10, n_estimators=100, random_state=42)

rmse_cv_GB = rmse_cv(model_GB)
print('Gradient Boosted Regressor RMSE is: {}'.format(rmse_cv_GB))
# Specifies the kernel type to be used in the algorithm
kernel = ['linear', 'poly', 'rbf', 'sigmoid', 'rbf']

# Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels
degree = [2, 3, 4, 5, 6]

# Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’
gamma = ['scale', 'auto']

# Regularization parameter. The strength of the regularization is inversely proportional to C
C = [0.1, 1, 1.5, 2]

# Define the grid of hyperparameters to search
hyperparameter_grid = {'kernel': kernel,
                       'degree': degree,
                       'gamma': gamma,
                       'C': C}

# Create the model to use for hyperparameter tuning
model = SVR()

# Set up the random search with 10-fold cross validation
random_cv = RandomizedSearchCV(estimator=model,
                               param_distributions=hyperparameter_grid,
                               cv=10, n_iter=40, 
                               scoring = 'neg_mean_squared_error',
                               n_jobs = -1, verbose = 1, 
                               return_train_score = True)

# Fit on the training data
random_cv.fit(X_train, y_train)

# Best Estimators
best_estimators = random_cv.best_estimator_
print(best_estimators)
model_SVR = SVR(kernel='rbf', degree=6, gamma='auto', C=2)

rmse_cv_SVR = rmse_cv(model_SVR)
print('Support Vector Regressor RMSE is: {}'.format(rmse_cv_SVR))
# Dataframe to hold the results
model_comparison = pd.DataFrame({'model': ['Ridge Regression', 'Lasso Regression', 'ElasticNet Regression', 
                                           'Decision Tree Regressor', 'Random Forest Regressor', 'Gradient Boosted Regressor',
                                           'Support Vector Regressor'],
                                 'rmse': [rmse_cv_Ridge, rmse_cv_Lasso, rmse_cv_ElasticNet, rmse_cv_DecisionTree, 
                                         rmse_cv_RandomForest, rmse_cv_GB, rmse_cv_SVR]})

# Plot
f, ax = plt.subplots(figsize=(10, 8))
sns.barplot(x='rmse', y='model', data=model_comparison.sort_values('rmse', ascending = True), color='red', edgecolor='black')
plt.ylabel('')
plt.yticks(size = 14)
plt.xlabel('Root Mean Squared Error')
plt.xticks(size = 14)
plt.title('Model Comparison on CV RMSE', size = 20);
model_Lasso.fit(X_train,y_train)
coef = pd.Series(model_Lasso.coef_, index = X_train.columns)

imp_coef = pd.concat([coef.sort_values().head(10), coef.sort_values().tail(10)])

f, ax = plt.subplots(figsize=(10, 8))
sns.barplot(x=imp_coef[:], y=imp_coef.index, color='blue', edgecolor='black')
plt.ylabel('')
plt.yticks(size = 14)
plt.xlabel('')
plt.xticks(size = 14)
plt.title('Coefficients in Lasso Model', size = 20)
print("Lasso picked " + str(sum(coef != 0)) + " features and eliminated the other " +  str(sum(coef == 0)) + " features")
submission_Lasso = pd.DataFrame()
submission_Lasso['Id'] = test_ID
model_Lasso.fit(X_train,y_train)
submission_Lasso['SalePrice'] = np.expm1(model_Lasso.predict(X_test))
submission_Lasso.to_csv('submission_Lasso.csv',index=False)

submission_Lasso.head()