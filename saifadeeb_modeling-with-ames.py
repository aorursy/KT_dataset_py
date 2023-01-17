#loading in necessary packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
#loading data sets
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
#printing dimensions of data sets
print ("Dimensions of training data:", train.shape)
print ("Dimensions of testing data:", test.shape)
#preview of training set
train.head()
#printing all integer type variables in the training set
print(train.dtypes[train.dtypes=='int64'])
#printing all float type variables in the training set
print(train.dtypes[train.dtypes=='float64'])
#printing all string types variables in training set
print(train.dtypes[train.dtypes=='object'])
#printing preview of testing set
test.head()
#printing all integer type variables in the testing set
print(test.dtypes[test.dtypes=='int64'])
#printing all float type variables in the testing set
print(test.dtypes[test.dtypes=='float64'])
#printing all string type variables in the testing set
print(test.dtypes[test.dtypes=='object'])
#Checking to see if Null values present in training set

#Displaying ratio of missing variables to total number of rows in training data
training_data_missing_values_ratio = np.round(train.isnull().sum().loc[train.isnull().sum()>0,]/(len(train)) * 100.0,1)
training_data_missing_values_ratio = training_data_missing_values_ratio.reset_index()
training_data_missing_values_ratio.columns = ['column', 'ratio']
training_data_missing_values_ratio.sort_values(by=['ratio'], ascending=False)
#Checking to see if Null values present in testing set

#Displaying ratio of missing variables to total number of rows in testing data
testing_data_missing_values_ratio = np.round(test.isnull().sum().loc[test.isnull().sum()>0,]/(len(test)) * 100.0,1)
testing_data_missing_values_ratio = testing_data_missing_values_ratio.reset_index()
testing_data_missing_values_ratio.columns = ['column', 'ratio']
testing_data_missing_values_ratio.sort_values(by=['ratio'], ascending=False)
#Displaying a few plots with Sale Price
f, ax = plt.subplots(figsize=(8, 4))
sns.regplot(x=train.GrLivArea, y=train.SalePrice)
plt.title("GrLivArea vs SalePrice")
#Noticed a few outliers, so I will be dropping them from our data set
train = train.drop(train.loc[(train.GrLivArea>4000) & (train.SalePrice < 200000),].index)

f, ax = plt.subplots(figsize=(8, 4))
sns.regplot(x=train.GrLivArea, y=train.SalePrice)
plt.title("GrLivArea vs SalePrice")
#PLotting distribution of Sale Price
f, ax = plt.subplots(figsize=(10, 6))
sns.distplot(train.SalePrice)
#Heavily right skewed
from scipy import stats
f, ax = plt.subplots(figsize=(10, 6))
stats.probplot(train.SalePrice, plot=plt)
#kurtosis validates that the distribution is not normal
from scipy.stats import kurtosis
kurtosis(train.SalePrice)
#Take the logarithm of Sale Price to get a more normal distribution
f, ax = plt.subplots(figsize=(10, 6))
sns.distplot(np.log(train.SalePrice))
#QQ-Plot appears more normal
f, ax = plt.subplots(figsize=(10, 6))
stats.probplot(np.log(train.SalePrice), plot=plt)
#kurtosis is closer to 0, it appears more normal
kurtosis(np.log(train.SalePrice))
#Getting all variables that have numerical values
numeric_features = train.select_dtypes(include=[np.number])
numeric_features.columns
#Getting all variables that have string values
categorical_features = train.select_dtypes(include=[np.object])
categorical_features.columns
#Examining the correlation between all numerical values with Sale Price
numeric_features.corr()['SalePrice'].sort_values(ascending = False)
#Correlation Matrix
corr = train.iloc[:,1:].corr()
f, ax = plt.subplots(figsize=(20, 8))
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns,
        cmap = "YlGnBu")
plt.title("Correlation Matrix")
#Looking at highly correlated variables 
high_corr_cols = numeric_features.corr()['SalePrice'].sort_values(ascending = False)[numeric_features.corr()[
    'SalePrice'].sort_values(ascending = False)>0.5].index.tolist()
high_corr_cols
#Heat map of highly correlated variables with Sale Price
corr = train[high_corr_cols].corr()
f, ax = plt.subplots(figsize=(20, 8))
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns,
            annot=True,
        cmap = "YlGnBu")
plt.title("Correlation Matrix")
#Imputing missing values in categorical variables for plotting purposes
for cols in categorical_features:
    train[cols] = train[cols].astype('category')
    if train[cols].isnull().any():
        train[cols] = train[cols].cat.add_categories(['MISSING'])
        train[cols] = train[cols].fillna('MISSING')

#Mass plotting of categorical variables to see if relationships with Sale Price exist
def mass_boxplot(x, y, **kwargs):
    sns.boxplot(x=x, y=y)
    x=plt.xticks(rotation=90)
f = pd.melt(train, id_vars=['SalePrice'], value_vars=categorical_features)
g = sns.FacetGrid(f, col="variable",  col_wrap=2, sharex=False, sharey=False, size=5)
g = g.map(mass_boxplot, "value", "SalePrice")
train2 = train
#some numerical values that are actually categorical values
cols_to_conv_to_categorical = ["MSSubClass", "OverallQual", "OverallCond", "MoSold", "YrSold"]
for cols in cols_to_conv_to_categorical:
    train2[cols] = train2[cols].apply(str)

#plotting many boxplots to see relationship with Sale Price
def mass_boxplot(x, y, **kwargs):
    sns.boxplot(x=x, y=y)
    x=plt.xticks(rotation=90)

f = pd.melt(train2, id_vars=['SalePrice'], value_vars=cols_to_conv_to_categorical)
g = sns.FacetGrid(f, col="variable",  col_wrap=2, sharex=False, sharey=False, size=5)
g = g.map(mass_boxplot, "value", "SalePrice")
#plotting more categorical variables with Sale Price
for cols in ["YearRemodAdd", "YearBuilt", "GarageYrBlt"]: 
    data = pd.concat([train2['SalePrice'], train2[cols]], axis=1)
    f, ax = plt.subplots(figsize=(40, 20))
    fig = sns.boxplot(x=cols, y="SalePrice", data=data)
    plt.xticks(rotation=90, fontsize = 20)
    title_name = "SalePrice Across " + cols
    plt.title(str(title_name), fontsize = 25)
#Taking a closer look at the distribution of Sale Price across OverAllQual
data = pd.concat([train['SalePrice'], train['OverallQual']], axis=1)
data["OverallQual"] = data["OverallQual"].apply(int)
f, ax = plt.subplots(figsize=(12, 8))
fig = sns.boxplot(x='OverallQual', y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
plt.title("SalePrice vs. OverAllQual")
#getting the order of neighborhood based by median sale price
neighborhood_order = train[['Neighborhood','SalePrice']].groupby([
    'Neighborhood']).describe()['SalePrice']['50%'].sort_values(ascending=True).index
#Plotting Sale Price across Neighborhoods
data = pd.concat([train['SalePrice'], train['Neighborhood']], axis=1)
f, ax = plt.subplots(figsize=(20, 12))
fig = sns.boxplot(x='Neighborhood', y="SalePrice", data=data, order = neighborhood_order)
fig.axis(ymin=0, ymax=800000);
plt.title("SalePrice Across Neighborhoods")
train_feature_check = pd.read_csv('../input/train.csv')
train_feature_check = train_feature_check.drop(train_feature_check.loc[(train_feature_check.GrLivArea>4000) & (train_feature_check.SalePrice < 200000),].index)
print("Original dimension:", train_feature_check.shape)
def conv_to_numeric(df, column_list, mapper):
    for cols in column_list:
        df[str("o")+cols] = df[cols].map(mapper)
        df[str("o")+cols].fillna(0, inplace = True)         
#remapping categorical variables with numerical values
convert_col_1 = ["ExterQual", "ExterCond","BsmtQual", "BsmtCond", "HeatingQC","KitchenQual","FireplaceQu",
                  "GarageQual","GarageCond","PoolQC"]
mapper = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None':0}

conv_to_numeric(train_feature_check, convert_col_1, mapper)

convert_col_2 = ["BsmtExposure"]
mapper = {'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, 'None':0}
conv_to_numeric(train_feature_check, convert_col_2, mapper)

convert_col_3 = ["BsmtFinType1", "BsmtFinType2"]
mapper = {'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1 ,'None':0}
conv_to_numeric(train_feature_check, convert_col_3, mapper)

convert_col_4 = ["GarageFinish"]
mapper = {'Fin': 3, 'RFn': 2, 'Unf': 1, 'None':0}
conv_to_numeric(train_feature_check, convert_col_4, mapper)

convert_col_5 = ["Fence"]
mapper = {'GdPrv': 4, 'MnPrv': 3, 'GdWo': 2, 'MnWw': 1, 'None':0}
conv_to_numeric(train_feature_check, convert_col_5, mapper)
convert_col = convert_col_1 + convert_col_2 + convert_col_3 + convert_col_4 + convert_col_5
new_features_col=[]

for cols in convert_col:
    cols = str("o")+cols
    new_features_col.append(cols)
#plotting variables
for cols in new_features_col:
    data = pd.concat([train_feature_check['SalePrice'], train_feature_check[cols]], axis=1)
    f, ax = plt.subplots(figsize=(12, 8))
    fig = sns.boxplot(x=cols, y="SalePrice", data=data)
    fig.axis(ymin=0, ymax=800000)
#created new variable TotalSqFt 
train_feature_check["TotalSqFt"] = train_feature_check["TotalBsmtSF"] + train_feature_check["1stFlrSF"] + train_feature_check["2ndFlrSF"]
new_features_col.append("TotalSqFt")
new_features_col.append("SalePrice")
print("Training dimension:", train.shape)
print("Training with new features dimension:", train_feature_check.shape)
#correlation matrix with new features
corr = train_feature_check[new_features_col].corr()
f, ax = plt.subplots(figsize=(20, 8))
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns,
        annot=True,
        cmap = "YlGnBu")
plt.title("Correlation Matrix for New Features")
#combining training and testing set to address missing values
train_row = train.shape[0]
test_row = test.shape[0]
y = train["SalePrice"]
all_data = pd.concat([train,test]).reset_index(drop=True)
all_data = all_data.drop(['SalePrice'], axis = 1)
print("Size of concatenated train and test datasets:", all_data.shape)
train.shape
test.shape
data_missing_values_ratio = np.round(
    all_data.isnull().sum().loc[all_data.isnull().sum()>0,]/(len(all_data)) * 100.0,3
)
data_missing_values_counts = all_data.isnull().sum().loc[all_data.isnull().sum()>0,].reset_index()
data_missing_values_counts.columns = ['column', 'counts']

data_missing_values_ratio = data_missing_values_ratio.reset_index()
data_missing_values_ratio.columns = ['column', 'ratio']

data_missing_values = pd.merge(left = data_missing_values_ratio , right = data_missing_values_counts, on = 'column', how='inner')
data_missing_values = data_missing_values.sort_values(by=['ratio'], ascending=False).reset_index(drop=True)
data_missing_values
ind = np.arange(data_missing_values.shape[0])
width = 0.1
fig, ax = plt.subplots(figsize=(20,8))
rects = ax.barh(ind, data_missing_values.counts.values[::-1], color='b')
ax.set_yticks(ind)
ax.set_yticklabels(data_missing_values.column.values[::-1], rotation='horizontal')
ax.set_xlabel("Missing Observations Count")
ax.set_title("Missing Observations Counts")
plt.show()
all_data["PoolQC"] = all_data["PoolQC"].fillna("None")
all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")
all_data["Alley"] = all_data["Alley"].fillna("None")
all_data["Fence"] = all_data["Fence"].fillna("None")
all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")
all_data["LotFrontage"].describe()
all_data["LotFrontage"] = all_data["LotFrontage"].fillna(all_data["LotFrontage"].describe()['50%'])
all_data["GarageCond"] = all_data["GarageCond"].fillna("None")
all_data["GarageQual"] = all_data["GarageQual"].fillna("None")
np.sort(all_data["GarageYrBlt"].unique().tolist())
all_data.loc[all_data["GarageYrBlt"] > 2016,]["GarageYrBlt"] 
all_data.loc[(all_data["GarageYrBlt"] > 2016),]["YearBuilt"]
all_data.loc[all_data["GarageYrBlt"] > 2016,"GarageYrBlt"]  =  all_data.loc[(all_data["GarageYrBlt"] > 2016),"YearBuilt"]
all_data["GarageYrBlt"] = all_data["GarageYrBlt"].fillna(0)
all_data["GarageFinish"] = all_data["GarageFinish"].fillna("None")
all_data["GarageType"] = all_data["GarageType"].fillna("None")
all_data["BsmtExposure"] = all_data["BsmtExposure"].fillna("None")
all_data["BsmtCond"] = all_data["BsmtCond"].fillna("None")
all_data["BsmtQual"] = all_data["BsmtQual"].fillna("None")
all_data["BsmtFinType2"] = all_data["BsmtFinType2"].fillna("None")
all_data["BsmtFinType1"] = all_data["BsmtFinType1"].fillna("None")
all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
all_data["MasVnrArea"].describe()
all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(all_data["MasVnrArea"].describe()['50%'])
all_data["MSZoning"].value_counts()
all_data["MSZoning"] = all_data["MSZoning"].fillna(all_data["MSZoning"].value_counts()[0])
all_data["Utilities"].value_counts()
all_data = all_data.drop(["Utilities"], axis = 1)
all_data["Functional"].value_counts()
all_data["Functional"] = all_data["Functional"].fillna(all_data["Functional"].value_counts()[0])
all_data["BsmtHalfBath"] = all_data["BsmtHalfBath"].fillna(0)
all_data["BsmtFullBath"] = all_data["BsmtFullBath"].fillna(0)
all_data["GarageCars"] = all_data["GarageCars"].fillna(0)
all_data["Exterior2nd"].value_counts()
all_data["Exterior2nd"] = all_data["Exterior2nd"].fillna(all_data["Exterior2nd"].value_counts()[0])
all_data["Exterior1st"].value_counts()
all_data["Exterior1st"] = all_data["Exterior1st"].fillna(all_data["Exterior1st"].value_counts()[0])
all_data["KitchenQual"].value_counts()
all_data["KitchenQual"] = all_data["KitchenQual"].fillna(all_data["KitchenQual"].value_counts()[0])
all_data["Electrical"].value_counts()
all_data["Electrical"] = all_data["Electrical"].fillna(all_data["Electrical"].value_counts()[0])
all_data["BsmtUnfSF"] = all_data["BsmtUnfSF"].fillna(0)
all_data["BsmtFinSF2"] = all_data["BsmtFinSF2"].fillna(0)
all_data["BsmtFinSF1"] = all_data["BsmtFinSF1"].fillna(0)
all_data["SaleType"].value_counts()
all_data["SaleType"] = all_data["SaleType"].fillna(all_data["SaleType"].value_counts()[0])
all_data["TotalBsmtSF"] = all_data["TotalBsmtSF"].fillna(0)
all_data["GarageArea"] = all_data["GarageArea"].fillna(0)
#Check to see if any other missing values
all_data.isnull().sum().loc[all_data.isnull().sum()>0,]
print("All Data Dimensions:", all_data.shape)
all_data["TotalSqFt"] = all_data["TotalBsmtSF"] + all_data["1stFlrSF"] + all_data["2ndFlrSF"]
#remapping categorical variables for training and testing set
convert_col_1 = ["ExterQual", "ExterCond","BsmtQual", "BsmtCond", "HeatingQC","KitchenQual","FireplaceQu",
                  "GarageQual","GarageCond","PoolQC"]
mapper = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None':0}

conv_to_numeric(all_data, convert_col_1, mapper)

convert_col_2 = ["BsmtExposure"]
mapper = {'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, 'None':0}
conv_to_numeric(all_data, convert_col_2, mapper)

convert_col_3 = ["BsmtFinType1", "BsmtFinType2"]
mapper = {'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1 ,'None':0}
conv_to_numeric(all_data, convert_col_3, mapper)

convert_col_4 = ["GarageFinish"]
mapper = {'Fin': 3, 'RFn': 2, 'Unf': 1, 'None':0}
conv_to_numeric(all_data, convert_col_4, mapper)

convert_col_5 = ["Fence"]
mapper = {'GdPrv': 4, 'MnPrv': 3, 'GdWo': 2, 'MnWw': 1, 'None':0}
conv_to_numeric(all_data, convert_col_5, mapper)
print("All Data New Dimensions:", all_data.shape)
cols_to_conv_to_categorical = cols_to_conv_to_categorical + ["YearRemodAdd", "YearBuilt", "GarageYrBlt"]

for cols in cols_to_conv_to_categorical:
    all_data[cols] = all_data[cols].astype(str)
from scipy.stats import skew
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

# Check the skew of all numerical features
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness.head(10)
#Displaying an example of skewness in data
sns.distplot(all_data["LotArea"])
from scipy.stats import skew
skew(all_data["LotArea"])
#After applying the Box Cox Transformation, we eliminate majority of the skewness and normalize the variable.
from scipy.special import boxcox1p
all_data_LotArea_boxcox_transform = boxcox1p(all_data["LotArea"], 0.15)
skew(all_data_LotArea_boxcox_transform)
sns.distplot(all_data_LotArea_boxcox_transform)
#Apply this to all features that exhibit skewness of over 
skewness = skewness[abs(skewness) > 0.75]
print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    all_data[feat] = boxcox1p(all_data[feat], lam)
all_data = pd.get_dummies(all_data)
print(all_data.shape)
ntrain = all_data[:train_row]
ntest = all_data[train_row:]
ntrain = ntrain.drop(['Id'], axis = 1)
ntest = ntest.drop(['Id'], axis = 1)
#import required packages
from sklearn.linear_model import ElasticNet, Lasso,Ridge
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor, plot_importance
import time
from mlxtend.regressor import StackingCVRegressor

RANDOM_SEED = 1
#defining number of folds
n_folds = 5

def cross_val_rmse(model):
    """This function will be used to perform cross validation and gather the average RMSE across five-folds for a model"""
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(ntrain.values)
    rmse= np.sqrt(-cross_val_score(model, ntrain.values, np.log(y).values, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)
models = [
          Ridge(alpha=0.5, random_state=RANDOM_SEED),
          ElasticNet(alpha=0.005, random_state=RANDOM_SEED),
          Lasso(alpha = 0.005, random_state=RANDOM_SEED),
          XGBRegressor(random_state=RANDOM_SEED)
         ]
model_name = ["Ridge","ElasticNet","Lasso","XGBoost"]

for name, model in zip(model_name, models):
    model_test = make_pipeline(RobustScaler(), model)
    score = cross_val_rmse(model_test)
    print(name, ": {:.4f} ({:.4f})".format(score.mean(), score.std()))
def grid_search_function(func_X_train, func_X_test, func_y_train, func_y_test, parameters, model):
    grid_search = GridSearchCV(model, parameters,  scoring='neg_mean_squared_error')
    regressor = grid_search.fit(func_X_train,func_y_train)
    return regressor
def train_test_split_function(X,y, test_size_percent):
    """Fucntion to perform train_test_split"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_percent, random_state=RANDOM_SEED)
    return X_train, X_test, y_train, y_test
starttime = time.monotonic()
parameters = {'ridge__alpha':[1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20], 'ridge__random_state':[RANDOM_SEED],
             'ridge__max_iter':[100000000]}
pipe = Pipeline(steps=[('rscale',RobustScaler()), ('ridge',Ridge())])
X_train, X_test, y_train, y_test = train_test_split(ntrain, 
                                                             np.log(y), 
                                                             test_size = 0.20,
                                                             random_state=RANDOM_SEED
                                                            )

ridge_regressor = grid_search_function(X_train, X_test, y_train, y_test, 
                                     parameters, 
                                     model = pipe)

print("That took ", (time.monotonic()-starttime)/60, " minutes")

print("\nBest Params:",ridge_regressor.best_estimator_)

print("\nBest Score:",np.sqrt(-ridge_regressor.best_score_))
ridge_regressor.best_estimator_.steps
starttime = time.monotonic()
parameters = {}
pipe = Pipeline(steps=[('ridge',Ridge(alpha=10, copy_X=True, fit_intercept=True, max_iter=100000000, normalize=False, random_state=1, solver='auto', tol=0.001))])

X_train, X_test, y_train, y_test = train_test_split(ntrain, 
                                                             np.log(y), 
                                                             test_size = 0.20,
                                                             random_state=RANDOM_SEED
                                                            )
ridge_model = grid_search_function(X_train, X_test, y_train, y_test, 
                                     parameters, 
                                     model = pipe)

print("That took ", (time.monotonic()-starttime)/60, " minutes")

print("\nBest Params:",ridge_model.best_estimator_)

print("\nBest Score:",np.sqrt(-ridge_model.best_score_))
starttime = time.monotonic()
parameters = {}
pipe = Pipeline(steps=[('enet',ElasticNet(alpha=0.0005, copy_X=True, fit_intercept=True, l1_ratio=0.5,
      max_iter=100000000, normalize=False, positive=False,
      precompute=False, random_state=1, selection='cyclic', tol=0.0001,
      warm_start=False))])
X_train, X_test, y_train, y_test = train_test_split(ntrain, 
                                                             np.log(y), 
                                                             test_size = 0.20,
                                                             random_state=RANDOM_SEED
                                                            )

enet_model = grid_search_function(X_train, X_test, y_train, y_test, 
                                     parameters, 
                                     model = pipe)

print("That took ", (time.monotonic()-starttime)/60, " minutes")

print("\nBest Params:",enet_model.best_estimator_)

print("\nBest Score:",np.sqrt(-enet_model.best_score_))
starttime = time.monotonic()
parameters = {}
pipe = Pipeline(steps=[('lasso',Lasso(alpha=0.0005, copy_X=True, fit_intercept=True, max_iter=100000000,
   normalize=False, positive=False, precompute=False, random_state=1,
   selection='cyclic', tol=0.0001, warm_start=False))])

X_train, X_test, y_train, y_test = train_test_split(ntrain, 
                                                             np.log(y), 
                                                             test_size = 0.20,
                                                             random_state=RANDOM_SEED
                                                            )

lasso_model = grid_search_function(X_train, X_test, y_train, y_test, 
                                     parameters, 
                                     model = pipe)

print("That took ", (time.monotonic()-starttime)/60, " minutes")

print("\nBest Params:",lasso_model.best_estimator_)

print("\nBest Score:",np.sqrt(-lasso_model.best_score_))
##UNCOMMENT to run

#starttime = time.monotonic()
#parameters = {'xgb__random_state':[RANDOM_SEED],
#             'xgb__gamma':[0,0.1], 
#              'xgb__learning_rate':[0.01,0.05,0.1],
#             'xgb__n_jobs':[-1], 
#              'xgb__n_estimators':[500,1000,2000],
#             'xgb__reg_lambda':[0,0.5,1],
#              'xgb__reg_alpha':[0,0.5,1]
#              }
#
#pipe = Pipeline(steps=[('xgb',XGBRegressor())])
#X_train, X_test, y_train, y_test = train_test_split(ntrain, 
#                                                             np.log(y), 
#                                                             test_size = 0.20,
# random_state=RANDOM_SEED)
#
#xgb_regressor = grid_search_function(X_train, X_test, y_train, y_test, 
#                                     parameters, 
#                                     model = pipe)
#
#print("That took ", (time.monotonic()-starttime)/60, " minutes")
#
#print("\nBest Params:",xgb_regressor.best_estimator_)
#
#print("\nBest Score:",np.sqrt(-xgb_regressor.best_score_))

#####Output###
####That took  66.38542943511857  minutes
####
####Best Params: Pipeline(memory=None,
####     steps=[('xgb', XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
####       colsample_bytree=1, gamma=0, learning_rate=0.05, max_delta_step=0,
####       max_depth=3, min_child_...
####       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
####       silent=True, subsample=1))])
####
####Best Score: 0.12259807102070201
##
###xgb_regressor.best_estimator_.steps
####[
#### ('xgb', XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
####         colsample_bytree=1, gamma=0, learning_rate=0.05, max_delta_step=0,
####         max_depth=3, min_child_weight=1, missing=None, n_estimators=1000,
####         n_jobs=-1, nthread=None, objective='reg:linear', random_state=1,
####         reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
####         silent=True, subsample=1))]
starttime = time.monotonic()
parameters = {}
pipe = Pipeline(steps=[
 ('xgb', XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
         colsample_bytree=1, gamma=0, learning_rate=0.05, max_delta_step=0,
         max_depth=3, min_child_weight=1, missing=None, n_estimators=1000,
         n_jobs=-1, nthread=None, objective='reg:linear', random_state=1,
         reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
         silent=True, subsample=1))])

X_train, X_test, y_train, y_test = train_test_split(ntrain, 
                                                             np.log(y), 
                                                             test_size = 0.20,
                                                             random_state=RANDOM_SEED
                                                            )

xgb_regressor = grid_search_function(X_train, X_test, y_train, y_test, 
                                     parameters, 
                                     model = pipe)

print("That took ", (time.monotonic()-starttime)/60, " minutes")

print("\nBest Params:",xgb_regressor.best_estimator_)

print("\nBest Score:",np.sqrt(-xgb_regressor.best_score_))
Ridge_model = Ridge(alpha=10, copy_X=True, fit_intercept=True, max_iter=100000000,
   normalize=False, random_state=1, solver='auto', tol=0.001)

Enet_model = ElasticNet(alpha=0.0005, copy_X=True, fit_intercept=True, l1_ratio=0.5,
      max_iter=100000000, normalize=False, positive=False,
      precompute=False, random_state=1, selection='cyclic', tol=0.0001,
      warm_start=False)

lasso_model = Lasso(alpha=0.0005, copy_X=True, fit_intercept=True, max_iter=100000000,
   normalize=False, positive=False, precompute=False, random_state=1,
   selection='cyclic', tol=0.0001, warm_start=False)

xgb_model = XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
         colsample_bytree=1, gamma=0, learning_rate=0.05, max_delta_step=0,
         max_depth=3, min_child_weight=1, missing=None, n_estimators=1000,
         n_jobs=-1, nthread=None, objective='reg:linear', random_state=1,
         reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
         silent=True, subsample=1)
#running a stacking model
regression_stacker = StackingCVRegressor(regressors = [
    Enet_model, Ridge_model, xgb_model],
                                         meta_regressor = lasso_model,
                                         cv=3)

score = cross_val_rmse(regression_stacker)
print(" Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
Ridge_model.fit(ntrain.values, np.log(y).values)
y_pred = Ridge_model.predict(ntest.values)
y_train_pred =  Ridge_model.predict(ntrain.values)
exp_y_pred = np.exp(y_pred)
submission = pd.read_csv("../input/sample_submission.csv")
submission.SalePrice = exp_y_pred
submission.to_csv("ridge_submission.csv",index = False)
print("Training Score:", np.sqrt(mean_squared_error(np.log(y).values, y_train_pred)))
#Examining magnitudes of features for Ridge regression
predictors = ntrain.columns

coef = pd.Series(Ridge_model.coef_,predictors).sort_values()
coef2 = coef[coef!=0]
f, ax = plt.subplots(figsize=(60, 20))
coef2.plot(kind='bar', title='Model Coefficients')

Enet_model.fit(ntrain.values, np.log(y).values)
y_pred = Enet_model.predict(ntest.values)
y_train_pred =  Enet_model.predict(ntrain.values)
exp_y_pred = np.exp(y_pred)
submission = pd.read_csv("../input/sample_submission.csv")
submission.SalePrice = exp_y_pred
submission.to_csv("elastic_net_submission.csv",index = False)
print("Training Score:", np.sqrt(mean_squared_error(np.log(y).values, y_train_pred)))
#Examining magnitudes of features for Elastic Net regression
predictors = ntrain.columns

coef = pd.Series(Enet_model.coef_,predictors).sort_values()
coef2 = coef[coef!=0]
f, ax = plt.subplots(figsize=(30, 12))
coef2.plot(kind='bar', title='Model Coefficients')

lasso_model.fit(ntrain.values, np.log(y).values)
y_pred = lasso_model.predict(ntest.values)
y_train_pred =  lasso_model.predict(ntrain.values)
exp_y_pred = np.exp(y_pred)
submission = pd.read_csv("../input/sample_submission.csv")
submission.SalePrice = exp_y_pred
submission.to_csv("lasso_submission.csv",index = False)
print("Training Score:", np.sqrt(mean_squared_error(np.log(y).values, y_train_pred)))
#Examining magnitudes of features for Lasso regression
predictors = ntrain.columns

coef = pd.Series(lasso_model.coef_,predictors).sort_values()
coef2 = coef[coef!=0]
f, ax = plt.subplots(figsize=(30, 12))
coef2.plot(kind='bar', title='Model Coefficients')

xgb_model.fit(ntrain.values, np.log(y).values)
y_pred = xgb_model.predict(ntest.values)
y_train_pred =  xgb_model.predict(ntrain.values)
exp_y_pred = np.exp(y_pred)
submission = pd.read_csv("../input/sample_submission.csv")
submission.SalePrice = exp_y_pred
submission.to_csv("xgboost_submission.csv",index = False)
print("Training Score:", np.sqrt(mean_squared_error(np.log(y).values, y_train_pred)))
#Examining feature importance (most important features) of XGBoost
xgb_feature_importance_df = pd.DataFrame({"column_names":ntrain.columns, "feature_importance": xgb_model.feature_importances_})
xgb_feature_importance_filtered_df = xgb_feature_importance_df.loc[xgb_feature_importance_df.feature_importance>0.003].sort_values('feature_importance',ascending=False)
f, ax = plt.subplots(figsize=(20, 12))
sns.barplot(x="feature_importance", y="column_names", data=xgb_feature_importance_filtered_df)
regression_stacker.fit(ntrain.values, np.log(y).values)
y_pred = regression_stacker.predict(ntest.values)
y_train_pred =  regression_stacker.predict(ntrain.values)
exp_y_pred = np.expm1(y_pred)
submission = pd.read_csv("../input/sample_submission.csv")
submission.SalePrice = exp_y_pred
submission.to_csv("stacking_exp1m_submission.csv",index = False)
print("Training Score:", np.sqrt(mean_squared_error(np.log(y).values, y_train_pred)))