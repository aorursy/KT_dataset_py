# This first set of packages include Pandas, for data manipulation, numpy for mathematical computation and matplotlib & seaborn, for visualisation.
import pandas as pd
import numpy as np
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
sns.set(style='white', context='notebook', palette='deep')
print('Data Manipulation, Mathematical Computation and Visualisation packages imported!')

# Statistical packages used for transformations
from scipy import stats
from scipy.stats import skew, norm
from scipy.special import boxcox1p
from scipy.stats.stats import pearsonr
print('Statistical packages imported!')

# Metrics used for measuring the accuracy and performance of the models
#from sklearn import metrics
#from sklearn.metrics import mean_squared_error
print('Metrics packages imported!')

# Algorithms used for modeling
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
print('Algorithm packages imported!')

# Pipeline and scaling preprocessing will be used for models that are sensitive
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
print('Pipeline and preprocessing packages imported!')

# Model selection packages used for sampling dataset and optimising parameters
from sklearn import model_selection
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
print('Model selection packages imported!')

# Set visualisation colours
mycols = ["#66c2ff", "#5cd6d6", "#00cc99", "#85e085", "#ffd966", "#ffb366", "#ffb3b3", "#dab3ff", "#c2c2d6"]
sns.set_palette(palette = mycols, n_colors = 4)
print('My colours are ready! :)')

# To ignore annoying warning
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)
warnings.filterwarnings("ignore", category=DeprecationWarning)
print('Deprecation warning will be ignored!')


import os
print(os.listdir("../input"))

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

# Save the 'Id' column
train_ID = train['Id']
test_ID = test['Id']

# Now drop the  'Id' column as it's redundant for modeling
train.drop("Id", axis = 1, inplace = True)
test.drop("Id", axis = 1, inplace = True)

print(train.shape)
print(test.shape)
train.head()
# First of all, save the length of the training and test data for use later
ntrain = train.shape[0]
ntest = test.shape[0]

# Also save the target value, as we will remove this
y_train = train.SalePrice.values

# concatenate training and test data into all_data
all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop(['SalePrice'], axis=1, inplace=True)

print("all_data shape: {}".format(all_data.shape))
all_data.describe()
plt.subplots(figsize=(15, 5))

plt.subplot(1, 2, 1)
g = sns.regplot(x=train['GrLivArea'], y=train['SalePrice'], fit_reg=False).set_title("")
# aggregate all null values 
all_data_na = all_data.isnull().sum()

# get rid of all the values with 0 missing values
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
plt.subplots(figsize =(15, 10))
all_data_na.plot(kind='bar');
# Using data description, fill these missing values with "None"
for col in ("PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",
           "GarageType", "GarageFinish", "GarageQual", "GarageCond",
           "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1",
            "BsmtFinType2", "MSSubClass", "MasVnrType"):
    all_data[col] = all_data[col].fillna("None")
print("'None' - treated...")

# The area of the lot out front is likely to be similar to the houses in the local neighbourhood
# Therefore, let's use the median value of the houses in the neighbourhood to fill this feature
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))
print("'LotFrontage' - treated...")

# Using data description, fill these missing values with 0 
for col in ("GarageYrBlt", "GarageArea", "GarageCars", "BsmtFinSF1", 
           "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "MasVnrArea",
           "BsmtFullBath", "BsmtHalfBath"):
    all_data[col] = all_data[col].fillna(0)
print("'0' - treated...")


# Fill these features with their mode, the most commonly occuring value. This is okay since there are a low number of missing values for these features
all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])
all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])
all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])
all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])
all_data["Functional"] = all_data["Functional"].fillna(all_data['Functional'].mode()[0])
print("'mode' - treated...")

all_data_na = all_data.isnull().sum()
print("Features with missing values: ", all_data_na.drop(all_data_na[all_data_na == 0].index))

# From inspection, we can remove Utilities because it does not vary in the test data
all_data = all_data.drop(['Utilities'], axis=1)
all_data_na = all_data.isnull().sum()
print("Features with missing values: ", len(all_data_na.drop(all_data_na[all_data_na == 0].index)))
#Some more eda and visualizations
corr = train.corr()
plt.subplots(figsize=(30, 30))
cmap = sns.diverging_palette(150, 250, as_cmap=True)
sns.heatmap(corr, cmap="RdYlBu", vmax=1, vmin=-0.6, center=0.2, square=True, linewidths=0, cbar_kws={"shrink": .5}, annot = True);
#Check if categorical variables are ordinal?
#I assume that the algorithm can figure out the value of basement quality and will keep as categorical
#BsmtQual
plt.subplots(figsize =(20, 5))

plt.subplot(1, 3, 1)
sns.boxplot(x="BsmtQual", y="SalePrice", data=train, order=['Fa', 'TA', 'Gd', 'Ex']);

plt.subplot(1, 3, 2)
sns.stripplot(x="BsmtQual", y="SalePrice", data=train, size = 5, jitter = True, order=['Fa', 'TA', 'Gd', 'Ex']);

plt.subplot(1, 3, 3)
sns.barplot(x="BsmtQual", y="SalePrice", data=train, order=['Fa', 'TA', 'Gd', 'Ex']);
grid = plt.GridSpec(2, 3, wspace=0.1, hspace=0.15)
plt.subplots(figsize =(30, 15))

plt.subplot(grid[0, 0])
g = sns.regplot(x=train['BsmtFinSF1'], y=train['SalePrice'], fit_reg=False, label = "corr: %2f"%(pearsonr(train['BsmtFinSF1'], train['SalePrice'])[0]))
g = g.legend(loc="best")

plt.subplot(grid[0, 1:])
sns.boxplot(x="Neighborhood", y="BsmtFinSF1", data=train, palette = mycols)

plt.subplot(grid[1, 0]);
sns.barplot(x="BldgType", y="BsmtFinSF1", data=train, palette = mycols)

plt.subplot(grid[1, 1]);
sns.barplot(x="HouseStyle", y="BsmtFinSF1", data=train, palette = mycols)

plt.subplot(grid[1, 2]);
sns.barplot(x="LotShape", y="BsmtFinSF1", data=train, palette = mycols);
#Replace this with a flag

grid = plt.GridSpec(2, 3, wspace=0.1, hspace=0.15)
plt.subplots(figsize =(30, 15))

plt.subplot(grid[0, 0])
g = sns.regplot(x=train['BsmtFinSF2'], y=train['SalePrice'], fit_reg=False, label = "corr: %2f"%(pearsonr(train['BsmtFinSF2'], train['SalePrice'])[0]))
g = g.legend(loc="best")

plt.subplot(grid[0, 1:])
sns.boxplot(x="Neighborhood", y="BsmtFinSF2", data=train, palette = mycols)

plt.subplot(grid[1, 0]);
sns.barplot(x="BldgType", y="BsmtFinSF2", data=train, palette = mycols)

plt.subplot(grid[1, 1]);
sns.barplot(x="HouseStyle", y="BsmtFinSF2", data=train, palette = mycols)

plt.subplot(grid[1, 2]);
sns.barplot(x="LotShape", y="BsmtFinSF2", data=train, palette = mycols);
plt.subplots(figsize =(20, 5))

plt.subplot(1, 3, 1)
sns.boxplot(x="YearRemodAdd", y="SalePrice", data=train, palette = mycols);

plt.subplot(1, 3, 2)
sns.stripplot(x="YearRemodAdd", y="SalePrice", data=train, size = 5, jitter = True, palette = mycols);

plt.subplot(1, 3, 3)
sns.barplot(x="YearRemodAdd", y="SalePrice", data=train, palette = mycols);
#create new feature for the difference between build year and remodel year
all_data['Remod_Diff'] = all_data['YearRemodAdd'] - all_data['YearBuilt']
all_data.drop('YearRemodAdd', axis=1, inplace=True)
#combine all proch data and plot
all_data['TotalPorchSF'] = all_data['OpenPorchSF'] + all_data['OpenPorchSF'] + all_data['EnclosedPorch'] + all_data['3SsnPorch'] + all_data['ScreenPorch']
train['TotalPorchSF'] = train['OpenPorchSF'] + train['OpenPorchSF'] + train['EnclosedPorch'] + train['3SsnPorch'] + train['ScreenPorch']
grid = plt.GridSpec(2, 3, wspace=0.1, hspace=0.15)
plt.subplots(figsize =(30, 15))

plt.subplot(grid[0, 0])
g = sns.regplot(x=train['TotalPorchSF'], y=train['SalePrice'], fit_reg=False, label = "corr: %2f"%(pearsonr(train['TotalPorchSF'], train['SalePrice'])[0]))
g = g.legend(loc="best")

plt.subplot(grid[0, 1:])
sns.boxplot(x="Neighborhood", y="TotalPorchSF", data=train, palette = mycols)

plt.subplot(grid[1, 0]);
sns.barplot(x="BldgType", y="TotalPorchSF", data=train, palette = mycols)

plt.subplot(grid[1, 1]);
sns.barplot(x="HouseStyle", y="TotalPorchSF", data=train, palette = mycols)

plt.subplot(grid[1, 2]);
sns.barplot(x="LotShape", y="TotalPorchSF", data=train, palette = mycols);
grid = plt.GridSpec(2, 3, wspace=0.1, hspace=0.15)
plt.subplots(figsize =(30, 15))

plt.subplot(grid[0, 0])
g = sns.regplot(x=train['LotArea'], y=train['SalePrice'], fit_reg=False, label = "corr: %2f"%(pearsonr(train['LotArea'], train['SalePrice'])[0]))
g = g.legend(loc="best")

plt.subplot(grid[0, 1:])
sns.boxplot(x="Neighborhood", y="LotArea", data=train, palette = mycols)

plt.subplot(grid[1, 0]);
sns.barplot(x="BldgType", y="LotArea", data=train, palette = mycols)

plt.subplot(grid[1, 1]);
sns.barplot(x="HouseStyle", y="LotArea", data=train, palette = mycols)

plt.subplot(grid[1, 2]);
sns.barplot(x="LotShape", y="LotArea", data=train, palette = mycols);
#Correlation, but skewed. I will bin this
all_data['LotArea_Band'] = pd.cut(all_data['LotArea'], 4)
print (all_data['MiscFeature'].unique())
print (all_data['MiscVal'].unique())
plt.subplots(figsize =(20, 5))

plt.subplot(1, 3, 1)
sns.boxplot(x="MiscFeature", y="SalePrice", data=train, palette = mycols)

plt.subplot(1, 3, 2)
sns.stripplot(x="MiscFeature", y="SalePrice", data=train, size = 5, jitter = True, palette = mycols);

plt.subplot(1, 3, 3)
sns.barplot(x="MiscFeature", y="SalePrice", data=train, palette = mycols);

plt.subplots(figsize =(20, 5))

plt.subplot(1, 3, 1)
sns.boxplot(x="MiscVal", y="SalePrice", data=train, palette = mycols)

plt.subplot(1, 3, 2)
sns.stripplot(x="MiscVal", y="SalePrice", data=train, size = 5, jitter = True, palette = mycols);

plt.subplot(1, 3, 3)
sns.barplot(x="MiscVal", y="SalePrice", data=train, palette = mycols);
#create flags features that have a lot of 0's
#can look at these by changing the columns plotted in the prevous cells
all_data['BsmtFinSf2_Flag'] = all_data['BsmtFinSF2'].map(lambda x:0 if x==0 else 1)
all_data.drop('BsmtFinSF2', axis=1, inplace=True)

all_data['MiscVal_Flag'] = all_data['MiscVal'].map(lambda x:0 if x==0 else 1)
all_data.drop('MiscVal', axis=1, inplace=True)

all_data['MiscFeature_Flag'] = all_data['MiscFeature'].map(lambda x:0 if x=='None' else 1)
all_data.drop('MiscFeature', axis=1, inplace=True)

def WoodDeckFlag(col):
    if col['WoodDeckSF'] == 0:
        return 1
    else:
        return 0    
all_data['NoWoodDeck_Flag'] = all_data.apply(WoodDeckFlag, axis=1)

def PorchFlag(col):
    if col['TotalPorchSF'] == 0:
        return 1
    else:
        return 0
all_data['NoPorch_Flag'] = all_data.apply(PorchFlag, axis=1)

def PoolFlag(col):
    if col['PoolArea'] == 0:
        return 0
    else:
        return 1
all_data['HasPool_Flag'] = all_data.apply(PoolFlag, axis=1)
all_data.drop('PoolArea', axis=1, inplace=True)

def Slope(col):
    if col['LandSlope'] == 1:
        return 1
    else:
        return 0
all_data['GentleSlope_Flag'] = all_data.apply(Slope, axis=1)
all_data.drop('LandSlope', axis=1, inplace=True)

all_data['GasA_Flag'] = all_data['Heating'].map({"GasA":1, "GasW":0, "Grav":0, "Wall":0, "OthW":0, "Floor":0})
all_data.drop('Heating', axis=1, inplace=True)

all_data['LowQualFinSF_Flag'] = all_data['LowQualFinSF'].map(lambda x:0 if x==0 else 1)
all_data.drop('LowQualFinSF', axis=1, inplace=True)


#Some values are numeric, but should be categorical, fix this.
all_data['MSSubClass'] = all_data['MSSubClass'].astype(str)
all_data['OverallQual'] = all_data['OverallQual'].astype(str)
all_data['OverallCond'] = all_data['OverallCond'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)
all_data['YrSold'] = all_data['YrSold'].astype(str)


#get dummy variables for all of these
all_data = pd.get_dummies(all_data)
all_data.head(10)
#look at the target variable
plt.subplots(figsize=(15, 10))
g = sns.distplot(train['SalePrice'], fit=norm, label = "Skewness : %.2f"%(train['SalePrice'].skew()));
g = g.legend(loc="best")
#The distribution of the target variable is positively skewed, meaning that the mode is always less than the mean and median.
train["SalePrice"] = np.log1p(train["SalePrice"])
y_train = train["SalePrice"]
# First lets single out the numeric features
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

# Check how skewed they are
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

plt.subplots(figsize =(65, 20))
skewed_feats.plot(kind='bar');
skewness = skewed_feats[abs(skewed_feats) > 0.5]

skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    all_data[feat] = boxcox1p(all_data[feat], lam)

print(skewness.shape[0],  "skewed numerical features have been Box-Cox transformed")
# First, re-create the training and test datasets
train = all_data[:ntrain]
test = all_data[ntrain:]

print(train.shape)
print(test.shape)

# Next we want to sample our training data to test for performance of robustness and accuracy, before applying to the test data
scaler=StandardScaler()
X = scaler.fit_transform(train)
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, y_train, test_size=0.3, random_state=42)

# X_train = predictor features for estimation dataset
# X_test = predictor variables for validation dataset
# Y_train = target variable for the estimation dataset
# Y_test = target variable for the estimation dataset

print('X_train: ', X_train.shape, '\nX_test: ', X_test.shape, '\nY_train: ', Y_train.shape, '\nY_test: ', Y_test.shape)
# First I will use ShuffleSplit as a way of randomising the cross validation samples.
shuff = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

#Try random forest with default
rf=RandomForestRegressor()
rf.fit(X_train,Y_train)
training_results = np.sqrt((-cross_val_score(rf, X_train, Y_train, cv = shuff, scoring= 'neg_mean_squared_error')).mean())
test_results = np.sqrt(((Y_test-rf.predict(X_test))**2).mean())
print ('Default fit params: Training error:', training_results,'Test error:',test_results)


#Best fit params from random search cross-validation
params={'n_estimators': 1000, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'auto', 'max_depth': 20}
rf_random=RandomForestRegressor(**params)
rf_random.fit(X_train,Y_train,)
training_results = np.sqrt((-cross_val_score(rf_random, X_train, Y_train, cv = shuff, scoring= 'neg_mean_squared_error')).mean())
test_results = np.sqrt(((Y_test-rf_random.predict(X_test))**2).mean())
print ('Best fit params: Training error:', training_results,'Test error:',test_results)
                     
#Use random search to tune the Random Forest model
#THIS TAKES A LONG TIME!!!
from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]

rfr=RandomForestRegressor()

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,}

#Uncomment to run this tuning!!!
# rf_random = RandomizedSearchCV(estimator = rfr, param_distributions = random_grid, n_iter = 50, cv = shuff, verbose=2, random_state=42, n_jobs = -1, scoring= 'neg_mean_squared_error')
# # Fit the random search model
# rf_random.fit(X_train, Y_train)

# #What were the best params?
# print (rf_random.best_params_)
from sklearn.linear_model import LassoCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

#Try LASSO with default
lass=Lasso()
lass.fit(X_train,Y_train)
training_results = np.sqrt((-cross_val_score(lass, X_train, Y_train, cv = shuff, scoring= 'neg_mean_squared_error')).mean())
test_results = np.sqrt(((Y_test-lass.predict(X_test))**2).mean())
print ('Default fit params: Training error:', training_results,'Test error:',test_results)

#Best model in validation set
#Alpha from CV below
alpha=0.0045931063731713632
lasscv=Lasso(alpha=alpha)
lasscv.fit(X_train,Y_train)
training_results = np.sqrt((-cross_val_score(lasscv, X_train, Y_train, cv = shuff, scoring= 'neg_mean_squared_error')).mean())
test_results = np.sqrt(((Y_test-lasscv.predict(X_test))**2).mean())
print ('Best fit params, alpha=',alpha, 'Training error:', training_results,'Test error:',test_results)

#Best fit param with cross-validation
# lasscv=LassoCV()
# lasscv.fit(X_train,Y_train,)
# training_results = np.sqrt((-cross_val_score(lasscv, X_train, Y_train, cv = shuff, scoring= 'neg_mean_squared_error')).mean())
# test_results = np.sqrt(((Y_test-lasscv.predict(X_test))**2).mean())
# print ('Best fit params, alpha=',lasscv.alpha_, 'Training error:', training_results,'Test error:',test_results)

#Now use the previous lasso to downselect features, get their polynommial interactions, fit new LASSO model
poly_alpha=0.0093650422881921096
lasspoly=Pipeline([
    ('feature_selection',SelectFromModel(Lasso(alpha=alpha))),
    ('poly',PolynomialFeatures(2)),
    ('reg',Lasso(alpha=poly_alpha))])
lasspoly.fit(X_train, Y_train)
training_results = np.sqrt((-cross_val_score(lasspoly, X_train, Y_train, cv = shuff, scoring= 'neg_mean_squared_error')).mean())
test_results = np.sqrt(((Y_test-lasspoly.predict(X_test))**2).mean())
print ('Lasso with feature selection and polynomials: Training error:', training_results,'Test error:',test_results)

#fitting
#uncomment for testing
# model=Pipeline([
#     ('feature_selection',SelectFromModel(Lasso(alpha=alpha))),
#     ('poly',PolynomialFeatures(2)),
#     ('reg',LassoCV(n_jobs=-1,verbose=2))])
# model.fit(X_train,Y_train)
# poly_alpha=model.named_steps['reg'].alpha_

#Try polynomial interactions for all features
all_poly_alpha=0.011545661849734707
lassallpoly=Pipeline([
    ('poly',PolynomialFeatures(2)),
    ('reg',Lasso(alpha=all_poly_alpha))])
lassallpoly.fit(X_train, Y_train)
training_results = np.sqrt((-cross_val_score(lassallpoly, X_train, Y_train, cv = shuff, scoring= 'neg_mean_squared_error')).mean())
test_results = np.sqrt(((Y_test-lassallpoly.predict(X_test))**2).mean())
print ('Lasso with all polynomials: Training error:', training_results,'Test error:',test_results)

#Fit
# all_model=Pipeline([
#     ('poly',PolynomialFeatures(2)),
#     ('reg',LassoCV(n_jobs=-1,verbose=2))])
# all_model.fit(X_train,Y_train)
# all_poly_alpha=all_model.named_steps['reg'].alpha_

alpha=0.0045931063731713632
lasscv=Lasso(alpha=alpha)
lasscv.fit(X,y_train)
pred=lasscv.predict(scaler.transform(test))

submission = pd.DataFrame()
submission['Id'] = test_ID
submission['SalePrice'] = np.expm1(pred)
submission.to_csv('final_submission.csv',index=False)
print("Submission file, created!")
print (submission.head())

