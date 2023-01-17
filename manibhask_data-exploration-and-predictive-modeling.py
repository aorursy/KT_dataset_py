# Importing DataScience libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm,skew
from scipy import stats
%matplotlib inline
from sklearn.linear_model import LinearRegression,Ridge,RidgeCV, ElasticNetCV, LassoCV,BayesianRidge
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler,StandardScaler,MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split, GridSearchCV
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
# Reading csv file and making'Id' as Index
df_train = pd.read_csv('../input/train.csv',index_col = 'Id')
df_test = pd.read_csv('../input/test.csv',index_col = 'Id')
# Let's look at the top few rows
df_train.head()
# The rows and columns of our dataset 
df_train.shape
# Well we have to deal with plenty of attributes 
df_train.columns
# Datatype of each attribute
df_train.dtypes
# Descriptive Statistics of Numerical Variables
df_train.describe()
# Statistics of our Categorical variables
df_train.describe(include=['O'])
# Checking for the Missing values
# Using isnull fuction to count the total null values in each field
total = df_train.isnull().sum().sort_values(ascending=False) 
# Percent of missing values is estimated by dividing total missing and the original total
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
# Concatenating the Total and Percent fields sing pandas concat fucntion
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
# Displays top 20 from our max sorted list
missing_data.head(20)
# Pandas fillna method to fill missing values with None
for col in ('PoolQC', 'MiscFeature', 'Alley', 'Fence','FireplaceQu'):
    df_train[col] = df_train[col].fillna('None')
#Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood
df_train["LotFrontage"] = df_train.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
# For loop to replace the missing data in the 4 attributes to None 
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    df_train[col] = df_train[col].fillna('None')
# Filling 0 for GarageYrBlt since it depicts No Garage for missing values
df_train['GarageYrBlt'] = df_train['GarageYrBlt'].fillna(0)
# Filling with None since missing values means there is no Basement
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    df_train[col] = df_train[col].fillna('None')
df_train["MasVnrType"] = df_train["MasVnrType"].fillna("None")
df_train["MasVnrArea"] = df_train["MasVnrArea"].fillna(0)
df_train["Functional"] = df_train["Functional"].fillna("Typ")
# Imputing the most occuring field for Electrical since mostly all houses have Electricity
df_train['Electrical'] = df_train['Electrical'].fillna(df_train['Electrical'].mode()[0])
# Well! Things look better now! 
df_train.isnull().sum().max()
#Scatterplot of Area(sq ft) and SalePrice
var = 'GrLivArea'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
# set ylimit to limit our data
data.plot.scatter(x=var, y='SalePrice', ylim=(0,1000000));
#Sorted descending to pick the values of GrLivArea for them to drop it
df_train.sort_values(by = 'GrLivArea', ascending = False)[:2]
df_train = df_train.drop(df_train[df_train['GrLivArea'] == 5642].index)
df_train = df_train.drop(df_train[df_train['GrLivArea'] == 4676].index)
# Rechecking the plot after outlier elimination
var = 'GrLivArea'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,1000000));
# Total sq feet of Basement area and SalePrice comparison
var = 'TotalBsmtSF'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
#Let's look at some more variables which affect/influence the Target variable
g = sns.PairGrid(df_train,
                 x_vars=['OverallQual','GarageCars','FullBath'],
                 y_vars=["SalePrice"],
                 aspect=.75, size=6)
#plt.xticks(rotation=90)
g.map(sns.barplot, palette="coolwarm");
# seaborn pairplot
sns.set(style= 'ticks')
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(df_train[cols], size = 2.5)
plt.show();
# Using Seaborn to create a distplot with 20 bins
sns.distplot(df_train['SalePrice'],bins = 20);
#We use the numpy fuction log1p which  applies log(1+x) to all elements of the column
df_train["SalePrice"] = np.log1p(df_train["SalePrice"])

#Check the new distribution 
sns.distplot(df_train['SalePrice'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(df_train['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(df_train['SalePrice'], plot=plt)
plt.show()
sns.set(style="white")
corr = df_train.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(20,20))
cmap = sns.diverging_palette(1000, 50, as_cmap=True)
sns.heatmap(corr,annot=True, mask=mask,vmax =.8, cmap=cmap, center=0,fmt= '.2f',
            square=True, linewidths=.1, cbar_kws={"shrink": .5});
# Using datatype to seperate numerical features
numeric_feats = df_train.dtypes[df_train.dtypes != "object"].index

# Computing skewness using lambda function
skewed_feats = df_train[numeric_feats].apply(lambda x: skew(x.dropna()))
# Filtering highly skewed features
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index
#Log transform to normalize skewed features
df_train[skewed_feats] = np.log1p(df_train[skewed_feats])
# pandas get_dummies function to convert categorical values into binary
df_train = pd.get_dummies(df_train)
# X array - drop target
X= df_train.drop(['SalePrice'],axis=1).values
# y array - target only
y=df_train['SalePrice'].values
# Split Train into Train and Validate Sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20,random_state=42)
# Size of our newly split datasets 
print('Train size: %i' % X_train.shape[0])
print('Validation size: %i' % X_val.shape[0])
# No. of folds for Cross Validation
n_folds = 5
# Defining a function which returns the mean squared error between the cross validation score of X and y arrays
def rmse_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42)
    rmse= np.sqrt(-cross_val_score(model,X_train, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)
# Intialize the Cross Validation function with Shuffle
n_folds = 5
custom_cv = KFold(n_folds, shuffle=True, random_state=42)
#define  our  GB model
GBR = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)
#fitting he model on the train set
GBR.fit(X_train,y_train)
#Cross Validation Error
rmse_cv(GBR).mean()
# R squared, RMSE and Mean Absolute Error obtained by using actual and predicted values from Validate set
print('Validation R^2: %.5f'  % r2_score(y_val,GBR.predict(X_val)))
print('Validation RMSE: %.5f\n'   % mean_squared_error(y_val,GBR.predict(X_val)))
print('Validation Mean Absolute Error: %.5f\n'   % mean_absolute_error(y_val,GBR.predict(X_val)))
# ridgecv model
ridgecv = RidgeCV(alphas=[0.1, 1.0, 10.0],cv =custom_cv)
#fitting our model on the training data
ridgecv.fit(X_train,y_train)
# let's call our function to calculate the mean cross validation error
rmse_cv(ridgecv).mean()
# R- squared, RMSE and Mean Absolute Error for the validation sets
print('Validation R^2: %.5f'  % r2_score(y_val,ridgecv.predict(X_val)))
print('Validation RMSE: %.5f\n'   % mean_squared_error(y_val,ridgecv.predict(X_val)))
print('Validation Mean Absolute Error: %.5f\n'   % mean_absolute_error(y_val,ridgecv.predict(X_val)))
# kernel model
KRR = KernelRidge(alpha=5)
# fitting on train
KRR.fit(X_train,y_train)
#  model estimatores
print('Validation R^2: %.5f'  % r2_score(y_val,KRR.predict(X_val)))
print('Validation RMSE: %.5f\n'   % mean_squared_error(y_val,KRR.predict(X_val)))
print('Validation Mean Absolute Error: %.5f\n'   % mean_absolute_error(y_val,KRR.predict(X_val)))
# CV error
rmse_cv(KRR).mean()
# define the model with alpha values in a numpy array with cross-validation function included
lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005],cv =custom_cv)
# fit the model
lasso.fit(X_train,y_train)
# Model scoring
print('Validation R^2: %.5f'  % r2_score(y_val,lasso.predict(X_val)))
print('Validation RMSE: %.5f\n'   % mean_squared_error(y_val,lasso.predict(X_val)))
print('Validation Mean Absolute Error: %.5f\n'   % mean_absolute_error(y_val,lasso.predict(X_val)))
#CV error
rmse_cv(lasso).mean()
# defining the model with alphas and Elasticnet mixing parameter l1_ratio
Elastic = ElasticNetCV(alphas=[0.0001, 0.0005, 0.001, 0.01, 0.1, 1, 10], l1_ratio=[.01, .1, .5, .9, .99], max_iter=5000
                       ,cv=custom_cv)
# fit the model
Elastic.fit(X_train,y_train)
# Model scores
print('Validation R^2: %.5f'  % r2_score(y_val,Elastic.predict(X_val)))
print('Validation RMSE: %.5f\n'   % mean_squared_error(y_val,Elastic.predict(X_val)))
print('Validation Mean Absolute Error: %.5f\n'   % mean_absolute_error(y_val,Elastic.predict(X_val)))
# CV error
rmse_cv(Elastic).mean()
# import pipeline
from sklearn.pipeline import Pipeline
# initialize pipeline with the model and here we have function which takes cares of outlier
GBRPipe = Pipeline([
        ('outlier', RobustScaler()),
        ('gbm', GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5))])
# set up the parameter grid with parameters you like to iterate on. Here we have 3
param_grid = {'gbm__learning_rate': [0.01, 0.05],
             'gbm__max_depth': [1,5],
             'gbm__min_samples_leaf': [10, 15]}
             
# setup gridsearchcv with estimator, param_grid
GBRGrid = GridSearchCV(estimator = GBRPipe,param_grid=param_grid)
GBRGrid.fit(X_train,y_train)
rmse_cv(GBRGrid).mean()
print('Validation R^2: %.5f'  % r2_score(y_val,GBRGrid.predict(X_val)))
print('Validation RMSE: %.5f\n'   % mean_squared_error(y_val,GBRGrid.predict(X_val)))
print('Validation Mean Absolute Error: %.5f\n'   % mean_absolute_error(y_val,GBRGrid.predict(X_val)))