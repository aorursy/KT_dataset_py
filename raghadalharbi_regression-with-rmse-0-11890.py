# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# all the used libraries in this project

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from matplotlib.colors import ListedColormap

from scipy import stats



from sklearn import datasets, metrics

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression, RidgeCV, LassoCV, ElasticNetCV

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

from sklearn import decomposition, datasets

from sklearn.pipeline import Pipeline

from sklearn.metrics import mean_squared_error, mean_squared_log_error, mean_absolute_error

from math import sqrt

from sklearn.tree import DecisionTreeRegressor

from sklearn import neighbors

from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor

from xgboost import XGBRegressor # just for fun :) 

from scipy.special import boxcox, inv_boxcox

 

plt.style.use('ggplot')

sns.set(font_scale = 1.5)

%config InlineBackend.figure_format = 'retina'

%matplotlib inline

import warnings

warnings.filterwarnings("ignore")



# Pallets used for visualizations

color= "Spectral"

color_plt = ListedColormap(sns.color_palette(color).as_hex())

color_hist = 'teal'
# Both train and test files

df_full = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test_df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
df = df_full



test_df['SalePrice'] = 0

# saving the IDs for the first and last data point in the test set 

# because we will be merging both train and test

test_first_id = test_df['Id'].iloc[0]

test_last_id = test_df['Id'].iloc[-1]
pd.set_option('display.max_columns', None)
df.head()
test_df.head()
# Combining both train and test datasets

df = df.append(test_df, ignore_index = True)
#Making sure that test set was appended to the main df

df.tail()
df.shape
df.info()
df.isna().sum()[df.isnull().sum() > 0].sort_values(ascending = False)
#Finding missing data and the percentage of it in each column

total = df.isnull().sum().sort_values(ascending = False)

percent = (df.isnull().sum() / df.isnull().count()).sort_values(ascending = False)

missing_data = pd.concat([total, percent], axis = 1, keys = ['Total_NaN', 'Percent_Nan'])

missing_data.head(20)
#visualize the missing data 

plt.figure(figsize = (19, 10))

sns.heatmap(data = df.isnull())
df.columns
df.describe()
df.dtypes
# fill missing values with NA in Categorical Columns

cat_bsmt_col = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']

cat_multi_col = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu']

cat_Garage_col = ['GarageType', 'GarageCond', 'GarageFinish', 'GarageQual']



df[cat_bsmt_col] = df[cat_bsmt_col].fillna('No_Basement')

df[cat_multi_col] = df[cat_multi_col].fillna('No')

df[cat_Garage_col] = df[cat_Garage_col].fillna('No_Garage')

df['MasVnrType']= df['MasVnrType'].fillna('No_MasVnr')



# numerical values

df['Electrical'].fillna(df['Electrical'].mode().iloc[0], inplace = True)

df['LotFrontage'].fillna(df['LotFrontage'].median(), inplace = True) #right skewed

df['GarageYrBlt'].fillna(df['YearBuilt'], inplace = True) #left skewed

df['MasVnrArea'].fillna(df['MasVnrArea'].median(), inplace = True) 
# check if all columns are  filled 

df.isna().sum().sum()
# These are the null in test

df.isna().sum()[df.isnull().sum() > 0].sort_values(ascending = False)
# fill missing data with mode as there are very few missing data

# we do not need to use other complex methods for filling data

df.fillna(df.mode().iloc[0], inplace = True)

df.isna().sum().sum()
# head of categorical columns

df[df.select_dtypes('object').columns].head()
# head of numerical columns

df[df.select_dtypes('number').columns].head()
####################################################### Mapping all quality columns to ranking numbers

ranking_columns = ['ExterQual', 'BsmtQual', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond']

# Removed: 'PoolQC', 'ExterCond', 'BsmtCond',



qual_dict = {"NA": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}



for column in ranking_columns:

    col = np.array(df[column].map(qual_dict), np.int16)

    df[column] = col

    

#Corr with Saleprice: 

#ExterQual    =     0.686756

#KitchenQual  =     0.662236

#BsmtQual     =     0.586674

#FireplaceQu  =     0.521144

#HeatingQC    =     0.428024

#GarageQual   =     0.273898

#GarageCond   =     0.263249

#BsmtCond     =     0.212632

#ExterCond    =     0.018865

#PoolQC       =     0.124084



####################################################### Mapping basement quality columns to ranking numbers  

#BsmtExposure    = 0.376309

#'BsmtFinType1'  = 0.305372

#'BsmtFinType2'  = -0.011422



basement_columns = ['BsmtFinType1']

basement_dict = {'NA':0, 'Unf':1, 'LwQ':2, 'Rec':3, 'BLQ':4, 'ALQ':5, 'GLQ':6 }    

#Removed: , 'BsmtFinType2'

for column in basement_columns:

    col = np.array(df[column].map(basement_dict), np.int16)

    df[column] = col  

    

# BsmtExposure   =   0.376309 

BsmtExposure_col = np.array(df['BsmtExposure'].map({'NA':0, 'No':1, 'Mn':2, 'Av':3, 'Gd':4}), np.int16)

df['BsmtExposure'] = BsmtExposure_col



####################################################### Mapping garage quality columns to ranking numbers  

# GarageFinish  = 0.550255

GarageFinish_col = np.array(df['GarageFinish'].map({'NA':0, 'Unf':1, 'RFn':2, 'Fin':3 }), np.int16)

df['GarageFinish'] = GarageFinish_col    



#CentralAir =    0.251328

df['CentralAir'] = df['CentralAir'].map({'N':0, 'Y':1}).astype(int) 



#PavedDrive =    0.233281

df['PavedDrive'] = df['PavedDrive'].map({'N':0, 'P':1, 'Y':3}).astype(int) 



#LandSlope =  -0.051779

df['LandSlope'] = df['LandSlope'].map({'Sev':0, 'Mod':1, 'Gtl':3}).astype(int)
# Just to make sure again no Null values after conversion 

df.isna().sum().sum()
# new dataframe for only train set to visualize relationships and corelations 

visual_df = df.iloc[0:(df[df['Id'] == test_first_id].index[0]), :] # train set

visual_df.head()
# Finding correlation of all numerical columns with target

visual_df.corr()['SalePrice'].sort_values(ascending = False)
# Just to check if any of the categorical columns has a high correlation with SalePrice

cat_columns = visual_df[visual_df.select_dtypes('object').columns]

for column in cat_columns:

    cat_columns[column] = visual_df[column].astype('category').cat.codes

    

cat_columns['SalePrice'] = visual_df['SalePrice']

cat_columns.corr()['SalePrice'].sort_values(ascending = False)
fig, ax = plt.subplots( figsize=(15, 6))

ax.hist(visual_df['SalePrice'], bins = 300, color = color_hist)



ax.set_xlabel('SalePrice')

ax.set_ylabel('Frequency')

fig.suptitle('The Distribution of Sale Price Before Transformation', fontsize = 20)



ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))



plt.show()
fig, ax = plt.subplots(figsize = (14, 6))

res = stats.probplot(visual_df['SalePrice'], plot = plt)

fig.suptitle('Probability Plot of Sale Price Before Transformation', fontsize = 20)



ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))

plt.show()
fig, ax = plt.subplots( figsize = (15, 6))

ax.hist(np.log(visual_df['SalePrice']), bins = 300, color = color_hist)



ax.set_xlabel('SalePrice')

ax.set_ylabel('Frequency')



fig.suptitle('The Distribution of Sale Price After  log Transformation', fontsize = 20)



ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))



plt.show()
fig, ax = plt.subplots(figsize = (14, 6))

res = stats.probplot(np.log1p(visual_df['SalePrice']), plot = plt)



fig.suptitle('Probability Plot of Sale Price After log Transformation', fontsize = 20)



ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))

plt.show()
# applying log to target

df['SalePrice'] = np.log(df['SalePrice'])
# getting the columns with the highest corelation with salePrice, and clean them from outliers

high_corr = visual_df.corr()['SalePrice'].sort_values(ascending = False).head(10)

high_corr.index.to_list()
# overview of all plots

sns.pairplot(visual_df[high_corr.index.to_list()])
fig, ax = plt.subplots( figsize = (12, 8))

ax = sns.scatterplot(x = 'ExterQual', 

                     y = 'SalePrice', 

                     data = visual_df, 

                     marker = 'o', s = 200, palette = color)



ax.set_ylabel('Sale Price')

ax.set_xlabel('The quality of the material on the exterior')

fig.suptitle('The Guality of the Material on the Exterior vs. Sales Price', fontsize = 20)



ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))

ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))

plt.show()
fig, ax = plt.subplots( figsize = (12, 8))

ax = sns.scatterplot(x = 'OverallQual', 

                     y = 'SalePrice', 

                     data = visual_df, 

                     marker = 'o', s = 200, palette = color)



ax.set_ylabel('Sale Price')

ax.set_xlabel('Overall Quality')

fig.suptitle('Overall Quality vs. Sales Price', fontsize = 20)



ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))

ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))

plt.show()
fig, ax = plt.subplots( figsize = (12, 8))

ax = sns.scatterplot(x = 'GrLivArea', 

                     y = 'SalePrice', 

                     data = visual_df, 

                     marker = 'o', s = 200, palette = color)



ax.set_ylabel('Sale Price')

ax.set_xlabel('Ground living area')

fig.suptitle('Ground living Area vs. Sales Price', fontsize = 20)



ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))

ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))

plt.show()
fig, ax = plt.subplots( figsize = (12, 8))

ax = sns.scatterplot(x = 'TotalBsmtSF', 

                     y = 'SalePrice', 

                     data = visual_df, 

                     marker = 'o', s = 200, palette = color)



ax.set_ylabel('Sale Price')

ax.set_xlabel('TotalBsmtSF')

fig.suptitle('Total square feet of basement area vs. Sales Price', fontsize = 20)



ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))

ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))

plt.show()
fig, ax = plt.subplots( figsize = (12, 8))

ax = sns.scatterplot(x = '1stFlrSF', 

                     y = 'SalePrice', 

                     data = visual_df, 

                     marker = 'o', s = 200, palette = color)



ax.set_ylabel('Sale Price')

ax.set_xlabel('1stFlrSF')

fig.suptitle('First Floor Square Feet vs. Sales Price', fontsize = 20)



ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))

ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))

plt.show()
cat_visual_df = visual_df[visual_df.select_dtypes('object').columns]

cat_visual_df.columns
# find corelations between all columns and the target

df.corr()['SalePrice'].sort_values(ascending = False)
fig, axs = plt.subplots(figsize = (16, 14)) 

mask = np.triu(np.ones_like(visual_df.corr(), dtype = np.bool))

g = sns.heatmap(visual_df.corr(), ax = axs, mask=mask, cmap = sns.diverging_palette(180, 10, as_cmap = True), square = True)



plt.title('Correlation between Features')



# fix for mpl bug that cuts off top/bottom of seaborn viz

b, t = plt.ylim() # discover the values for bottom and top

b += 0.5 # Add 0.5 to the bottom

t -= 0.5 # Subtract 0.5 from the top

plt.ylim(b, t) # update the ylim(bottom, top) values

plt.show()
df = df.drop(['GarageYrBlt', 'TotRmsAbvGrd', 'TotalBsmtSF'], axis = 1)
corr_matrix = visual_df.corr()

top_corr_features = corr_matrix.index[abs(corr_matrix['SalePrice']) > 0.5]



fig, axs = plt.subplots(figsize = (13, 8)) 

mask = np.triu(np.ones_like(visual_df[top_corr_features].corr(), dtype = np.bool))

sns.heatmap(visual_df[top_corr_features].corr(), ax = axs, annot = True, mask = mask, cmap = sns.diverging_palette(180, 10, as_cmap = True))

plt.title('Correlation of high correlated columns with Sale Price')



# fix for mpl bug that cuts off top/bottom of seaborn viz

b, t = plt.ylim() # discover the values for bottom and top

b += 0.5 # Add 0.5 to the bottom

t -= 0.5 # Subtract 0.5 from the top

plt.ylim(b, t) # update the ylim(bottom, top) values

plt.show()
# a function that takes a dataframe and transforms it into a standard form after dropping nun_numirical columns

def to_standard (df):

    

    num_df = df[df.select_dtypes(include = np.number).columns.tolist()]

    

    ss = StandardScaler()

    std = ss.fit_transform(num_df)

    

    std_df = pd.DataFrame(std, index = num_df.index, columns = num_df.columns)

    return std_df
ax, fig = plt.subplots(1, 1, figsize = (18, 18))

plt.title('The distribution of All Numeric Variable in the Dataframe', fontsize = 20) #Change please



sns.boxplot(y = "variable", x = "value", data = pd.melt(to_standard(visual_df)), palette = color)

plt.xlabel('Range after Standarization', size = 16)

plt.ylabel('Attribue', size = 16)





# fix for mpl bug that cuts off top/bottom of seaborn viz

b, t = plt.ylim() # discover the values for bottom and top

b += 0.5 # Add 0.5 to the bottom

t -= 0.5 # Subtract 0.5 from the top

plt.ylim(b, t) # update the ylim(bottom, top) values



plt.show()
ax, fig = plt.subplots(1, 1, figsize = (18, 8))

plt.title('The distribution of All Numeric Variable in the Dataframe', fontsize = 20) #Change please



sns.boxplot(y = "variable", x = "value", data = pd.melt(to_standard(visual_df[top_corr_features])), palette = color)

plt.xlabel('Range after Standarization', size = 16)

plt.ylabel('Attribue', size = 16)





# fix for mpl bug that cuts off top/bottom of seaborn viz

b, t = plt.ylim() # discover the values for bottom and top

b += 0.5 # Add 0.5 to the bottom

t -= 0.5 # Subtract 0.5 from the top

plt.ylim(b, t) # update the ylim(bottom, top) values



plt.show()
numeric_feats = visual_df.dtypes[visual_df.dtypes != "object"].index



# Check the skew of all numerical features

skewed_feats = visual_df[numeric_feats.tolist()].apply(lambda x:stats.skew(x.dropna())).sort_values(ascending = False)

print("\nSkew in numerical features: \n")

skewness = pd.DataFrame({'Skew':skewed_feats})

skewness.head()

skewed_features =['MiscVal', 'PoolArea', 'LotArea', '3SsnPorch', 'LowQualFinSF', 

                  'KitchenAbvGr','BsmtFinSF2', 'ScreenPorch', 'GrLivArea', 'ExterQual',

                  'BsmtHalfBath']



skewness = skewness[abs(skewness) > 0.75]

print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))





#skewed_features = skewness.index

lam = 0.15

for feat in skewed_features:

    df[feat] = boxcox(df[feat], lam)
# changing all Categorical columns to dummies (0,1)

df = pd.get_dummies(df, columns = df.select_dtypes('object').columns, drop_first = True)
# split data from from test_first_id to the end to be in the test

test_df = df.iloc[df[df['Id'] == test_first_id].index[0]:, :]

test_df.head()
# deleting the test dataset from the main df

df = df.iloc[0:(df[df['Id'] == test_first_id].index[0]), :]

df.tail()
# removing GrLivArea outliers, also dropping SalePrice > 700,000 by default, which is good.

print(df.shape)

df.drop(df.index[[523, 1298]], inplace = True)

df = df.drop(df[(df['GrLivArea'] > 4000) & (df['SalePrice'] < 300000)].index)

df = df.drop(df[df['1stFlrSF'] >= 3000].index)

print(df.shape)
#No need for the ID column

df = df.drop('Id', axis = 1)



test_df = test_df.drop(['Id','SalePrice'] , axis = 1)
# a function that gets the predictions and saves them into a csv file with the correct format

def submission_file (test_pred):

    for_id = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')



    my_submission = pd.DataFrame({'Id':for_id.Id, 'SalePrice':test_pred.reshape(1459)})

    my_submission.to_csv('submission.csv', index = False) # dropping the index column before saving it
BOLD = '\033[1m'

END = '\033[0m'
y = df['SalePrice']

X = df.drop('SalePrice', axis = 1)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .10, shuffle = True, random_state = 42)
# a function that gets all datasets and model, and will fit and calculates all metrics, and return predictions

def model_metrics(model, kfold, X_train, X_test, y_train, y_test, test_df):



    model.fit(X_train, y_train)



    #metrics -> R squared

    results = cross_val_score(model, X_train, y_train, cv = kfold, scoring = 'r2')

    print("CV scores: ", results); print("CV Standard Deviation: ", results.std()); print();

    print('CV Mean score: ', results.mean()); 

    print('Train score:   ', model.score(X_train, y_train))

    print('Test score:    ', model.score(X_test, y_test))

      

    MSE = -(cross_val_score(model, X_train, y_train, cv = kfold, scoring = 'neg_mean_squared_error').mean())

    print("CV MSE:        ",MSE)

    

    RMSE = sqrt(MSE)

    print("CV RMSE:       ",RMSE)

    

    test_pred = model.predict(test_df)

    test_pred_exp = np.exp(test_pred)

    

    return test_pred_exp
def Multi_models (X_train, X_test, y_train, y_test, test_df):

    kfold = 5

#     # Create an scaler object

#     ss = StandardScaler()

#     X_train = ss.fit_transform(X_train)

#     X_test = ss.transform(X_test)

#     test_df = ss.transform(test_df)

#     y_train = ss.transform(y_train)

#     y_test = ss.transform(y_test)

###################################################################################################### Linear Regression model

    print(BOLD + 'Linear Regression model:' + END)

    

    lr = LinearRegression()

    lr_pred= model_metrics(lr, kfold, X_train, X_test, y_train, y_test, test_df)

    

######################################################################################################  Lasso model

    print(); print(BOLD + 'Lasso model:' + END)

    

    alpha = np.arange(0, 3, 200)

    lasso = Lasso(alpha = alpha, max_iter = 50000)

    lasso_pred = model_metrics(lasso, kfold, X_train, X_test, y_train, y_test, test_df)



######################################################################################################  Ridge model

    print(); print(BOLD + 'Ridge model:' + END)

    

    ridge_alpha_values = np.logspace(0, 5, 200)

    ridgecv_optimal = RidgeCV(alphas = ridge_alpha_values, cv = 10)

    ridge_pred = model_metrics(ridgecv_optimal, kfold, X_train, X_test, y_train, y_test, test_df)

    

######################################################################################################  Elastic Net model

    print(); print(BOLD + 'Elastic Net model:' + END)

    

    elasticnet = ElasticNet(alpha = 0.01)

    elasticnet_pred = model_metrics(elasticnet, kfold, X_train, X_test, y_train, y_test, test_df)

    

######################################################################################################  Decision Tree Regressor model

    print(); print(BOLD + 'Decision Tree Regressor model:' + END)

    

    dtr = DecisionTreeRegressor()

    dtr_pred = model_metrics(dtr, kfold, X_train, X_test, y_train, y_test, test_df)

    

######################################################################################################  K Neighbors Regressor model

    print(); print(BOLD + 'K Neighbors Regressor model:' + END)

    

    KNN = neighbors.KNeighborsRegressor()

    KNN_pred = model_metrics(KNN, kfold, X_train, X_test, y_train, y_test, test_df)

    

######################################################################################################  Random Forest Regressor model   

    print(); print(BOLD + 'Random Forest Regressor model:' + END)



    rfr = RandomForestRegressor(n_estimators = 100, oob_score = True, random_state = 42)

    rfr_pred = model_metrics(rfr, kfold, X_train, X_test, y_train, y_test, test_df)

    

    #submission_file (dtr_pred)

    
Multi_models (X_train, X_test, y_train, y_test, test_df)
def ridge__optimizer(X_train, X_test, y_train, y_test, test_df):

    print(); print(BOLD + 'Ridge model (best so far with 0.12100 kaggle score):' + END)

    kfold = 5

    

    

    ridge_alpha_values = np.logspace(0, 5, 200)



    ridgecv_optimal = RidgeCV(alphas = ridge_alpha_values, cv = 10)

    ridgecv_optimal.fit(X_train, y_train)



    print('Optimal Alpha:   ' , ridgecv_optimal.alpha_)

    

    # Create a logistic regression object with an L2 penalty

    ridge = Ridge(alpha = ridgecv_optimal.alpha_)



    

    ridge_opt_pred = model_metrics(ridge, kfold, X_train, X_test, y_train, y_test, test_df)

    print(ridge_opt_pred)

    

    #submission_file (ridge_opt_pred) 

    #The line for saving the predictions for submission gives an error in Kaggle, therefor we commented it here
ridge__optimizer(X_train, X_test, y_train, y_test, test_df)
rmse = []

# check the below alpha values for Ridge Regression

alpha = np.arange(0.0001, 10, 200)



for alph in alpha:

    ridge = Ridge(alpha = alph, copy_X = True, fit_intercept = True)

    ridge.fit(X_train, y_train)

    predict = ridge.predict(X)

    rmse.append(np.sqrt(mean_squared_error(predict, y)))

print(rmse)

plt.scatter(alpha, rmse)

rmse = pd.Series(rmse, index = alpha)

print(rmse.argmin())

print(rmse.min())
# trying KNN regression model with a stander scaler and grid search for multiple k values

def KNN_opt_model (X_train, X_test, y_train, y_test, test_df):

    kfold = 5

    

    print(); print(BOLD + 'K Neighbors Regressor model:' + END)



    # Create an scaler object

    ss = StandardScaler()



    # Create a logistic regression object with an L2 penalty

    KNN = neighbors.KNeighborsRegressor()



    # Create a pipeline of three steps. First, standardize the data.

    # Second, tranform the data with PCA.

    # Third, train a Decision Tree Classifier on the data.

    pipe = Pipeline(steps = [('ss', ss),

                           ('KNN', KNN)])

    

    # Create lists of parameter for KNeighborsRegressor()

    n_neighbors = [5, 10, 15]

    algorithm = ['auto', 'ball_tree', 'kd_tree', 'brute']



    # Create a dictionary of all the parameter options 

    # Note has you can access the parameters of steps of a pipeline by using '__’

    parameters = dict(KNN__n_neighbors = n_neighbors,

                      KNN__algorithm = algorithm)



    # Conduct Parameter Optmization With Pipeline

    # Create a grid search object

    clf = GridSearchCV(pipe, parameters)

    

    KNN_pred = model_metrics(KNN, kfold, X_train, X_test, y_train, y_test, test_df)

    #submission_file (KNN_pred)

   
KNN_opt_model(X_train, X_test, y_train, y_test, test_df)
def lasso_optimizer(X_train, X_test, y_train, y_test, test_df, X, y):

    print(); print(BOLD + 'Optimized Lasso model:' + END)

    kfold = 5

    optimal_lasso = LassoCV(n_alphas = 500, cv = 10, verbose = 1)

    optimal_lasso.fit(X_train, y_train)

    print('optimal_lasso:    ', optimal_lasso.alpha_)

    

    # Create a logistic regression object with an L2 penalty

    lasso = Lasso(alpha = optimal_lasso.alpha_)



    lasso_pred = model_metrics(lasso, kfold,  X_train, X_test, y_train, y_test, test_df)

    submission_file(lasso_pred)

    

    lasso.fit(X, y)



    lasso_coefs = pd.DataFrame()

    lasso_coefs['Column_name'] = X.columns

    lasso_coefs['coefficient'] = lasso.coef_

    lasso_coefs['absolute_coefficient'] = np.abs(lasso.coef_)



    lasso_coefs = lasso_coefs.sort_values('absolute_coefficient', ascending = False)

    print('Percent variables zeroed out:', np.sum(lasso.coef_ == 0) / X.iloc[:, 0].count())



    lasso_coefs.head(15)

    return lasso_coefs
lasso_coefs = lasso_optimizer(X_train, X_test, y_train, y_test, test_df, X, y);
lasso_coefs.head(10)
def ElasticNet__optimizer(X_train, X_test, y_train, y_test, test_df, X, y):

    

    kfold = 5

    

    l1_ratios = np.linspace(0.01, 1.0, 25)



    optimal_enet = ElasticNetCV(l1_ratio = l1_ratios, n_alphas = 100, cv = 10, verbose = 1)

    optimal_enet.fit(X, y)



    print(); print(BOLD + 'Elastic Net model:' + END)

    print('Optimal alpha:       ', optimal_enet.alpha_)

    print('Optimal l1 ratio  :  ', optimal_enet.l1_ratio_)

    

    enet = ElasticNet(alpha = optimal_enet.alpha_, l1_ratio = optimal_enet.l1_ratio_)

    



    enet_pred = model_metrics(enet, kfold,  X_train, X_test, y_train, y_test, test_df)

    #submission_file (enet_pred)
ElasticNet__optimizer(X_train, X_test, y_train, y_test, test_df, X, y)
#Applying DecisionTreeRegressor Model 



print(BOLD + 'Decision Tree Regressor model:' + END)





decision_tree = DecisionTreeRegressor( max_depth = 10, random_state = 33)

decision_tree.fit(X_train, y_train)



#Calculating Training & Testing Scores

print('Train Score: ', decision_tree.score(X_train, y_train))

print('Test Score is : ', decision_tree.score(X_test, y_test))

print('----------------------------------------------------')



#Calculating Prediction

y_pred = decision_tree.predict(X_test)



#----------------------------------------------------

#Calculating MAE

MAE_value = mean_absolute_error(y_test, y_pred, multioutput = 'uniform_average') 

print('MAE Score: ', MAE_value)



#----------------------------------------------------

#Calculating MSE

MSE_value = mean_squared_error(y_test, y_pred, multioutput = 'uniform_average') 

print('MSE Score: ', MSE_value)


