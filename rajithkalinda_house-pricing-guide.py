# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np 

import pandas as pd

from sklearn.preprocessing import StandardScaler

from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder, RobustScaler

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.preprocessing import LabelEncoder

from xgboost import XGBRegressor

from sklearn.linear_model import Lasso

from sklearn.linear_model import LassoCV

from sklearn.linear_model import LogisticRegression 

from sklearn.linear_model import LinearRegression

from sklearn import svm 

from sklearn.ensemble import RandomForestClassifier 

from sklearn.neighbors import KNeighborsClassifier 

from sklearn.naive_bayes import GaussianNB 

from sklearn.tree import DecisionTreeClassifier 

from sklearn.model_selection import train_test_split 

from sklearn import metrics 

from sklearn.metrics import confusion_matrix 

from sklearn.linear_model import Lasso

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge

from sklearn.linear_model import ElasticNet

from sklearn.ensemble import GradientBoostingRegressor

from sklearn import svm 

from sklearn.pipeline import make_pipeline

from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV, cross_val_score

import warnings

warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', None)

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

from scipy.stats import skew

from scipy.special import boxcox1p

from scipy.stats import boxcox_normmax



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
home = pd.read_csv('../input/home-data-for-ml-course/train.csv',index_col='Id')

test = pd.read_csv('../input/home-data-for-ml-course/test.csv',index_col='Id')
home.head()
test.head()
home.shape
test.shape
home.info()
test.info()
#missing values

missing = home.isnull().sum()

missing = missing[missing>0]

missing.sort_values(inplace=True)

missing.plot.bar()
numerical_features = home.select_dtypes(exclude=['object']).drop(['SalePrice'], axis=1).copy()

print(numerical_features.columns)
categorical_features = home.select_dtypes(include=['object']).copy()

print(categorical_features.columns)
fig = plt.figure(figsize=(12,18))

for i in range(len(numerical_features.columns)):

    fig.add_subplot(9,4,i+1)

    sns.distplot(numerical_features.iloc[:,i].dropna(), rug=True, hist=False, label='UW', kde_kws={'bw':0.1})

    plt.xlabel(numerical_features.columns[i])

plt.tight_layout()

plt.show()
fig = plt.figure(figsize=(12,18))

for i in range(len(numerical_features.columns)):

    fig.add_subplot(9,4,i+1)

    sns.boxplot(y=numerical_features.iloc[:,i])



plt.tight_layout()

plt.show()

fig = plt.figure(figsize=(12,18))

for i in range(len(numerical_features.columns)):

    fig.add_subplot(9, 4, i+1)

    sns.scatterplot(numerical_features.iloc[:, i],home['SalePrice'])

plt.tight_layout()

plt.show()
figure, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8), (ax9, ax10)) = plt.subplots(nrows=5, ncols=2)

figure.set_size_inches(16,28)

_ = sns.regplot(home['LotFrontage'], home['SalePrice'], ax=ax1)

_ = sns.regplot(home['LotArea'], home['SalePrice'], ax=ax2)

_ = sns.regplot(home['MasVnrArea'], home['SalePrice'], ax=ax3)

_ = sns.regplot(home['BsmtFinSF1'], home['SalePrice'], ax=ax4)

_ = sns.regplot(home['TotalBsmtSF'], home['SalePrice'], ax=ax5)

_ = sns.regplot(home['GrLivArea'], home['SalePrice'], ax=ax6)

_ = sns.regplot(home['1stFlrSF'], home['SalePrice'], ax=ax7)

_ = sns.regplot(home['EnclosedPorch'], home['SalePrice'], ax=ax8)

_ = sns.regplot(home['MiscVal'], home['SalePrice'], ax=ax9)

_ = sns.regplot(home['LowQualFinSF'], home['SalePrice'], ax=ax10)
home.shape
home = home.drop(home[home['LotFrontage']>200].index)

home = home.drop(home[home['LotArea']>100000].index)

home = home.drop(home[home['MasVnrArea']>1200].index)

home = home.drop(home[home['BsmtFinSF1']>4000].index)

home = home.drop(home[home['TotalBsmtSF']>4000].index)

home = home.drop(home[(home['GrLivArea']>4000) & (home['SalePrice']<300000)].index)

home = home.drop(home[home['1stFlrSF']>4000].index)

home = home.drop(home[home['EnclosedPorch']>500].index)

home = home.drop(home[home['MiscVal']>5000].index)

home = home.drop(home[(home['LowQualFinSF']>600) & (home['SalePrice']>400000)].index)
num_correlation = home.select_dtypes(exclude='object').corr()

plt.figure(figsize=(20,20))

plt.title('High Correlation')

sns.heatmap(num_correlation > 0.8, annot=True, square=True)
corr = num_correlation.corr()

print(corr['SalePrice'].sort_values(ascending=False))
home.drop(columns=['GarageArea','TotRmsAbvGrd','GarageYrBlt','1stFlrSF'],axis=1,inplace=True) 

test.drop(columns=['GarageArea','TotRmsAbvGrd','GarageYrBlt','1stFlrSF'],axis=1,inplace=True)
# Useless Columns...

home=home.drop(columns=['Street','Utilities','Condition2','RoofMatl','Heating']) 

test=test.drop(columns=['Street','Utilities','Condition2','RoofMatl','Heating']) 
home.isnull().mean().sort_values(ascending=False).head(3)
home.drop(columns=['Alley','MiscFeature','PoolQC','PoolArea', 'YrSold', 'MoSold'], axis=1, inplace=True)

test.drop(columns=['Alley','MiscFeature','PoolQC','PoolArea', 'YrSold', 'MoSold'], axis=1, inplace=True)
test.isnull().mean().sort_values(ascending=False).head(3)
# Checking Home and Test data missing value percentage

null = pd.DataFrame(data={'Home Null Percentage': home.isnull().sum()[home.isnull().sum() > 0], 'Test Null Percentage': test.isnull().sum()[test.isnull().sum() > 0]})

null = (null/len(home)) * 100



null.index.name='Feature'

null
home.isnull().sum().sort_values(ascending=False)[:50]
home_num_features = home.select_dtypes(exclude='object').isnull().mean()

test_num_features = test.select_dtypes(exclude='object').isnull().mean()



num_null_features = pd.DataFrame(data={'Missing Num Home Percentage: ': home_num_features[home_num_features>0], 'Missing Num Test Percentage: ': test_num_features[test_num_features>0]})

num_null_features.index.name = 'Numerical Features'

num_null_features
for df in [home, test]:

    for col in ('GarageCars', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 

                'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotalBsmtSF',

                'Fireplaces', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'MiscVal',

                'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea'):

                    df[col] = df[col].fillna(0)
_=sns.regplot(home['LotFrontage'],home['SalePrice'])

home_num_features = home.select_dtypes(exclude='object').isnull().mean()

test_num_features = test.select_dtypes(exclude='object').isnull().mean()



num_null_features = pd.DataFrame(data={'Missing Num Home Percentage: ': home_num_features[home_num_features>0], 'Missing Num Test Percentage: ': test_num_features[test_num_features>0]})

num_null_features.index.name = 'Numerical Features'

num_null_features
cat_col = home.select_dtypes(include='object').columns

print(cat_col)
home_cat_features = home.select_dtypes(include='object').isnull().mean()

test_cat_features = test.select_dtypes(include='object').isnull().mean()



cat_null_features = pd.DataFrame(data={'Missing Cat Home Percentage: ': home_cat_features[home_cat_features>0], 'Missing Cat Test Percentage: ': test_cat_features[test_cat_features>0]})

cat_null_features.index.name = 'Categorical Features'

cat_null_features
cat_col = home.select_dtypes(include='object').columns



columns = len(cat_col)/4+1



fg, ax = plt.subplots(figsize=(20, 30))



for i, col in enumerate(cat_col):

    fg.add_subplot(columns, 4, i+1)

    sns.countplot(home[col])

    plt.xlabel(col)

    plt.xticks(rotation=90)



plt.tight_layout()

plt.show()
var = home['KitchenQual']

f, ax = plt.subplots(figsize=(10,6))

sns.boxplot(y=home.SalePrice, x=var)

plt.show()
f, ax = plt.subplots(figsize=(12,8))

sns.boxplot(y=home.SalePrice, x=home.Neighborhood)

plt.xticks(rotation=45)

plt.show()
## Count of categories within Neighborhood attribute

fig = plt.figure(figsize=(12.5,4))

sns.countplot(x='Neighborhood', data=home)

plt.xticks(rotation=90)

plt.ylabel('Frequency')

plt.show()
for df in [home, test]:

    for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',

                  'BsmtFinType2', 'Neighborhood', 'BldgType', 'HouseStyle', 'MasVnrType', 'FireplaceQu', 'Fence'):

        df[col] = df[col].fillna('None')
for df in [home, test]:

    for col in ('LotShape', 'LandContour', 'LotConfig', 'LandSlope', 'Condition1', 'RoofStyle',

                  'Electrical', 'Functional', 'KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType', 'ExterQual', 'ExterCond',

                  'Foundation', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'PavedDrive', 'SaleType', 'SaleCondition'):

        df[col] = df[col].fillna(df[col].mode()[0])
home_cat_features = home.select_dtypes(include='object').isnull().mean()

test_cat_features = test.select_dtypes(include='object').isnull().mean()



cat_null_features = pd.DataFrame(data={'Missing Cat Home Percentage: ': home_cat_features[home_cat_features>0], 'Missing Cat Test Percentage: ': test_cat_features[test_cat_features>0]})

cat_null_features.index.name = 'Categorical Features'

cat_null_features
_=sns.regplot(home['LotFrontage'],home['SalePrice'])

home['LotFrontage'] = home.groupby('Neighborhood')['LotFrontage'].apply(lambda x: x.fillna(x.median()))

test['LotFrontage'] = test.groupby('Neighborhood')['LotFrontage'].apply(lambda x: x.fillna(x.median()))
home.corr()['SalePrice'].sort_values(ascending=False)
home.isnull().sum().sort_values(ascending=False)
test.isnull().sum().sort_values(ascending=False)
list(home.select_dtypes(exclude='object').columns)
figure, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)

figure.set_size_inches(20,10)

_ = sns.regplot(home['TotalBsmtSF'], home['SalePrice'], ax=ax1)

_ = sns.regplot(home['2ndFlrSF'], home['SalePrice'], ax=ax2)

_ = sns.regplot(home['TotalBsmtSF'] + home['2ndFlrSF'], home['SalePrice'], ax=ax3)
home['TotalSF']=home['TotalBsmtSF']  + home['2ndFlrSF']

test['TotalSF']=test['TotalBsmtSF']  + test['2ndFlrSF']
figure, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)

figure.set_size_inches(14,10)

_ = sns.barplot(home['BsmtFullBath'], home['SalePrice'], ax=ax1)

_ = sns.barplot(home['FullBath'], home['SalePrice'], ax=ax2)

_ = sns.barplot(home['BsmtHalfBath'], home['SalePrice'], ax=ax3)

_ = sns.barplot(home['BsmtFullBath'] + home['FullBath'] + home['BsmtHalfBath'] + home['HalfBath'], home['SalePrice'], ax=ax4)

home['TotalBath']=home['BsmtFullBath'] + home['FullBath'] + (0.5*home['BsmtHalfBath']) + (0.5*home['HalfBath'])

test['TotalBath']=test['BsmtFullBath'] + test['FullBath'] + test['BsmtHalfBath'] + test['HalfBath']

figure, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)

figure.set_size_inches(18,8)

_ = sns.regplot(home['YearBuilt'], home['SalePrice'], ax=ax1)

_ = sns.regplot(home['YearRemodAdd'], home['SalePrice'], ax=ax2)

_ = sns.regplot((home['YearBuilt']+home['YearRemodAdd'])/2, home['SalePrice'], ax=ax3)
home['YrBltAndRemod']=home['YearBuilt']+(home['YearRemodAdd']/2)

test['YrBltAndRemod']=test['YearBuilt']+(test['YearRemodAdd']/2)
figure, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3)

figure.set_size_inches(20,10)

_ = sns.regplot(home['OpenPorchSF'], home['SalePrice'], ax=ax1)

_ = sns.regplot(home['3SsnPorch'], home['SalePrice'], ax=ax2)

_ = sns.regplot(home['EnclosedPorch'], home['SalePrice'], ax=ax3)

_ = sns.regplot(home['ScreenPorch'], home['SalePrice'], ax=ax4)

_ = sns.regplot(home['WoodDeckSF'], home['SalePrice'], ax=ax5)

_ = sns.regplot((home['OpenPorchSF']+home['3SsnPorch']+home['EnclosedPorch']+home['ScreenPorch']+home['WoodDeckSF']), home['SalePrice'], ax=ax6)
home['Porch_SF'] = (home['OpenPorchSF'] + home['3SsnPorch'] + home['EnclosedPorch'] + home['ScreenPorch'] + home['WoodDeckSF'])

test['Porch_SF'] = (test['OpenPorchSF'] + test['3SsnPorch'] + test['EnclosedPorch'] + test['ScreenPorch'] + test['WoodDeckSF'])
home['Has2ndfloor'] = home['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)

home['HasBsmt'] = home['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)

home['HasFirePlace'] = home['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

home['Has2ndFlr']=home['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)

home['HasBsmt']=home['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)



test['Has2ndfloor'] = test['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)

test['HasBsmt'] = test['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)

test['HasFirePlace'] = test['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

test['Has2ndFlr']=test['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)

test['HasBsmt']=test['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
home['MSSubClass'] = home['MSSubClass'].apply(str)

home['LotArea'] = home['LotArea'].astype(np.int64)



test['MSSubClass'] = test['MSSubClass'].apply(str)

test['LotArea'] = test['LotArea'].astype(np.int64)
fig = plt.figure(figsize=(11,11))



print ("Skew of SalePrice:", home.SalePrice.skew())

plt.hist(home.SalePrice, normed=1, color='red')

plt.show()
fig = plt.figure(figsize=(11,11))



print ("Skew of Log-Transformed SalePrice:", np.log1p(home.SalePrice).skew())

plt.hist(np.log1p(home.SalePrice), color='green')

plt.show()
X = home.drop(['SalePrice'], axis=1)

y = np.log1p(home['SalePrice'])
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=2)
test.head()
categorical_cols = [cname for cname in X.columns if

                    X[cname].nunique() <= 30 and

                    X[cname].dtype == "object"] 

                





numerical_cols = [cname for cname in X.columns if

                 X[cname].dtype in ['int64','float64']]





my_cols = numerical_cols + categorical_cols



X_train = X_train[my_cols].copy()

X_valid = X_valid[my_cols].copy()

X_test = test[my_cols].copy()

num_transformer = Pipeline(steps=[

    ('num_imputer', SimpleImputer(strategy='constant'))

    ])



cat_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

    ])



preprocessor = ColumnTransformer(

    transformers=[

        ('num', num_transformer, numerical_cols),       

        ('cat',cat_transformer,categorical_cols),

        ])
# Reversing log-transform on y

def inv_y(transformed_y):

    return np.exp(transformed_y)



n_folds = 10



# XGBoost

model = XGBRegressor(learning_rate=0.01, n_estimators=3460, max_depth=3, min_child_weight=0,gamma=0, subsample=0.7,colsample_bytree=0.7,objective='reg:squarederror', nthread=-1,scale_pos_weight=1, seed=27, reg_alpha=0.00006)

clf = Pipeline(steps=[('preprocessor', preprocessor),

                          ('model', model)])

clf.fit(X_train, y_train)

predict = clf.predict(X_valid)

print('XGBoost: ' + str(mean_absolute_error(inv_y(predict), inv_y(y_valid))))



      

# Lasso   

model = LassoCV(max_iter=1e7,  random_state=14, cv=n_folds)

clf = Pipeline(steps=[('preprocessor', preprocessor),

                          ('model', model)])

clf.fit(X_train, y_train)

predict = clf.predict(X_valid)

print('Lasso: ' + str(mean_absolute_error(inv_y(predict), inv_y(y_valid))))

  

      

      

# GradientBoosting   

model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=5)

clf = Pipeline(steps=[('preprocessor', preprocessor),

                          ('model', model)])

clf.fit(X_train, y_train)

predict = clf.predict(X_valid)

print('Gradient: ' + str(mean_absolute_error(inv_y(predict), inv_y(y_valid))))
# model = XGBRegressor(learning_rate=0.01, n_estimators=3460,

#                      max_depth=3, min_child_weight=0,

#                      gamma=0, subsample=0.7,

#                      colsample_bytree=0.7,

#                      objective='reg:squarederror', nthread=-1,

#                      scale_pos_weight=1, seed=27,

#                      reg_alpha=0.00006)



# clf = Pipeline(steps=[('preprocessor', preprocessor),

#                           ('model', model)])





# scores = cross_val_score(clf, X, y, scoring='neg_mean_squared_error', 

#                          cv=n_folds)

# gbr_mae_scores = -scores



# print('Mean RMSE: ' + str(gbr_mae_scores.mean()))

# print('Error std deviation: ' +str(gbr_mae_scores.std()))
model = XGBRegressor(learning_rate=0.01, n_estimators=3460,

                     max_depth=3, min_child_weight=0,

                     gamma=0, subsample=0.7,

                     colsample_bytree=0.7,

                     objective='reg:squarederror', nthread=-1,

                     scale_pos_weight=1, seed=27,

                     reg_alpha=0.00006)



final_model = Pipeline(steps=[('preprocessor', preprocessor),

                          ('model', model)])



final_model.fit(X_train, y_train)



final_predictions = final_model.predict(X_test)
output = pd.DataFrame({'Id': X_test.index,

                       'SalePrice': inv_y(final_predictions)})



output.to_csv('submission.csv', index=False)