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

from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from xgboost import XGBRegressor

from sklearn.linear_model import LassoCV

from sklearn import metrics 

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.pipeline import make_pipeline

from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV, cross_val_score

from scipy.stats import skew  # for some statistics

from scipy.special import boxcox1p

from scipy.stats import boxcox_normmax



from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.svm import SVR

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.model_selection import KFold, cross_val_score

from sklearn.metrics import mean_squared_error, mean_absolute_error



from mlxtend.regressor import StackingCVRegressor





import warnings

warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', None)

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

from scipy.stats import skew



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv('../input/train.csv',index_col='PassengerId')

test = pd.read_csv('../input/test.csv',index_col='PassengerId')
print('Training data:\n ')

train.head()
print('Test data:\n ')

test.head()
train.shape
test.shape
train.info()
test.info()
#missing values

missing = train.isnull().sum()

missing = missing[missing>0]

missing.sort_values(inplace=True)

missing.plot.bar()
#missing values

missing = test.isnull().sum()

missing = missing[missing>0]

missing.sort_values(inplace=True)

missing.plot.bar()
numerical_features = train.select_dtypes(exclude=['object']).drop(['Survived'], axis=1).copy()

print(numerical_features.columns)
categorical_features = train.select_dtypes(include=['object']).copy()

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

    sns.scatterplot(numerical_features.iloc[:, i],train['Survived'])

plt.tight_layout()

plt.show()
figure, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(nrows=3, ncols=2)

figure.set_size_inches(16,28)

_ = sns.regplot(train['Pclass'], train['Survived'], ax=ax1)

_ = sns.regplot(train['Age'],  train['Survived'], ax=ax2)

_ = sns.regplot(train['SibSp'],train['Survived'], ax=ax3)

_ = sns.regplot(train['Parch'], train['Survived'], ax=ax4)

_ = sns.regplot(train['Fare'], train['Survived'], ax=ax5)

_ = sns.regplot(train['Fare'], train['Survived'], ax=ax6)
train.shape
# Notseeing outliers worth removing at this point????? Can return to this step later.

# home = home.drop(home[home['LotFrontage']>200].index)

# home = home.drop(home[home['LotArea']>100000].index)

# home = home.drop(home[home['MasVnrArea']>1200].index)

# home = home.drop(home[home['BsmtFinSF1']>4000].index)

# home = home.drop(home[home['TotalBsmtSF']>4000].index)

# home = home.drop(home[(home['GrLivArea']>4000) & (home['SalePrice']<300000)].index)

# home = home.drop(home[home['1stFlrSF']>4000].index)

# home = home.drop(home[home['EnclosedPorch']>500].index)

# home = home.drop(home[home['MiscVal']>5000].index)

# home = home.drop(home[(home['LowQualFinSF']>600) & (home['SalePrice']>400000)].index)
num_correlation = train.select_dtypes(exclude='object').corr()

plt.figure(figsize=(10,10))

plt.title('High Correlation')

sns.heatmap(num_correlation > 0.8, annot=True, square=True)
corr = num_correlation.corr()

print(corr['Survived'].sort_values(ascending=False))
# home.drop(columns=['GarageArea','TotRmsAbvGrd','GarageYrBlt','1stFlrSF'],axis=1,inplace=True) 

# test.drop(columns=['GarageArea','TotRmsAbvGrd','GarageYrBlt','1stFlrSF'],axis=1,inplace=True)



# # Useless Columns...

# home=home.drop(columns=['Street','Utilities']) 

# test=test.drop(columns=['Street','Utilities']) 
train.isnull().mean().sort_values(ascending=False).head(3)
train.drop(columns=['Cabin'], axis=1, inplace=True)

test.drop(columns=['Cabin'], axis=1, inplace=True)
test.isnull().mean().sort_values(ascending=False).head(3)
# Checking Training and Test data missing value percentage

null = pd.DataFrame(data={'Train Null Percentage': train.isnull().sum()[train.isnull().sum() > 0], 'Test Null Percentage': test.isnull().sum()[test.isnull().sum() > 0]})

null = (null/len(train)) * 100



null.index.name='Feature'

null
train.isnull().sum().sort_values(ascending=False)[:50]
test.isnull().sum().sort_values(ascending=False)[:50]
train_num_features = train.select_dtypes(exclude='object').isnull().mean()

test_num_features = test.select_dtypes(exclude='object').isnull().mean()



num_null_features = pd.DataFrame(data={'Missing Num Train Percentage: ': train_num_features[train_num_features>0], 'Missing Num Test Percentage: ': test_num_features[test_num_features>0]})

num_null_features.index.name = 'Numerical Features'

num_null_features
for df in [train, test]:

    for col in ('Pclass', 'Age', 'SibSp', 'Parch', 'Fare'):

                    df[col] = df[col].fillna(0)
train.isnull().sum().sort_values(ascending=False)[:50]
test.isnull().sum().sort_values(ascending=False)[:50]
train_num_features = train.select_dtypes(exclude='object').isnull().mean()

test_num_features = test.select_dtypes(exclude='object').isnull().mean()



num_null_features = pd.DataFrame(data={'Missing Num Train Percentage: ': train_num_features[train_num_features>0], 'Missing Num Test Percentage: ': test_num_features[test_num_features>0]})

num_null_features.index.name = 'Numerical Features'

num_null_features
cat_col = train.select_dtypes(include='object').columns

print(cat_col)
train_cat_features = train.select_dtypes(include='object').isnull().mean()

test_cat_features = test.select_dtypes(include='object').isnull().mean()



cat_null_features = pd.DataFrame(data={'Missing Cat Train Percentage: ': train_cat_features[train_cat_features>0], 'Missing Cat Test Percentage: ': test_cat_features[test_cat_features>0]})

cat_null_features.index.name = 'Categorical Features'

cat_null_features
cat_col = train.select_dtypes(include='object').columns



columns = len(cat_col)/4+1



fg, ax = plt.subplots(figsize=(20, 30))



for i, col in enumerate(cat_col):

    fg.add_subplot(columns, 4, i+1)

    sns.countplot(train[col])

    plt.xlabel(col)

    plt.xticks(rotation=90)



plt.tight_layout()

plt.show()
var = train['Sex']

f, ax = plt.subplots(figsize=(10,6))

sns.boxplot(y=train.Survived, x=var)

plt.show()
var = train['Embarked']

f, ax = plt.subplots(figsize=(10,6))

sns.boxplot(y=train.Survived, x=var)

plt.show()
f, ax = plt.subplots(figsize=(12,8))

sns.boxplot(y=train.Survived, x=train.Sex)

plt.xticks(rotation=45)

plt.show()
f, ax = plt.subplots(figsize=(12,8))

sns.boxplot(y=train.Survived, x=train.Embarked)

plt.xticks(rotation=45)

plt.show()
## Count of categories within Sex attribute

fig = plt.figure(figsize=(12.5,4))

sns.countplot(x='Sex', data=train)

plt.xticks(rotation=90)

plt.ylabel('Frequency')

plt.show()
## Count of categories within Embarked attribute

fig = plt.figure(figsize=(12.5,4))

sns.countplot(x='Embarked', data=train)

plt.xticks(rotation=90)

plt.ylabel('Frequency')

plt.show()
for df in [train, test]:

    for col in ('Name', ):

        df[col] = df[col].fillna('None')
# changing missing values to mode value -- note 'cabin' was dropped earlier as a feature

for df in [train, test]:

    for col in ('Sex', 'Ticket', 'Embarked'):

        df[col] = df[col].fillna(df[col].mode()[0])
# below code verifies no remaining missing values in categorical data (via an empty table)

train_cat_features = train.select_dtypes(include='object').isnull().mean()

test_cat_features = test.select_dtypes(include='object').isnull().mean()



cat_null_features = pd.DataFrame(data={'Missing Cat Train Percentage: ': train_cat_features[train_cat_features>0], 'Missing Cat Test Percentage: ': test_cat_features[test_cat_features>0]})

cat_null_features.index.name = 'Categorical Features'

cat_null_features
train.isnull().sum().sort_values(ascending=False)
test.isnull().sum().sort_values(ascending=False)
list(train.select_dtypes(exclude='object').columns) # note: still has target within the dataframe; also note we revoved cabin as an example
list(test.select_dtypes(exclude='object').columns)
X = train.drop(['Survived'], axis=1)

y = train['Survived']

# y = np.log1p(home['Survived']) Note: See example where they discuss using log of saleprice
X.head()
y.head()
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=1)
X_train.head()
X_valid.head()
test.head()
# We select every numerical column from X and the categorical columns with unique values under 20



categorical_cols = [cname for cname in X.columns if

                    X[cname].nunique() <= 20 and

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

print('XGBoost: ' + str(mean_absolute_error(predict, y_valid)))



      

# Lasso   

model = LassoCV(max_iter=1e7,  random_state=14, cv=n_folds)

clf = Pipeline(steps=[('preprocessor', preprocessor),

                          ('model', model)])

clf.fit(X_train, y_train)

predict = clf.predict(X_valid)

print('Lasso: ' + str(mean_absolute_error(predict, y_valid)))

  

      

      

# GradientBoosting   

model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=5)

clf = Pipeline(steps=[('preprocessor', preprocessor),

                          ('model', model)])

clf.fit(X_train, y_train)

predict = clf.predict(X_valid)

print('Gradient: ' + str(mean_absolute_error(predict, y_valid)))
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

                       'Survived': final_predictions})



output.to_csv('submission.csv', index=False)
output