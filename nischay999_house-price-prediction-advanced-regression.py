import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')
pd.options.display.max_columns = None

pd.options.display.max_rows = None

sns.set(style="whitegrid", color_codes=True)
df = pd.read_csv('../input/train.csv')
copy = pd.read_csv('../input/train.csv')
df.head()
df.info()
df.describe()
# Checking for any duplicate values

df.duplicated().sum()
round(df.isna().mean().sort_values(ascending = False)*100,2).head(20)
plt.figure(figsize = (18,9))

sns.heatmap(df.isna())
# First looking at categorical columns

round(df.select_dtypes(exclude = ['int64', 'float64']).isna().mean().sort_values(ascending = False)*100,2).head(20)
# Dropping the columns which have significantly high NA values

df.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu'], axis = 1, inplace = True)
# Dropping the NA rows from the remaining columns

df = df[~df['BsmtFinType1'].isna()]
df = df[~df['GarageCond'].isna()]
df = df[~df['MasVnrType'].isna()]
df = df[~df['Electrical'].isna()]
df = df[~df['BsmtFinType2'].isna()]
df = df[~df['BsmtExposure'].isna()]
round(df.select_dtypes(exclude = ['int64', 'float64']).isna().mean().sort_values(ascending = False)*100,2).head()
# Checking % of retained rows

round(len(df)/1460,2)
# Solving for NA values in Numeric columns
round(df.select_dtypes(include = ['int64', 'float64']).isna().mean().sort_values(ascending = False)*100,2).head(20)
# Filling the NA values in LotFrontage with mean

df['LotFrontage'].fillna(np.mean(df['LotFrontage']), inplace = True)
plt.figure(figsize = (18,9))

sns.heatmap(df.isna())
for col in df.select_dtypes(exclude = ['int64', 'float64']).columns:

    print(df[col].value_counts())

    print('___________________________________')
df.select_dtypes(exclude = ['int64', 'float64']).shape
# Selecting the columns with very low variance

col_to_drop = ['Street', 'Utilities', 'Condition2', 'RoofMatl', 'BsmtCond', 'LandSlope', 'Heating', 'GarageQual', 'GarageCond']
df.drop(col_to_drop, axis = 1, inplace = True)
# Dropping the id column

df.drop('Id', axis = 1, inplace = True)
df.info()
plt.figure(figsize = (15,80))

for i,col in enumerate(df.select_dtypes(exclude = ['int64', 'float64'])):

    plt.subplot(20,2,i+1)

    sns.boxplot(x = col, y = 'SalePrice', data = df)

plt.tight_layout()
plt.figure(figsize = (15,80))

for i,col in enumerate(df.select_dtypes(include = ['int64', 'float64'])):

    plt.subplot(20,2,i+1)

    sns.scatterplot(x = col, y = 'SalePrice', data = df)

plt.tight_layout()
# Finding correlated features

plt.figure(figsize = (20, 15))

sns.heatmap(df.select_dtypes(include = ['int64', 'float64']).corr(), cmap="YlGnBu")
plt.figure(figsize=(15,8))

df.select_dtypes(include = ['int64', 'float64']).corrwith(df['SalePrice']).sort_values(ascending = False).plot(kind = 'bar')
# Creating dummy variables for categorical columns

cat_col = df.select_dtypes(exclude = ['int64', 'float64']).columns

int_col = df.select_dtypes(include = ['int64', 'float64']).drop('SalePrice', axis = 1).columns
df_f = pd.get_dummies(df, columns = cat_col, drop_first= True)
df_f.shape
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_f.drop('SalePrice', axis=1), 

                                                    df_f['SalePrice'], test_size=0.3)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train[int_col] = scaler.fit_transform(X_train[int_col]) 
round(X_train.describe(),2)
X_test[int_col] = scaler.transform(X_test[int_col])
X_test.describe()
from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import r2_score

from sklearn.feature_selection import RFE
# List of alphas to tune

params = {'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 3.0, 

 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 20, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]}
ridge = Ridge()
folds = 5

model_cv_r = GridSearchCV(ridge, param_grid = params, 

                       scoring = 'r2',

                       cv = folds, return_train_score=True,

                        verbose = 1)

model_cv_r.fit(X_train, y_train)

cv_results_r = pd.DataFrame(model_cv_r.cv_results_)
# plotting mean test and train scoes with alpha

plt.plot(cv_results_r['param_alpha'], cv_results_r['mean_train_score'])

plt.plot(cv_results_r['param_alpha'], cv_results_r['mean_test_score'])

plt.xlabel('alpha')

plt.ylabel('R2')

plt.title("R2 and alpha")

plt.legend(['train score', 'test score'], loc='upper left')

plt.show()
model_cv_r.best_params_
model_cv_r.best_score_
lasso = Lasso()
# cross validation

model_cv_l = GridSearchCV(estimator = lasso, 

                        param_grid = params, 

                        scoring= 'r2', 

                        cv = folds, 

                        return_train_score=True,

                        verbose = 1)            



model_cv_l.fit(X_train, y_train)

cv_results_l = pd.DataFrame(model_cv_l.cv_results_)
# plotting mean test and train scoes with alpha 

plt.plot(cv_results_l['param_alpha'], cv_results_l['mean_train_score'])

plt.plot(cv_results_l['param_alpha'], cv_results_l['mean_test_score'])

plt.xlabel('alpha')

plt.ylabel('R2')

plt.title("R2 and alpha")

plt.legend(['train score', 'test score'], loc='upper left')

plt.show()
model_cv_l.best_params_
model_cv_l.best_score_
# Taking alpha = 300

alpha = 300
lasso = Lasso(alpha=alpha)

lasso.fit(X_train, y_train)
# Plotting the feature coefficients
feature_df_l = pd.DataFrame(list(zip(X_train.columns, lasso.coef_)), columns = ['Feature', 'coeff'])
plt.figure(figsize = (15,5))

sns.barplot(x = 'Feature', y = 'coeff', data = feature_df_l[feature_df_l['coeff']>=0.01].sort_values(by = 'coeff', 

                                                                                                  ascending = False))

plt.xticks(rotation = 90)

plt.tight_layout()
plt.figure(figsize = (15,5))

sns.barplot(x = 'Feature', y = 'coeff', data = feature_df_l[feature_df_l['coeff']<0].sort_values(by = 'coeff', 

                                                                                                 ascending = True))

plt.xticks(rotation = 90)

plt.tight_layout()
feature_df_l[feature_df_l['coeff']!=0].shape
feature_df_l[feature_df_l['coeff']!=0].sort_values(by = 'coeff', ascending = False)
feature_df_l.shape
alpha = 300

lasso = Lasso(alpha=alpha)

lasso.fit(X_train, y_train)
# predict

y_train_pred = lasso.predict(X_train)

print(r2_score(y_true=y_train, y_pred=y_train_pred))

y_test_pred = lasso.predict(X_test)

print(r2_score(y_true=y_test, y_pred=y_test_pred))