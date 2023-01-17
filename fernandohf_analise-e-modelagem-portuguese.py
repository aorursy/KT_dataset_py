# Import EDA libraries

import pandas as pd

import seaborn as sns

import numpy as np

import matplotlib.pyplot as plt

import missingno as msno



# Ignore warnings

import warnings

warnings.filterwarnings("ignore")
# Load data

train = pd.read_csv("../input/train.csv", infer_datetime_format=True)
# Brief visualization

train.head()
# Brief infomatin

train.info()
# Brief description

train.describe()
# Features

train.columns
# Size

train.shape
# Look for duplicates

train.duplicated().sum()
# Look for missing data

train.isna().sum().sum()
msno.matrix(train)
miss_count = train.isna().sum()

miss_count = miss_count[miss_count > 0]

miss_count
# Shows which missings values occurs simultaneously on both columns

ax = msno.heatmap(train, cmap='RdBu')
# Show unique values in missing columns

def get_unique_values_col(df):

    data = {}

    for col, s in df.iteritems():

        data[col] = s.unique()

    return pd.Series(data)

unique_mc = get_unique_values_col(train[miss_count.index])

unique_mc
# Cleaned data

c_train = train.copy()



# Imput categorical columns

miss_cat_cols = ['Alley', 'MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure',

                 'BsmtFinType1', 'BsmtFinType2', 'Electrical', 'FireplaceQu',

                 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',

                 'PoolQC', 'Fence', 'MiscFeature']

c_train[miss_cat_cols] = train[miss_cat_cols].fillna('None')



# Tranform numerical variables in categorical variables

cat_feat = ['MSSubClass', 'OverallQual', 'OverallCond'] + list(train.select_dtypes('object').columns)

c_train[cat_feat] = c_train[cat_feat].astype('category')



print(f"Empty entries: {c_train[miss_cat_cols].isna().sum().sum()}")

c_train[miss_cat_cols].head()
from sklearn.preprocessing import Imputer



# Imput numeric columns

c_train['LotFrontage'] = train.LotFrontage.fillna(train.LotFrontage.median())

c_train['MasVnrArea'] = train.MasVnrArea.fillna(0)

c_train['GarageYrBlt'] = train.GarageYrBlt.fillna(0)



print(f"Empty entries: {c_train.isna().sum().sum()}")

c_train.head()
c_train.describe()
# Look at Target variable

fig, ax = plt.subplots(figsize=(12, 8))

sns.distplot(c_train.SalePrice, ax=ax)
c_train.drop('SalePrice', axis=1).hist(bins=40, figsize=(20, 16))

plt.tight_layout()
# Some of the main categorical features

fig, ax = plt.subplots(figsize=(12,10))

sns.violinplot(x='OverallQual', y='SalePrice', data=c_train, ax=ax)
# Some of the main categorical features

fig, ax = plt.subplots(figsize=(12,10))

sns.violinplot(x='OverallCond', y='SalePrice', data=c_train, ax=ax)
# Some of the main categorical features

fig, ax = plt.subplots(figsize=(12,10))

sns.violinplot(x='MSSubClass', y='SalePrice', data=c_train, ax=ax)
num_corr_matrix = c_train.corr()

num_corr_target = num_corr_matrix['SalePrice'].sort_values(ascending=False)
num_corr_target.plot.barh(color='steelblue', figsize=(20, 12));

plt.title('Correlation with Sale Price')
fig, ax = plt.subplots(figsize=(22, 12))

sns.heatmap(num_corr_matrix, ax=ax, annot=False, cmap='coolwarm', vmin=-1., vmax=1.)
# Correlation ratio definition

def correlation_ratio(categories, measurements):

    """

    Calculate the correlation ratio between the categoryes and the measurement

    

    Args: 

        categories: Iterable with categories

        measurements: Iterable with the measurements

    

    Return:

        eta: Correlation ratio

    """

    fcat, _ = pd.factorize(categories)

    cat_num = np.max(fcat)+1

    y_avg_array = np.zeros(cat_num)

    n_array = np.zeros(cat_num)

    for i in range(0,cat_num):

        cat_measures = measurements[np.argwhere(fcat == i).flatten()]

        n_array[i] = len(cat_measures)

        y_avg_array[i] = np.average(cat_measures)

    y_total_avg = np.sum(np.multiply(y_avg_array,n_array))/np.sum(n_array)

    numerator = np.sum(np.multiply(n_array,np.power(np.subtract(y_avg_array,y_total_avg),2)))

    denominator = np.sum(np.power(np.subtract(measurements,y_total_avg),2))

    if numerator == 0:

        eta = 0.0

    else:

        eta = numerator/denominator

    return eta
# Categorical feature correlated with numeric target

cat_cols = c_train.select_dtypes('category').columns

cat_corr_target = pd.Series([correlation_ratio(c_train[c], c_train['SalePrice']) for c in cat_cols], index=cat_cols)



cat_corr_target.sort_values(ascending=False).plot.barh(color='steelblue', figsize=(20, 12));

plt.title('Correlation with Sale Price')
# Removendo Variáveis de baixa correlação com o alvo

thres_t = 0.1

num_corr_matrix = c_train.corr()

cols_target = num_corr_matrix['SalePrice'][(num_corr_matrix['SalePrice'] > thres_t) |

                                           (num_corr_matrix['SalePrice'] < -thres_t)].sort_values(ascending=True).index

cols_target
new_num_corr_matrix = c_train[cols_target].corr()
# Removing numeric variable highly correlated

thres_i = 0.8

cols_corr = {}

for c in cols_target:

    series = new_num_corr_matrix[c].drop(c)

    if np.any([np.any(series > thres_i), np.any(series < -thres_i)]):

        cols_corr[c] = series.idxmax()

# Select just one of the pair with highest correlation

remove_cols = set([k if new_num_corr_matrix['SalePrice'][k] < new_num_corr_matrix['SalePrice'][v] else v for k, v in cols_corr.items() ])

n_cols = [c for c in cols_target if c not in remove_cols]

n_cols
# Categorical columns

thres_c = 0.1

c_cols = [c for c, v in cat_corr_target.iteritems() if v > thres_c]

c_cols
# Filters data

f_cols = c_cols + n_cols

f_train = c_train[f_cols]

f_train.head()
# Check new correlations

f_num_corr_matrix = f_train[n_cols].corr()

fig, ax = plt.subplots(figsize=(22, 12))

sns.heatmap(f_num_corr_matrix, ax=ax, annot=True, cmap='coolwarm', vmin=-1., vmax=1.)
# Filtered results

f_cat_corr_target = pd.Series([correlation_ratio(f_train[c], f_train['SalePrice']) for c in c_cols], index=c_cols)



f_cat_corr_target.sort_values(ascending=False).plot.barh(color='steelblue', figsize=(20, 12));

plt.title('Correlation with Sale Price')
# Dynamic plot

from bokeh.plotting import output_file, figure, show

from bokeh.models.tools import HoverTool

from bokeh.io import output_notebook



# Notebook mode

output_notebook()



# New sample for manipulated df

x = f_train['GrLivArea']

y = f_train['SalePrice']

TOOLS = "hover,pan,wheel_zoom,box_zoom,reset,save"

p = figure(title="Ground Living Area X Sale Price", tools=TOOLS,

           y_range=(y.min(), y.max()), x_range=(x.min(), x.max()))



p.circle('GrLivArea', 'SalePrice', source=f_train)

# Linear Regression

slope, intercept = np.polyfit(x, y, 1)

reg_x = np.linspace(x.min(), x.max())

p.line(reg_x, reg_x*slope+intercept, color='red')

p.xaxis[0].axis_label = 'Square Feet (ft²)'

p.yaxis[0].axis_label = 'Price ($)'



p.hover.tooltips = [

    # add to this

    ("(Ground Living Area, Sale Price)", "($x, $y)"),

]

show(p)



# Check correlation between variables

# sns.scatterplot(x='distance', y='fare_amount', data=f_train[columns].sample(100000))
# Generate dynamic boxplot

y = f_train["SalePrice"].copy()

x = f_train["OverallQual"].copy()

# Boxplot of categorical feature

fig, ax = plt.subplots(figsize=(12, 8))

sns.boxenplot(x, y, color='steelblue', ax=ax)

ax.set_title("Overall Quality x Price")
# Recap of data types

f_train.info()
# Recap of data format

print(f_train.shape)

f_train.head()
from sklearn.preprocessing import RobustScaler

from sklearn.model_selection import train_test_split
# Managing categorical variables

cat_cols = f_train.select_dtypes('category').columns

df_dummies = pd.get_dummies(f_train[cat_cols], prefix=cat_cols, drop_first=True)

X_cat = df_dummies

X_cat.head()
# Scaling numeric variables

scaler = RobustScaler()

scale_features = list(f_train.select_dtypes(exclude='category').columns)

scale_features.remove('SalePrice')

X_num = f_train[scale_features].copy()

X_num[scale_features] = scaler.fit_transform(f_train[scale_features].values)

X_num.head()
# Features and target

X = pd.concat([X_num, X_cat], axis=1)

y = f_train['SalePrice']

X.head()
print(X.shape)

print(y.shape)
# Split data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25)
print(X_train.shape)

print(y_train.shape)
from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import cross_val_score

from sklearn.metrics import mean_squared_error

import xgboost as xgb
# SKlearn Models

lin_reg = LinearRegression()

tree_reg = DecisionTreeRegressor()

forest_reg = RandomForestRegressor()



# Scores with x-validation

lin_scores = cross_val_score(lin_reg, 

                             X_train, 

                             y_train,

                             scoring = "neg_mean_squared_error", 

                             cv = 10)



decision_scores = cross_val_score(tree_reg,

                                  X_train, 

                                  y_train,

                                  scoring = "neg_mean_squared_error", 

                                  cv = 10)



forest_scores = cross_val_score(forest_reg,

                                X_train, 

                                y_train,

                                scoring = "neg_mean_squared_error", 

                                cv = 10)
# Boost Models

xgb_reg = xgb.XGBRegressor(n_jobs=12).fit(X_train, y_train)



# Scores with x-validation

xgb_scores = cross_val_score(xgb_reg, 

                             X_train, 

                             y_train,

                             scoring = "neg_mean_squared_error", 

                             cv = 10)
# Test Results

lin_rmse_scores = np.sqrt(-lin_scores)

decision_rmse_scores = np.sqrt(-decision_scores)

forest_rmse_scores = np.sqrt(-forest_scores)

xgb_rmse_scores = np.sqrt(-xgb_scores)

# decision_rmse_test = mean_squared_error(y_test, tree_reg.predict(X_test))

# forest_rmse_test = mean_squared_error(y_test, forest_reg.predict(X_test))

# xgb_rmse_test = mean_squared_error(y_test, xgb_reg.predict(X_test))



# Results

print("Linear Regression Results:")

# print(f"Test MSE: {lin_rmse_test:.2f}")

print(f"CV RMSE: {lin_rmse_scores.mean():.2f} +/- {lin_rmse_scores.std() * 2:.2f}\n")

print("Decision Tree Regressor Results:")

# print(f"Test MSE: {decision_rmse_test:.2f}")

print(f"CV RMSE: {decision_rmse_scores.mean():.2f} +/- {decision_rmse_scores.std() * 2:.2f}\n")

print("Random Forest Regressor Results:")

# print(f"Test MSE: {forest_rmse_test:.2f}")

print(f"CV RMSE: {forest_rmse_scores.mean():.2f} +/- {forest_rmse_scores.std() * 2:.2f}\n")

print("X-Gradient Boosting Results:")

# print(f"Test MSE: {xgb_rmse_test:.2f}")

print(f"CV RMSE: {xgb_rmse_scores.mean():.2f} +/- {xgb_rmse_scores.std() * 2:.2f}\n")
from sklearn.model_selection import GridSearchCV
y_train.shape
# Grid of parameters

parameters = {'max_depth': [3, 6],

              'learning_rate': [0.15, 0.75],

              'n_estimators': [250, 300],

              }

fit_parameters = {'early_stopping_rounds':range(4, 12, 2),

                  'eval_set': (X_test, y_test)

                  }

grid_reg = GridSearchCV(xgb_reg,

                        parameters,

                        scoring='neg_mean_squared_error',

                        cv= 10,

                        n_jobs=12)

grid_reg.fit(X_train, y_train)
xgb_rmse_test = mean_squared_error(y_test, grid_reg.best_estimator_.predict(X_test))



# Results

print(f"Best Parameters: {grid_reg.best_params_}")

print(f"Best Mean CV Score: {np.sqrt(-grid_reg.best_score_):.2f}")

print(f"Test MSE: {np.sqrt(xgb_rmse_test):.2f}")

modelos = ["CV Decision Tree Regressor", "CV Random Forest Regressor", "CV Linear Regression",

           "CV XGBoost Regressor", "CV Tunned XGBoost Regressor"]

mses = [decision_rmse_scores.mean(), forest_rmse_scores.mean(), lin_rmse_scores.mean(),

        xgb_rmse_scores.mean(), np.sqrt(-grid_reg.best_score_)]



fig, ax = plt.subplots(figsize=(8, 6))

ax.set_xlabel("MSE")

ax.set_ylabel("Model")

ax.set_title("MSE Error - Lower Better")

sns.barplot(y=modelos, x=mses, color="steelblue", ax=ax)

afig, ax = plt.subplots(figsize=(16, 16))

xgb.plot_importance(xgb_reg, height=.5, ax=ax)
# Save model

import pickle

pickle.dump(grid_reg.best_estimator_, open("model_house.pt", "wb"))
# Load model

import pickle

best_model = pickle.load(open("model_house.pt", "rb"))