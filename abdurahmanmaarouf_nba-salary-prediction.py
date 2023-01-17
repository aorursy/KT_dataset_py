# Data Manipulation

import pandas as pd

pd.options.mode.chained_assignment = None

import numpy as np

import missingno as msno



# Visualization

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



# Machine Learning

from sklearn.model_selection import train_test_split, KFold, cross_val_score

from sklearn import model_selection, metrics

from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler, MaxAbsScaler

from sklearn.decomposition import PCA

from sklearn.linear_model import LinearRegression, Ridge, Lasso

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor

from xgboost import XGBRegressor

from lightgbm import LGBMRegressor



# Other packages

from scipy.stats import norm

from scipy import stats



# Import data

df_salary = pd.read_csv('../input/nba-player-salaries-as-at-2020/NBA Players Salaries 1920.csv')

df_stats1718 = pd.read_csv('../input/nba-players-stats-2016-2017/NBA Players Stats 201718.csv')

df_stats1819 = pd.read_csv('../input/nba-players-stats-2016-2017/NBA Players Stats 201819.csv')

df_list = [df_stats1718, df_stats1819, df_salary]
# Salary

df_salary.head()
# Season Stats 17/18

df_stats1718.head()
# Season Stats 18/19

df_stats1819.head()
# Correct player names

for df in df_list:

    df[['Player', 'Del']] = df.Player.str.split("\\", expand = True)



df_stats1718 = df_stats1718.drop(['Del'], axis = 1)

df_stats1819 = df_stats1819.drop(['Del'], axis = 1)

df_salary = df_salary.drop(['Del'], axis = 1)
# Delete $ signs and turn column into float

df_salary['2019-20'] = df_salary['2019-20'].str[:-2].astype(float)



# Rename salary column

df_salary = df_salary.rename(columns = {'2019-20': 'Salary 19/20'})



# Transform salary to 1000

df_salary['Salary 19/20'] = df_salary['Salary 19/20']/1000
# As total stats always is in the top row we can simply use the drop_duplicates function

df_stats1718 = df_stats1718.drop_duplicates(['Player'])

df_stats1819 = df_stats1819.drop_duplicates(['Player'])
# Add season year to corresponding columns

columns_renamed = [s + ' 17/18' for s in list(df_stats1718.columns)]

df_stats1718.columns = list(df_stats1718.columns)[:3] + columns_renamed[3:]



columns_renamed = [s + ' 18/19' for s in list(df_stats1819.columns)]

df_stats1819.columns = list(df_stats1819.columns)[:3] + columns_renamed[3:]



# Delete Pos column from 17/18 df; we need it only once

df_stats1718 = df_stats1718.drop('Pos', axis = 1)
# Merge datasets

df_stats = df_stats1718.merge(df_stats1819, how = 'outer',left_on = ['Player'],right_on = ['Player'])

df = df_stats.merge(df_salary, how = 'outer', left_on = ['Player'],right_on = ['Player'])



df.head()
len(df)
df.dtypes
# Columns of dataset

df.columns
# Drop unnecessary columns

df = df.drop(['Rk_x', 'Rk_y', 'Rk', 'Tm', '2020-21', '2021-22', '2022-23', '2023-24', '2024-25',

              'Signed Using', 'Guaranteed'], axis = 1)
# Number of missing values for each column

df.isnull().sum()
# Drop rows with NaN

# Create Dataframe without stats for season 17/18

df1 = df.dropna(subset = ['Salary 19/20', 'PTS 18/19', 'eFG% 18/19'])

df1 = df1.reset_index()

columns = list(df1.columns)

for i in columns:

    if '17/18' in i:

        df1 = df1.drop([i], axis = 1)

df1 = df1.reset_index()



# Create Dataframe with stats for season 17/18

df2 = df.dropna(subset = ['Salary 19/20', 'PTS 18/19', 'eFG% 18/19', 'PTS 17/18', 'eFG% 17/18'])

df2 = df2.reset_index()
# Get unique positions

print(df1.Pos.unique())

print(df2.Pos.unique())
# Replace duplicate positions with first position.

df1 = df1.replace({'SF-SG': 'SF', 'PF-SF': 'PF', 'SG-PF': 'SG', 'C-PF': 'C', 'SG-SF': 'SG'})

df2 = df2.replace({'SF-SG': 'SF', 'PF-SF': 'PF', 'SG-PF': 'SG', 'C-PF': 'C', 'SG-SF': 'SG'})
# Get absolute growth

# List of stats of which we want the growth

list_growth = ['eFG%','TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']



# Add absolute growth columns

for i in list_growth:

    df2[i + ' +-'] = df2[i + ' 18/19'] - df2[i + ' 17/18']

    

# Drop 17/18 columns

columns = list(df2.columns)

for i in columns:

    if '17/18' in i:

        df2 = df2.drop([i], axis = 1)
print(df1.shape)

print(df2.shape)
# Setup dataframe

df_sal = df1[['Player', 'Salary 19/20']]

df_sal.sort_values(by = 'Salary 19/20', ascending = False, inplace = True)



# Create barchart

sns.catplot(x = 'Player', y = 'Salary 19/20', kind = 'bar', data = df_sal.head()).set(xlabel = None)

plt.title('Players with highest salary (in 1000)')

plt.ylim([35000, 40000])

plt.xticks(rotation = 90)
# Statistics summary

df1['Salary 19/20'].describe()
# Histogram

sns.distplot(df1['Salary 19/20'])
# Setup dataframes

df_pts = df1[['Player', 'PTS 18/19']]

df_pts.sort_values(by = 'PTS 18/19', ascending = False, inplace = True)

df_ast = df1[['Player', 'AST 18/19']]

df_ast.sort_values(by = 'AST 18/19', ascending = False, inplace = True)

df_stl = df1[['Player', 'STL 18/19']]

df_stl.sort_values(by = 'STL 18/19', ascending = False, inplace = True)

df_trb = df1[['Player', 'TRB 18/19']]

df_trb.sort_values(by = 'TRB 18/19', ascending = False, inplace = True)



# Set up figure

f, axes = plt.subplots(2, 2, figsize=(20, 15))

sns.despine(left=True)



# Create barcharts

sns.barplot(x = 'PTS 18/19', y = 'Player', data = df_pts.head(), color = "b", ax = axes[0, 0]).set(ylabel = None)

sns.barplot(x = 'AST 18/19', y = 'Player', data = df_ast.head(), color = "r", ax = axes[0, 1]).set(ylabel = None)

sns.barplot(x = 'STL 18/19', y = 'Player', data = df_stl.head(), color = "g", ax = axes[1, 0]).set(ylabel = None)

sns.barplot(x = 'TRB 18/19', y = 'Player', data = df_trb.head(), color = "m", ax = axes[1, 1]).set(ylabel = None)
# Set up figure

f, axes = plt.subplots(2, 2, figsize=(20, 15))

sns.despine(left=True)



# Histograms

sns.distplot(df1['PTS 18/19'], color = "b", ax = axes[0, 0])

sns.distplot(df1['AST 18/19'], color = "r", ax = axes[0, 1])

sns.distplot(df1['STL 18/19'], color = "g", ax = axes[1, 0])

sns.distplot(df1['TRB 18/19'], color = "m", ax = axes[1, 1])
# Set up figure

f, axes = plt.subplots(2, 2, figsize=(20, 15))



# Regressionplot

sns.regplot(x = df1['PTS 18/19'], y = df1['Salary 19/20'], color="b", ax=axes[0, 0])

sns.regplot(x = df1['AST 18/19'], y = df1['Salary 19/20'], color="r", ax=axes[0, 1])

sns.regplot(x = df1['STL 18/19'], y = df1['Salary 19/20'], color="g", ax=axes[1, 0])

sns.regplot(x = df1['TRB 18/19'], y = df1['Salary 19/20'], color="m", ax=axes[1, 1])
# Relationship with effecitve field goal percentage

sns.regplot(x = df1['eFG% 18/19'], y = df1['Salary 19/20'])
# Relationship with minutes played per game

sns.regplot(x = df1['MP 18/19'], y = df1['Salary 19/20'])
# Relationship with age

sns.regplot(x = df1['Age 18/19'], y = df1['Salary 19/20'])
# Relationship with Position

sns.boxplot(x = 'Pos', y = 'Salary 19/20', data = df1, order = ['PG', 'SG', 'SF', 'PF', 'C'])
sns.set(style = "white")

cor_matrix = df1.loc[:, 'Age 18/19': 'Salary 19/20'].corr()



# Generate a mask for the upper triangle

mask = np.triu(np.ones_like(cor_matrix, dtype = np.bool))



plt.figure(figsize = (15, 12))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap = True)



sns.heatmap(cor_matrix, mask = mask, cmap = cmap, center = 0,

            square = True, linewidths = .5, cbar_kws = {"shrink": .5})
cor_matrix = df2.loc[:, ['eFG% +-','TRB +-', 'AST +-', 'STL +-', 'BLK +-', 'TOV +-', 'PF +-', 'PTS +-', 

                        'Salary 19/20']].corr()



# Generate a mask for the upper triangle

mask = np.triu(np.ones_like(cor_matrix, dtype = np.bool))



plt.figure(figsize = (10, 8))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap = True)



sns.heatmap(cor_matrix, mask = mask, cmap = cmap, center = 0,

            square = True, linewidths = .5, cbar_kws = {"shrink": .5})
y = df1.loc[:, 'Salary 19/20']



x = df1.loc[:, ['Pos', 'Age 18/19', 'G 18/19', 'GS 18/19', 'MP 18/19', 'FG 18/19', 'FGA 18/19',

                'FG% 18/19', '3P 18/19', '3PA 18/19', '2P 18/19', '2PA 18/19', '2P% 18/19', 

                'eFG% 18/19', 'FT 18/19', 'FTA 18/19', 'ORB 18/19', 'DRB 18/19', 'TRB 18/19', 

                'AST 18/19', 'STL 18/19', 'BLK 18/19', 'TOV 18/19', 'PF 18/19', 'PTS 18/19']] 



print(x.shape)

print(y.shape)
# Instantiate OneHotEncoder

ohe = OneHotEncoder(categories = [['PG', 'SG', 'SF', 'PF', 'C']])



# Apply one-hot encoder

x_ohe = pd.DataFrame(ohe.fit_transform(x['Pos'].to_frame()).toarray())



# Get feature names

x_ohe.columns = ohe.get_feature_names(['Pos'])



# One-hot encoding removed index; put it back

x_ohe.index = x.index



# Add one-hot encoded columns to numerical features and remove categorical column

x = pd.concat([x, x_ohe], axis=1).drop(['Pos'], axis=1)



# How does it look like?

x.head()
# Split data using train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)



print(x_train.shape)

print(y_train.shape)

print(x_test.shape)

print(y_test.shape)
#Apply cube-root transformation

y_train = pd.DataFrame(np.cbrt([y_train])).T

y_test = pd.DataFrame(np.cbrt([y_test])).T

y = pd.DataFrame(np.cbrt([y])).T



#transformed histogram and normal probability plot

f, axes = plt.subplots(1, 2, figsize = (10, 5), sharex = True)

sns.distplot(y_train, color = "skyblue", fit = norm, ax = axes[0], axlabel = "y_train")

sns.distplot(y_test, color = "olive",fit = norm, ax = axes[1], axlabel = "y_test")

#sns.distplot(y, color = "olive",fit = norm, axlabel = "y")
# Use Robustscaler



#scaler = RobustScaler()

#x_train_scaled = pd.DataFrame(scaler.fit_transform(x_train), index = x_train.index, columns = x_train.columns)

#x_test_scaled = pd.DataFrame(scaler.transform(x_test), index = x_test.index, columns = x_test.columns)



#x_train_scaled.head()
# Function which uses an algorithm as input and returns the desired accuracy metrics and some predictions



def alg_fit(alg, x_train, y_train, x_test, name, y_true, df, mse, r2):

    

    # Model selection

    mod = alg.fit(x_train, y_train)

    

    # Prediction

    y_pred = mod.predict(x_test)

    

    # Accuracy

    acc1 = round(mse(y_test, y_pred), 4)

    acc2 = round(r2(y_test, y_pred), 4)

    

    # Accuracy table

    x_test['y_pred'] = mod.predict(x_test)

    df_acc = pd.merge(df, x_test, how = 'right')

    x_test.drop(['y_pred'], axis = 1, inplace = True)

    df_acc = df_acc[[name, y_true, 'y_pred']]

    df_acc.sort_values(by = y_true, ascending = False, inplace = True)

    df_acc['y_pred'] = df_acc['y_pred']**3

    

    return y_pred, acc1, acc2, df_acc
# Linear Regression

y_pred_lin, mse_lin, r2_lin, df_acc_lin = alg_fit(LinearRegression(), x_train, y_train, x_test, 'Player', 'Salary 19/20', 

                                                  df1, metrics.mean_squared_error, metrics.r2_score)



print("Root Mean Squared Error: %s" % round(np.sqrt(mse_lin), 4))

print("R-squared: %s" % r2_lin)

df_acc_lin.head(10)
# Ridge Regression

y_pred_rid, mse_rid, r2_rid, df_acc_rid = alg_fit(Ridge(alpha = 1), x_train, y_train, x_test, 'Player', 'Salary 19/20',

                                                  df1, metrics.mean_squared_error, metrics.r2_score)



print("Root Mean Squared Error: %s" % round(np.sqrt(mse_rid), 4))

print("R-squared: %s" % r2_rid)

df_acc_rid.head(10)
# Lasso Regression

y_pred_las, mse_las, r2_las, df_acc_las = alg_fit(Lasso(alpha = 0.001), x_train, y_train, x_test, 'Player', 'Salary 19/20',

                                                  df1, metrics.mean_squared_error, metrics.r2_score)



print("Root Mean Squared Error: %s" % round(np.sqrt(mse_las), 4))

print("R-squared: %s" % r2_las)

df_acc_las.head(10)
def alg_fit_cv(alg, x, y, mse, r2):

    

    # Cross validation

    cv = KFold(shuffle = True, random_state = 0, n_splits = 5)

    

    # Accuracy

    scores1 = cross_val_score(alg, x, y, cv = cv, scoring = mse)

    scores2 = cross_val_score(alg, x, y, cv = cv, scoring = r2)

    acc1_cv = round(scores1.mean(), 4)

    acc2_cv = round(scores2.mean(), 4)

    

    return acc1_cv, acc2_cv
# Linear Regression



mse_cv_lin, r2_cv_lin = alg_fit_cv(LinearRegression(), x, y, 'neg_mean_squared_error', 'r2')



print("Root Mean Squared Error: %s" % round(np.sqrt(mse_cv_lin*-1), 4))

print("R-squared: %s" % r2_cv_lin)
# Ridge Regression

mse_cv_rid, r2_cv_rid = alg_fit_cv(Ridge(alpha = 23), x, y, 'neg_mean_squared_error', 'r2')



print("Root Mean Squared Error: %s" % round(np.sqrt(mse_cv_rid*-1), 4))

print("R-squared: %s" % r2_cv_rid)
# Lasso Regression

mse_cv_las, r2_cv_las = alg_fit_cv(Ridge(alpha = 23), x, y, 'neg_mean_squared_error', 'r2')



print("Root Mean Squared Error: %s" % round(np.sqrt(mse_cv_las*-1), 4))

print("R-squared: %s" % r2_cv_las)
# LightGBM Regressor (after some parameter tuning)

lgbm = LGBMRegressor(objective = 'regression',

                     num_leaves = 20,

                     learning_rate = 0.03,

                     n_estimators = 200,

                     max_bin = 50,

                     bagging_fraction = 0.85,

                     bagging_freq = 4,

                     bagging_seed = 6,

                     feature_fraction = 0.2,

                     feature_fraction_seed = 7,

                     verbose = -1)



mse_cv_lgbm, r2_cv_lgbm = alg_fit_cv(lgbm, x, y, 'neg_mean_squared_error', 'r2')



print("Root Mean Squared Error: %s" % round(np.sqrt(mse_cv_lgbm*-1), 4))

print("R-squared: %s" % r2_cv_lgbm)
# XGB-Regressor (after some parameter tuning)

xgb = XGBRegressor(n_estimators = 300,

                   max_depth = 2,

                   min_child_weight = 0,

                   gamma = 8,

                   subsample = 0.6,

                   colsample_bytree = 0.9,

                   objective = 'reg:squarederror',

                   nthread = -1,

                   scale_pos_weight = 1,

                   seed = 27,

                   learning_rate = 0.02,

                   reg_alpha = 0.006)



mse_cv_xgb, r2_cv_xgb = alg_fit_cv(xgb, x, y, 'neg_mean_squared_error', 'r2')



print("Root Mean Squared Error: %s" % round(np.sqrt(mse_cv_xgb*-1), 4))

print("R-squared: %s" % r2_cv_xgb)
# Merge y and x back together

#df_new = pd.concat([y, x], axis=1)



# Compute Z-score for the dataframe

#z = np.abs(stats.zscore(df_new))



# Delete rows with outliers

#df_new = df_new[(z < 4).all(axis = 1)].reset_index()



# Split into y and x again

#y_new = df_new.loc[:, 0]

#x_new = df_new.loc[:, 'PTS 18/19':] 

#print(x_new.shape)

#print(y_new.shape)
# Model

mod = xgb.fit(x, y)



# Feature importance

df_feature_importance = pd.DataFrame(xgb.feature_importances_, index = x.columns, 

                                     columns = ['feature importance']).sort_values('feature importance', 

                                                                                   ascending = False)

df_feature_importance
# Drop out features with low importance or which are redundant

x_new = x.loc[:, ['PTS 18/19', 'Pos_PG', 'Pos_SG', 'Pos_SF', 'Pos_PF', 'Pos_C', 'Age 18/19', 'STL 18/19', 

                  'G 18/19', 'TRB 18/19', 'AST 18/19', 'PF 18/19', 'MP 18/19']]
# XGB-Regressor (after some parameter tuning)

xgb_new = XGBRegressor(n_estimators = 270,

                       max_depth = 2,

                       min_child_weight = 0,

                       gamma = 18,

                       subsample = 0.7,

                       colsample_bytree = 0.9,

                       objective = 'reg:squarederror',

                       nthread = -1,

                       scale_pos_weight = 1,

                       seed = 27,

                       learning_rate = 0.023,

                       reg_alpha = 0.02)



mse_cv_xgb, r2_cv_xgb = alg_fit_cv(xgb_new, x_new, y, 'neg_mean_squared_error', 'r2')



print("Root Mean Squared Error: %s" % round(np.sqrt(mse_cv_xgb*-1), 4))

print("R-squared: %s" % r2_cv_xgb)
# Split data now with x_new

x_train, x_test, y_train, y_test = train_test_split(x_new, y, test_size = 0.2, random_state = 0)



# Use function to fit algorithm

y_pred_xgb, mse_xgb, r2_xgb, df_acc_xgb = alg_fit(xgb_new, x_train, y_train, x_test, 'Player', 'Salary 19/20', 

                                                  df1, metrics.mean_squared_error, metrics.r2_score)



print("Root Mean Squared Error: %s" % round(np.sqrt(mse_xgb), 4))

print("R-squared: %s" % r2_xgb)

df_acc_xgb.head(10)