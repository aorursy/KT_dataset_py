import numpy as np 

import pandas as pd 



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.model_selection import KFold, cross_validate, GridSearchCV, train_test_split

from sklearn.metrics import mean_squared_error



from sklearn.pipeline import Pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.preprocessing import OneHotEncoder, LabelEncoder



import statsmodels.formula.api as sm

from sklearn.linear_model import Lasso

from sklearn.tree import DecisionTreeRegressor





import warnings



# seaborn and scipy versions are not aligned in the docker image

warnings.filterwarnings("ignore", message="Using a non-tuple sequence for multidimensional indexing is deprecated")
df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')



df_train = df_train[df_train.GrLivArea < 4500].copy()  # the documentation says they are outliers



combine = [df_train, df_test]

df_train.name = 'Train'

df_test.name = 'Test'



for df in combine:

    # LotFrontage

    df.loc[df.LotFrontage.isnull(), 'LotFrontage'] = 0

    # Alley

    df.loc[df.Alley.isnull(), 'Alley'] = "NoAlley"

    # MSSubClass

    df['MSSubClass'] = df['MSSubClass'].astype(str)

    # MissingBasement

    fil = ((df.BsmtQual.isnull()) & (df.BsmtCond.isnull()) & (df.BsmtExposure.isnull()) &

          (df.BsmtFinType1.isnull()) & (df.BsmtFinType2.isnull()))

    fil1 = ((df.BsmtQual.notnull()) | (df.BsmtCond.notnull()) | (df.BsmtExposure.notnull()) |

          (df.BsmtFinType1.notnull()) | (df.BsmtFinType2.notnull()))

    df.loc[fil1, 'MisBsm'] = 0

    df.loc[fil, 'MisBsm'] = 1

    # BsmtQual

    df.loc[fil, 'BsmtQual'] = "NoBsmt" # missing basement

    # BsmtCond

    df.loc[fil, 'BsmtCond'] = "NoBsmt" # missing basement

    # BsmtExposure

    df.loc[fil, 'BsmtExposure'] = "NoBsmt" # missing basement

    # BsmtFinType1

    df.loc[fil, 'BsmtFinType1'] = "NoBsmt" # missing basement

    # BsmtFinType2

    df.loc[fil, 'BsmtFinType2'] = "NoBsmt" # missing basement

    # FireplaceQu

    df.loc[(df.Fireplaces == 0) & (df.FireplaceQu.isnull()), 'FireplaceQu'] = "NoFire" # missing

    # MisGarage

    fil = ((df.GarageYrBlt.isnull()) & (df.GarageType.isnull()) & (df.GarageFinish.isnull()) &

          (df.GarageQual.isnull()) & (df.GarageCond.isnull()))

    fil1 = ((df.GarageYrBlt.notnull()) | (df.GarageType.notnull()) | (df.GarageFinish.notnull()) |

          (df.GarageQual.notnull()) | (df.GarageCond.notnull()))

    df.loc[fil1, 'MisGarage'] = 0

    df.loc[fil, 'MisGarage'] = 1

    # GarageYrBlt

    df.loc[df.GarageYrBlt > 2200, 'GarageYrBlt'] = 2007  # correct mistake

    df.loc[fil, 'GarageYrBlt'] = 0

    # GarageType

    df.loc[fil, 'GarageType'] = "NoGrg" # missing garage

    # GarageFinish

    df.loc[fil, 'GarageFinish'] = "NoGrg" # missing

    # GarageQual

    df.loc[fil, 'GarageQual'] = "NoGrg" # missing

    # GarageCond

    df.loc[fil, 'GarageCond'] = "NoGrg" # missing

    # Fence

    df.loc[df.Fence.isnull(), 'Fence'] = "NoFence" # missing fence

    # Dropping stuff

    del df['PoolQC']

    del df['MiscFeature']



# Fixing some entries in test

df_test[['BsmtUnfSF', 

         'TotalBsmtSF', 

         'BsmtFinSF1', 

         'BsmtFinSF2']] = df_test[['BsmtUnfSF', 

                                   'TotalBsmtSF', 

                                   'BsmtFinSF1', 

                                   'BsmtFinSF2']].fillna(0) # checked



# eliminating entries in train with missing values

for f in df_train.columns:

    df_train = df_train[pd.notnull(df_train[f])]

    

    

df_train['target'] = np.log1p(df_train.SalePrice)
#To find the segment of the missing values, can be useful to impute the missing values

def find_segment(df, feat): 

    mis = df[feat].isnull().sum()

    cols = df.columns

    seg = []

    for col in cols:

        vc = df[df[feat].isnull()][col].value_counts(dropna=False).iloc[0]

        if (vc == mis): #returns the columns for which the missing entries have only 1 possible value

            seg.append(col)

    return seg



# to find the mode of the missing feature, by choosing the right segment to compare (uses find_segment)

def find_mode(df, feat): #returns the mode to fill in the missing feat

    md = df[df[feat].isnull()][find_segment(df, feat)].dropna(axis=1).mode()

    md = pd.merge(df, md, how='inner')[feat].mode().iloc[0]

    return md



# identical to the previous one, but with the median

def find_median(df, feat): #returns the median to fill in the missing feat

    md = df[df[feat].isnull()][find_segment(df, feat)].dropna(axis=1).mode()

    md = pd.merge(df, md, how='inner')[feat].median()

    return md



# find the mode in a segment defined by the user

def similar_mode(df, col, feats): #returns the mode in a segment made by similarity in feats

    sm = df[df[col].isnull()][feats]

    md = pd.merge(df, sm, how='inner')[col].mode().iloc[0]

    return md



# Find the median in a segment defined by the user

def similar_median(df, col, feats): #returns the median in a segment made by similarity in feats

    sm = df[df[col].isnull()][feats]

    md = pd.merge(df, sm, how='inner')[col].median()

    return md
# Cleaning of test 



# MSZoning

md = find_mode(df_test, 'MSZoning')

print("MSZoning {}".format(md))

df_test[['MSZoning']] = df_test[['MSZoning']].fillna(md)

# Utilities

md = 'AllPub'

df_test[['Utilities']] = df_test[['Utilities']].fillna(md)

# MasVnrType

md = find_mode(df_test, 'MasVnrType')

print("MasVnrType {}".format(md))

df_test[['MasVnrType']] = df_test[['MasVnrType']].fillna(md)

# MasVnrArea

md = find_mode(df_test, 'MasVnrArea')

print("MasVnrArea {}".format(md))

df_test[['MasVnrArea']] = df_test[['MasVnrArea']].fillna(md)

# BsmtQual

simi = ['BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']

md = similar_mode(df_test, 'BsmtQual', simi)

print("BsmtQual {}".format(md))

df_test[['BsmtQual']] = df_test[['BsmtQual']].fillna(md)

# BsmtCond

simi = ['BsmtQual', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']

md = similar_mode(df_test, 'BsmtCond', simi)

print("BsmtCond {}".format(md))

df_test[['BsmtCond']] = df_test[['BsmtCond']].fillna(md)

# BsmtCond

simi = ['BsmtQual', 'BsmtCond', 'BsmtFinType1', 'BsmtFinType2']

md = similar_mode(df_test, 'BsmtExposure', simi)

print("BsmtExposure {}".format(md))

df_test[['BsmtExposure']] = df_test[['BsmtExposure']].fillna(md)

# BsmtFullBath

simi = ['BsmtQual', 'BsmtCond', 'BsmtFinType1', 'BsmtFinType2']

md = similar_median(df_test, 'BsmtFullBath', simi)

print("BsmtFullBath {}".format(md))

df_test[['BsmtFullBath']] = df_test[['BsmtFullBath']].fillna(md)

# BsmtHalfBath

simi = ['BsmtQual', 'BsmtCond', 'BsmtFinType1', 'BsmtFinType2']

md = similar_median(df_test, 'BsmtHalfBath', simi)

print("BsmtHalfBath {}".format(md))

df_test[['BsmtHalfBath']] = df_test[['BsmtHalfBath']].fillna(md)

# KitchenQual

md = df_test.KitchenQual.mode().iloc[0]

print("KitchenQual {}".format(md))

df_test[['KitchenQual']] = df_test[['KitchenQual']].fillna(md)

# Functional

md = 'Typ'

df_test[['Functional']] = df_test[['Functional']].fillna(md)

# GarageYrBlt

simi = ['GarageType', 'MisGarage']

md = similar_median(df_test, 'GarageYrBlt', simi)

print("GarageYrBlt {}".format(md))

df_test[['GarageYrBlt']] = df_test[['GarageYrBlt']].fillna(md)

# GarageFinish

md = 'Unf'

print("GarageFinish {}".format(md))

df_test[['GarageFinish']] = df_test[['GarageFinish']].fillna(md)

# GarageArea

simi = ['GarageType', 'MisGarage']

md = similar_median(df_test, 'GarageArea', simi)

print("GarageArea {}".format(md))

df_test[['GarageArea']] = df_test[['GarageArea']].fillna(md)

# GarageQual

simi = ['GarageType', 'MisGarage', 'GarageFinish']

md = similar_mode(df_test, 'GarageQual', simi)

print("GarageQual {}".format(md))

df_test[['GarageQual']] = df_test[['GarageQual']].fillna(md)

# GarageCond

simi = ['GarageType', 'MisGarage', 'GarageFinish']

md = similar_mode(df_test, 'GarageCond', simi)

print("GarageCond {}".format(md))

df_test[['GarageCond']] = df_test[['GarageCond']].fillna(md)

# GarageCars

simi = ['GarageType', 'MisGarage']

md = similar_median(df_test, 'GarageCars', simi)

print("GarageCars {}".format(md))

df_test[['GarageCars']] = df_test[['GarageCars']].fillna(md)



cols = df_test.columns

mis_test = []

print("Start printing the missing values...")

for col in cols:

    mis = df_test[col].isnull().sum()

    if mis > 0:

        print("{}: {} missing, {}%".format(col, mis, round(mis/df_test.shape[0] * 100, 3)))

        mis_test.append(col)

print("...done printing the missing values")
sel_cols = ['MSZoning', 'Alley', 'LotShape', 'Foundation', 

             'Heating', 'GarageQual', 'MasVnrType', 'ExterQual']



for col in sel_cols:

    print(col)

    print(df_train[col].value_counts())

    print('_'*20)

    print(df_test[col].value_counts())

    print('_'*40)

    print('\n')
X_train, X_test, y_train, y_test = train_test_split(df_train, df_train.target, test_size=0.20, random_state=42)



kfolds = KFold(n_splits=5, shuffle=True, random_state=14)



print(f"Original train set shape: \t {df_train.shape}")

print(f"New train set shape: \t {X_train.shape}")

print(f"Validation set shape: \t {X_test.shape}")
def OLS_experiment(train, target, validate, val_target):

    train = train.copy()

    validate = validate.copy()

    train['intercept'] = 1 

    validate['intercept'] = 1

    

    regressor_OLS = sm.OLS(endog = target, exog = train).fit()

    print(regressor_OLS.summary())

    

    in_pred = regressor_OLS.predict(train)

    score = mean_squared_error(y_pred=in_pred, y_true=target)

    print('\n')

    print(f'Score in-sample: \t {score}')

    

    pred = regressor_OLS.predict(validate)

    score = mean_squared_error(y_pred=pred, y_true=val_target)

    

    print('\n')

    print(f'Score out of sample: \t {score}')

    

    return pred, in_pred





def get_coef(clsf, ftrs):

    imp = clsf.coef_.tolist() 

    feats = ftrs

    result = pd.DataFrame({'feat':feats,'score':imp})

    result = result.sort_values(by=['score'],ascending=False)

    return result





def lasso_experiment(train, target, validate, val_target, folds):

    param_grid = [{'alpha' : [0.0001, 0.0005, 0.00075,

                                 0.001, 0.005, 0.0075, 

                                 0.01, 0.05, 0.075,

                                 0.1, 0.5, 0.75, 

                                 1, 5, 7.5]}]



    grid = GridSearchCV(Lasso(), param_grid=param_grid,

                        cv=folds, scoring='neg_mean_squared_error', 

                        return_train_score=True, n_jobs=-1)

    grid.fit(train, target)

    

    best_lasso = grid.best_estimator_

    print(best_lasso)

    print("_"*40)

    #with its score

    print(np.sqrt(-grid.best_score_))

    print("_"*40)

    

    print(get_coef(best_lasso, train.columns))

    

    pred = best_lasso.predict(validate)

    score = mean_squared_error(y_pred=pred, y_true=val_target)

    

    print(f'Validation score: \t {score}')

    

    return pred





def get_feature_importance(clsf, ftrs):

    imp = clsf.feature_importances_.tolist() 

    feats = ftrs

    result = pd.DataFrame({'feat':feats,'score':imp})

    result = result.sort_values(by=['score'],ascending=False)

    return result





def tree_experiment(train, target, validate, val_target, folds):

    param_grid = [{'max_depth': [2, 3, 5, 8, 10, 20], 

                   'max_leaf_nodes': [None, 5, 10, 20]}]

    

    grid = GridSearchCV(DecisionTreeRegressor(), param_grid=param_grid,

                        cv=folds, scoring='neg_mean_squared_error', 

                        return_train_score=True, n_jobs=-1)

    grid.fit(train, target)

    

    best_tree = grid.best_estimator_

    print(best_tree)

    print("_"*40)

    #with its score

    print(np.sqrt(-grid.best_score_))

    print("_"*40)

    

    print(get_feature_importance(best_tree, train.columns))

    

    pred = best_tree.predict(validate)

    score = mean_squared_error(y_pred=pred, y_true=val_target)

    

    print(f'Validation score: \t {score}')

    

    return pred





def plot_predictions(val_target, ols, lasso, tree):

    line = pd.DataFrame({'x': np.arange(10.5,13.5,0.01), # small hack for a diagonal line

                         'y': np.arange(10.5,13.5,0.01)})

    plt.figure(figsize=(10,6))

    plt.scatter(val_target, ols, label='OLS')

    plt.scatter(val_target, lasso, label='Lasso')

    plt.scatter(val_target, tree, label='Tree')

    plt.plot(line.x, line.y, color='black')

    plt.xlabel('True value', fontsize=12)

    plt.ylabel('Prediction', fontsize=12)

    plt.legend()
exp_train = X_train[['OverallQual', 'OverallCond', 'GrLivArea', 'GarageArea']].copy()

exp_test = X_test[['OverallQual', 'OverallCond', 'GrLivArea', 'GarageArea']].copy()



print('OLS experiment')

ols_pred, ols_i_pred = OLS_experiment(exp_train, y_train, exp_test, y_test)

print('\n')

print('_'*40)

print('Lasso experiment')

lasso_pred = lasso_experiment(exp_train, y_train, exp_test, y_test, kfolds)

print('\n')

print('_'*40)

print('Tree experiment')

tree_pred = tree_experiment(exp_train, y_train, exp_test, y_test, kfolds)
plot_predictions(y_test, ols_pred, lasso_pred, tree_pred)
dum_test = df_train[['MasVnrType']].copy()

dum_test.head(10)
dum_test.MasVnrType.value_counts(dropna=False)
dum_transf = pd.get_dummies(dum_test)

dum_transf.head()
for col in dum_transf.columns:

    print(col)

    print(dum_transf[col].value_counts())

    print('\n')
encoder = OneHotEncoder()

dum_transf = encoder.fit_transform(dum_test)

pd.DataFrame(dum_transf.todense(), columns=encoder.get_feature_names()).head(10)
exp_train = X_train[['MasVnrType', 'Alley', 'LotShape']].copy()

exp_test = X_test[['MasVnrType', 'Alley', 'LotShape']].copy()



exp_train = pd.get_dummies(exp_train)

exp_test = pd.get_dummies(exp_test)



exp_train.head()
res_ols, res_ols_i = OLS_experiment(exp_train, y_train, exp_test, y_test)
exp_train = X_train[['MasVnrType', 'Alley', 'LotShape']].copy()

exp_test = X_test[['MasVnrType', 'Alley', 'LotShape']].copy()



exp_train = pd.get_dummies(exp_train, drop_first=True)

exp_test = pd.get_dummies(exp_test, drop_first=True)



exp_train.head()
res_ols_o, res_ols_i = OLS_experiment(exp_train, y_train, exp_test, y_test)
exp_train = X_train[['MasVnrType', 'Alley', 'LotShape']].copy()

exp_test = X_test[['MasVnrType', 'Alley', 'LotShape']].copy()



exp_train = pd.get_dummies(exp_train)

exp_test = pd.get_dummies(exp_test)



res_lasso = lasso_experiment(exp_train, y_train, exp_test, y_test, kfolds)



print('\n')

print('_'*40)

print('Now, let\'s drop one dummy')

print('\n')



exp_train = X_train[['MasVnrType', 'Alley', 'LotShape']].copy()

exp_test = X_test[['MasVnrType', 'Alley', 'LotShape']].copy()



exp_train = pd.get_dummies(exp_train, drop_first=True)

exp_test = pd.get_dummies(exp_test, drop_first=True)



res_lasso = lasso_experiment(exp_train, y_train, exp_test, y_test, kfolds)
exp_train = X_train[['MasVnrType', 'Alley', 'LotShape']].copy()

exp_test = X_test[['MasVnrType', 'Alley', 'LotShape']].copy()



exp_train = pd.get_dummies(exp_train)

exp_test = pd.get_dummies(exp_test)



res_tree = tree_experiment(exp_train, y_train, exp_test, y_test, kfolds)



print('\n')

print('_'*40)

print('Now, let\'s drop one dummy')

print('\n')



exp_train = X_train[['MasVnrType', 'Alley', 'LotShape']].copy()

exp_test = X_test[['MasVnrType', 'Alley', 'LotShape']].copy()



exp_train = pd.get_dummies(exp_train, drop_first=True)

exp_test = pd.get_dummies(exp_test, drop_first=True)



res_tree = tree_experiment(exp_train, y_train, exp_test, y_test, kfolds)
plot_predictions(y_test, res_ols_o, res_lasso, res_tree)
le = LabelEncoder()

dum_test['encoded'] = le.fit_transform(dum_test.MasVnrType)

dum_test.head(10)
exp_train = X_train[['MasVnrType', 'Alley', 'LotShape']].copy()

exp_test = X_test[['MasVnrType', 'Alley', 'LotShape']].copy()



exp_train = exp_train.apply(le.fit_transform)

exp_test = exp_test.apply(le.fit_transform)



exp_train.head()
tmp,tmp_1 = OLS_experiment(exp_train, y_train, exp_test, y_test)



print('\n')



tmp = tree_experiment(exp_train, y_train, exp_test, y_test, kfolds)
exp_train = X_train[['MasVnrType']].copy()

exp_train = pd.get_dummies(exp_train)

exp_train.head()
from sklearn import tree

import graphviz



dt = DecisionTreeRegressor().fit(exp_train, y_train)



dot_data = tree.export_graphviz(dt, out_file=None, filled=True)  

graph = graphviz.Source(dot_data)  

graph
exp_train = X_train[['MasVnrType']].copy()

exp_train['MasVnrType'] = le.fit_transform(exp_train.MasVnrType)

exp_train.head()
dt = DecisionTreeRegressor().fit(exp_train, y_train)



dot_data = tree.export_graphviz(dt, out_file=None, filled=True)  

graph = graphviz.Source(dot_data)  

graph
df_train.ExterQual.value_counts()
ord_test = df_train[['ExterQual']].copy()

ord_test['encoded'] = le.fit_transform(ord_test.ExterQual)

ord_test.head()
pd.crosstab(ord_test.encoded, ord_test.ExterQual)
ord_test = df_train[['ExterQual', 'target']].copy()

ord_test.groupby('ExterQual').agg(['mean', 'median', 'max', 'min', 'count'])
ord_test.loc[ord_test.ExterQual == 'Fa', 'ExtQuGroup'] = 0

ord_test.loc[ord_test.ExterQual == 'TA', 'ExtQuGroup'] = 1

ord_test.loc[ord_test.ExterQual == 'Gd', 'ExtQuGroup'] = 2

ord_test.loc[ord_test.ExterQual == 'Ex', 'ExtQuGroup'] = 3



ord_test['LabelEncoded'] = le.fit_transform(ord_test.ExterQual)



x1 = ord_test['ExtQuGroup']

x2 = ord_test['LabelEncoded']

y = ord_test['target']



print('Correlations with target')

print(ord_test[['ExtQuGroup','LabelEncoded' ,'target' ]].corr())



fig, ax= plt.subplots(1,2, figsize=(15, 6))



sns.regplot(x = x1, y = y, x_estimator = np.mean, ax=ax[0])

sns.regplot(x = x2, y = y, x_estimator = np.mean, ax=ax[1])



ax[0].set_title('Manual encoding', fontsize=16)

ax[1].set_title('Automatic encoding', fontsize=16)
ord_test = df_train[['HeatingQC', 'target']].copy()



ord_test.loc[ord_test.HeatingQC == 'Po', 'HeatQGroup'] = 1

ord_test.loc[ord_test.HeatingQC == 'Fa', 'HeatQGroup'] = 2

ord_test.loc[ord_test.HeatingQC == 'TA', 'HeatQGroup'] = 3

ord_test.loc[ord_test.HeatingQC == 'Gd', 'HeatQGroup'] = 4

ord_test.loc[ord_test.HeatingQC == 'Ex', 'HeatQGroup'] = 5



ord_test['LabelEncoded'] = le.fit_transform(ord_test.HeatingQC)



x1 = ord_test['HeatQGroup']

x2 = ord_test['LabelEncoded']

y = ord_test['target']



print('Correlations with target')

print(ord_test[['HeatQGroup','LabelEncoded' ,'target' ]].corr())



fig, ax= plt.subplots(1,2, figsize=(15, 6))



sns.regplot(x = x1, y = y, x_estimator = np.mean, ax=ax[0])

sns.regplot(x = x2, y = y, x_estimator = np.mean, ax=ax[1])



ax[0].set_title('Manual encoding', fontsize=16)

ax[1].set_title('Automatic encoding', fontsize=16)
ord_test = df_train[['HeatingQC', 'target']].copy()



ord_test.loc[ord_test.HeatingQC == 'Po', 'HeatQGroup'] = 1

ord_test.loc[ord_test.HeatingQC == 'Fa', 'HeatQGroup'] = 2

ord_test.loc[ord_test.HeatingQC == 'TA', 'HeatQGroup'] = 3

ord_test.loc[ord_test.HeatingQC == 'Gd', 'HeatQGroup'] = 4

ord_test.loc[ord_test.HeatingQC == 'Ex', 'HeatQGroup'] = 5



ord_test.loc[ord_test.HeatingQC == 'Po', 'HeatQGroup_2'] = 1

ord_test.loc[ord_test.HeatingQC == 'Fa', 'HeatQGroup_2'] = 1

ord_test.loc[ord_test.HeatingQC == 'TA', 'HeatQGroup_2'] = 3

ord_test.loc[ord_test.HeatingQC == 'Gd', 'HeatQGroup_2'] = 4

ord_test.loc[ord_test.HeatingQC == 'Ex', 'HeatQGroup_2'] = 7



x1 = ord_test['HeatQGroup']

x2 = ord_test['HeatQGroup_2']

y = ord_test['target']



print('Correlations with target')

print(ord_test[['HeatQGroup','HeatQGroup_2' ,'target' ]].corr())



fig, ax= plt.subplots(1,2, figsize=(15, 6))



sns.regplot(x = x1, y = y, x_estimator = np.mean, ax=ax[0])

sns.regplot(x = x2, y = y, x_estimator = np.mean, ax=ax[1])



ax[0].set_title('Old encoding', fontsize=16)

ax[1].set_title('New encoding', fontsize=16)
print('With a normal encoding')

print('\n')



exp_train = X_train[['HeatingQC']].copy() # , 'ExterQual', 'KitchenQual'

exp_test = X_test[['HeatingQC']].copy()



norm_encode = {'Po': 1, 'Fa': 2, 'TA':3, 'Gd':4, 'Ex': 5}



exp_train['HeatingQC'] = exp_train.HeatingQC.map(norm_encode).astype(int)

exp_test['HeatingQC'] = exp_test.HeatingQC.map(norm_encode).astype(int)

#exp_train['ExterQual'] = exp_train.ExterQual.map(norm_encode).astype(int)

#exp_test['ExterQual'] = exp_test.ExterQual.map(norm_encode).astype(int)

#exp_train['KitchenQual'] = exp_train.KitchenQual.map(norm_encode).astype(int)

#exp_test['KitchenQual'] = exp_test.KitchenQual.map(norm_encode).astype(int)



lasso_pred_norm = lasso_experiment(exp_train, y_train, exp_test, y_test, kfolds)



print('_'*40)



print('\n Now by encoding HeatingQC differently')

print('\n')



exp_train = X_train[['HeatingQC']].copy()

exp_test = X_test[['HeatingQC']].copy()



norm_encode = {'Po': 1, 'Fa': 2, 'TA':3, 'Gd':4, 'Ex': 5}

new_encode = {'Po': 1, 'Fa': 1, 'TA':3, 'Gd':4, 'Ex': 7}



exp_train['HeatingQC'] = exp_train.HeatingQC.map(new_encode).astype(int)

exp_test['HeatingQC'] = exp_test.HeatingQC.map(new_encode).astype(int)

#exp_train['ExterQual'] = exp_train.ExterQual.map(norm_encode).astype(int)

#exp_test['ExterQual'] = exp_test.ExterQual.map(norm_encode).astype(int)

#exp_train['KitchenQual'] = exp_train.KitchenQual.map(norm_encode).astype(int)

#exp_test['KitchenQual'] = exp_test.KitchenQual.map(norm_encode).astype(int)



lasso_pred_new = lasso_experiment(exp_train, y_train, exp_test, y_test, kfolds)



print('_'*40)



print('\n Now by getting dummies')

print('\n')



exp_train = X_train[['HeatingQC']].copy()

exp_test = X_test[['HeatingQC']].copy()



# dirty fix for dummies mismatch

exp_test.loc[exp_test.HeatingQC == 'Po', 'HeatingQC'] = 'Fa' 



exp_train = pd.get_dummies(exp_train, drop_first=True)

exp_test = pd.get_dummies(exp_test, drop_first=True)



lasso_pred_dum = lasso_experiment(exp_train, y_train, exp_test, y_test, kfolds)
print('With a normal encoding')

print('\n')



exp_train = X_train[['HeatingQC']].copy() # , 'ExterQual', 'KitchenQual'

exp_test = X_test[['HeatingQC']].copy()



norm_encode = {'Po': 1, 'Fa': 2, 'TA':3, 'Gd':4, 'Ex': 5}



exp_train['HeatingQC'] = exp_train.HeatingQC.map(norm_encode).astype(int)

exp_test['HeatingQC'] = exp_test.HeatingQC.map(norm_encode).astype(int)

#exp_train['ExterQual'] = exp_train.ExterQual.map(norm_encode).astype(int)

#exp_test['ExterQual'] = exp_test.ExterQual.map(norm_encode).astype(int)

#exp_train['KitchenQual'] = exp_train.KitchenQual.map(norm_encode).astype(int)

#exp_test['KitchenQual'] = exp_test.KitchenQual.map(norm_encode).astype(int)



tree_pred_norm = tree_experiment(exp_train, y_train, exp_test, y_test, kfolds)



print('_'*40)



print('\n Now by encoding HeatingQC differently')

print('\n')



exp_train = X_train[['HeatingQC']].copy()

exp_test = X_test[['HeatingQC']].copy()



norm_encode = {'Po': 1, 'Fa': 2, 'TA':3, 'Gd':4, 'Ex': 5}

new_encode = {'Po': 1, 'Fa': 1, 'TA':3, 'Gd':4, 'Ex': 7}



exp_train['HeatingQC'] = exp_train.HeatingQC.map(new_encode).astype(int)

exp_test['HeatingQC'] = exp_test.HeatingQC.map(new_encode).astype(int)

#exp_train['ExterQual'] = exp_train.ExterQual.map(norm_encode).astype(int)

#exp_test['ExterQual'] = exp_test.ExterQual.map(norm_encode).astype(int)

#exp_train['KitchenQual'] = exp_train.KitchenQual.map(norm_encode).astype(int)

#exp_test['KitchenQual'] = exp_test.KitchenQual.map(norm_encode).astype(int)



tree_pred_new = tree_experiment(exp_train, y_train, exp_test, y_test, kfolds)



print('_'*40)



print('\n Now by getting dummies')

print('\n')



exp_train = X_train[['HeatingQC']].copy()

exp_test = X_test[['HeatingQC']].copy()



# dirty fix for dummies mismatch

exp_test.loc[exp_test.HeatingQC == 'Po', 'HeatingQC'] = 'Fa' 



exp_train = pd.get_dummies(exp_train, drop_first=True)

exp_test = pd.get_dummies(exp_test, drop_first=True)



tree_pred_dum = tree_experiment(exp_train, y_train, exp_test, y_test, kfolds)
print('Training set')

exp_train = X_train[['HeatingQC']].copy()

exp_train = pd.get_dummies(exp_train, drop_first=True)

exp_train.head()
print('Validation set')

exp_test= X_test[['HeatingQC']].copy()

exp_test = pd.get_dummies(exp_test, drop_first=True)

exp_test.head()
df_train.HeatingQC.value_counts()
mis_dum = df_train[['HeatingQC']].copy()

mis_dum = pd.get_dummies(mis_dum)



mis_d_train, mis_d_test, mis_d_y_train,  mis_d_y_test = train_test_split(mis_dum, df_train.target, test_size=0.20, random_state=42)



print(mis_d_train.columns)

print(f'Number of `Po`: \t {mis_d_train.HeatingQC_Po.sum()}')

print('\n')

print(mis_d_test.columns)

print(f'Number of `Po`: \t {mis_d_test.HeatingQC_Po.sum()}')
dum_col = ['HeatingQC', 'GarageQual', 'Condition2', 'Utilities', 'Heating']



exp_train = X_train[dum_col].copy()

exp_test = X_test[dum_col].copy()

exp_train = pd.get_dummies(exp_train, drop_first=True)

exp_test = pd.get_dummies(exp_test, drop_first=True)



print('Mismatched dummies:')

print(list(set(exp_train.columns) - set(exp_test.columns)))



exp_train = exp_train[[col for col in exp_test.columns if col in exp_train.columns]]

exp_test = exp_test[[col for col in exp_train.columns]]  # yes, it has to be done twice



print('\nCommon columns: ')

print(list(exp_train.columns))
rare_cat = ['MSZoning', 'LotShape', 'Foundation', 'MasVnrType']

rare_ord = ['GarageQual', 'ExterQual']
df_train.Foundation.value_counts()
exp_train = X_train[rare_cat].copy()

exp_test = X_test[rare_cat].copy()



exp_train = pd.get_dummies(exp_train, drop_first=True)

exp_test = pd.get_dummies(exp_test, drop_first=True)



del exp_train['Foundation_Wood']



print('Lasso experiment')

lasso_pred = lasso_experiment(exp_train, y_train, exp_test, y_test, kfolds)

print('\n')

print('Tree experiment')

tree_pred = tree_experiment(exp_train, y_train, exp_test, y_test, kfolds)
exp_train = X_train[rare_cat].copy()

exp_test = X_test[rare_cat].copy()



# floating villages can be low density residential area

exp_train.loc[exp_train.MSZoning == 'FV', 'MSZoning'] = 'RL'  

exp_test.loc[exp_test.MSZoning == 'FV', 'MSZoning'] = 'RL'

# medium and high density residential areas can be considered, with commercial area, a non low density residential area

exp_train.loc[exp_train.MSZoning != 'RL', 'MSZoning'] = 'notRL'  

exp_test.loc[exp_test.MSZoning != 'RL', 'MSZoning'] = 'notRL'



# LotShape is either regular or irregular

exp_train.loc[exp_train.LotShape != 'Reg', 'LotShape'] = 'IRR'

exp_test.loc[exp_test.LotShape != 'Reg', 'LotShape'] = 'IRR'



# making foundations simpler

fond = ['PConc', 'CBlock']

exp_train.loc[~exp_train.Foundation.isin(fond), 'Foundation'] = 'Other'

exp_test.loc[~exp_test.Foundation.isin(fond), 'Foundation'] = 'Other'



# MasVnrType doesn't need 2 types of bricks

exp_train.loc[exp_train.MasVnrType == 'BrkCmn', 'MasVnrType'] = 'BrkFace'

exp_test.loc[exp_test.MasVnrType == 'BrkCmn', 'MasVnrType'] = 'BrkFace'



exp_train = pd.get_dummies(exp_train, drop_first=True)

exp_test = pd.get_dummies(exp_test, drop_first=True)



tree_pred = tree_experiment(exp_train, y_train, exp_test, y_test, kfolds)
exp_train = X_train[rare_cat].copy()

exp_test = X_test[rare_cat].copy()



# floating villages can be low density residential area

exp_train.loc[exp_train.MSZoning == 'FV', 'MSZoning'] = 'RL'  

exp_test.loc[exp_test.MSZoning == 'FV', 'MSZoning'] = 'RL'



# Smoothing out the levels or irregularity

exp_train.loc[exp_train.LotShape == 'IR2', 'LotShape'] = 'IR3'

exp_test.loc[exp_test.LotShape == 'IR2', 'LotShape'] = 'IR3'



# making foundations simpler

#exp_train.loc[exp_train.Foundation == 'Stone', 'Foundation'] = 'PConc' # Similar coefficients, but not similar categories

#exp_test.loc[exp_test.Foundation == 'Stone', 'Foundation'] = 'PConc'

#exp_train.loc[exp_train.Foundation == 'Wood', 'Foundation'] = 'PConc'





exp_train = pd.get_dummies(exp_train, drop_first=True)

exp_test = pd.get_dummies(exp_test, drop_first=True)

del exp_train['Foundation_Wood']



print('Lasso experiment')

lasso_pred = lasso_experiment(exp_train, y_train, exp_test, y_test, kfolds)

print('\n')

print('Tree experiment')

tree_pred = tree_experiment(exp_train, y_train, exp_test, y_test, kfolds)
exp_train = X_train[['GarageQual']].copy()

exp_test = X_test[['GarageQual']].copy()



to_num = {'NoGrg': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}



exp_train['GarageQual'] = exp_train['GarageQual'].map(to_num).astype(int)

exp_test['GarageQual'] = exp_test['GarageQual'].map(to_num).astype(int)



print('Tree experiment')

tree_pred = tree_experiment(exp_train, y_train, exp_test, y_test, kfolds)
exp_train = X_train[['GarageQual']].copy()

exp_test = X_test[['GarageQual']].copy()



to_num = {'NoGrg': 0, 'Po': 2, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 4}



exp_train['GarageQual'] = exp_train['GarageQual'].map(to_num).astype(int)

exp_test['GarageQual'] = exp_test['GarageQual'].map(to_num).astype(int)



print('Tree experiment')

tree_pred = tree_experiment(exp_train, y_train, exp_test, y_test, kfolds)
g = sns.FacetGrid(df_train, hue="GarageCars", height=8)

g.map(plt.scatter, "GarageArea", "target", edgecolor="w")

g.add_legend()
exp_train = X_train[['GarageArea']].copy()

exp_test = X_test[['GarageArea']].copy()



print('OLS experiment')  # lasso would give the same

ols_pred, ols_i_pred = OLS_experiment(exp_train, y_train, exp_test, y_test)

print('\n')

print('Tree experiment')

tree_pred = tree_experiment(exp_train, y_train, exp_test, y_test, kfolds)



plot_predictions(y_test, ols_pred, ols_pred, tree_pred)  
exp_train = X_train[['GarageCars']].copy()

exp_test = X_test[['GarageCars']].copy()



print('OLS experiment')

ols_pred, ols_i_pred = OLS_experiment(exp_train, y_train, exp_test, y_test)

print('\n')

print('Tree experiment')

tree_pred = tree_experiment(exp_train, y_train, exp_test, y_test, kfolds)



plot_predictions(y_test, ols_pred, ols_pred, tree_pred)
exp_train = X_train[['GarageCars']].copy()

exp_test = X_test[['GarageCars']].copy()



exp_train['GarageCars'] = exp_train['GarageCars'].astype(str)

exp_test['GarageCars'] = exp_test['GarageCars'].astype(str)



exp_train = pd.get_dummies(exp_train)

exp_test = pd.get_dummies(exp_test)



del exp_train['GarageCars_4']



print('OLS experiment')

ols_pred, ols_i_pred = OLS_experiment(exp_train, y_train, exp_test, y_test)

print('\n')

print('Tree experiment')

tree_pred = tree_experiment(exp_train, y_train, exp_test, y_test, kfolds)



plot_predictions(y_test, ols_pred, ols_pred, tree_pred)
df_train.Neighborhood.value_counts()
exp_train = X_train[['Neighborhood']].copy()

exp_test = X_test[['Neighborhood']].copy()

# getting the frequency in the training set only, you can also use the full set if it is available

freq_train = exp_train.groupby('Neighborhood').size() / len(exp_train) 



exp_train['Enc_Neigh'] = exp_train['Neighborhood'].map(freq_train)

exp_test['Enc_Neigh'] = exp_test['Neighborhood'].map(freq_train)



exp_train.sample(10)
from sklearn.feature_extraction import FeatureHasher



m = int(len(set(df_train.Neighborhood)))  # number of unique neighboorhoods



h = FeatureHasher(n_features=m, input_type='string')



f = h.transform(df_train['Neighborhood'])



f
f.toarray()
m = int(len(set(df_train.Neighborhood))/2)

h = FeatureHasher(n_features=m, input_type='string')

f = h.transform(df_train['Neighborhood'])

f
f.toarray()
exp_train = X_train[['GrLivArea', 'Neighborhood']].copy()

exp_test = X_test[['GrLivArea', 'Neighborhood']].copy()



avg_area = exp_train.groupby('Neighborhood', as_index=False).mean()



conversion = dict(zip(avg_area.Neighborhood, avg_area.GrLivArea))



exp_train['Neigh_avg_area'] = exp_train['Neighborhood'].map(conversion)

exp_test['Neigh_avg_area'] = exp_test['Neighborhood'].map(conversion)



exp_test.sample(10)