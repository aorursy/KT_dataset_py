import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats



from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.base import TransformerMixin



import os



df_train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

df_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
print(df_train.shape)

df_train.head()
print(df_test.shape)

df_test.head()
#Info on our target variable

df_train.SalePrice.describe()
# function to check distribution



def skew_distribution(data, col='SalePrice'):

    fig, ax1 = plt.subplots()

    sns.distplot(data[col], ax=ax1, fit=stats.norm)

    (mu, sigma) = stats.norm.fit(data[col])

    ax1.set(title='Normal distribution ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma))



    fig, ax2 = plt.subplots()

    stats.probplot(data[col], plot=plt)



    print('The {} skewness is {:.2f}'.format(col, stats.skew(data[col])))
# distribution of the Price and fit of normal distribution

skew_distribution(df_train, 'SalePrice')
df_train['SalePrice'] = np.log1p(df_train['SalePrice'])



skew_distribution(df_train, 'SalePrice')
#Finding the correlations in numeric features

corr = df_train.corr()   # or df_train[num_columns].corr()

top_corr_feat = corr['SalePrice'].sort_values(ascending=False)[:25]

print(top_corr_feat)
# Most correlated variables

threshold = 0.51

top_corr = corr.index[np.abs(corr["SalePrice"]) > threshold]



plt.figure(figsize=(10,8))

sns.heatmap(df_train[top_corr].corr(),annot=True,cmap="RdBu_r")
# Inspect numeric / categorical correlated features

for col in top_corr_feat.index[:15]:

    print('{} - unique values: {} - mean: {:.2f}'.format(col, df_train[col].unique()[:5], np.mean(df_train[col])))
# we prefer select non categorical values for a scatter matrix plot

cols = 'SalePrice GrLivArea GarageArea TotalBsmtSF YearBuilt 1stFlrSF MasVnrArea TotRmsAbvGrd'.split()



with plt.rc_context(rc={'font.size':14}): 

    fig, ax = plt.subplots(figsize=(16,13), tight_layout=True)    

    pd.plotting.scatter_matrix(df_train[cols], ax=ax, diagonal='kde', alpha=0.8)
cut_area = 4600



fig, ax = plt.subplots(figsize=(8,5))

ax.scatter(df_train['SalePrice'], df_train['GrLivArea'], s=18)

ax.set(xlabel='SalePrice', ylabel='GrLivArea')

ax.axhline(cut_area, ls='--', lw=2.5, c='green', alpha=0.5)   

ax.grid()
# remove points in the SalePrice - GrLivArea scatter plot

print('size of train dataset {}'.format(df_train.shape))

df_train = df_train.loc[df_train['GrLivArea'] < cut_area]

print('size of train dataset {} after removing outliers'.format(df_train.shape))
# Let's dig now into some of the most interpretable correlated features

decades = (df_train['YearBuilt'] // 10) * 10   # construct decade construction

decades.name = 'Decades'



with plt.rc_context(rc={'font.size':14}): 

    fig, ((ax1,ax2),(ax3,ax4), (ax5,ax6)) = plt.subplots(3,2, figsize=(20,12), tight_layout=True)

    

    sns.violinplot(x=df_train['OverallQual'], y=df_train['SalePrice'], ax=ax1)

    sns.violinplot(x=df_train['GarageCars'], y=df_train['SalePrice'], ax=ax2)

    sns.violinplot(x=df_train['FullBath'], y=df_train['SalePrice'], ax=ax3)

    sns.violinplot(x=df_train['Fireplaces'], y=df_train['SalePrice'], ax=ax4)

    sns.violinplot(x=df_train['TotRmsAbvGrd'], y=df_train['SalePrice'], ax=ax5)

    sns.violinplot(x=decades, y=df_train['SalePrice'], ax=ax6)
months = 'Jan. Feb. March April May June July Aug. Sept. Oct. Nov. Dec.'.split()

with plt.rc_context(rc={'font.size':14}): 

    fig, ax = plt.subplots(figsize=(8,4))

    df_train.groupby('MoSold')['SalePrice'].count().plot(kind='bar', alpha=0.3, ax=ax)

    ax.set(xlabel='Month Sold', ylabel='Number of houses sold / month', 

           xticklabels=months)

    ax.tick_params(axis='x', rotation=15)
fig, ax = plt.subplots(figsize=(10,4))

sns.violinplot(x=df_train['MoSold'], y=df_train['SalePrice'], ax=ax)

_ = ax.set(xticklabels=months)
# We don't need the Id column so we save it for final submission

df_train_id = df_train['Id']

df_test_id = df_test['Id']



df_train.drop("Id", axis=1, inplace=True)

df_test.drop("Id", axis=1, inplace=True)
# same transformation to the train / test datasets to avoid irregularities

size_train = len(df_train.index)

size_test = len(df_test.index)



df_tot = pd.concat([df_train, df_test], sort=False).reset_index(drop=True)

df_tot.drop(['SalePrice'], axis=1, inplace=True)



y_train = df_train['SalePrice'].values
df_na = (df_tot.isnull().sum()) / len(df_tot) * 100

df_na = df_na.drop(df_na[df_na==0].index).sort_values(ascending=False)

df_na.head(15)
with plt.rc_context(rc={'font.size':14}): 

    fig, ax = plt.subplots(figsize=(16, 6))



    sns.barplot(df_na.index, df_na, palette="pastel", ax=ax)

    ax.set(xlabel='Features', ylabel='Missing values percentages')

    ax.tick_params(axis='x', rotation=55)
# According to the data description

for col in 'PoolQC MiscFeature Alley Fence FireplaceQu'.split():

    df_tot[col].fillna('None', inplace=True)
# Get the LotFrontage from its median values from the Neighborhood

print(df_tot.groupby("Neighborhood")["LotFrontage"].agg(np.median).head())



df_tot["LotFrontage"] = df_tot.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
# Inspect Garage properties columns 

for col in df_tot.columns:

    if col.startswith('Garage'):

        print('{} - unique values: {}'.format(col, df_tot[col].unique()[:5]))
# Replace Garage categorical values by None

for col in 'GarageType GarageFinish GarageQual GarageCond'.split():

    df_tot[col].fillna('None', inplace=True)

    

# Replace Garage numeric values by 0

for col in 'GarageYrBlt GarageCars GarageArea'.split():

    df_tot[col].fillna(0, inplace=True)
# Inspect Basement properties columns 

for col in df_tot.columns:

    if 'Bsmt' in col:

        print('{} - unique values: {}'.format(col, df_tot[col].unique()[:5]))
# Same replacements (that Garage columns)

for col in 'BsmtQual BsmtCond BsmtExposure BsmtFinType1 BsmtFinType2'.split():

    df_tot[col].fillna('None', inplace=True)

    

# Replace numeric values by 0

for col in 'BsmtFinSF1 BsmtFinSF2 BsmtUnfSF TotalBsmtSF BsmtFullBath BsmtHalfBath'.split():

    df_tot[col].fillna(0, inplace=True)
df_tot['MasVnrArea'].fillna(0, inplace=True)

df_tot['MasVnrType'].fillna('None', inplace=True)
# The most frequent value is RL

df_tot['MSZoning'].value_counts() / len(df_tot) * 100
df_tot['MSZoning'].fillna('RL', inplace=True)
print(df_tot['Utilities'].value_counts())



# Since the values are almost only AllPub (except 1 line) this column is useless for SalePrice prediction

df_tot.drop(columns='Utilities', inplace=True)
print(df_tot['Functional'].value_counts() / len(df_tot) * 100)

df_tot['Functional'].fillna('Typ', inplace=True)
print('{} missing values from {} column'.format(df_tot['Exterior1st'].isnull().sum(), 'Exterior1st'))



# Fill in with the most common value --> 'Vinyl1Sd'

df_tot['Exterior1st'].fillna(df_tot['Exterior1st'].mode()[0], inplace=True)

print('{} missing values from {} column'.format(df_tot['Exterior2nd'].isnull().sum(), 'Exterior2nd'))



# Fill in with the most common value

df_tot['Exterior2nd'].fillna(df_tot['Exterior2nd'].mode()[0], inplace=True)
# Fill in with the most common value

df_tot['KitchenQual'].fillna(df_tot['KitchenQual'].mode()[0], inplace=True)

df_tot['Electrical'].fillna(df_tot['Electrical'].mode()[0], inplace=True)

df_tot['SaleType'].fillna(df_tot['SaleType'].mode()[0], inplace=True)

df_tot['MSSubClass'].fillna('None', inplace=True)
df_na = (df_tot.isnull().sum()) / len(df_tot) * 100

df_na = df_na.drop(df_na[df_na==0].index).sort_values(ascending=False)

df_na.head()
num_cols = df_tot.select_dtypes(exclude='object').columns

print('{} Numeric columns \n{}'.format(len(num_cols), num_cols))



categ_cols = df_tot.select_dtypes(include='object').columns

print('\n{} Categorical columns \n{}'.format(len(categ_cols), categ_cols))
# Basic cleaning of the data

# numeric cols --> mean value and non-numeric cols --> most frequent value

# Inspired from the nice code by 'sveitser' at http://stackoverflow.com/a/25562948



class DataImputer(TransformerMixin):

    """First data cleaning operation."""

    

    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0] if X[c].dtype == np.dtype('O') 

                               else X[c].median() for c in X], index=X.columns)

        return self

    

    def transform(self, X, y=None):

        return X.fillna(self.fill)



### same transformation to the train / test datasets to avoid irregularities

#tot_X = df_train.append(df_test, sort=False)

#tot_X_imputed = DataImputer().fit_transform(tot_X)



#le = LabelEncoder()

#for feat in object_cols:

#    tot_X_imputed[feat] = le.fit_transform(tot_X_imputed[feat])
# Inspect numeric / categorical correlated features

for col in top_corr_feat.index[:25]:

    print('{} - unique values: {} - mean: {:.2f}'.format(col, df_train[col].unique()[:5], np.mean(df_train[col])))
cols = 'MSSubClass OverallQual GarageCars YrSold MoSold Fireplaces HalfBath'.split()



for col in cols:

    df_tot[col] = df_tot[col].astype(str)
cols_le = ['BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'BsmtQual', 'BsmtCond', 

           'GarageQual', 'GarageCond', 'GarageFinish', 'GarageType', 

           'FireplaceQu', 'ExterQual', 'ExterCond', 

           'HeatingQC', 'PoolQC', 'KitchenQual', 

           'Functional', 'Fence', 'LandSlope',

           'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 

           'MSSubClass', 'OverallCond', 'GarageCars', 'YrSold', 'MoSold', 'Fireplaces', 'HalfBath'] 



le = LabelEncoder() 

for col in cols_le:

    df_tot[col] = le.fit_transform(df_tot[col])
# Combine total square foot area

df_tot['TotalSF'] = df_tot['TotalBsmtSF'] + df_tot['1stFlrSF'] + df_tot['2ndFlrSF'] + df_tot['GrLivArea'] + df_tot['GarageArea']



# Combine the bathrooms

df_tot['Bathrooms'] = df_tot['FullBath'] + df_tot['HalfBath']* 0.5 



# Combine Year built, Garage Year Built and Year Remod 

# (with a coeff 0.5 since it's less correlated to Year Built than the Garage year built).

df_tot['YearMean'] = df_tot['YearBuilt'] + df_tot['YearRemodAdd'] * 0.5 + df_tot['GarageYrBlt']
# Compute the skew of all numerical features

new_num_cols = df_tot.select_dtypes(exclude='object').columns



feat_skews = df_tot[new_num_cols].apply(stats.skew).sort_values(ascending=False)

skew_df = pd.DataFrame({'skewness' :feat_skews})

skew_df.head()
# Check MiscVal distribution before transformation

skew_distribution(df_tot, 'MiscVal')
from scipy.special import boxcox1p



cols = skew_df[np.abs(skew_df['skewness']) > 0.8].index

print('We use the boxcop1p transformation on {} numeric features'.format(len(cols)))

for col in cols:

    df_tot[col] = boxcox1p(df_tot[col], 0.15)
skew_distribution(df_tot, 'MiscVal')
print(df_tot.shape, 'before dummy categories')

df_tot = pd.get_dummies(df_tot)

print(df_tot.shape, 'after dummy categories')
df_train = df_tot[:size_train]

df_test = df_tot[size_train:]
from warnings import filterwarnings

filterwarnings('ignore')



from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold, StratifiedShuffleSplit, StratifiedKFold

from sklearn.preprocessing import RobustScaler

from sklearn.pipeline import make_pipeline

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

from mlxtend.regressor import StackingRegressor



from sklearn.linear_model import Ridge, Lasso, ElasticNet

from sklearn.svm import SVR

from lightgbm import LGBMRegressor

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor

from xgboost import XGBRegressor
def rmse(ypred, ytrue):

    """

    Compute the RMSE between true labels and predictions.

    """

    return np.sqrt(mean_squared_error(ypred, ytrue))
#train and test (for validation) from the train dataset

Xtrain, Xtest, ytrain, ytest = train_test_split(df_train, y_train, shuffle=True, 

                                                test_size=0.3, random_state=28)



scale = RobustScaler()

kf = KFold(n_splits=5, shuffle=True, random_state=28)
# Function to compute the RMSE

def result_GridCV(name, model):

    """

    Display the results on the RMSE after the GridSearchCV.

    """

    model.fit(Xtrain, ytrain)

    ytrain_pred = model.predict(Xtrain)

    ytest_pred = model.predict(Xtest)

    

    rmse_train = rmse(ytrain, ytrain_pred)

    rmse_test = rmse(ytest, ytest_pred)



    print("{} - TRAIN score: {:.4f}" .format(name, rmse_train))

    print("{} - TEST score: {:.4f}" .format(name, rmse_test))
score = 'neg_mean_squared_error'
# First a fast GridSearchCV to find the optimal alpha

param_grid = {'alpha': np.logspace(0, 2, 50)}



grid_ridge = GridSearchCV(Ridge(), param_grid, cv=40, scoring=score, verbose=0, n_jobs=-1)  

grid_ridge.fit(Xtrain, ytrain)

print(grid_ridge.best_params_)
# Compute RMSE with best Ridge

ridge = make_pipeline(RobustScaler(), grid_ridge.best_estimator_)

result_GridCV('Ridge', ridge)
# First a fast GridSearchCV to find the optimal alpha

param_grid = {'alpha': np.logspace(-4, 0, 30)}

score = 'neg_mean_squared_error'



grid_lasso = GridSearchCV(Lasso(), param_grid, cv=30, scoring=score, verbose=0, n_jobs=-1)       

grid_lasso.fit(Xtrain, ytrain)

print(grid_lasso.best_params_)
print('{}/{} coefficients not null with the Lasso method'.format((grid_lasso.best_estimator_.coef_ !=0).sum(), len(df_train.columns)))

#df_train.columns[grid_lasso.best_estimator_.coef_ !=0]
# Compute RMSE with best Lasso

lasso = make_pipeline(RobustScaler(), grid_lasso.best_estimator_)

result_GridCV('Lasso', lasso)
grid_net = {'alpha': np.logspace(-4, -3, 20),

            'l1_ratio': [0.5,0.55,0.6,0.65]}

score = 'neg_mean_squared_error'



grid_net = GridSearchCV(ElasticNet(), grid_net, cv=20, scoring=score, verbose=0, n_jobs=-1)       

grid_net.fit(Xtrain, ytrain)

print(grid_net.best_params_)
# Compute RMSE with best Ridge

enet = make_pipeline(RobustScaler(), grid_net.best_estimator_)

result_GridCV('ElasticNET', enet)
# SVR kernel - 

#params_svr = {'kernel': ['rbf', 'poly'], 'gamma': np.logspace(-3,2,10),

#                     'C': [1, 10, 100]},

#score = 'neg_mean_squared_error'



#grid_svr = GridSearchCV(SVR(), params_svr, scoring=score)

#grid_svr.fit(df_train, y_train)

#print(grid_svr.best_params_)



# Compute RMSE with best SVR

#svr_score = rmse_CV(grid_svr.best_estimator_)

#print("SVR score: {:.4f} +/- {:.4f}\n" .format(svr_score.mean(), svr_score.std()))
#grid_forr = {'n_estimators': [1000,1500,2000], 'max_features': ['auto'],

#             'min_samples_leaf': [4,8,10,20], 'min_samples_split': [4,8,16], 

#             'max_depth': [4,8,10,20]

#            }

#score = 'neg_mean_squared_error'



#grid_forrest = GridSearchCV(RandomForestRegressor(), grid_forr, cv=5, scoring=score, verbose=1, n_jobs=-1)

#grid_forrest.fit(df_train, y_train)

#print(grid_forrest.best_params_)
# Because the above GridSeachCV took several minutes, we dit get the optimal hyperparameters to run the score

rfc = RandomForestRegressor(n_estimators=2000, n_jobs=-1, random_state=28,

                           max_features='auto', min_samples_leaf=10, 

                           min_samples_split=8, max_depth=20, 

                           bootstrap=True)



result_GridCV('RandomForrest Regr.', rfc)
top_n_feat = 24



feat_imp = pd.Series(rfc.feature_importances_, index=df_train.columns)[:top_n_feat]     # random forrest regr.

coefs = pd.Series(grid_lasso.best_estimator_.coef_, index=df_train.columns)     # lasso 

feat_coeff = pd.concat([coefs.sort_values().head(top_n_feat//2), coefs.sort_values().tail(top_n_feat//2)])



colors, alpha = sns.color_palette('bright'), 0.6

with plt.rc_context(rc={'font.size':14}): 

    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(16,6), tight_layout=True)

    feat_imp.sort_values().plot(kind='barh', color=colors, ax=ax1, alpha=alpha) 

    #feat_coeff.reindex(feat_imp.sort_values().index).plot(kind='barh', color=colors, ax=ax2, alpha=alpha)  # if same cols. as forrest

    feat_coeff.plot(kind='barh', color=colors, ax=ax2, alpha=alpha) 

    

    ax1.set(xlabel='Feature importances', title='Random Forrest Regression - feature importances')    

    ax2.set(xlabel='Lasso coeff.', title='Lasso - feature coeff. weights')

    for ax in [ax1,]:

        ax.set_xscale('log')
gboost = GradientBoostingRegressor(n_estimators=800, learning_rate=0.05,

                                   max_depth=4, max_features='auto',

                                   min_samples_leaf=10, min_samples_split=10, 

                                   loss='huber', random_state=28)

result_GridCV('GBoost', gboost)
#params_lgb = {'num_leaves': [4,8,32], 'max_depth': [4,8], 'reg_alpha': [0,0.2], 'reg_lambda': [0.5,1], 

#              'n_estimators': [750], 'learning_rate': [0.02,0.04,0.08]}



#grid_lgb = GridSearchCV(LGBMRegressor(n_jobs=-1), params_lgb, cv=5, scoring=score, verbose=1, n_jobs=-1)

#grid_lgb.fit(Xtrain, ytrain)

#print(grid_lgb.best_params_)
params_lgb = {'learning_rate': 0.04, 'max_depth': 4, 'n_estimators': 750, 'num_leaves': 4, 'reg_alpha': 0, 'reg_lambda': 1}

lgb = LGBMRegressor(n_jobs=-1, **params_lgb)

result_GridCV('LBGMRegressor', lgb)
#params_xgb = {'gamma': [0.03,0.04,0.05,], 'max_depth': [3,4,8], 'reg_alpha': [0,], 'reg_lambda': [1,], 

#              'n_estimators': [500,1000,1500], 'learning_rate': [0.02,0.04,0.08], 'subsample': [0.5,], 

#              'min_child_weight': [1,2,4,8]}



#params_xgb = {'gamma': [0.025,0.03,0.035,], 'max_depth': [3,4,8]}



#grid_xgb = GridSearchCV(XGBRegressor(n_jobs=-1, random_state=28, n_estimators=1500, objective='reg:squarederror'), 

#                        params_xgb, cv=3, scoring=score, verbose=1, n_jobs=-1)

#grid_xgb.fit(Xtrain, ytrain)

#print('XGBoosting best RMSE: %f using %s\n' % (-grid_xgb.best_score_, grid_xgb.best_params_))
# Because the above GridSeachCV took quite a few time, we dit get the optimal hyperparameters to run the score.

xgb = XGBRegressor(base_score=0.5, colsample_bylevel=1,

                   colsample_bynode=1, colsample_bytree=0.5, 

                   learning_rate=0.02, gamma=0.025,

                   max_depth=4, n_estimators=1500, min_child_weight=2,

                   nthread=-1, reg_alpha=0., reg_lambda=1, subsample=0.5, 

                   objective='reg:squarederror', random_state=28)

                    

result_GridCV('XGBoost', xgb)



# Avoid overfitting with early-stops method

#pred = model.predict(Xtest, ntree_limit=model.best_ntree_limit)
eval_set = [(Xtrain, ytrain), (Xtest, ytest)]

xgb.fit(Xtrain, ytrain, early_stopping_rounds=100, eval_metric='rmse', eval_set=eval_set, verbose=None)
# Check evolution of the loss function



results = xgb.evals_result()

epochs = len(results['validation_0']['rmse'])

x_axis = range(0, epochs)



fig, ax = plt.subplots(figsize=(8,6))

ax.plot(x_axis, results['validation_0']['rmse'], label='Train')

ax.plot(x_axis, results['validation_1']['rmse'], label='Test')

ax.set(title='XGBoost Regression Error', ylabel='RMSE', ylim=([0,1]))

ax.legend()
class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):

    def __init__(self, models):

        self.models = models

        

    # clones of the actual model 

    def fit(self, X, y):

        self.models_ = [clone(x) for x in self.models]

        

        # Training of the clone model

        for model in self.models_:

            model.fit(X, y)

        return self

    

    # prediction and average of the trained clone models

    def predict(self, X):

        predictions = np.column_stack([model.predict(X) for model in self.models_])

        return np.mean(predictions, axis=1)   
averaged_models = AveragingModels(models=(ridge, lasso, enet, gboost, lgb, xgb))

result_GridCV('Averaged models', averaged_models)
stack = StackingRegressor(regressors=[ridge, lasso, enet, averaged_models], 

                          meta_regressor=xgb, 

                          use_features_in_secondary=True)



# Training the stacking regr.

result_GridCV('Stack regressors', stack)
# We can try to average the models

pred_net = enet.predict(Xtest)

pred_averaged = averaged_models.predict(Xtest)

pred_stack = stack.predict(Xtest)
def weighted_average(pred1, pred2, pred3, weights=[0.5, 0.4,0.1]):

    """

    Compute the average with weights (a+b+c=1) of the top 3 predictions.

    """

    

    a, b, c = weights

    return pred1 * a + pred2 * b + pred3 * c
pred_tot = weighted_average(pred_net, pred_averaged, pred_stack, [0.5,0.4,0.1])

RMSE = rmse(ytest, pred_tot)

print('RMSE of average models: {:.4f}'.format(RMSE))
# Check what is the optimal combinaison of weights

delta = 0.1

c = 0.05

for a,b in zip(np.arange(0,1-c,delta), 1-c - np.arange(0,1-c,delta)):

    pred = weighted_average(pred_net, pred_averaged, pred_stack, [a,b,c])

    RMSE = rmse(ytest, pred)

    print('RMSE (a={:.2f}, b={:.2f}) : {:.4f}'.format(a, b, RMSE))
pred_tot = weighted_average(pred_net, pred_averaged, pred_stack, [0.05,0.90,0.05])

RMSE = rmse(ytest, pred_tot)

print('RMSE of average models: {:.4f}'.format(RMSE))
# Final prediction on the SalePrice



enet.fit(df_train, y_train)

averaged_models.fit(df_train, y_train)

stack.fit(df_train, y_train)
enet_final = np.expm1(enet.predict(df_test))

average_final = np.expm1(averaged_models.predict(df_test))

stack_final = np.expm1(stack.predict(df_test))
pred_final = weighted_average(enet_final, average_final, stack_final, [0.05,0.9,0.05])
df_sub = pd.DataFrame({'Id': df_test_id, 'SalePrice': pred_final})

print(df_sub.head())

df_sub.to_csv('submission.csv',index=False)
from keras.models import Sequential

from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization

from keras.callbacks import ModelCheckpoint

from keras.optimizers import Adam, RMSprop, Adagrad, Nadam, SGD, Adadelta, Adamax

from keras.constraints import maxnorm

from keras.regularizers import l2

np.random.seed(28)    # for reproductibily
#train, validation and test from the train dataset

Xtrain, Xtest, ytrain, ytest = train_test_split(df_train, y_train, shuffle=True, 

                                                test_size=0.3, random_state=28)



Xval, Xtest, yval, ytest = train_test_split(Xtest, ytest, test_size=0.5, 

                                            shuffle=True, random_state=28)
# function to plot the evolution of loss (use of it later)

def loss_plot(info_model, ymax=5):

    train_loss = info_model.history['loss']

    val_loss = info_model.history['val_loss']    

    epochs = range(len(train_loss))

    

    with plt.rc_context({'font.size':13}):

        fig, ax = plt.subplots(figsize=(8,5))

        ax.plot(epochs, train_loss, label='Train')

        ax.plot(epochs, val_loss, label='Validation')

        ax.set(xlabel='Epochs', ylabel='Loss - RMSE', 

               title='Model loss', ylim=(0,ymax))

        ax.legend()       
model1 = Sequential()



# First layer

model1.add(Dense(128, input_dim=Xtrain.shape[1], activation='relu'))



# Hidden layers

model1.add(Dense(256, activation='relu'))

model1.add(Dense(256, activation='relu'))



# Final layer

model1.add(Dense(1, activation='linear'))
model1.summary()
# Optimizer

opt1 = Adam(lr=0.001, beta_1=0.95)    # default values



# Since the metrics RMSE does not exists in Keras, we need to implement it

import keras.backend as K



def rmse_keras(y_true, y_pred):

        return K.sqrt(K.mean(K.square(y_pred - y_true)))
# Let's compile our model

model1.compile(optimizer=opt1, loss=rmse_keras) 
def init_callback(model_id='1', verbose=0):

    """

    Create a callback per model id. 

    """



    filepath = 'weights{}.best.hdf5'.format(model_id)

    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=verbose, 

                                 save_best_only=True, mode='auto')

    return [checkpoint]



# Callback MODEL 1

callbacks1 = init_callback(model_id='1', verbose=0)
info_model1 = model1.fit(Xtrain, ytrain, epochs=2000, 

                       batch_size=32, verbose=0,         # default values

                       validation_data=(Xval,yval), 

                       use_multiprocessing=True, 

                       shuffle=True, 

                       callbacks=callbacks1)
# Evolution of the training

loss_plot(info_model1, 0.8)
# Load weights file of the best model :



#val_loss = info_model1.history['val_loss']

#best_weight = np.argmin(val_loss) + 1

#best_val_loss = np.min(val_loss)

#file_w = 'weights-{:03d}--{:.5f}.hdf5'.format(best_weight, best_val_loss) # best checkpoint



def load_best_weights(model, model_id='1', opt='adam'):

    """

    Load the model with best weight during the training and compile it.

    """

    model.load_weights('weights{}.best.hdf5'.format(model_id))

    model.compile(loss=rmse_keras, optimizer=opt)

    

    return model
# Load best model

model1 = load_best_weights(model1, model_id='1', opt=opt1)

loss1 = model1.evaluate(Xtest, ytest)

print('MODEL 1  --  RMSE (test data): {:.4f}'.format(loss1))
# Optimizer

opt2 = Adam(lr=0.001, beta_1=0.9)    # higher learning rate (dropping)
# kernel contraints

weight_constraint = maxnorm(3)

activation = 'relu'

weight_init = 'normal'





model2 = Sequential()

# First layer

model2.add(Dropout(0.2, input_shape=(Xtrain.shape[1],))) 

model2.add(Dense(128, activation=activation, kernel_constraint=weight_constraint, 

                 kernel_initializer=weight_init, ))



# Hidden layers

model2.add(Dense(256, activation=activation, kernel_constraint=weight_constraint, 

                 kernel_initializer=weight_init))

model2.add(Dropout(0.3))

model2.add(Dense(256, activation=activation, kernel_constraint=weight_constraint, 

                kernel_initializer=weight_init))

model2.add(Dropout(0.3))



# Final layer

model2.add(Dense(1, kernel_initializer='normal', activation='linear'))
# Let's compile our model

model2.compile(optimizer=opt2, loss=rmse_keras) 



# Callback MODEL 2

callbacks2= init_callback(model_id='2', verbose=0)
info_model2 = model2.fit(Xtrain, ytrain, epochs=2000, 

                       batch_size=32, verbose=0,         # default values

                       validation_data=(Xval,yval), 

                       use_multiprocessing=True, 

                       shuffle=True, 

                       callbacks=callbacks2)
# Evolution of the training

loss_plot(info_model2, 1)
# Load best model

model2 = load_best_weights(model2, model_id='2', opt=opt2)

loss2 = model2.evaluate(Xtest, ytest)

print('MODEL 2  --  RMSE (test data): {:.4f}'.format(loss2))
from sklearn.model_selection import GridSearchCV

from keras.wrappers.scikit_learn import KerasRegressor



def create_mlp(lr=0.001, beta1=0.9, beta2=0.999, 

              activation='relu',

              dropout_input=0.2, 

              dropout_rate=0.3,

              weight_constraint=3,

              weight_init='normal',

              input_neurons=128,

              hidden_neurons=256):

    """

    Function to create a Keras model that can be used with sklearn methods.

    """

    

    model = Sequential()

    # First layer

    model.add(Dropout(dropout_input, input_shape=(Xtrain.shape[1],))) 

    model.add(Dense(input_neurons, activation=activation, 

                    kernel_constraint=maxnorm(weight_constraint), 

                    kernel_initializer=weight_init, 

                    ))   

    # Hidden layers

    model.add(Dense(hidden_neurons, activation=activation, 

                    kernel_constraint=maxnorm(weight_constraint), 

                    kernel_initializer=weight_init))

    model.add(Dropout(dropout_rate))

    model.add(Dense(hidden_neurons, activation=activation, 

                    kernel_constraint=maxnorm(weight_constraint), 

                    kernel_initializer=weight_init)) 

    model.add(Dropout(dropout_rate))   

    # Final layer

    model.add(Dense(1, kernel_initializer=weight_init, activation='linear'))



    # Compile model

    optimizer = Adam(learning_rate=lr, beta_1=beta1, beta_2=beta2)

    model.compile(loss=rmse_keras, optimizer=optimizer)

    return model
# Scoring for the GridSearchCV

score = 'neg_mean_squared_error'



fit_grid = False         # avoid the notebook to take 1 day to run
cv = 3  # only cross-val=3 for now since it takes time 

model = KerasRegressor(build_fn=create_mlp, epochs=1500, batch_size=32, 

                       verbose=1, shuffle=True, use_multiprocessing=True)



# Grid search params 

lr = [0.001, 0.01, 0.1]

beta1 = [0., 0.4, 0.9, 0.95]

param_grid = dict(lr=lr, beta1=beta1)



if fit_grid:

    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=score, n_jobs=-1, cv=cv, verbose=0)   

    grid_result = grid.fit(Xtrain, ytrain)

    print('Adam GridSearch best MSE: {:.4f} using {}'.format(-grid_result.best_score_, grid_result.best_params_))
cv = 3

model = KerasRegressor(build_fn=create_mlp, epochs=1500, batch_size=32, 

                       verbose=1, shuffle=True, use_multiprocessing=True)



# Grid search params 

weight_constraint = [2,3,4] 

dropout_rate = np.arange(0,0.5,0.1)  # hidden layers

param_grid = dict(weight_constraint=weight_constraint, 

                  dropout_rate=dropout_rate)



if fit_grid:

    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=score, n_jobs=-1, cv=cv, verbose=0)   

    grid_result = grid.fit(Xtrain, ytrain)

    print('MaxNorm constraint GridSearch best MSE: %f using %s\n' % (-grid_result.best_score_, grid_result.best_params_))
#means = grid_result.cv_results_['mean_test_score']

#stds = grid_result.cv_results_['std_test_score']

#params = grid_result.cv_results_['params']

#for mean, stdev, param in zip(means, stds, params):

#    print('{:.4f} +/- {:.4f} with: {}'.format(-mean, stdev, param))
cv = 3

model = KerasRegressor(build_fn=create_mlp, epochs=1500, batch_size=32, 

                       verbose=1, shuffle=True, use_multiprocessing=True)



# Grid search params 

weight_init = ['uniform', 'normal', 'zero', 'he_normal', 'he_uniform']

param_grid = dict(weight_init=weight_init)



if fit_grid:

    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=score, n_jobs=-1, cv=cv, verbose=0)   

    grid_result = grid.fit(Xtrain, ytrain)

    print('Weight initialization GridSearch best MSE: %f using %s\n' % (-grid_result.best_score_, grid_result.best_params_))
cv = 3

model = KerasRegressor(build_fn=create_mlp, epochs=1500, batch_size=32, 

                       verbose=1, shuffle=True, use_multiprocessing=True)



# Grid search params 

activation = ['softplus', 'softsign', 'relu', 'tanh', 'linear']

param_grid = dict(activation=activation)



if fit_grid:

    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=score, n_jobs=-1, cv=cv, verbose=0)   

    grid_result = grid.fit(Xtrain, ytrain)

    print('Activation functions GridSearch best MSE: %f using %s\n' % (-grid_result.best_score_, grid_result.best_params_))
cv = 3

model = KerasRegressor(build_fn=create_mlp, epochs=1500, batch_size=32, 

                       verbose=1, shuffle=True, use_multiprocessing=True)



# Grid search params 

dropout_input = [0.1,0.2,0.3]

param_grid = dict(dropout_input=dropout_input)



if fit_grid:

    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=score, n_jobs=-1, cv=cv, verbose=0)   

    grid_result = grid.fit(Xtrain, ytrain)

    print('Activation functions GridSearch best MSE: %f using %s\n' % (-grid_result.best_score_, grid_result.best_params_))
# Get the best model and then fit and select best weight

best_model = create_mlp(lr=0.001, beta1=0.95, 

                        weight_constraint=4, 

                        weight_init='uniform', 

                        activation='linear', 

                        dropout_input=0.1, 

                        dropout_rate=0.4)



# Callback best MODEL

callbacks = init_callback(model_id='0', verbose=0)



# Train

info = best_model.fit(Xtrain, ytrain, epochs=4000, 

                       batch_size=32, verbose=0,         

                       validation_data=(Xval,yval), 

                       use_multiprocessing=True, 

                       shuffle=True, 

                       callbacks=callbacks)
# Evolution of the training

loss_plot(info, 2)
# Load best model

best_model = load_best_weights(best_model, model_id='0', opt=Adam(0.001,0.95))

loss = best_model.evaluate(Xtest, ytest)

print('FINAL MODEL  --  RMSE (test data): {:.4f}'.format(loss))
# Get the best model and then fit and select best weight

best_model = create_mlp(lr=0.001, beta1=0.99, 

                        weight_constraint=4, 

                        weight_init='uniform', 

                        activation='linear', 

                        dropout_input=0.1, 

                        dropout_rate=0.4)



# Callback best MODEL

callbacks99 = init_callback(model_id='99', verbose=0)



# Train

info99 = best_model.fit(Xtrain, ytrain, epochs=4000, 

                       batch_size=32, verbose=0,         

                       validation_data=(Xval,yval), 

                       use_multiprocessing=True, 

                       shuffle=True, 

                       callbacks=callbacks99)
loss_plot(info99, 2)



# Load best model

best_model = load_best_weights(best_model, model_id='99', opt=Adam(0.001,0.99))

loss = best_model.evaluate(Xtest, ytest)

print('FINAL MODEL  --  RMSE (test data): {:.4f}'.format(loss))
#model.add(BatchNormalization())   # use_biais=False into Dense layer before
# Test to prediction



#best_model.fit(df_train, y_train)

#prediction = np.expm1(best_model.predict(df_test)).flatten()



#df_sub = pd.DataFrame({'Id': df_test_id, 'SalePrice': prediction})

#print(df_sub.head())

#df_sub.to_csv('submission.csv',index=False)