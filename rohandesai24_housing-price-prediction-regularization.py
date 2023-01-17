# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Read the data
X = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv', index_col='Id')
X_test_full = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv', index_col='Id')
print("Shape of training set:",format(X.shape))
print("Shape of test set:",format(X_test_full.shape))
X.describe()
num_X = X.select_dtypes(exclude='object')
num_X_cor = num_X.corr()
f,ax=plt.subplots(figsize=(20,2))
sns.heatmap(num_X_cor.sort_values(by=['SalePrice'], ascending=False).head(1), cmap='Blues')
plt.xticks(weight='bold')
plt.yticks(weight='bold', color='dodgerblue', rotation=0)
high_cor_num = num_X_cor['SalePrice'].sort_values(ascending=False).head(10).to_frame()
high_cor_num
f,ax=plt.subplots(figsize=(20,5))

nbr_count = X.Neighborhood.value_counts().to_frame().reset_index() #getting the number of houses for each neighborhood

nbr_plt = ax.barh(nbr_count.iloc[:,0], nbr_count.iloc[:,1], color=sns.color_palette('Blues',len(nbr_count)))

ax.invert_yaxis()
plt.yticks(weight='bold')
plt.xlabel('Count')
plt.title('Number of houses by Neighborhood')
plt.show()
f,ax=plt.subplots(figsize=(20,5))

lotshape_count = X.LotShape.value_counts().to_frame().reset_index() #getting the number of houses for each neighborhood

nbr_plt = ax.barh(lotshape_count.iloc[:,0], lotshape_count.iloc[:,1], color=sns.color_palette('Reds',len(nbr_count)))

ax.invert_yaxis()
plt.yticks(weight='bold')
plt.xlabel('Count')
plt.title('Number of houses by Lot Shape')
plt.show()
X.Utilities.value_counts()
nX = X.shape[0]
nX_test = X_test_full.shape[0]
y_train = X['SalePrice'].to_frame()
#Combine train and test sets
combined_df = pd.concat((X,X_test_full), sort=False).reset_index(drop=True)

#Drop the target "SalePrice" and Id columns
combined_df.drop(['SalePrice'], axis=1, inplace=True)
print(f"Total size is {combined_df.shape}")

nullpercentage = (combined_df.isnull().mean())*100
nullpercentage = nullpercentage.sort_values(ascending=False).to_frame()
nullpercentage.head()
def msv1(data, thresh=20, color='black', edgecolor='black', width=15, height=3):
    """
    SOURCE: https://www.kaggle.com/amiiiney/price-prediction-regularization-stacking
    """
    
    plt.figure(figsize=(width,height))
    percentage=(data.isnull().mean())*100
    percentage.sort_values(ascending=False).plot.bar(color=color, edgecolor=edgecolor)
    
    plt.axhline(y=thresh, color='r', linestyle='-')
    plt.title('Missing values percentage per column', fontsize=20, weight='bold' )
    plt.text(len(data.isnull().sum()/len(data))/1.7, thresh+12.5, f'Columns with more than {thresh}% missing values', fontsize=12, color='crimson',
         ha='left' ,va='top')
    plt.text(len(data.isnull().sum()/len(data))/1.7, thresh - 5, f'Columns with less than {thresh} missing values', fontsize=12, color='green',
         ha='left' ,va='top')
    plt.xlabel('Columns', size=15, weight='bold')
    plt.ylabel('Missing values percentage')
    plt.yticks(weight ='bold')
    
    return plt.show()
msv1(combined_df, color=sns.color_palette('Blues'))
combined_df1 = combined_df.dropna(thresh=len(combined_df)*0.8, axis=1) #dropping columns with more than 20% null values
print(f"We dropped {combined_df.shape[1]-combined_df1.shape[1]} features in the combined set")
combined_df1.select_dtypes(exclude='object').isnull().sum().sort_values(ascending=False).head(11)
def msv2(data, width=12, height=8, color=('silver', 'gold','lightgreen','skyblue','lightpink'), edgecolor='black'):
    """
    SOURCE: https://www.kaggle.com/amiiiney/price-prediction-regularization-stacking
    """
    fig, ax = plt.subplots(figsize=(width, height))

    allna = (data.isnull().sum() / len(data))*100
    tightout= 0.008*max(allna)
    allna = allna.drop(allna[allna == 0].index).sort_values().reset_index()
    mn= ax.barh(allna.iloc[:,0], allna.iloc[:,1], color=color, edgecolor=edgecolor)
    ax.set_title('Missing values percentage per column', fontsize=15, weight='bold' )
    ax.set_xlabel('Percentage', weight='bold', size=15)
    ax.set_ylabel('Features with missing values', weight='bold')
    plt.yticks(weight='bold')
    plt.xticks(weight='bold')
    for i in ax.patches:
        ax.text(i.get_width()+ tightout, i.get_y()+0.1, str(round((i.get_width()), 2))+'%',
            fontsize=10, fontweight='bold', color='grey')
    return plt.show()
msv2(combined_df1)
#LotFrontage has 16% missing values. Filling with median.
combined_df1['LotFrontage'] = combined_df1.LotFrontage.fillna(combined_df1.LotFrontage.median())
#Masonry Veneer Area having missing values would mean that there is no veneer. Hence we fill with 0.
combined_df1['MasVnrArea'] = combined_df1.MasVnrArea.fillna(0)
#No value for GarageYrBlt would mean no garage exists. But since this is a year value, we cannot fill it with 0s. Hence we fill with median.
combined_df1['GarageYrBlt']=combined_df1["GarageYrBlt"].fillna(1980)

combined_df1.shape
df_Cat = combined_df1.select_dtypes(include='object')
nan_cols = df_Cat.columns[df_Cat.isnull().any()]
NA_Cat= df_Cat[nan_cols]
NA_Cat.isnull().sum().sort_values(ascending=False)
cols_to_fill = ['SaleType','Exterior1st','Exterior2nd','KitchenQual',
                'Electrical','Utilities','Functional','MSZoning']

combined_df1[cols_to_fill] = combined_df1[cols_to_fill].fillna(method='ffill')
cols = combined_df1.columns
for col in cols:
    if combined_df1[col].dtype == 'object':
        combined_df1[col] = combined_df1[col].fillna('None')
    elif combined_df1[col].dtype != 'object':
        combined_df1[col] = combined_df1[col].fillna(0)
combined_df1.isnull().sum().sort_values(ascending=False).head() #checking whether all the null values have been dealt with
combined_df1.shape
combined_df1['TotalArea'] = combined_df1['TotalBsmtSF'] + combined_df1['1stFlrSF'] + combined_df1['2ndFlrSF'] + combined_df1['GrLivArea'] + combined_df1['GarageArea']
# Creating a new feature 'TotalArea' by adding up the area of all the floors and basement
combined_df1['MSSubClass'] = combined_df1['MSSubClass'].apply(str)
combined_df1['YrSold'] = combined_df1['YrSold'].astype(str)
combined_df_onehot = pd.get_dummies(combined_df1)
print(f"the shape of the original dataset {combined_df1.shape}")
print(f"the shape of the encoded dataset {combined_df_onehot.shape}")
print(f"We have {combined_df_onehot.shape[1]- combined_df1.shape[1]} new encoded features")
X_train = combined_df_onehot[:nX]   #nX is the number of rows in the original training set
test = combined_df_onehot[nX:]

print(X_train.shape, test.shape)
fig = plt.figure(figsize=(15,15))

ax1 = plt.subplot2grid((3,2),(0,0))
plt.scatter(x=X['GrLivArea'], y=y_train['SalePrice'], color=('yellowgreen'), alpha=0.5)
plt.axvline(x=4600, color='r', linestyle='-')
plt.title('Greater living Area - Price scatter plot', fontsize=15, weight='bold' )

ax1 = plt.subplot2grid((3,2),(0,1))
plt.scatter(x=X['TotalBsmtSF'], y=y_train['SalePrice'], color=('red'),alpha=0.5)
plt.axvline(x=5900, color='r', linestyle='-')
plt.title('Basement Area - Price scatter plot', fontsize=15, weight='bold' )

ax1 = plt.subplot2grid((3,2),(1,0))
plt.scatter(x=X['MasVnrArea'], y=y_train['SalePrice'], color=('blue'),alpha=0.5)
plt.axvline(x=1200, color='r', linestyle='-')
plt.title('Masonry Veneer Area - Price scatter plot', fontsize=15, weight='bold' )

ax1 = plt.subplot2grid((3,2),(1,1))
plt.scatter(x=X['1stFlrSF'], y=y_train['SalePrice'], color=('green'),alpha=0.5)
plt.axvline(x=4500, color='r', linestyle='-')
plt.title('1st Floor Area - Price scatter plot', fontsize=15, weight='bold' )

ax1 = plt.subplot2grid((3,2),(2,0))
plt.scatter(x=X['GarageArea'], y=y_train['SalePrice'], color=('orchid'),alpha=0.5)
plt.axvline(x=1300, color='r', linestyle='-')
plt.title('Garage Area - Price scatter plot', fontsize=15, weight='bold' )

ax1 = plt.subplot2grid((3,2),(2,1))
plt.scatter(x=X['TotRmsAbvGrd'], y=y_train['SalePrice'], color=('deepskyblue'),alpha=0.5)
plt.axvline(x=13, color='r', linestyle='-')
plt.title('Total Rooms - Price scatter plot', fontsize=15, weight='bold' )
print(X['TotRmsAbvGrd'].sort_values(ascending=False).head(1))
print(X['GarageArea'].sort_values(ascending=False).head(3))
print(X['1stFlrSF'].sort_values(ascending=False).head(1))
print(X['MasVnrArea'].sort_values(ascending=False).head(2))
print(X['TotalBsmtSF'].sort_values(ascending=False).head(1))
print(X['GrLivArea'].sort_values(ascending=False).head(2))

x_train = X_train[(X_train['TotRmsAbvGrd'] < 13) & (X_train['MasVnrArea']<1200) & (X_train['1stFlrSF']<4000) & (X_train['TotalBsmtSF']<5000)
                 & (X_train['GarageArea']<1300) & (X_train['GrLivArea']<4600)]

print(f'We removed {X_train.shape[0]- x_train.shape[0]} outliers')
target = X[['SalePrice']]
target.shape
pos = [635,1298,581,1190,297,1169,523]
target.drop(target.index[pos], inplace=True)
print('We make sure that both train and target sets have the same row number after removing the outliers:')
print( 'Train: ',x_train.shape[0], 'rows')
print('Target:', target.shape[0],'rows')
print("Skewness before log transform: ", X['LotFrontage'].skew())
print("Kurtosis before log transform: ", X['LotFrontage'].kurt())
from scipy.stats import skew

#num_feats = combined_df1.dtypes[combined_df1.dtypes != "object"].index


#skewed_feats = x_train[num_feats].apply(lambda x: skew(x.dropna())) #compute skewness

#skewed_feats = skewed_feats[skewed_feats > 0.55]
#skewed_feats = skewed_feats.index

#x_train[skewed_feats] = np.log1p(x_train[skewed_feats])
print(f"Skewness after log transform: {x_train['LotFrontage'].skew()}")
print(f"Kurtosis after log transform: {x_train['LotFrontage'].kurt()}")
plt.style.use('seaborn')
sns.set_style('whitegrid')
fig = plt.figure(figsize=(15,10))
#1 rows 2 cols
#first row, first col
ax1 = plt.subplot2grid((2,2),(0,0))
sns.distplot(X.LotFrontage, color='plum')
plt.title('Before: Distribution of Lot Frontage',weight='bold', fontsize=18)
#first row sec col
ax1 = plt.subplot2grid((2,2),(0,1))
sns.distplot(X['GrLivArea'], color='tan')
plt.title('Before: Distribution of GrLivArea',weight='bold', fontsize=18)


ax1 = plt.subplot2grid((2,2),(1,0))
sns.distplot(x_train.LotFrontage, color='plum')
plt.title('After: Distribution of Lot Frontage',weight='bold', fontsize=18)
#first row sec col
ax1 = plt.subplot2grid((2,2),(1,1))
sns.distplot(x_train['GrLivArea'], color='tan')
plt.title('After: Distribution of GrLivArea',weight='bold', fontsize=18)
plt.show()
target["SalePrice"] = np.log1p(target["SalePrice"])
plt.style.use('seaborn')
sns.set_style('whitegrid')
fig = plt.figure(figsize=(15,5))
#1 rows 2 cols
#first row, first col
ax1 = plt.subplot2grid((1,2),(0,0))
plt.hist(X.SalePrice, bins=10, color='red',alpha=0.5)
plt.title('Sale price distribution before log transform',weight='bold', fontsize=18)
#first row sec col
ax1 = plt.subplot2grid((1,2),(0,1))
plt.hist(target.SalePrice, bins=10, color='darkgreen',alpha=0.5)
plt.title('Sale price distribution after log transform',weight='bold', fontsize=18)
plt.show()
print(f"Skewness after log transform: {target['SalePrice'].skew()}")
print(f"Kurtosis after log transform: {target['SalePrice'].kurt()}")
from sklearn.model_selection import train_test_split
#train-test split
target = np.array(target)
X_train_full, X_valid_full, y_train, y_valid = train_test_split(x_train, target, train_size=0.8, test_size=0.2,
                                                                random_state=0)
print(X_train_full.shape,y_train.shape)
X_train
from sklearn.preprocessing import RobustScaler
scaler= RobustScaler()
# transform "x_train"
X_train_full = scaler.fit_transform(X_train_full)
# transform "x_test"
X_valid_full = scaler.transform(X_valid_full)
#Transform the test set
X_test= scaler.transform(test)
X_test
from sklearn.metrics import mean_squared_error
def score(prediction): #creating a function to get RMSE for predictions
    return str(math.sqrt(mean_squared_error(y_valid, prediction)))
import sklearn.model_selection as ms
from sklearn.linear_model import Ridge
import math

ridge=Ridge()
parameters= {'alpha':[x for x in range(1,101)]} 

ridge_reg= ms.GridSearchCV(ridge, param_grid=parameters, scoring='neg_mean_squared_error', cv=15)
ridge_reg.fit(X_train_full,y_train)
print(f"The best value of Alpha is: {ridge_reg.best_params_}")
print(f"The best score achieved with Alpha=10 is: {math.sqrt(-ridge_reg.best_score_)}")
ridge_pred=math.sqrt(-ridge_reg.best_score_)


ridge_mod=Ridge(alpha=15)
ridge_mod.fit(X_train_full,y_train)
y_pred_train=ridge_mod.predict(X_train_full)
y_pred_test=ridge_mod.predict(X_valid_full)

print(f'Root Mean Square Error train =  {str(math.sqrt(mean_squared_error(y_train, y_pred_train)))}')
print(f'Root Mean Square Error test =  {score(y_pred_test)}')   
from sklearn.model_selection import cross_val_score

#Ridge regression
Ridge_CV=Ridge(alpha=15)
MSEs=cross_val_score(Ridge_CV, x_train, target, scoring='neg_mean_squared_error', cv=5)

#RMSE score of the 5 folds
print("RMSE scores of the 5 folds:")
for i,j in enumerate(MSEs):
    j= math.sqrt(np.mean(-j))
    print(f'Fold {i}: {round(j,4)}')

#Final RMSE score with Lasso
print(f'Mean RMSE with Ridge: {round(math.sqrt(np.mean(-MSEs)),4)}')

from sklearn.linear_model import Lasso

params= {'alpha':[0.0001,0.0009,0.001,0.002,0.003,0.01,0.1,1,10,100]}

lasso=Lasso(tol=0.01)
lasso_reg=ms.GridSearchCV(lasso, param_grid=params, scoring='neg_mean_squared_error', cv=15)
lasso_reg.fit(X_train_full,y_train)

print(f'The best value of Alpha is: {lasso_reg.best_params_}')
lasso_mod=Lasso(alpha=0.0009)
lasso_mod.fit(X_train_full,y_train)
y_lasso_train=lasso_mod.predict(X_train_full)
y_lasso_test_pred=lasso_mod.predict(X_valid_full)

print(f'Root Mean Square Error train  {str(math.sqrt(mean_squared_error(y_train, y_lasso_train)))}')
print(f'Root Mean Square Error test  {score(y_lasso_test_pred)}')
coefs = pd.Series(lasso_mod.coef_, index = x_train.columns)

imp_coefs = pd.concat([coefs.sort_values().head(10),
                     coefs.sort_values().tail(10)])
imp_coefs.plot(kind = "barh", color='darkcyan')
plt.xlabel("Lasso coefficient", weight='bold')
plt.title("Feature importance in the Lasso Model", weight='bold')
plt.show()
print(f"Lasso kept {sum(coefs != 0)} important features and dropped the other  {sum(coefs == 0)} features")
#Lasso regression
Lasso_CV=Lasso(alpha=0.0009, tol=0.001)
MSEs=ms.cross_val_score(Lasso_CV, x_train, target, scoring='neg_mean_squared_error', cv=5)

#RMSE score of the 5 folds
print("RMSE scores of the 5 folds:")
for i,j in enumerate(MSEs):
    j= math.sqrt(np.mean(-j))
    print(f'Fold {i}: {round(j,4)}')

#Final RMSE score with Lasso
print(f'Mean RMSE with Lasso: {round(math.sqrt(np.mean(-MSEs)),4)}')
from sklearn.linear_model import ElasticNetCV

alphas = [10,1,0.1,0.01,0.001,0.002,0.003,0.004,0.005,0.00056]
l1ratio = [0.1, 0.3,0.5, 0.9, 0.95]

elastic_cv = ElasticNetCV(cv=5, max_iter=1e7, alphas=alphas,  l1_ratio=l1ratio)

elasticmod = elastic_cv.fit(X_train_full, y_train.ravel())
ela_pred=elasticmod.predict(X_valid_full)
print('Root Mean Square Error test = ' + str(math.sqrt(mean_squared_error(y_valid, ela_pred))))
print(elastic_cv.alpha_)
print(elastic_cv.l1_ratio_)
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
%matplotlib inline

def get_score(n_estimators):
    
    
    xgb = XGBRegressor(n_estimators=n_estimators, learning_rate = 0.02)
    
    scores_new = -1 * cross_val_score(xgb, X_train_full, y_train,
                              cv=5,
                              scoring='neg_mean_squared_error')

    print("Average RMSE score for n_estimators : {} is :".format(n_estimators), round(math.sqrt(np.mean(scores_new)),4))
    return math.sqrt(np.mean(scores_new))

results = {} #dict to store results
for i in range(100,800,100):  #checking different values of n_estimators
        results[i] = get_score(i)

plt.plot(list(results.keys()), list(results.values()))
plt.show()
from sklearn.model_selection import GridSearchCV

#xg_reg = XGBRegressor()
#xgparam_grid= {'learning_rate' : [0.01],'n_estimators':[2000,3000,4000],
#                                     'reg_alpha':[0.0001,0.01],
#                                    'reg_lambda':[1,0.01]}
#
#xg_grid=GridSearchCV(xg_reg, param_grid=xgparam_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
#xg_grid.fit(X_train_full,y_train)
#print(xg_grid.best_estimator_)
#print(xg_grid.best_score_)
from xgboost import XGBRegressor

xgb = XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=0.6, gamma=0, gpu_id=-1,
             importance_type='gain',
             learning_rate=0.01, max_delta_step=0, max_depth=3,
            objective='reg:squarederror',
             min_child_weight=1, missing=None, monotone_constraints='()',
             n_estimators=4000, n_jobs=1, random_state=0,
             reg_alpha=0.0001, reg_lambda=0.01, scale_pos_weight=1, subsample=1,
              validate_parameters=1, verbosity=1)
xgmod=xgb.fit(X_train_full,y_train)
xg_pred=xgmod.predict(X_valid_full)
print(f'Root Mean Square Error test = {score(xg_pred)}')
from sklearn.ensemble import VotingRegressor

vote_mod = VotingRegressor([('Ridge', ridge_mod), ('Lasso', lasso_mod), ('Elastic', elastic_cv), 
                            ('XGBRegressor', xgb)])
vote= vote_mod.fit(X_train_full, y_train.ravel())
vote_pred=vote.predict(X_valid_full)

print(f'Root Mean Square Error test = {score(vote_pred)}')
from mlxtend.regressor import StackingRegressor


stackreg = StackingRegressor(regressors=[elastic_cv,ridge_mod, lasso_mod, vote_mod], 
                           meta_regressor=xgb, use_features_in_secondary=True
                          )

stack_mod=stackreg.fit(X_train_full, y_train.ravel())
stacking_pred=stack_mod.predict(X_valid_full)

print(f'Root Mean Square Error test = {score(stacking_pred)}')
final_test=(0.4*vote_pred+0.4*stacking_pred+ 0.2*y_lasso_test_pred)
print(f'Root Mean Square Error test=  {score(final_test)}')
#VotingRegressor to predict the final Test
vote_test = vote_mod.predict(X_test)
final1=np.expm1(vote_test)

#StackingRegressor to predict the final Test
stack_test = stackreg.predict(X_test)
final2=np.expm1(stack_test)

#LassoRegressor to predict the final Test
lasso_test = lasso_mod.predict(X_test)
final3=np.expm1(lasso_test)

# averaging the predictions before submitting
final=(0.4*final1+0.4*final2+0.2*final3)
final
# creating submission file for submitting to the competition
output = pd.DataFrame({'Id': X_test_full.index,
                       'SalePrice': final})
output.to_csv('submission.csv', index=False)
output.head()
