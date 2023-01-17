#import libraries

import warnings

warnings.filterwarnings('ignore')

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import StandardScaler,MinMaxScaler,PolynomialFeatures,RobustScaler

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet

from sklearn.linear_model import LassoCV,RidgeCV,ElasticNetCV

from sklearn.feature_selection import RFE

from sklearn.model_selection import GridSearchCV,KFold,RandomizedSearchCV,StratifiedKFold,cross_val_score

from sklearn.metrics import r2_score

sns.set_context("paper", font_scale = 1, rc={"grid.linewidth": 3})

pd.set_option('display.max_rows', 100, 'display.max_columns', 400)

from scipy.stats import skew,boxcox_normmax

from scipy.special import boxcox1p

from sklearn.pipeline import make_pipeline

from sklearn.metrics import mean_squared_error

from sklearn.svm import SVR

from sklearn.ensemble import GradientBoostingRegressor

from xgboost import XGBRFRegressor,XGBRegressor

from lightgbm import LGBMRegressor

from mlxtend.regressor import StackingCVRegressor

from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,BaggingRegressor,ExtraTreesRegressor
#loading data

train= pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test=pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

train.head()
test.head()
# Let's look at the number of rows and columns in the dataset

print(train.shape)

print(test.shape)
# Getting insights of the features and outliers

train.describe([0.25,0.50,0.75,0.99])
# Summary of the training dataset

print(train.info())
#summary of testing dataset

print(test.info())
#Checking percentage of null values present in training dataset 

missing_num= train[train.columns].isna().sum().sort_values(ascending=False)

missing_perc= (train[train.columns].isna().sum()/len(train)*100).sort_values(ascending=False)

missing= pd.concat([missing_num,missing_perc],keys=['Total','Percentage'],axis=1)

missing_train= missing[missing['Percentage']>0]

missing_train
#Checking percentage of null values present in testing dataset 

missing_num= test[test.columns].isna().sum().sort_values(ascending=False)

missing_perc= (test[test.columns].isna().sum()/len(test)*100).sort_values(ascending=False)

missing= pd.concat([missing_num,missing_perc],keys=['Total','Percentage'],axis=1)

missing_test= missing[missing['Percentage']>0]

missing_test
numerical = train.select_dtypes(include=['int64','float64']).drop(['SalePrice','Id'],axis=1)

numerical.head()
categorical = train.select_dtypes(exclude=['int64','float64'])

categorical.head()
def showvalues(ax,m=None):

    for p in ax.patches:

        ax.annotate("%.1f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),\

                    ha='center', va='center', fontsize=14, color='k', rotation=0, xytext=(0, 7),\

                    textcoords='offset points',fontweight='light',alpha=0.9) 
plt.figure(figsize=(20,20))

plt.subplot(2,1,1)

plt.xticks(fontsize=14)

ax1=sns.barplot(x=missing_train.index,y='Percentage',data=missing_train)

showvalues(ax1)

ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, horizontalalignment='right')

plt.subplot(2,1,2)

ax2=sns.barplot(x=missing_test.index,y='Percentage',data=missing_test)

showvalues(ax2)

ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, horizontalalignment='right')

plt.show()
# Dropping Id column from train and test test

train.drop('Id',axis=1,inplace=True)

test.drop('Id',axis=1,inplace=True)



print(train.shape)

print(test.shape)
len(train.select_dtypes(include=['int64','float64']).columns)
#Visualising numerical predictor variables with Target Variables

train_num = train.select_dtypes(include=['int64','float64'])

fig,axs= plt.subplots(12,3,figsize=(20,80))

#adjust horizontal space between plots 

fig.subplots_adjust(hspace=0.6)

for i,ax in zip(train_num.columns,axs.flatten()):

    sns.scatterplot(x=i, y='SalePrice', hue='SalePrice',data=train_num,ax=ax,palette='viridis_r')

    plt.xlabel(i,fontsize=12)

    plt.ylabel('SalePrice',fontsize=12)

    #ax.set_yticks(np.arange(0,900001,100000))

    ax.set_title('SalePrice'+' - '+str(i),fontweight='bold',size=20)
##Visualising Categorical predictor variables with Target Variables

def facetgrid_boxplot(x, y, **kwargs):

    sns.boxplot(x=x, y=y)

    x=plt.xticks(rotation=90)

    



f = pd.melt(train, id_vars=['SalePrice'], value_vars=sorted(train[categorical.columns]))

g = sns.FacetGrid(f, col="variable", col_wrap=3, sharex=False, sharey=False, size=5)

g = g.map(facetgrid_boxplot, "value", "SalePrice")
# Distribution of Target variable (SalePrice)

plt.figure(figsize=(8,6))

sns.distplot(train['SalePrice'],hist_kws={"edgecolor": (1,0,0,1)})
# Skew and kurtosis for SalePrice 

print("Skewness: %f" % train['SalePrice'].skew())

print("Kurtosis: %f" % train['SalePrice'].kurt())
#Applying log transformation to remove skewness and make target variable normally distributed

train['SalePrice'] = np.log1p(train['SalePrice'])
#Plotting graph again to see if its normally distributed or not and see outliers

# Distribution of Target variable (SalePrice)

plt.figure(figsize=(8,6))

sns.distplot(train['SalePrice'],hist_kws={"edgecolor": (1,0,0,1)})
#Correlation between variables to check multicollinearity 

# Generate a mask for the upper triangle (taken from seaborn example gallery)

plt.subplots(figsize = (30,20))

mask = np.zeros_like(train_num.corr(), dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

#Plotting heatmap

sns.heatmap(train_num.corr(), cmap=sns.diverging_palette(20, 220, n=200), mask = mask, annot=True, center = 0)
## Deleting those two values with outliers. 

train = train[train.GrLivArea < 4500]

train.reset_index(drop = True, inplace = True)

y=train['SalePrice']

train_df=train.drop('SalePrice',axis=1)



test_df = test

df_all= pd.concat([train_df,test_df]).reset_index(drop=True)
df_all['age']=df_all['YrSold']-df_all['YearBuilt']

# Some of the non-numeric predictors are stored as numbers; convert them into strings 

#will convert those columns into dummy variables later.

df_all[['MSSubClass']] = df_all[['MSSubClass']].astype(str) 

df_all['YrSold'] = df_all['YrSold'].astype(str) #year

df_all['MoSold'] = df_all['MoSold'].astype(str) #month

#Functional: Home functionality (Assume typical unless deductions are warranted)

df_all['Functional'] = df_all['Functional'].fillna('Typ')

df_all['Electrical'] = df_all['Electrical'].fillna('SBrkr') #Filling with modef

# data description states that NA refers to "No Pool"

df_all["PoolQC"] = df_all["PoolQC"].fillna("None")



# Replacing the missing values with 0, since no garage = no cars in garage inferred from data dictionary

df_all['GarageYrBlt'] = df_all['GarageYrBlt'].fillna(0)

df_all['KitchenQual'] = df_all['KitchenQual'].fillna("TA")

df_all['Exterior1st'] = df_all['Exterior1st'].fillna(df_all['Exterior1st'].mode()[0])

df_all['Exterior2nd'] = df_all['Exterior2nd'].fillna(df_all['Exterior2nd'].mode()[0])

df_all['SaleType'] = df_all['SaleType'].fillna(df_all['SaleType'].mode()[0])

# Replacing the missing values with None inferred from data dictionary 

for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:

    df_all[col] = df_all[col].fillna('None')

# Replacing the missing values with None 

# NaN values for these categorical basement df_all, means there's no basement

for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

    df_all[col] = df_all[col].fillna('None')

#Replacing missing value it with median beacuse of outliers

df_all['LotFrontage'] = df_all.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))

# Replacing the missing values with None 

# We have no particular intuition around how to fill in the rest of the categorical df_all

# So we replace their missing values with None

objects = []

for i in df_all.columns:

    if df_all[i].dtype == object:

        objects.append(i)

df_all.update(df_all[objects].fillna('None'))



numeric_dtypes = [ 'int64','float64']

numerics = []

for i in df_all.columns:

    if df_all[i].dtype in numeric_dtypes:

        numerics.append(i)

df_all.update(df_all[numerics].fillna(0))



df_all['MSZoning'] = df_all.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))
#Checking percentage of null values present in dataset 

missing_perc= (df_all[df_all.columns].isna().sum()/len(df_all)*100).sort_values(ascending=False)

print(missing_perc[missing_perc>0].sum()) #No missing values

# We have already removed skewness from target variable (SalePrice) and made it normally distributed.

# Lets find out if numerical predictor variables are largely skewed or not

df_all_num = df_all.select_dtypes(include=['int64','float64'])

skew_features = df_all_num.apply(lambda x: skew(x)).sort_values(ascending=False)

high_skew = skew_features[skew_features > 0.5]

skew_index = high_skew.index

skewness = pd.DataFrame({'Skew' :high_skew})

skew_features
f, ax = plt.subplots(figsize=(8, 7))

ax.set_xscale("log")

ax = sns.boxplot(data=df_all_num , orient="h", palette="Set1")

ax.xaxis.grid(False)

ax.set(ylabel="Feature names")

ax.set(xlabel="Numeric values")

ax.set(title="Numeric Distribution of Features")

sns.despine(trim=True, left=True)
# Normalize skewed features using a Box-Cox power transformation, we can use other techniques but am using boxpox

# as it works very well on this dataset

for i in skew_index:

    df_all[i] = boxcox1p(df_all[i], boxcox_normmax(df_all[i] + 1.002))
# Let's make sure we handled all the skewed values

f, ax = plt.subplots(figsize=(8, 7))

ax.set_xscale("log")

ax = sns.boxplot(data=df_all[skew_index] , orient="h", palette="Set1")

ax.xaxis.grid(False)

ax.set(ylabel="Feature names")

ax.set(xlabel="Numeric values")

ax.set(title="Numeric Distribution of Features")

sns.despine(trim=True, left=True)
# NOt useful columns in our predictions, more than 99% rows have same value.

print(df_all['Utilities'].value_counts())

# NOt useful columns in our predictions, more than 99% rows have same value.

print(df_all['Street'].value_counts())

# NOt useful columns in our predictions, more than 99% rows have same value.

print(df_all['PoolQC'].value_counts())
cols_to_drop = ['Utilities', 'Alley','FireplaceQu','PoolQC','Fence','MiscFeature']

df_all=df_all.drop(cols_to_drop, axis=1) # not useful df_all, evident from above



# df_all=df_all.drop(['Utilities', 'Street', 'PoolQC'], axis=1) # not useful df_all, evident from above

# vintage house with remodified version of it plays a important role in prediction(i.e. high price )

df_all['YrBltAndRemod']=df_all['YearBuilt']+df_all['YearRemodAdd']

#Overall area for all floors and basement plays an important role, hence creating total area in square foot column

df_all['Total_sqr_footage'] = (df_all['BsmtFinSF1'] + df_all['BsmtFinSF2'] +

                                 df_all['1stFlrSF'] + df_all['2ndFlrSF'])

# Creating derived column for total number of bathrooms column

df_all['Total_Bathrooms'] = (df_all['FullBath'] + (0.5 * df_all['HalfBath']) +

                               df_all['BsmtFullBath'] + (0.5 * df_all['BsmtHalfBath']))

#Creating derived column for total porch area 

df_all['Total_porch_sf'] = (df_all['OpenPorchSF'] + df_all['3SsnPorch'] + df_all['EnclosedPorch'] + \

                              df_all['ScreenPorch'] + df_all['WoodDeckSF'])

df_all['has_pool'] = df_all['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

df_all['has_2ndfloor'] = df_all['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)

df_all['has_garage'] = df_all['GarageArea'].apply(lambda x: 1 if x > 0 else 0)

df_all['has_bsmt'] = df_all['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)

df_all['has_fireplace'] = df_all['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

df_all['has_openporch'] =df_all['OpenPorchSF'].apply(lambda x: 1 if x > 0 else 0)

df_all['has_wooddeck'] =df_all['WoodDeckSF'].apply(lambda x: 1 if x > 0 else 0)

df_all['has_enclosedporch'] = df_all['EnclosedPorch'].apply(lambda x: 1 if x > 0 else 0)

df_all['has_3ssnporch']=df_all['3SsnPorch'].apply(lambda x: 1 if x > 0 else 0)

df_all['has_openporch'] = df_all['OpenPorchSF'].apply(lambda x: 1 if x > 0 else 0)

df_all['has_screenporch'] = df_all['ScreenPorch'].apply(lambda x: 1 if x > 0 else 0)



#<-------------------------- Check Again ----------------------->

df_all['TotalBsmtSF'] = df_all['TotalBsmtSF'].apply(lambda x: np.exp(6) if x <= 0.0 else x)

df_all['2ndFlrSF'] = df_all['2ndFlrSF'].apply(lambda x: np.exp(6.5) if x <= 0.0 else x)

df_all['LotFrontage'] = df_all['LotFrontage'].apply(lambda x: np.exp(4.2) if x <= 0.0 else x)

df_all['MasVnrArea'] = df_all['MasVnrArea'].apply(lambda x: np.exp(4) if x <= 0.0 else x)

df_all['BsmtFinSF1'] = df_all['BsmtFinSF1'].apply(lambda x: np.exp(6.5) if x <= 0.0 else x)



def log_transform(result, features):

    m = result.shape[1]

    for feature in features:

        result = result.assign(newcol=pd.Series(np.log(1.01+result[feature])).values)   

        result.columns.values[m] = feature + '_log'

        m += 1

    return result



log_features = ['LotFrontage','LotArea','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF',

                 'TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea',

                 'BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr',

                 'TotRmsAbvGrd','Fireplaces','GarageCars','GarageArea','WoodDeckSF','OpenPorchSF',

                 'EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal','YearRemodAdd']



df_all = log_transform(df_all, log_features)
df_all_num= df_all.select_dtypes(include=['float64','int64']).columns  # Numerical columns

df_all_temp = df_all.select_dtypes(exclude=['float64','int64']) # selecting object and categorical features only

df_all_dummy= pd.get_dummies(df_all_temp)

df_all=pd.concat([df_all,df_all_dummy],axis=1) # joining converted dummy feature and original df_all dataset

df_all= df_all.drop(df_all_temp.columns,axis=1) #removing original categorical columns

df_all.shape
X= df_all[:len(train)] #converted into train data

Z_test= df_all[len(train):] #test data

print('Train Data Shape:',X.shape) #train set shape

print('Test Data Shape:',Z_test.shape)  #test set shape
#based on describe method and scatter plot, removing outliers

outl_col = ['GrLivArea','GarageArea','TotalBsmtSF','LotArea']



def drop_outliers(x):

    list = []

    for col in outl_col:

        Q1 = x[col].quantile(.25)

        Q3 = x[col].quantile(.99)

        IQR = Q3-Q1

        x =  x[(x[col] >= (Q1-(1.5*IQR))) & (x[col] <= (Q3+(1.5*IQR)))] 

    return x   

X = drop_outliers(X)

outliers = [30, 88, 462, 631, 1322]

X = X.drop(X.index[outliers])

y = y.drop(y.index[outliers])

print(X.shape)
def redundant_feature(df):

    redundant = []

    for i in df.columns:

        counts = df[i].value_counts()

        count_max = counts.iloc[0]

        if count_max / len(df) * 100 > 99.94:

            redundant.append(i)

    redundant = list(redundant)

    return redundant





redundant_features = redundant_feature(X)



X = X.drop(redundant_features, axis=1)

Z_test = Z_test.drop(redundant_features, axis=1)
print('Train Data Shape:',X.shape) #train set shape

print('Test Data Shape:',Z_test.shape)  #test set shape
kfold= KFold(n_splits=11,random_state=42,shuffle=True) #kfold cross validation
# Error function to compute error

def rmsle(y, y_pred):

    return np.sqrt(mean_squared_error(y, y_pred))



#Assigning scoring paramter to 'neg_mean_squared_error' beacause 'mean_squared_error' is not 

# available inside cross_val_score method

def cv_rmse(model, X=X):

    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=kfold))

    return (rmse)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(X_train.shape)

print(y_train.shape)

print(X_test.shape)
ridge=Ridge()

params= {'alpha': [5,8,10,10.1,10.2,10.3,10.35,10.36,11,12,15]}

scaler=RobustScaler()

X_train=scaler.fit_transform(X_train)

X_test=scaler.transform(X_test)

grid_ridge=GridSearchCV(ridge, param_grid=params,cv=kfold,scoring='neg_mean_squared_error')

grid_ridge.fit(X_train,y_train)

alpha = grid_ridge.best_params_

ridge_score = grid_ridge.best_score_

print("The best alpha value found is:",alpha['alpha'],'with score:',ridge_score)



ridge_alpha=Ridge(alpha=alpha['alpha'])

ridge_alpha.fit(X_train,y_train)

y_pred_train=ridge_alpha.predict(X_train)

y_pred_test=ridge_alpha.predict(X_test)



print('RMSE train = ',rmsle(y_train,y_pred_train))

print('RMSE test = ',rmsle(y_test,y_pred_test))
scores={}

alphas_ridge = [15, 15.1, 15.2, 15.3, 15.4, 15.5] #Best value of alpha parmaters for Ridge regression

ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=alphas_ridge, cv=kfold))

score = cv_rmse(ridge)

print(score)

print("ridge RMSE: {:.4f} ({:.4f})".format(score.mean(), score.std()))

scores['ridge'] = (score.mean(), score.std()) #Printing standard deviation to check deviation of scores



alphas_lasso = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008] #Best value of alpha parmaters for lasso

lasso = make_pipeline(RobustScaler(), LassoCV(alphas=alphas_lasso, cv=kfold))

score = cv_rmse(lasso)

print(score)

print("lasso RMSE: {:.4f} ({:.4f})".format(score.mean(), score.std()))

scores['lasso'] = (score.mean(), score.std()) #Printing standard deviation to check deviation of scores

alpha_elnet= [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]

l1ratio_elnet = [0.8, 0.85, 0.9, 0.95, 0.99, 1]

elasticnet = make_pipeline(RobustScaler(), ElasticNetCV(max_iter=1000000, alphas=alpha_elnet, \

                                                        cv=kfold, l1_ratio=l1ratio_elnet))

score=cv_rmse(elasticnet)

print(score)

print("Elasticnet RMSE: {:.4f} ({:.4f})".format(score.mean(), score.std()))

scores['elasticnet'] = (score.mean(), score.std()) #Printing standard deviation to check deviation of scores
svr = make_pipeline(RobustScaler(), SVR(C= 19, epsilon= 0.008, gamma=0.00015))

score=cv_rmse(svr)

print(score)

print("SVR RMSE: {:.4f} ({:.4f})".format(score.mean(), score.std()))

scores['svr'] = (score.mean(), score.std()) #Printing standard deviation to check deviation of scores
gbr= GradientBoostingRegressor(n_estimators=6000,learning_rate=0.01,max_depth=3,\

                              min_samples_leaf=15,max_features='sqrt',min_samples_split=10,loss='huber',\

                              random_state=42)

score=cv_rmse(gbr)

print(score)

print("GBR RMSE: {:.4f} ({:.4f})".format(score.mean(), score.std()))

scores['gbr'] = (score.mean(), score.std()) #Printing standard deviation to check deviation of scores
lgbm =  LGBMRegressor(objective='regression', num_leaves=4,learning_rate=0.01, n_estimators=6000,

                                       max_bin=200, bagging_fraction=0.75,bagging_freq=5, bagging_seed=7,

                                       feature_fraction=0.2,feature_fraction_seed=7,verbose=-1)

score=cv_rmse(lgbm)

print(score)

print("LGBM RMSE: {:.4f} ({:.4f})".format(score.mean(), score.std()))

scores['lgbm'] = (score.mean(), score.std()) #Printing standard deviation to check deviation of scores
xgb = XGBRegressor(learning_rate=0.01,n_estimators=3460, max_depth=3, min_child_weight=0, gamma=0, subsample=0.7,

                                     colsample_bytree=0.7, objective='reg:squarederror', nthread=-1,

                                     scale_pos_weight=1, seed=27, reg_alpha=0.00006)

score=cv_rmse(xgb)

print(score)

print("XGBOOST RMSE: {:.4f} ({:.4f})".format(score.mean(), score.std()))

scores['xgb'] = (score.mean(), score.std()) #Printing standard deviation to check deviation of xgb
#bc = BaggingRegressor(n_estimators=3000,max_features=280)

#score=cv_rmse(bc)

#print(score)

#print("XGBOOST RMSE: {:.4f} ({:.4f})".format(score.mean(), score.std()))

#scores['bc'] = (score.mean(), score.std()) #Printing standard deviation to check deviation of xgb
etr = ExtraTreesRegressor(n_estimators=60,random_state=42)

score=cv_rmse(etr)

print(score)

print("XGBOOST RMSE: {:.4f} ({:.4f})".format(score.mean(), score.std()))

scores['etr'] = (score.mean(), score.std()) #Printing standard deviation to check deviation of xgb
stack_reg = StackingCVRegressor(regressors=(ridge, lasso, elasticnet, gbr, xgb, lgbm),

                                meta_regressor=xgb,

                                use_features_in_secondary=True)
lasso_final= lasso.fit(X,y)
ridge_final=ridge.fit(X,y)
elasticnet_final=elasticnet.fit(X,y)
svr_final=svr.fit(X,y)
gbr_final=gbr.fit(X,y)
lgbm_final=lgbm.fit(X,y)
xgb_final=xgb.fit(X,y)
stack_reg_final=stack_reg.fit(X,y)
def blend_models_predict(X):

    return ((0.025* elasticnet_final.predict(X)) + \

            (0.025 * lasso_final.predict(X)) + \

            (0.025 * ridge_final.predict(X)) + \

            (0.025* svr_final.predict(X)) + \

            (0.62 * gbr_final.predict(X)) + \

            (0.03 * xgb_final.predict(X)) + \

            (0.03 * lgbm_final.predict(X)) + \

            (0.22 * stack_reg_final.predict(np.array(X))))  ##best best best best *5
print('RMSLE score on train data:')

blended_score=rmsle(y, blend_models_predict(X))

print(blended_score)

scores['blended'] = (blended_score, 0)
# Plot the predictions for each model

#sns.set_style("white")

fig = plt.figure(figsize=(20, 8))



ax = sns.pointplot(x=list(scores.keys()), y=[score for score, _ in scores.values()], markers=['o'], linestyles=['-'])

for i, score in enumerate(scores.values()):

    ax.text(i, score[0] + 0.002, '{:.6f}'.format(score[0]), horizontalalignment='left', size='large', color='black', weight='semibold')



plt.ylabel('Score (RMSE)', size=20, labelpad=12.5)

plt.xlabel('Model', size=20, labelpad=12.5)

plt.tick_params(axis='x', labelsize=13.5)

plt.tick_params(axis='y', labelsize=12.5)



plt.title('Scores of Models', size=20)



plt.show()
print('Prediction_Submission')

submission = submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")

submission.iloc[:,1] = np.floor(np.expm1(blend_models_predict(Z_test)))
q1 = submission['SalePrice'].quantile(0.005)

q2 = submission['SalePrice'].quantile(0.995)

submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x > q1 else x*0.77)

submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x < q2 else x*1.1)

submission.to_csv("./submission_prediction.csv", index=False)