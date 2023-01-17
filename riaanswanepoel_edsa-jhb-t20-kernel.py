import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import scipy.stats as st

from scipy.stats import skew



from sklearn import metrics

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split, RandomizedSearchCV

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, RidgeCV, LassoCV, ElasticNetCV

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import cross_val_score, KFold, RandomizedSearchCV

from sklearn.pipeline import make_pipeline



import warnings

warnings.filterwarnings('ignore')
#These functions will be used to assess the performance of the models

def rmse_cv(model, X, y):

    '''Calculates the cross validation rmse of the prediction

    

    Parameters

    model (sklearn model): The model for which the cross validation should be performed

    X (Pandas dataframe): The dataframe that the model should perform the predictions on

    y (Series): The values that the prediction should be compared against

    '''

    rmse = np.sqrt(-cross_val_score(model, X_train, y_train, scoring="neg_mean_squared_error", cv=kfolds))

    return (rmse)

  



def rmsle (y, y_pred):

    '''Calculates the rmsle of the prediction

    

    Parameters

    y (Series): The target values to assess the prediction aganst

    y_pred (Series): The predicted values to assess

    '''

    return (np.sqrt(metrics.mean_squared_error(y, y_pred)))



  

def CVscore(model, X, y):

    '''Calculates the cross validation score of the prediction

    

    Parameters

    model (sklearn model): The model for which the cross validation should be performed

    X (Pandas dataframe): The dataframe that the model should perform the predictions on

    y (Series): The values that the prediction should be compared against

    '''

    result = cross_val_score(model, X, y, cv=kfold)

    return result.mean()

  

  

#This function will be used to combine different models into an ensembled model  

def blend_models_predict(X, models):

    '''Takes list of models and returns a blended set of the models

    

    Parameters

    X (Pandas dataframe): The dataframe that the model should perform the predictions on

    models (list of sklearn pipelines): A list of model pipelines for which the predictions will be blended

    '''

    blend = []

    for model in models:

        blend.append(1/len(models)*model.predict(X))

    return sum(blend)
df_train = pd.read_csv('../input/train.csv',index_col='Id')

df_test = pd.read_csv('../input/test.csv',index_col='Id')
sns.scatterplot(df_train['GrLivArea'], df_train['SalePrice'])

plt.title('Identifying outliers')

plt.xlabel('GrLivArea')

plt.ylabel('SalePrice')
df_train.drop(df_train[(df_train['GrLivArea']>4000) & (df_train['SalePrice']<300000)].index, inplace=True)
#This variable is used to keep track of the training dataset when we combine it with the test dataset

len_train = len(df_train)
df_train.head()
y = df_train['SalePrice']
y.describe()
sns.distplot(y, fit=st.norm)
#Log transformation is one of the simplest transformations to normalise positively skewed data

y = np.log1p(df_train['SalePrice'])

sns.distplot(y, fit=st.norm)
df_all = pd.concat([df_train, df_test], sort=False)
print(df_all.shape)

print(df_all.info())
df_all['Electrical'].unique()
df_all['Neighborhood'].unique()
#Check number of missing values

missing = df_all.isnull().sum()

missing = missing[missing > 0]

missing.sort_values(inplace=True)

missing.plot.bar()



#Note that SalePrice also have missing values, but this is due to the absence of SalePrice from the test dataset. We will therefore not fill in the SalePrice
#The following features are categorical features that needs to be filled in with 'None'

Cat_toNone = ('PoolQC','Fence','MiscFeature','Alley','FireplaceQu','GarageType','GarageFinish',

              'GarageQual','GarageCond','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1',

              'BsmtFinType2','MasVnrType','MSSubClass')



#This loop takes the tuple of features and fills all the missing values with 'None'

for c in Cat_toNone:

    df_all[c] = df_all[c].fillna('None')
#The following features are categorical features that needs to be filled in with the mode

Cat_toMode = ('MSZoning','Electrical','KitchenQual','Exterior1st','Exterior2nd',

              'Utilities','SaleType','Functional')



#This loop takes the tuple of features and fills all the missing values with the mode

for c in Cat_toMode:

    df_all[c] = df_all[c].fillna(df_all[c].mode()[0])
#The following features are numerical features that needs to be filled in with 0

Num_toZero = ('GarageArea','GarageCars','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF',

              'BsmtFullBath','BsmtHalfBath','GarageYrBlt','MasVnrArea')



#This loop takes the tuple of features and fills all the missing values with 0

for c in Num_toZero:

    df_all[c] = df_all[c].fillna(0)
#In this case, neigborhood might have a large influence on the lot frontage. Therefore we group by Neighborhood and fill the LotFrontage with the median of the groups

df_all["LotFrontage"] = df_all.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
#Recheck number of missing values to identify any missed features

missing = df_all.isnull().sum()

missing = missing[missing > 0]

missing.sort_values(inplace=True)

missing.plot.bar()



#Note that SalePrice also have missing values, but this is due to the absence of SalePrice from the test dataset
df_all['MSSubClass'] = df_all['MSSubClass'].astype(str)
#We make a list of all the numerical values here in order to exclude the numerical variables from the ordinal variables that we will create in the next step

numeric = df_all.select_dtypes(exclude = ['object']).columns
df_all.hist(figsize = (12, 7), bins = 40)

plt.tight_layout()

plt.show()
skewed_feats = df_all[numeric].apply(lambda x: skew(x.dropna())) #compute skewness

skewed_feats = skewed_feats[skewed_feats > 0.75]

skewed_feats = skewed_feats.index



df_all[skewed_feats] = np.log1p(df_all[skewed_feats])



#df_all[numeric] = np.log1p(df_all[numeric])
df_all.hist(figsize = (12, 7), bins = 40)

plt.tight_layout()

plt.show()
#We replace all ordinal categorical variables with an ordered set of integers

df_all = df_all.replace({"Alley" : {"None" : 0, "Grvl" : 1, "Pave" : 2},

                     "BsmtCond" : {"None" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

                     "BsmtExposure" : {"None" : 0, "Mn" : 1, "Av": 2, "Gd" : 3},

                     "BsmtFinType1" : {"None" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4, 

                                       "ALQ" : 5, "GLQ" : 6},

                     "BsmtFinType2" : {"None" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4, 

                                       "ALQ" : 5, "GLQ" : 6},

                     "BsmtQual" : {"None" : 0, "Po" : 1, "Fa" : 2, "TA": 3, "Gd" : 4, "Ex" : 5},

                     "ExterCond" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},

                     "ExterQual" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},

                     "FireplaceQu" : {"None" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

                     "Functional" : {"Sal" : 1, "Sev" : 2, "Maj2" : 3, "Maj1" : 4, "Mod": 5, 

                                     "Min2" : 6, "Min1" : 7, "Typ" : 8},

                     "GarageCond" : {"None" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

                     "GarageQual" : {"None" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

                     "HeatingQC" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

                     "KitchenQual" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

                     "LandSlope" : {"Sev" : 1, "Mod" : 2, "Gtl" : 3},

                     "LotShape" : {"IR3" : 1, "IR2" : 2, "IR1" : 3, "Reg" : 4},

                     "PavedDrive" : {"N" : 0, "P" : 1, "Y" : 2},

                     "PoolQC" : {"None" : 0, "Fa" : 1, "TA" : 2, "Gd" : 3, "Ex" : 4},

                     "Street" : {"Grvl" : 1, "Pave" : 2},

                     "Utilities" : {"ELO" : 1, "NoSeWa" : 2, "NoSewr" : 3, "AllPub" : 4}}

                    )
#Combining bathrooms into single bathroom feature

df_all['Bathrooms'] = df_all['FullBath'] + (df_all['HalfBath']*0.5) + df_all['BsmtFullBath'] + (df_all['BsmtHalfBath']*0.5)



#Combining fireplace and fireplace quality into a fireplace score

df_all['FireplaceScore'] = df_all['Fireplaces'] * df_all['FireplaceQu']



#Creating a single GarageScore

#GarageCond does not add to the correlation, and is therefore excluded

df_all['GarageScore'] = df_all['GarageCars'] * df_all['GarageQual'] * df_all['GarageArea']



#Determining house age

df_all['HouseAge'] = df_all['YrSold'] - df_all['YearBuilt']



#Total Living square feet

df_all['TotalLivingSF'] = df_all['GrLivArea'] + df_all['TotalBsmtSF'] - df_all['LowQualFinSF']
print("Find most important features relative to target")

corr = df_all.corr()

plt.figure(figsize=(12,12))

sns.heatmap(corr, cmap='coolwarm')

corr.sort_values(["SalePrice"], ascending = False, inplace = True)

print(corr.SalePrice)
df_all = df_all.drop(['FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath',

                      'GarageCars','GarageArea','GarageQual','GarageCond',

                      'FireplaceQu','Fireplaces', 

                      'PoolArea', #Pool Quality is more important than pool size

                      'Utilities', 'Street',

                      'GarageYrBlt'

                     ], axis = 1)
df_all = pd.get_dummies(df_all, drop_first=True)
df_all = df_all.drop(['SalePrice'], axis = 1)

train = df_all[:len_train]

test = df_all[len_train:]
X_train, X_test, y_train, y_test = train_test_split(train, y, test_size = 0.3, random_state = 0)
#Multiple Linear Regression

regressor = LinearRegression()

regressor.fit(X_train, y_train)

#Predicting the Test set results

train_lm = regressor.predict(X_train)

test_lm = regressor.predict(X_test)



#Multiple Linear Regression using RIDGE regularization

ridge = Ridge(alpha = 10)

ridge.fit(X_train, y_train)

#Predicting the Test set results

ridge_train_lm = ridge.predict(X_train)

ridge_test_lm = ridge.predict(X_test)



#Multiple Linear Regression using LASSO regularization

lasso = Lasso(max_iter=500,alpha = 0.001)

lasso.fit(X_train, y_train)

#Predicting the Test set results

lasso_train_lm = lasso.predict(X_train)

lasso_test_lm = lasso.predict(X_test)



#https://www.kaggle.com/fugacity/ridge-lasso-elasticnet-regressions-explained

#Multiple Linear Regression using ELASTICNET regularization

en = ElasticNet(max_iter=500,alpha = 0.001)

en.fit(X_train, y_train)

#Predicting the Test set results

en_train_lm = en.predict(X_train)

en_test_lm = en.predict(X_test)



#Decision Tree Regression

dtr = DecisionTreeRegressor(max_depth=5)

dtr.fit(X_train, y_train)

#Predicting the Test set results

train_dtr = dtr.predict(X_train)

test_dtr = dtr.predict(X_test)



#Random Forest Regression

rf = RandomForestRegressor(random_state = 0)

rf.fit(X_train, y_train)

#Predicting the Test set results

train_rf = rf.predict(X_train)

test_rf = rf.predict(X_test)
scorer = metrics.make_scorer(metrics.mean_squared_error, greater_is_better = False)

kfolds = KFold(n_splits=10, shuffle=True, random_state=42)
#Assessing the Linear Regression model 

print('Linear Regression')

print("RMSE on Training set :", rmse_cv(regressor,X_train,y_train).mean())

print("RMSE on Test set :", rmse_cv(regressor,X_test,y_test).mean())

print('RMSLE score on train data: ', rmsle(y_train, regressor.predict(X_train)))

print('RMSLE score on test data: ', rmsle(y_test, regressor.predict(X_test)))

print('-'*25)



#Assessing the Ridge Linear Regression model 

print('Linear Regression Ridge')

print("RMSE on Training set :", rmse_cv(ridge,X_train,y_train).mean())

print("RMSE on Test set :", rmse_cv(ridge,X_test,y_test).mean())

print('RMSLE score on train data: ', rmsle(y_train, ridge.predict(X_train)))

print('RMSLE score on test data: ', rmsle(y_test, ridge.predict(X_test)))

print('-'*25)



#Assessing the Lasso Linear Regression model 

print('Linear Regression Lasso')

print("RMSE on Training set :", rmse_cv(lasso,X_train,y_train).mean())

print("RMSE on Test set :", rmse_cv(lasso,X_test,y_test).mean())

print('RMSLE score on train data: ', rmsle(y_train, lasso.predict(X_train)))

print('RMSLE score on test data: ', rmsle(y_test, lasso.predict(X_test)))

print('-'*25)



#Assessing the ElasticNet Linear Regression model 

print('Linear Regression ElasticNet')

print("RMSE on Training set :", rmse_cv(en,X_train,y_train).mean())

print("RMSE on Test set :", rmse_cv(en,X_test,y_test).mean())

print('RMSLE score on train data: ', rmsle(y_train, en.predict(X_train)))

print('RMSLE score on test data: ', rmsle(y_test, en.predict(X_test)))

print('-'*25)



#Assessing the Decision Tree model 

print('Decision Tree Regression')

print("RMSE on Training set :", rmse_cv(dtr,X_train,y_train).mean())

print("RMSE on Test set :", rmse_cv(dtr,X_test,y_test).mean())

print('RMSLE score on train data: ', rmsle(y_train, dtr.predict(X_train)))

print('RMSLE score on test data: ', rmsle(y_test, dtr.predict(X_test)))

print('-'*25)



#Assessing the Random Forest model 

print('Random Forest Regression')

print("RMSE on Training set :", rmse_cv(rf,X_train,y_train).mean())

print("RMSE on Test set :", rmse_cv(rf,X_test,y_test).mean())

print('RMSLE score on train data: ', rmsle(y_train, rf.predict(X_train)))

print('RMSLE score on test data: ', rmsle(y_test, rf.predict(X_test)))
alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]

cv_ridge = [rmse_cv(Ridge(alpha = alpha),X_train,y_train).mean() 

            for alpha in alphas]



cv_ridge = pd.Series(cv_ridge, index = alphas)

cv_ridge.plot(title = "Validation")

plt.xlabel("alpha")

plt.ylabel("rmse")

print('Best alpha parameter: ', cv_ridge[cv_ridge == min(cv_ridge)].index[0])
alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001, 0.0011, 0.0012]

cv_lasso = [rmse_cv(Lasso(max_iter=500, alpha = alpha),X_train,y_train).mean() 

            for alpha in alphas]



cv_lasso = pd.Series(cv_lasso, index = alphas)

cv_lasso.plot(title = "Validation")

plt.xlabel("alpha")

plt.ylabel("rmse")

print('Best alpha parameter: ', cv_lasso[cv_lasso == min(cv_lasso)].index[0])
alphas = [0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001, 0.0011, 0.0012, 0.0013, 0.0014, 0.0015, 0.0016, 0.0017, 0.0018, 0.0019, 0.002]

cv_elasticnet = [rmse_cv(ElasticNet(max_iter=500,alpha = alpha),X_train,y_train).mean() 

            for alpha in alphas]



cv_elasticnet = pd.Series(cv_elasticnet, index = alphas)

cv_elasticnet.plot(title = "Validation")

plt.xlabel("alpha")

plt.ylabel("rmse")

print('Best alpha parameter: ', cv_elasticnet[cv_elasticnet == min(cv_elasticnet)].index[0])
#Multiple Linear Regression using RIDGE regularization

ridge = Ridge(alpha = cv_ridge[cv_ridge == min(cv_ridge)].index[0])

ridge.fit(X_train, y_train)

#Predicting the Test set results

ridge_train_lm = ridge.predict(X_train)

ridge_test_lm = ridge.predict(X_test)



#Multiple Linear Regression using LASSO regularization

lasso = Lasso(max_iter=500,alpha = cv_lasso[cv_lasso == min(cv_lasso)].index[0])

lasso.fit(X_train, y_train)

#Predicting the Test set results

lasso_train_lm = lasso.predict(X_train)

lasso_test_lm = lasso.predict(X_test)



#https://www.kaggle.com/fugacity/ridge-lasso-elasticnet-regressions-explained

#Multiple Linear Regression using ELASTICNET regularization

en = ElasticNet(max_iter=500,alpha = cv_elasticnet[cv_elasticnet == min(cv_elasticnet)].index[0])

en.fit(X_train, y_train)

#Predicting the Test set results

en_train_lm = en.predict(X_train)

en_test_lm = en.predict(X_test)
#Assessing the Ridge Linear Regression model 

print('Linear Regression Ridge')

print("RMSE on Training set :", rmse_cv(ridge,X_train,y_train).mean())

print("RMSE on Test set :", rmse_cv(ridge,X_test,y_test).mean())

print('RMSLE score on train data: ', rmsle(y_train, ridge.predict(X_train)))

print('RMSLE score on test data: ', rmsle(y_test, ridge.predict(X_test)))

print('-'*25)



#Assessing the Lasso Linear Regression model 

print('Linear Regression Lasso')

print("RMSE on Training set :", rmse_cv(lasso,X_train,y_train).mean())

print("RMSE on Test set :", rmse_cv(lasso,X_test,y_test).mean())

print('RMSLE score on train data: ', rmsle(y_train, lasso.predict(X_train)))

print('RMSLE score on test data: ', rmsle(y_test, lasso.predict(X_test)))

print('-'*25)



#Assessing the ElasticNet Linear Regression model 

print('Linear Regression ElasticNet')

print("RMSE on Training set :", rmse_cv(en,X_train,y_train).mean())

print("RMSE on Test set :", rmse_cv(en,X_test,y_test).mean())

print('RMSLE score on train data: ', rmsle(y_train, en.predict(X_train)))

print('RMSLE score on test data: ', rmsle(y_test, en.predict(X_test)))
alphas_ridge = [2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20, 22.5, 25, 27.5, 30]

alphas_lasso = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001, 0.0011]

alphas_en = [0.0006, 0.0007, 0.0008, 0.0009, 0.001, 0.0011, 0.0012, 0.0013, 0.0014, 0.0015, 0.0016]

l1ratio_en = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99, 1]



ridge = make_pipeline(RidgeCV(alphas=alphas_ridge, cv=kfolds))

lasso = make_pipeline(LassoCV(max_iter=500, alphas=alphas_lasso, random_state=42, cv=kfolds))

elasticnet = make_pipeline(ElasticNetCV(max_iter=500,alphas=alphas_en, cv=kfolds, l1_ratio=l1ratio_en))
ridge_full = ridge.fit(X_train,y_train)

lasso_full = lasso.fit(X_train,y_train)

en_full = elasticnet.fit(X_train,y_train)
#Assessing the Ridge Linear Regression model 

print('Linear Regression Ridge')

print("RMSE on Training set :", rmse_cv(ridge_full,X_train,y_train).mean())

print("RMSE on Test set :", rmse_cv(ridge_full,X_test,y_test).mean())

print('RMSLE score on train data: ', rmsle(y_train, ridge_full.predict(X_train)))

print('RMSLE score on test data: ', rmsle(y_test, ridge_full.predict(X_test)))

print('-'*25)



#Assessing the Lasso Linear Regression model 

print('Linear Regression Lasso')

print("RMSE on Training set :", rmse_cv(lasso_full,X_train,y_train).mean())

print("RMSE on Test set :", rmse_cv(lasso_full,X_test,y_test).mean())

print('RMSLE score on train data: ', rmsle(y_train, lasso_full.predict(X_train)))

print('RMSLE score on test data: ', rmsle(y_test, lasso_full.predict(X_test)))

print('-'*25)



#Assessing the ElasticNet Linear Regression model 

print('Linear Regression ElasticNet')

print("RMSE on Training set: ", rmse_cv(en_full,X_train,y_train).mean())

print("RMSE on Test set: ", rmse_cv(en_full,X_test,y_test).mean())

print('RMSLE score on train data: ', rmsle(y_train, en_full.predict(X_train)))

print('RMSLE score on test data: ', rmsle(y_test, en_full.predict(X_test)))
#Assessing the Ensembled model 

print('Ensembled Model')

print('RMSLE score on train data: ', rmsle(y_train, blend_models_predict(X_train,[ridge_full,lasso_full,en_full])))

print('RMSLE score on test data: ', rmsle(y_test, blend_models_predict(X_test,[ridge_full,lasso_full,en_full])))
blended_train = pd.DataFrame({'Id': X_train.index.values, 'SalePrice': np.expm1(blend_models_predict(X_train,[ridge_full,lasso_full,en_full]))})

blended_test = pd.DataFrame({'Id': X_test.index.values, 'SalePrice': np.expm1(blend_models_predict(X_test,[ridge_full,lasso_full,en_full]))})
#We only use the predictions at the extremes, i.e. the 1% quantile and the 99% quantile

q1_tr = blended_train['SalePrice'].quantile(0.01)

q2_tr = blended_train['SalePrice'].quantile(0.99)

q1_te = blended_test['SalePrice'].quantile(0.01)

q2_te = blended_test['SalePrice'].quantile(0.99)



#We scale the identified predictions by lowering the small predictions and increasing the large predictions

blended_train['SalePrice'] = blended_train['SalePrice'].apply(lambda x: x if x > q1_tr else x*0.8)

blended_train['SalePrice'] = blended_train['SalePrice'].apply(lambda x: x if x < q2_tr else x*1.1)

blended_test['SalePrice'] = blended_test['SalePrice'].apply(lambda x: x if x > q1_te else x*0.8)

blended_test['SalePrice'] = blended_test['SalePrice'].apply(lambda x: x if x < q2_te else x*1.1)
#Assessing the brute force Ensembled model

print('Ensembled Model')

print('RMSLE score on train data: ', rmsle(y_train, np.log1p(blended_train['SalePrice'])))

print('RMSLE score on test data: ', rmsle(y_test, np.log1p(blended_test['SalePrice'])))
ridge_full = ridge.fit(train, y)

lasso_full = lasso.fit(train, y)

en_full = elasticnet.fit(train, y)
submission = pd.DataFrame({'Id': df_test.index.values, 'SalePrice': np.expm1(blend_models_predict(test,[ridge_full,lasso_full,en_full]))})
#We use brute force again to deal with our predictions at the extremes

q1 = submission['SalePrice'].quantile(0.01)

q2 = submission['SalePrice'].quantile(0.99)



submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x > q1 else x*0.8)

submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x < q2 else x*1.1)
submission.to_csv("submission.csv", index=False)

print('Save submission',)