import numpy as np

import scipy as sp

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.linear_model import LinearRegression as LR, Lasso, Ridge, ElasticNet as EN

from sklearn.tree import DecisionTreeRegressor as DTR

from sklearn.ensemble import RandomForestRegressor as RFR

from sklearn.decomposition import PCA

from sklearn.metrics import mean_squared_error
train = pd.read_csv('../input/train.csv')

#finding columns with null values

nl = list()

datapoints = train.shape[0]

dropped = list()

print('Dropped: ')

for i in train.columns.values:

    n = train[i].isnull().sum()

    if(n > 0):

        if (n < 0.2*datapoints):

            nl.append([i,n])

        else:

            print(i)

            train = train.drop([i],axis=1) #discarding columns with more than 20% missing values

            dropped.append(i)



#features with categorical values don't have numerical categories

nl_cat = list()

nl_num = list()

for i in nl:

    if (train[i[0]].dtype != 'int64') and (train[i[0]].dtype != 'float64'):

        nl_cat.append(i[0])

    else:

        nl_num.append(i[0])
#filling numerical NaN with mean

for i in nl_num:

    train[i] = train[i].fillna(train[i].mean()) 

#filling categorical NaN with most frequent value

for i in nl_cat:

    train[i] = train[i].fillna(train[i].value_counts().index[0]) 

train.isnull().sum().sum()
X = train.drop(['SalePrice','Id'],axis=1)

y = train['SalePrice']



X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=24)

#finding categorical and numerical variables

cat = list()

num = list()

for i in X_train.columns.values:

    if(type(X_train[i].values[0]) == str):

        cat.append(i)

    else:

        num.append(i)
#plotting all numerical categories to separate the ones which are categorical in nature

for i in num:

    plt.scatter(X_train[i],y_train)

    plt.xlabel(i)

    plt.ylabel('Sale Price')

    plt.show()
#Some of the numerical data is categorical in nature.

numcat = ['YrSold','MoSold','GarageCars','Fireplaces','TotRmsAbvGrd','KitchenAbvGr','BedroomAbvGr',

         'HalfBath','FullBath','BsmtHalfBath','BsmtFullBath','OverallCond',

         'OverallQual','MSSubClass'] #excluding YearBuilt and YearRemodAdd

num = [x for x in num if x not in numcat]

#MSSubClass is a categorical feature

cat.append('MSSubClass')

numcat.remove('MSSubClass')

X_train['MSSubClass'] = X_train['MSSubClass'].astype(str)

X_test['MSSubClass'] = X_test['MSSubClass'].astype(str)
#plotting all the numercial features which are categorical

for i in numcat:

    data = pd.concat([y_train,X_train[i]],axis=1)

    sns.boxplot(x=i,y='SalePrice',data=data)

    plt.xlabel(i)

    plt.ylabel('Sale Price')

    plt.xticks(rotation=90)

    plt.show()
#Dropping MoSold and YrSold - there isn't much variability in SalePrice for these features

X_train = X_train.drop(['MoSold','YrSold'],axis=1)

X_test = X_test.drop(['MoSold','YrSold'],axis=1)

numcat.remove('MoSold')

numcat.remove('YrSold')

yeardrop = ['MoSold','YrSold']

#Prices by neighborhood

data = pd.concat([y_train,X_train['Neighborhood']],axis=1)

sns.boxplot(x='Neighborhood',y='SalePrice',data=data)

plt.xticks(rotation=45)

plt.show()

#Looking at the categorical data (excluding the numerical categorical data above)

for i in cat:

    if i == 'Neighborhood':

        continue

    data = pd.concat([y_train,X_train[i]],axis=1)

    sns.boxplot(x=i,y='SalePrice',data=data)

    plt.xlabel(i)

    plt.ylabel('Sale Price')

    plt.xticks(rotation=90)

    plt.show()
#Distributions of all purely numerical categories. Log transform is performed when skewed

skewed = list()

for i in num:

    sk = X_train[i].skew()

    print(i)

    print('Skew: ',sk)

    if (X_train[i].skew() < 0.75): #threshold of 0.75 skew to log transform

        sns.distplot(X_train[i])

        plt.xlabel(i+' (Not log transformed) ')

    else:

        sns.distplot(np.log1p(X_train[i]))

        plt.xlabel(i+' (log transformed) ')

        skewed.append(i)

    plt.show()
'''

Discarding - these all have most values (>1200) as zero:

MiscVal 

PoolArea

ScreenPorch 

3SsnPorch

EnclosedPorch

LowQualFinSF

BsmtFinSF2



Recheck if Log Transform - They have a significant number of zero values. Excluding them, they might not have

a large skew

OpenPorchSF

WoodDeckSF

2ndFlrSF

'''

discard = ['MiscVal','PoolArea','ScreenPorch','3SsnPorch','EnclosedPorch','LowQualFinSF','BsmtFinSF2']

train = train.drop(discard,axis=1)



num = [x for x in num if x not in discard]

skewed = [x for x in skewed if x not in discard]



checkskew = ['OpenPorchSF','WoodDeckSF','2ndFlrSF']

for i in checkskew:

    idx = X_train[i] > 0

    sk = sp.stats.skew(X_train[i].values[idx])

    print(i)

    print('Skew: ',sk)

    if(sk<0.75):

        sns.distplot(X_train[i].values[idx])

        plt.xlabel(i+' (Not log transformed) ')

        skewed.remove(i)

    else:

        sns.distplot(np.log1p(X_train[i].values[idx]))

        plt.xlabel(i+' (log transformed) ')

    plt.show()



for i in skewed:

    X_train[i] = np.log1p(X_train[i])

    X_test[i] = np.log1p(X_test[i])
#plotting pair correlation heatmap between the numerical quantities



correlation = X_train[num].corr()

plt.figure(figsize=(12,12))

sns.heatmap(correlation)

plt.title('Correlation Heatmap between numerical features',fontsize=14)

plt.xticks(fontsize=13)

plt.yticks(fontsize=13)

plt.show()
print('SalePrice')

print('Skew: ',sp.stats.skew(y_train))

sns.distplot(np.log1p(y_train))

plt.xlabel('SalePrice (log transformed)')

plt.show()

#since SalePrice is skewed, we'll log transform it
year = ['YearBuilt','YearRemodAdd','GarageYrBlt']

num = [x for x in num if x not in year]
'''

Now we know what to do with each of the features, we'll load the data again.

This is done because the pd.get_dummies is likely to give different ordering/features for X_train and X_test

'''

def cleanup(df):

    df = df.drop(dropped,axis = 1)

    df = df.drop(yeardrop,axis = 1)

    for i in nl_num:

        df[i] = df[i].fillna(df[i].mean())

    for i in nl_cat:

        df[i] = df[i].fillna(df[i].value_counts().index[0])

    df.MSSubClass = df.MSSubClass.astype(str)

    df = df.drop(discard,axis=1)

    df[skewed] = np.log1p(df[skewed])

    return df
data = pd.read_csv('../input/train.csv')

data = cleanup(data)

X = data.drop(['SalePrice','Id'],axis=1)

y = np.log1p(data['SalePrice'])



#generating indicator features for categorical data

X = pd.get_dummies(X)



X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=55)



minmax = MinMaxScaler()

minmax.fit(X_train)

X_train = minmax.transform(X_train)

X_test = minmax.transform(X_test)



rmsle_all = dict()
#Linear Regression Model

lr = LR()

lr.fit(X_train,y_train)

print('Linear Regression')

print('Test Score: ',lr.score(X_test,y_test))

print('Train Score: ',lr.score(X_train,y_train))

y_predict = lr.predict(X_test)

print('RMSLE: ',np.sqrt(mean_squared_error(y_test,y_predict)))
#Bad behavior of the linear regression model can be explained by the large number of categorical variables

#each having a lot of categories, some of which occur in small numbers

#We'll fit a linear model using just the numerical categories



X_num = data.drop('SalePrice',axis=1)

y = data['SalePrice']

y = np.log1p(y)



X_num = X_num.drop(numcat,axis=1)

X_num = X_num.drop(cat,axis=1)



X_num_train, X_num_test, y_num_train, y_num_test = train_test_split(X_num,y,test_size=0.2,random_state=55)



minmax_num = MinMaxScaler()

minmax_num.fit(X_num_train)

X_num_train = minmax_num.transform(X_num_train)

X_num_test = minmax_num.transform(X_num_test)



lr_num = LR()

lr_num.fit(X_num_train,y_num_train)

print('Linear Regression using only Numerical Features')

print('Test Score: ',lr_num.score(X_num_test,y_num_test))

print('Train Score: ',lr_num.score(X_num_train,y_num_train))

y_predict = lr_num.predict(X_num_test)

print('RMSLE: ',np.sqrt(mean_squared_error(y_num_test,y_predict)))



rmsle_all['LR - NUM'] = np.sqrt(mean_squared_error(y_num_test,y_predict))

#This gives a much better behaved model
#We'll try tree and random forest regressors which handle categorical data well

#Decision Tree Regressor

dtr = DTR()

dtr.fit(X_train,y_train)

print('Decision Tree Regression')

print('Test Score: ',dtr.score(X_test,y_test))

print('Train Score: ',dtr.score(X_train,y_train))

y_predict = dtr.predict(X_test)

print('RMSLE: ',np.sqrt(mean_squared_error(y_test,y_predict)))

rmsle_all['DTR - ALL'] = np.sqrt(mean_squared_error(y_test,y_predict))



idx = np.argwhere(dtr.feature_importances_ > 0.01)

feat_imp = dtr.feature_importances_[idx[:,0]]

imp_feat = X.columns.values[idx[:,0]]

dtr_feat = pd.DataFrame([feat_imp,imp_feat],index = ['Coefficient','Feature']).transpose()

sns.barplot(x='Coefficient', y='Feature',data=dtr_feat)

plt.xlabel('Feature Importance')

plt.show()
#Random Forest Regressor

rfr = RFR()

rfr.fit(X_train,y_train)

print('Random Forest Regression')

print('Test Score: ',rfr.score(X_test,y_test))

print('Train Score: ',rfr.score(X_train,y_train))

y_predict = rfr.predict(X_test)

print('RMSLE: ',np.sqrt(mean_squared_error(y_test,y_predict)))

rmsle_all['RFR - ALL'] = np.sqrt(mean_squared_error(y_test,y_predict))



idx = np.argwhere(rfr.feature_importances_ > 0.01)

feat_imp = rfr.feature_importances_[idx[:,0]]

imp_feat = X.columns.values[idx[:,0]]

rfr_feat = pd.DataFrame([feat_imp,imp_feat],index = ['Coefficient','Feature']).transpose()

sns.barplot(x='Coefficient', y='Feature',data=rfr_feat)

plt.xlabel('Feature Importance')

plt.show()
#Ridge Regression Model

ridge = Ridge(alpha=1.0) #gridsearchcv gave 1.0 as optimum alpha

ridge.fit(X_train,y_train)

print('Ridge Regression')

print('Test Score: ',ridge.score(X_test,y_test))

print('Train Score: ',ridge.score(X_train,y_train))

y_predict = ridge.predict(X_test)

print('RMSLE: ',np.sqrt(mean_squared_error(y_test,y_predict)))

rmsle_all['Ridge - ALL'] = np.sqrt(mean_squared_error(y_test,y_predict))



idx = np.argwhere(np.abs(ridge.coef_) > 0.2)

imp_coef = ridge.coef_[idx[:,0]]

imp_feat = X.columns.values[idx[:,0]]

ridge_feat = pd.DataFrame([imp_coef,imp_feat],index = ['Coefficient','Feature']).transpose()

sns.barplot(x='Coefficient', y='Feature',data=ridge_feat)

plt.show()
#Ridge Regression Model using only numerical features

ridge_num = Ridge(alpha=1.0) #gridsearchcv gave 1.0 as optimum alpha

ridge_num.fit(X_num_train,y_num_train)

print('Ridge Regression using only Numerical Features')

print('Test Score: ',ridge_num.score(X_num_test,y_test))

print('Train Score: ',ridge_num.score(X_num_train,y_train))

y_predict = ridge_num.predict(X_num_test)

print('RMSLE: ',np.sqrt(mean_squared_error(y_num_test,y_predict)))

rmsle_all['Ridge - NUM'] = np.sqrt(mean_squared_error(y_test,y_predict))



idx = np.argwhere(np.abs(ridge_num.coef_) > 0.2)

imp_coef = ridge_num.coef_[idx[:,0]]

imp_feat = X_num.columns.values[idx[:,0]]

ridge_feat = pd.DataFrame([imp_coef,imp_feat],index = ['Coefficient','Feature']).transpose()

sns.barplot(x='Coefficient', y='Feature',data=ridge_feat)

plt.show()
#We'll train a Decision Tree and a Random Forest model with only the categorical variables

X_cat = data.drop('SalePrice',axis=1)

y = data['SalePrice']

y = np.log1p(y)



X_cat = X_cat.drop(num,axis=1)

X_cat = X_cat.drop(year,axis=1)



X_cat = pd.get_dummies(X_cat)



X_cat_train, X_cat_test, y_cat_train, y_cat_test = train_test_split(X_cat,y,test_size=0.2,random_state=55)



dtr_cat = DTR()

dtr_cat.fit(X_cat_train,y_cat_train)

print('Decision Tree Regression using only Categorical Features')

print('Test Score: ',dtr_cat.score(X_cat_test,y_cat_test))

print('Train Score: ',dtr_cat.score(X_cat_train,y_cat_train))

y_predict = dtr_cat.predict(X_cat_test)

print('RMSLE: ',np.sqrt(mean_squared_error(y_cat_test,y_predict)))

rmsle_all['DTR - CAT'] = np.sqrt(mean_squared_error(y_cat_test,y_predict))



idx = np.argwhere(dtr_cat.feature_importances_ > 0.01)

feat_imp = dtr_cat.feature_importances_[idx[:,0]]

imp_feat = X_cat.columns.values[idx[:,0]]

dtr_feat = pd.DataFrame([feat_imp,imp_feat],index = ['Coefficient','Feature']).transpose()

sns.barplot(x='Coefficient', y='Feature',data=dtr_feat)

plt.xlabel('Feature Importance')

plt.show()



print()

rfr_cat = RFR()

rfr_cat.fit(X_cat_train,y_cat_train)

print('Random Forest Regression using only Categorical Features')

print('Test Score: ',rfr_cat.score(X_cat_test,y_cat_test))

print('Train Score: ',rfr_cat.score(X_cat_train,y_cat_train))

y_predict = rfr_cat.predict(X_cat_test)

print('RMSLE: ',np.sqrt(mean_squared_error(y_cat_test,y_predict)))

rmsle_all['RFR - CAT'] = np.sqrt(mean_squared_error(y_cat_test,y_predict))



idx = np.argwhere(rfr_cat.feature_importances_ > 0.01)

feat_imp = rfr_cat.feature_importances_[idx[:,0]]

imp_feat = X_cat.columns.values[idx[:,0]]

rfr_feat = pd.DataFrame([feat_imp,imp_feat],index = ['Coefficient','Feature']).transpose()

sns.barplot(x='Coefficient', y='Feature',data=rfr_feat)

plt.xlabel('Feature Importance')

plt.show()
#The Random forest performs much better here.

#We'll test a hybrid model where 

#y_predict = frac*y_predict_num_ridge + (1-frac)*y_predict_cat_rfr

#we'll choose a value of frac which gives us least RMSLE

frac = np.linspace(0,1.0,num=200)

rmsle = np.zeros(200)

y_predict_num_ridge = ridge_num.predict(X_num_test)

y_predict_cat_rfr = rfr_cat.predict(X_cat_test)



for i in range(200):

    y_predict = frac[i]*y_predict_num_ridge + (1-frac[i])*y_predict_cat_rfr

    rmsle[i] = np.sqrt(mean_squared_error(y_test,y_predict))



plt.scatter(frac,rmsle)

plt.xlabel('Fraction')

plt.ylabel('RMSLE')

plt.show()

opt_frac = frac[np.argwhere(rmsle[:] == min(rmsle))[0,0]]

print('Optimum fraction: %0.3f'%opt_frac)

print('Minimum RMSLE: %0.3f'%min(rmsle))

rmsle_all['Hybrid'] = min(rmsle)
df = pd.DataFrame([rmsle_all.keys(),rmsle_all.values()]).transpose()

df.columns = ['Model','RMSLE']

sns.barplot(x='RMSLE',y='Model',data=df)

plt.xlim(0,0.3)

plt.xlabel('RMSLE')

plt.title('RMSLE of Different Regressors')

plt.ylabel('')

plt.show()
#For the submission we'll use Ridge - ALL, RFR - ALL and Hybrid

test_data = pd.read_csv('../input/test.csv')

train_data = pd.read_csv('../input/train.csv')



train_data = cleanup(train_data)

test_data = cleanup(test_data)



test_null = test_data.columns.values[test_data.isnull().sum()!=0]

for i in test_null:

    if i in num:

        test_data[i] = test_data[i].fillna(test_data[i].mean())

    else:

        test_data[i] = test_data[i].fillna(test_data[i].value_counts().index[0])



test_data.index = test_data.index + train_data.shape[0]



data_all = pd.concat([train_data,test_data])



data_all_dummies = pd.get_dummies(data_all)



train_data_dummies = data_all_dummies[:train_data.shape[0]]

test_data_dummies = data_all_dummies[train_data.shape[0]:]



train_X = train_data_dummies.drop(['SalePrice','Id'],axis=1)

train_y = np.log1p(train_data_dummies['SalePrice'])



test_X = test_data_dummies.drop(['Id','SalePrice'],axis=1)



X_train, X_test, y_train, y_test = train_test_split(train_X,train_y,test_size=0.2,random_state=23)



cat_dummies = list()

for i in train_X.columns.values:

    if (i not in num) and (i not in year):

        cat_dummies.append(i)



X_num_train = X_train[num+year]

X_num_test = X_test[num+year]



X_cat_train = X_train[cat_dummies]

X_cat_test = X_test[cat_dummies]



X_submit = test_X

X_num_submit = test_X[num+year]

X_cat_submit = test_X[cat_dummies]
#Applying Ridge Regression using all features

ridge = Ridge(alpha=1.0)

ridge.fit(X_train,y_train)

y_predict = ridge.predict(X_test)

print('Ridge Regression Using All Features')

print('Test Score: ', ridge.score(X_test,y_test))

print('Train Score: ',ridge.score(X_train,y_train))

print('RMSLE: ', mean_squared_error(y_test,y_predict))



y_submit_ridge = np.expm1(ridge.predict(X_submit))

ridge_submission = pd.DataFrame()

ridge_submission['Id'] = test_data['Id'] 

ridge_submission['SalePrice'] = y_submit_ridge

ridge_submission.index = ridge_submission.index - train_data.shape[0]

ridge_submission.to_csv('ridge_submission.csv',index=False)
#Applying Random Forest Regression using all features

rfr = RFR()

rfr.fit(X_train,y_train)

y_predict = rfr.predict(X_test)

print('Random Forest Regression Using All Features')

print('Test Score: ', rfr.score(X_test,y_test))

print('Train Score: ',rfr.score(X_train,y_train))

print('RMSLE: ', mean_squared_error(y_test,y_predict))



y_submit_rfr = np.expm1(rfr.predict(X_submit))

rfr_submission = pd.DataFrame()

rfr_submission['Id'] = test_data['Id'] 

rfr_submission['SalePrice'] = y_submit_rfr

rfr_submission.index = rfr_submission.index - train_data.shape[0]

rfr_submission.to_csv('rfr_submission.csv',index=False)
#Hybrid Model

#Ridge

ridge_num = Ridge(alpha=1.0)

ridge_num.fit(X_num_train,y_train)

y_predict = ridge_num.predict(X_num_test)

print('Ridge Regression Using Numerical Features')

print('Test Score: ', ridge_num.score(X_num_test,y_test))

print('Train Score: ',ridge_num.score(X_num_train,y_train))

print('RMSLE: ', mean_squared_error(y_test,y_predict))

y_submit_ridge_num = ridge_num.predict(X_num_submit)



print()



#Random Forest

rfr_cat = RFR()

rfr_cat.fit(X_cat_train,y_train)

y_predict = rfr_cat.predict(X_cat_test)

print('Random Forest Regression Using Categorical Features')

print('Test Score: ', rfr_cat.score(X_cat_test,y_test))

print('Train Score: ',rfr_cat.score(X_cat_train,y_train))

print('RMSLE: ', mean_squared_error(y_test,y_predict))

y_submit_rfr_cat = rfr_cat.predict(X_cat_submit)



#optimizing fraction

frac = np.linspace(0,1.0,num=200)

rmsle = np.zeros(200)

y_predict_ridge_num = ridge_num.predict(X_num_test)

y_predict_rfr_cat = rfr_cat.predict(X_cat_test)



for i in range(200):

    y_predict = frac[i]*y_predict_ridge_num + (1-frac[i])*y_predict_rfr_cat

    rmsle[i] = np.sqrt(mean_squared_error(y_test,y_predict))



print()



plt.scatter(frac,rmsle)

plt.xlabel('Fraction')

plt.ylabel('RMSLE')

plt.show()

opt_frac = frac[np.argwhere(rmsle[:] == min(rmsle))[0,0]]

print('Optimum fraction: %0.3f'%opt_frac)

print('Hybrid Model RMSLE: %0.3f'%min(rmsle))



y_submit_hybrid = np.expm1(y_submit_ridge_num*opt_frac + y_submit_rfr_cat*(1-opt_frac))

hybrid_submission = pd.DataFrame()

hybrid_submission['Id'] = test_data['Id'] 

hybrid_submission['SalePrice'] = y_submit_hybrid

hybrid_submission.index = hybrid_submission.index - train_data.shape[0]

hybrid_submission.to_csv('hybrid_submission.csv',index=False)
plt.scatter(hybrid_submission['SalePrice'],ridge_submission['SalePrice'])

plt.xlabel('Hybrid Model Prediction')

plt.ylabel('Ridge Regression Prediction')

plt.show()



plt.scatter(hybrid_submission['SalePrice'],rfr_submission['SalePrice'])

plt.xlabel('Hybrid Model Prediction')

plt.ylabel('Random Forest Regression Prediction')

plt.show()



plt.scatter(ridge_submission['SalePrice'],rfr_submission['SalePrice'])

plt.xlabel('Ridge Regression Prediction')

plt.ylabel('Random Forest Regression Prediction')

plt.show()