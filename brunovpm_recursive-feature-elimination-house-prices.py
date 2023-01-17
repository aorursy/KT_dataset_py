import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import sklearn

import xgboost as xgb

from sklearn.preprocessing import StandardScaler,LabelEncoder

from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor

from sklearn.model_selection import cross_val_score, KFold,GridSearchCV,train_test_split

from lightgbm import LGBMRegressor

from sklearn.feature_selection import SelectKBest,f_regression,RFECV

from skopt import dummy_minimize

from sklearn.linear_model import ElasticNet

from sklearn.linear_model import Lasso 

from sklearn.linear_model import Ridge

from sklearn.metrics import mean_squared_error
#Read data

train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

test['SalePrice'] = np.nan

df = pd.concat([test,train],sort=False)

df.head()
#Deal with null values

missing = pd.DataFrame({'types':df.dtypes, 'percetange_of_missing': df.isna().sum()/len(df)*100})

missing = missing[missing['percetange_of_missing'] != 0]

missing.sort_values(by='percetange_of_missing', ascending=False)
#Drop coluns with more than 50% of null

df = df.drop(['PoolQC','MiscFeature','Alley','Fence'],axis=1)          



#Convert categoric features to object

df[['MSSubClass','OverallQual','OverallCond']] = df[['MSSubClass','OverallQual','OverallCond']].astype(object)
cols_num = missing[missing.types=="float64"].index

cols_num = cols_num.drop(['SalePrice'])

print(cols_num)

for i in cols_num:

    df[i].fillna(df[i].median(), inplace = True)
print(missing[missing.types=="object"][missing.percetange_of_missing<50])

cols_cat = missing[missing.types=="object"][missing.percetange_of_missing<50].index

for i in cols_cat:

    df[i].fillna(df[i].mode()[0], inplace = True)
df2 = df.drop(['Id'],axis=1)

# Threshold for removing correlated variables

threshold = 0.7

corr_matrix = df2.corr().abs()



# Upper triangle of correlations

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

upper.head()
# Select columns with correlations above threshold

to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

print('There are %d columns to remove.' % (len(to_drop)))

dataset = df.drop(columns = to_drop)
categorical= df.select_dtypes(include='object')



fig, axes = plt.subplots(round(len(categorical.columns) / 3), 2, figsize=(22, 80))



for i, ax in enumerate(fig.axes):

    if i < len(categorical.columns):

        sns.countplot(x=categorical.columns[i], data=categorical, ax=ax)



fig.tight_layout()
numerics = df.select_dtypes(include=['float64','float32','int64'])

numerics = numerics.drop(['Id'],axis=1)

fig, ax = plt.subplots(17,2,figsize=(22, 80))

for i, col in enumerate(numerics):

    plt.subplot(17,2,i+1)

    plt.xlabel(col, fontsize=10)

    sns.kdeplot(numerics[col].values, bw=0.5)

plt.show() 
fig, ax = plt.subplots(18,2,figsize=(22,80))

for i, col in enumerate(numerics):

    plt.subplot(17,2,i+1)

    plt.xlabel(col, fontsize=10)

    sns.scatterplot(x=numerics[col].values, y='SalePrice', data=numerics)

plt.show() 
import warnings

warnings.filterwarnings("ignore")

#Plot Categoric features 

categorical = df.select_dtypes(include=['object'])

categorical['SalePrice'] = train.SalePrice

fig, ax = plt.subplots(22,2,figsize=(22,150))

for i, col in enumerate(categorical):

    plt.subplot(22,2,i+1)

    plt.xlabel(col, fontsize=10)

    sns.boxplot(x=categorical.columns[i], y="SalePrice", data=categorical, dodge=False)

plt.show() 
#Check and print the correlation between features and target (top 10 positive correlations)

#Weak positive correlation

df2 = df2.select_dtypes(['float64','int64'])



corr = df2.corr()

sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values)

plt.title('Correlation Matrix')

plt.gcf().set_size_inches(11,9)

plt.show()
print(df.SalePrice.quantile([0.01,0.99]))

print('---')

print(df.SalePrice.describe())

df.SalePrice = np.where(df.SalePrice>442567.0,442567.0,df.SalePrice)

df.SalePrice = np.where(df.SalePrice<61815.97,61815.97,df.SalePrice)
num_col = numerics.columns

for col in df[num_col]:

    df[col]= np.where(df[col]>df[col].quantile(0.99),df[col].quantile(0.99),df[col])

    df[col]= np.where(df[col]<df[col].quantile(0.01),df[col].quantile(0.01),df[col])
df['OverallQual'] = LabelEncoder().fit_transform(df['OverallQual'])



#Create new features

df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']

df['TotalBath'] = df['FullBath']+df['BsmtFullBath']+(df['BsmtHalfBath']+df['HalfBath'])*0.5

df['TotalPorchSF'] = df['OpenPorchSF'] + df['3SsnPorch'] + df['EnclosedPorch'] + df['ScreenPorch']

df['HasPool'] = df['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

df['HasFireplace'] = df['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

df['isNewer'] = df['MSSubClass'].apply(lambda x: 1 if x in [20, 60, 120] else 0)



#Encoder in object columsn

for col in df.columns[df.dtypes == 'object']:

    le = LabelEncoder()

    le.fit(df[col])

    df[col] = le.transform(df[col])



#standard features

num_col = df2.select_dtypes(include=['float64','float32','int64']).columns

num_col = num_col.drop(['SalePrice'])

scaler = sklearn.preprocessing.StandardScaler().fit(df[num_col])

df[num_col]= scaler.transform(df[num_col])
#train

train = df[df['SalePrice'].isna()==False]



#Valid + submission

x_valid = df[df['SalePrice'].isna()==True]

submission = x_valid['Id'].to_frame()

x_valid = x_valid.drop(['Id'],axis=1)



#x e y

y =train.loc[:,['SalePrice','Id']]

x =train.drop(['SalePrice','Id'], axis = 1)



x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=42)
mdl1 = GradientBoostingRegressor(n_estimators = 100)

mdl1.fit(x_train,y_train['SalePrice'])

p = mdl1.predict(x_test) 

print((mean_squared_error(y_test['SalePrice'], p))**0.5)
mdl2 = RandomForestRegressor(n_estimators = 100)

mdl2.fit(x_train,y_train['SalePrice'])

p2 = mdl2.predict(x_test) 

print((mean_squared_error(y_test['SalePrice'], p2))**0.5)
mdl3 = Ridge() 

mdl3.fit(x_train,y_train['SalePrice'])

p3 = mdl3.predict(x_test) 

print((mean_squared_error(y_test['SalePrice'], p3))**0.5)
mdl4 = Lasso()

mdl4.fit(x_train,y_train['SalePrice'])

p4 = mdl4.predict(x_test) 

print((mean_squared_error(y_test['SalePrice'], p4))**0.5)
mdl5 = ElasticNet()

mdl5.fit(x_train,y_train['SalePrice'])

p5 = mdl5.predict(x_test) 

print((mean_squared_error(y_test['SalePrice'], p5))**0.5)
select_feature = SelectKBest(f_regression, k=30).fit(x_train, y_train['SalePrice'])

x_train2 = select_feature.transform(x_train)

x_test2 = select_feature.transform(x_test)



mdl6 = GradientBoostingRegressor(n_estimators = 100,random_state=42)

mdl6.fit(x_train2,y_train['SalePrice'])

p = mdl6.predict(x_test2) 

print((mean_squared_error(y_test['SalePrice'], p))**0.5)
mdl7 = GradientBoostingRegressor(n_estimators = 100)

rfecv = RFECV(estimator=mdl7, step=1, cv=5,scoring='neg_mean_squared_error')  

rfecv = rfecv.fit(x_train, y_train['SalePrice'])



print('Optimal number of features :', rfecv.n_features_)

print('Best features :', x_train.columns[rfecv.support_])

best_features = list(x_train.columns[rfecv.support_])



# Plot number of features VS. cross-validation scores

plt.figure()

plt.xlabel("Number of features selected")

plt.ylabel("Cross validation score of number of selected features")

plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)

plt.show()
x_train3 = x_train[best_features]

x_test3 = x_test[best_features]



mdl7 = GradientBoostingRegressor(n_estimators = 100,random_state=42)

mdl7.fit(x_train3,y_train['SalePrice'])

p = mdl7.predict(x_test3) 

print((mean_squared_error(y_test['SalePrice'], p))**0.5)
parameters = {

              'max_features': [0.7,0.8, 0.9],

              'min_samples_leaf' :[1,3,7],

              'learning_rate' : [0.01,0.03,0.1],

               'subsample': [0.8,0.9],

               'max_depth': [5,6,7]

              }





mdl8 = GradientBoostingRegressor(n_estimators = 100,random_state=42)

grid_search2 = GridSearchCV(mdl8, parameters, cv=5,n_jobs=-1,scoring='neg_root_mean_squared_error')

grid_search2.fit(x_train3,y_train['SalePrice'])



print(grid_search2.best_params_)

print(grid_search2.best_score_)
x_valid = x_valid[best_features]

SalePrice = mdl7.predict(x_valid)

submission['SalePrice'] = SalePrice  

submission['SalePrice'] = submission['SalePrice']

submission.to_csv('submission.csv' , index=False)