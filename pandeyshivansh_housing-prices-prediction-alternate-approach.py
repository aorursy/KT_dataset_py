import numpy as np
import pandas as pd
import seaborn as sns
sns.set(font_scale=1)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# Reading the data
X = pd.read_csv('../input/home-data-for-ml-course/train.csv', index_col='Id')
X_test = pd.read_csv('../input/home-data-for-ml-course/test.csv', index_col='Id')

# Removing rows with missing target, separate target from predictors
X.dropna(axis=0, subset=['SalePrice'], inplace=True)            

X.head()
# Selecting categorical columns with relatively low cardinality
low_cardinality_cols = [cname for cname in X.columns if X[cname].nunique() < 10 and 
                        X[cname].dtype == "object"]

# Selecting numeric columns
numeric_cols = [cname for cname in X.columns if X[cname].dtype in ['int64', 'float64'] and cname!='SalePrice']
#combining train and test data
combine = [X,X_test]
# Missng values of numrical columns
missing = X[numeric_cols].isnull().sum().sort_values(ascending=False)
missing[missing.values>0]
missing = X_test[numeric_cols].isnull().sum().sort_values(ascending=False)
missing[missing.values>0]
#LotFrontage: all house have linear connected feet so filling missing data with most mean value
for dataset in combine:
    dataset['LotFrontage']=dataset['LotFrontage'].fillna(dataset['LotFrontage'].dropna().mean())
#GarageYrBlt and MasVnrArea, can be 0, missing value may represnt that these 
#houses do not have Garage and Masonry Veeener
for dataset in combine:
    dataset['GarageYrBlt']=dataset['GarageYrBlt'].fillna(0)
    dataset['MasVnrArea']=dataset['MasVnrArea'].fillna(0)
#Similarly for BsmtHalfBath,BsmtFullBath, GarageArea, BsmtFinSF1, BsmtFinSF2, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF, GarageCars
for col in ['BsmtHalfBath','BsmtFullBath','GarageArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','GarageCars']:
    X_test[col]=X_test[col].fillna(0)
#No more missing values for numerical columns
missing = X[numeric_cols].isnull().sum().sort_values(ascending=False)
missing[missing.values>0]
missing = X_test[numeric_cols].isnull().sum().sort_values(ascending=False)
missing[missing.values>0]
#Missing values for Categorical columns
missing = X[low_cardinality_cols].isnull().sum().sort_values(ascending=False)
missing[missing.values>0]
missing = X_test[low_cardinality_cols].isnull().sum().sort_values(ascending=False)
missing[missing.values>0]
#PoolQC, MiscFeature : Since most houses do not have a pool we can fill missing values with NA
for dataset in combine:
    dataset['PoolQC']=dataset['PoolQC'].fillna('NA')
    dataset['MiscFeature']=dataset['MiscFeature'].fillna('NA')
# Similarly for Alley, Fence, FireplaceQu, GarageCond, GarageQual, GarageFinish and GarageType filling NA
for dataset in combine:
    for col in ['Alley','Fence','FireplaceQu','GarageCond','GarageQual','GarageFinish','GarageType']:
        dataset[col]=dataset[col].fillna('NA')
# Also for BsmtFinType2, BsmtExposure, BsmtCond, BsmtFinType1, BsmtQual, MasVnrType
for dataset in combine:
    for col in ['BsmtFinType2','BsmtExposure','BsmtCond','BsmtFinType1','BsmtQual','MasVnrType']:
        dataset[col]=dataset[col].fillna('NA')
# Electrical : There is single missing value for this feature so filling missing value with most occuring value
X['Electrical']=X['Electrical'].fillna(X['Electrical'].dropna().value_counts().index[0])
X['Electrical'].unique()
#Similarly for MSZoning, Utilities, Functional, KitchenQual, SaleType in test dataset
for col in ['MSZoning', 'Utilities', 'Functional', 'KitchenQual', 'SaleType']:
    X_test[col]=X_test[col].fillna(X_test[col].dropna().value_counts().index[0])
#No more missing values for Categorical columns
missing = X[low_cardinality_cols].isnull().sum().sort_values(ascending=False)
missing[missing.values>0]
missing = X_test[low_cardinality_cols].isnull().sum().sort_values(ascending=False)
missing[missing.values>0]
#Creating heatmap to check which numeric features are correlated with SalePrice
numeric_cols.append('SalePrice')
correlation=X[numeric_cols].corr().sort_values(by='SalePrice',ascending=False).round(2)
print(correlation['SalePrice'])
#Top correlated features with SalesPrice are OverallQual,GrLivArea,TotalBsmtSF,GarageCars,1stFlrSF,GarageArea 
plt.subplots(figsize=(12, 9))
sns.heatmap(correlation, vmax=.8, square=True);
#Creating heatmap for top 10 correlated features
cols =correlation['SalePrice'].head(10).index
cm = np.corrcoef(X[cols].values.T)
sns.set(font_scale=1)
hm = sns.heatmap(cm, annot=True, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
# It is obvious the Overall Qaulity has direct correlation with Sale price
plt.title("House sale price vs Overall Quality")
sns.barplot(x=X['OverallQual'], y=X['SalePrice'])
plt.title("House sale price vs GrLivArea")
sns.scatterplot(x=X['GrLivArea'],y=X['SalePrice'])
#Removing the outliers in above graph
X=X.drop(X.loc[(X['GrLivArea']>4000) & (X['SalePrice']<200000)].index,0)
X.reset_index(drop=True, inplace=True)
sns.scatterplot(x=X['GrLivArea'],y=X['SalePrice'])
sns.barplot(x=X['GarageCars'],y=X['SalePrice'])
sns.scatterplot(x=X['GarageArea'], y=X['SalePrice'])
# Removing outliers from above
X=X.drop(X.loc[(X['GarageArea']>1200) & (X['SalePrice']<300000)].index,0)
X.reset_index(drop=True, inplace=True)
sns.scatterplot(x=X['GarageArea'],y=X['SalePrice'])
y = X.SalePrice 
X.drop(['SalePrice'], axis=1, inplace=True)
#Creating new features based on highly correlated features, To increase weight of these features in the model
for col in ['GrLivArea','GarageCars','GarageArea','TotalBsmtSF','1stFlrSF']:
    X[col+'_2']=X[col]**2
    X_test[col+'_2']=X_test[col]**2
    X[col+'_3']=X[col]**3
    X_test[col+'_3']=X_test[col]**3
    X[col+'_4']=X[col]**4
    X_test[col+'_4']=X_test[col]**4
#adding 1stFlrSF and 2ndFlrSF and create new feature Totalfloorfeet
X['Totalfloorfeet']=X['1stFlrSF']+X['2ndFlrSF']
X_test['Totalfloorfeet']=X_test['1stFlrSF']+X['2ndFlrSF']
X=X.drop(['1stFlrSF','2ndFlrSF'],1)
X_test=X_test.drop(['1stFlrSF','2ndFlrSF'],1)
#adding BsmtFullBath, BsmtHalfBath, FullBath, HalfBath to create new feature TotalBath
for dataset in combine:
    dataset['Bath']=dataset['BsmtFullBath']+dataset['BsmtHalfBath']*.5+dataset['FullBath']+dataset['HalfBath']*.5
    dataset=dataset.drop(['BsmtFullBath','BsmtHalfBath','FullBath','HalfBath'],1)
#Test dataset has one more outlier in 'GarageYrBlt' feature
X_test.loc[X_test['GarageYrBlt']==2207.,'GarageYrBlt']=0

object_cols = [cname for cname in X.columns if X[cname].nunique() < 10 and 
                        X[cname].dtype == "object"]
numeric_cols = [cname for cname in X.columns if X[cname].dtype in ['int64', 'float64'] and cname!='SalePrice']
my_cols=object_cols+numeric_cols
X_final=X[my_cols].copy()
X_test_final=X_test[my_cols].copy()
combine=[X_final,X_test_final]

#Creating dummies for Categorical data
X_final=pd.get_dummies(X_final)
X_test=pd.get_dummies(X_test)

X_final, X_test_final = X_final.align(X_test_final, join='left', axis=1)
[cname for cname in X_final.columns if X_final[cname].nunique() < 10 and 
                        X_final[cname].dtype == "object"]
#[cname for cname in X.columns if X[cname].dtype in ['int64', 'float64'] and cname!='SalePrice']
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler=scaler.fit(X_final)
X_scaled=scaler.transform(X_final)
X_test_scaled=scaler.transform(X_test_final)
X_scaled=pd.DataFrame(X_scaled)
X_test_scaled=pd.DataFrame(X_test_scaled)
#1.LinearRegression
from sklearn.linear_model import LinearRegression
LR=LinearRegression()
LR.fit(X_scaled,y)
LR.score(X_scaled,y)
#2.LogisticRegression
from sklearn.linear_model import LogisticRegression
LogR=LogisticRegression()
LogR.fit(X_scaled,y)
print(LogR.score(X_scaled,y))
#3.Support Vector Regression
from sklearn import svm
svmR=svm.SVC()
svmR.fit(X_scaled,y)
print(svmR.score(X_scaled,y))
#4.Naive Bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
gnb=GaussianNB()
mnb=MultinomialNB()
gnb.fit(X_scaled,y)
mnb.fit(X_final,y)
print(gnb.score(X_scaled,y))
print(mnb.score(X_final,y))
#4.DecisionTree
from sklearn.tree import DecisionTreeRegressor
Dtree=DecisionTreeRegressor(criterion='mse',max_depth=5)
Dtree.fit(X_scaled,y)
print(Dtree.score(X_scaled,y))
#7.Random Forest
from sklearn.ensemble import RandomForestRegressor
RandomFR=RandomForestRegressor(n_estimators=500)
RandomFR.fit(X_scaled,y)
print(RandomFR.score(X_scaled,y))
#8.XGBoost
from xgboost import XGBRegressor

model = XGBRegressor(random_state=0, n_estimators=1000, learning_rate=0.05, n_jobs=4)
model.fit(X_scaled, y, 
             early_stopping_rounds=5, 
             eval_set=[(X_scaled, y)], 
             verbose=False)
print(model.score(X_scaled,y))
preds_test = model.predict(X_test_scaled)
output = pd.DataFrame({'Id': X_test_final.index,
                       'SalePrice': preds_test})
output.to_csv('submission.csv', index=False)