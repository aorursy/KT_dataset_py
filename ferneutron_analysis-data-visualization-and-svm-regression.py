import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import category_encoders as ce
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR



%matplotlib inline
prices = pd.read_csv('../input/train.csv')
prices.head()
print(prices.info())
prices.select_dtypes(include=['object']).head()
# prices = all features except those with object type value
# prices_objects = all features with object type value
prices_objects = prices.select_dtypes(include=['object']).copy()
prices = prices.select_dtypes(exclude=['object']).copy()
prices_objects.head()
prices.head()
prices = prices.drop(['Id'],axis=1)
prices.isnull().sum()
prices_objects.isnull().sum()
# Dropping not necessary features
prices_objects = prices_objects.drop(['Alley','PoolQC', 'Fence', 'MiscFeature'], axis=1)
BsmtQual_count = prices_objects['BsmtQual'].value_counts()
sb.set(style="darkgrid")
sb.barplot(BsmtQual_count.index, BsmtQual_count.values, alpha=0.9)
plt.title('Frequency Distribution of BsmtQual')
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('BsmtQual', fontsize=12)
plt.show()
prices_objects['BsmtQual'] = prices_objects['BsmtQual'].fillna(prices_objects['BsmtQual'].value_counts().index[0])
BsmtCond_count = prices_objects['BsmtCond'].value_counts()
sb.set(style="darkgrid")
sb.barplot(BsmtCond_count.index, BsmtCond_count.values, alpha=0.9)
plt.title('Frequency Distribution of BsmtCond')
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('BsmtCond', fontsize=12)
plt.show()
prices_objects['BsmtCond'] = prices_objects['BsmtCond'].fillna(prices_objects['BsmtCond'].value_counts().index[0])
prices_objects['BsmtExposure'] = prices_objects['BsmtExposure'].fillna(
                                        prices_objects['BsmtExposure'].value_counts().index[0])
prices_objects['BsmtFinType1'] = prices_objects['BsmtFinType1'].fillna(
                                        prices_objects['BsmtFinType1'].value_counts().index[0])
prices_objects['BsmtFinType2'] = prices_objects['BsmtFinType2'].fillna(
                                        prices_objects['BsmtFinType2'].value_counts().index[0])
prices_objects['Electrical'] = prices_objects['Electrical'].fillna(
                                        prices_objects['Electrical'].value_counts().index[0])
prices_objects['FireplaceQu'] = prices_objects['FireplaceQu'].fillna(
                                        prices_objects['FireplaceQu'].value_counts().index[0])
prices_objects['GarageType'] = prices_objects['GarageType'].fillna(
                                        prices_objects['GarageType'].value_counts().index[0])
prices_objects['GarageFinish'] = prices_objects['GarageFinish'].fillna(
                                        prices_objects['GarageFinish'].value_counts().index[0])
prices_objects['GarageQual'] = prices_objects['GarageQual'].fillna(
                                        prices_objects['GarageQual'].value_counts().index[0])
prices_objects['GarageCond'] = prices_objects['GarageCond'].fillna(
                                        prices_objects['GarageCond'].value_counts().index[0])
prices_objects['MasVnrType'] = prices_objects['MasVnrType'].fillna(
                                        prices_objects['MasVnrType'].value_counts().index[0])
prices_objects.isnull().sum()
prices['LotFrontage'] = prices['LotFrontage'].fillna(prices['LotFrontage'].value_counts().index[0])
prices['MasVnrArea'] = prices['MasVnrArea'].fillna(prices['MasVnrArea'].value_counts().index[0])
prices['GarageYrBlt'] = prices['GarageYrBlt'].fillna(prices['GarageYrBlt'].value_counts().index[0])
prices.isnull().sum()
# To show the count of every category in MSZoning column
print(prices_objects['MSZoning'].value_counts())
labels = prices_objects['MSZoning'].astype('category').cat.categories.tolist()
counts = prices_objects['MSZoning'].value_counts()
sizes = [counts[var_cat] for var_cat in labels]
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True) #autopct is show the % on plot
ax1.axis('equal')
plt.show()
# To show the count of every category in MSZoning column
print(prices_objects['Exterior2nd'].value_counts())
labels = prices_objects['Exterior2nd'].astype('category').cat.categories.tolist()
counts = prices_objects['Exterior2nd'].value_counts()
sizes = [counts[var_cat] for var_cat in labels]
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True) #autopct is show the % on plot
ax1.axis('equal')
plt.show()
encoder = ce.BackwardDifferenceEncoder(cols=['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities', 
            'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2','BldgType', 'HouseStyle',
            'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','ExterQual','ExterCond',
            'Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Heating','HeatingQC',
            'CentralAir','Electrical','KitchenQual','Functional','FireplaceQu','GarageType','GarageFinish',
            'GarageQual','GarageCond','PavedDrive','SaleType','SaleCondition'])
prices_objects_BackDiffEnc = encoder.fit_transform(prices_objects)
prices_objects_BackDiffEnc.head()
Ytrain = prices.SalePrice
prices = prices.drop(['SalePrice'], axis=1)
prices_corpus = pd.concat([prices, prices_objects_BackDiffEnc], axis=1)
prices_corpus.head()
min_max_scaler = preprocessing.MinMaxScaler()
X = pd.DataFrame(min_max_scaler.fit_transform(prices_corpus), columns = prices_corpus.columns)
X.head()
prices_test = pd.read_csv('../input/test.csv')
prices_objects_test = prices_test.select_dtypes(include=['object']).copy()
prices_test = prices_test.select_dtypes(exclude=['object']).copy()
prices_test = prices_test.drop(['Id'],axis=1)
prices_objects_test = prices_objects_test.drop(['Alley','PoolQC', 'Fence', 'MiscFeature'], axis=1)

prices_objects_test['MSZoning'] = prices_objects_test['MSZoning'].fillna(
                                        prices_objects_test['MSZoning'].value_counts().index[0])
prices_objects_test['Utilities'] = prices_objects_test['Utilities'].fillna(
                                        prices_objects_test['Utilities'].value_counts().index[0])
prices_objects_test['Exterior1st'] = prices_objects_test['Exterior1st'].fillna(
                                        prices_objects_test['Exterior1st'].value_counts().index[0])
prices_objects_test['Exterior2nd'] = prices_objects_test['Exterior2nd'].fillna(
                                        prices_objects_test['Exterior2nd'].value_counts().index[0])
prices_objects_test['BsmtCond'] = prices_objects_test['BsmtCond'].fillna(
                                        prices_objects_test['BsmtCond'].value_counts().index[0])
prices_objects_test['KitchenQual'] = prices_objects_test['KitchenQual'].fillna(
                                        prices_objects_test['KitchenQual'].value_counts().index[0])
prices_objects_test['Functional'] = prices_objects_test['Functional'].fillna(
                                        prices_objects_test['Functional'].value_counts().index[0])
prices_objects_test['SaleType'] = prices_objects_test['SaleType'].fillna(
                                        prices_objects_test['SaleType'].value_counts().index[0])
prices_objects_test['BsmtQual'] = prices_objects_test['BsmtQual'].fillna(
                                        prices_objects_test['BsmtQual'].value_counts().index[0])
prices_objects_test['BsmtExposure'] = prices_objects_test['BsmtExposure'].fillna(
                                        prices_objects_test['BsmtExposure'].value_counts().index[0])
prices_objects_test['BsmtFinType1'] = prices_objects_test['BsmtFinType1'].fillna(
                                        prices_objects_test['BsmtFinType1'].value_counts().index[0])
prices_objects_test['BsmtFinType2'] = prices_objects_test['BsmtFinType2'].fillna(
                                        prices_objects_test['BsmtFinType2'].value_counts().index[0])
prices_objects_test['Electrical'] = prices_objects_test['Electrical'].fillna(
                                        prices_objects_test['Electrical'].value_counts().index[0])
prices_objects_test['FireplaceQu'] = prices_objects_test['FireplaceQu'].fillna(
                                        prices_objects_test['FireplaceQu'].value_counts().index[0])
prices_objects_test['GarageType'] = prices_objects_test['GarageType'].fillna(
                                        prices_objects_test['GarageType'].value_counts().index[0])
prices_objects_test['GarageFinish'] = prices_objects_test['GarageFinish'].fillna(
                                        prices_objects_test['GarageFinish'].value_counts().index[0])
prices_objects_test['GarageQual'] = prices_objects_test['GarageQual'].fillna(
                                        prices_objects_test['GarageQual'].value_counts().index[0])
prices_objects_test['GarageCond'] = prices_objects_test['GarageCond'].fillna(
                                        prices_objects_test['GarageCond'].value_counts().index[0])
prices_objects_test['MasVnrType'] = prices_objects_test['MasVnrType'].fillna(
                                        prices_objects_test['MasVnrType'].value_counts().index[0])


prices_test['LotFrontage'] = prices_test['LotFrontage'].fillna(prices_test['LotFrontage'].value_counts().index[0])
prices_test['MasVnrArea'] = prices_test['MasVnrArea'].fillna(prices_test['MasVnrArea'].value_counts().index[0])
prices_test['GarageYrBlt'] = prices_test['GarageYrBlt'].fillna(prices_test['GarageYrBlt'].value_counts().index[0])
prices_test['BsmtFinSF1'] = prices_test['BsmtFinSF1'].fillna(prices_test['BsmtFinSF1'].value_counts().index[0])
prices_test['BsmtFinSF2'] = prices_test['BsmtFinSF2'].fillna(prices_test['BsmtFinSF2'].value_counts().index[0])
prices_test['BsmtUnfSF'] = prices_test['BsmtUnfSF'].fillna(prices_test['BsmtUnfSF'].value_counts().index[0])
prices_test['TotalBsmtSF'] = prices_test['TotalBsmtSF'].fillna(prices_test['TotalBsmtSF'].value_counts().index[0])
prices_test['BsmtFullBath'] = prices_test['BsmtFullBath'].fillna(prices_test['BsmtFullBath'].value_counts().index[0])
prices_test['BsmtHalfBath'] = prices_test['BsmtHalfBath'].fillna(prices_test['BsmtHalfBath'].value_counts().index[0])
prices_test['GarageCars'] = prices_test['GarageCars'].fillna(prices_test['GarageCars'].value_counts().index[0])
prices_test['GarageArea'] = prices_test['GarageArea'].fillna(prices_test['GarageArea'].value_counts().index[0])
prices_objects_BackDiffEnc_test = encoder.fit_transform(prices_objects_test)
prices_corpus_test = pd.concat([prices_test, prices_objects_BackDiffEnc_test], axis=1)

Xtest = pd.DataFrame(min_max_scaler.fit_transform(prices_corpus_test), columns = prices_corpus_test.columns)
pca = PCA()
pca.fit(X)
plt.figure(1, figsize=(9, 8))
plt.clf()
plt.axes([.2, .2, .7, .7])
plt.plot(pca.explained_variance_ratio_, linewidth=2)
plt.axis('tight')
plt.xlabel('Number of Feautres')
plt.ylabel('Variance Ratio')
pca = PCA(n_components=10)
pca.fit(X)
Xtrain_pca = pca.transform(X)
pca.fit(Xtest)
Xtest_pca = pca.transform(Xtest)
Ytrain = Ytrain.astype(float)
clf = SVR(C=0.8, epsilon=0.2, kernel='poly')
#regr = RandomForestRegressor(max_depth=3)
clf.fit(Xtrain_pca, Ytrain)
#regr.fit(Xtrain_pca, Ytrain)
Ypredict = clf.predict(Xtest_pca)
#Ypredict = regr.predict(Xtest_pca)
test = pd.read_csv('../input/test.csv')
test.head()
Ypred = pd.DataFrame({'SalePrice':Ypredict})
prediction = pd.concat([test['Id'], Ypred], axis=1)
prediction = pd.concat([test['Id'], Ypred], axis=1)
prediction.head()
prediction.to_csv('predictions.csv', sep=',', index=False)
