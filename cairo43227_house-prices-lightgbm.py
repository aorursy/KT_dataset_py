import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
train.shape
for col, value in train.iteritems():
    if train[col].isna().any():
        print(col, train[col].isna().mean())
# colunas = ["Alley", "PoolQC", "MiscFeature", "Fence", "Id", "MiscVal", "PoolArea", "ScreenPorch", "3SsnPorch",
#           "EnclosedPorch", "LowQualFinSF", "BsmtFinSF2", "Utilities"]

columns = ['Id','Alley', 'FireplaceQu', 'MiscFeature', 'Fence', 'PoolQC']

Y = train["SalePrice"]
X = train.drop(columns+["SalePrice"], axis=1)
X_test = test.drop(columns, axis=1)
train.dtypes.value_counts()
# X['FireplaceQu'] = X['FireplaceQu'].replace(['Fa','Ex','Po'], 'Rare')
# X['FireplaceQu'].fillna('Missing', inplace=True)
X['LotFrontage'].fillna(X['LotFrontage'].mean(), inplace=True)
X['BsmtQual'].fillna(X['BsmtQual'].mode()[0], inplace=True)
X['BsmtCond'].fillna(X['BsmtCond'].mode()[0], inplace=True)
X['BsmtExposure'].fillna(X['BsmtExposure'].mode()[0], inplace=True)
X['BsmtFinType1'].fillna(X['BsmtFinType1'].mode()[0], inplace=True)
X['MasVnrType'].fillna(X['MasVnrType'].mode()[0], inplace=True)
X['MasVnrArea'].fillna(0.0, inplace=True)
X['BsmtFinType2'].fillna(X['BsmtFinType2'].mode()[0], inplace=True)
X['BsmtExposure'].fillna(X['BsmtExposure'].mode()[0], inplace=True)
X['Electrical'].fillna(X['Electrical'].mode()[0], inplace=True)
X['GarageType'].fillna(X['GarageType'].mode()[0], inplace=True)
X['GarageYrBlt'].fillna(X['GarageYrBlt'].median(), inplace=True)
X['GarageFinish'].fillna(X['GarageFinish'].mode()[0], inplace=True)
X['GarageQual'].fillna(X['GarageQual'].mode()[0], inplace=True)
X['GarageCond'].fillna(X['GarageCond'].mode()[0], inplace=True)

# X_test['FireplaceQu'] = X_test['FireplaceQu'].replace(['Fa','Ex','Po'], 'Rare')
# X_test['FireplaceQu'].fillna('Missing', inplace=True)
X_test['LotFrontage'].fillna(X['LotFrontage'].mean(), inplace=True)
X_test['BsmtQual'].fillna(X['BsmtQual'].mode()[0], inplace=True)
X_test['BsmtCond'].fillna(X['BsmtCond'].mode()[0], inplace=True)
X_test['BsmtExposure'].fillna(X['BsmtExposure'].mode()[0], inplace=True)
X_test['BsmtFinType1'].fillna(X['BsmtFinType1'].mode()[0], inplace=True)
X_test['MasVnrType'].fillna(X['MasVnrType'].mode()[0], inplace=True)
X_test['MasVnrArea'].fillna(0.0, inplace=True)
X_test['BsmtFinType2'].fillna(X['BsmtFinType2'].mode()[0], inplace=True)
X_test['BsmtExposure'].fillna(X['BsmtExposure'].mode()[0], inplace=True)
X_test['Electrical'].fillna(X['Electrical'].mode()[0], inplace=True)
X_test['GarageType'].fillna(X['GarageType'].mode()[0], inplace=True)
X_test['GarageYrBlt'].fillna(X['GarageYrBlt'].median(), inplace=True)
X_test['GarageFinish'].fillna(X['GarageFinish'].mode()[0], inplace=True)
X_test['GarageQual'].fillna(X['GarageQual'].mode()[0], inplace=True)
X_test['GarageCond'].fillna(X['GarageCond'].mode()[0], inplace=True)

X_test['MSZoning'].fillna(X['MSZoning'].mode()[0], inplace=True)
X_test['Exterior1st'].fillna(X['Exterior1st'].mode()[0], inplace=True)
X_test['Exterior2nd'].fillna(X['Exterior2nd'].mode()[0], inplace=True)
X_test['BsmtFinSF1'].fillna(0.0, inplace=True)
X_test['BsmtUnfSF'].fillna(0.0, inplace=True)
X_test['TotalBsmtSF'].fillna(0.0, inplace=True)
X_test['BsmtFullBath'].fillna(0.0, inplace=True)
X_test['BsmtHalfBath'].fillna(0.0, inplace=True)
X_test['KitchenQual'].fillna(X['KitchenQual'].mode()[0], inplace=True)
X_test['Functional'].fillna(X['Functional'].mode()[0], inplace=True)
X_test['GarageCars'].fillna(X['GarageCars'].mean(), inplace=True)
X_test['GarageArea'].fillna(0.0, inplace=True)
X_test['SaleType'].fillna(X['SaleType'].mode()[0], inplace=True)
X_test['Utilities'].fillna(X['Utilities'].mode()[0], inplace=True)
X_test['BsmtFinSF2'].fillna(X['BsmtFinSF2'].mode()[0], inplace=True)

for col, value in X.iteritems():
    if X[col].isna().any():
        print(col, X[col].isna().sum())
for col, value in X_test.iteritems():
    if X_test[col].isna().any():
        print(col, X_test[col].isna().mean())
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

columns = X.select_dtypes('object').columns

lb = OrdinalEncoder()
lb.fit(X[columns])
X[columns] = lb.transform(X[columns])    
X_test[columns] = lb.transform(X_test[columns]) 
def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    return col_corr

corr_features = correlation(X, 0.8)

X.drop(labels=corr_features, axis=1, inplace=True)
X_test.drop(labels=corr_features, axis=1, inplace=True)

X.shape
from sklearn.feature_selection import VarianceThreshold

vt = VarianceThreshold(threshold=0.1)
vt.fit(X)

X = vt.transform(X)
X_test = vt.transform(X_test)

X.shape
!pip install feature_engine
from feature_engine.outlier_removers import Winsorizer

capper = Winsorizer(tail='both')
X = capper.fit_transform(pd.DataFrame(X))
#Pipeline

ss1=StandardScaler()
ss1.fit(X)
X = ss1.transform(X)
X_test = ss1.transform(X_test)

ss2=StandardScaler()
ss2.fit(Y.values.reshape(-1, 1))
Y = ss2.transform(Y.values.reshape(-1, 1))
Y = Y.reshape(Y.shape[0],)
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

X_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
import lightgbm as lgb

model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

sfs1 = SFS(model_lgb, 
           k_features=10, 
           forward=True, 
           floating=False, 
           verbose=2,
           scoring='r2',
           cv=3)

sfs1 = sfs1.fit(X_train, y_train)
sfs1.k_feature_idx_
X_train = X_train[:,list(sfs1.k_feature_idx_)]
x_test = x_test[:,list(sfs1.k_feature_idx_)]
X_test = X_test[:,list(sfs1.k_feature_idx_)]
X_ = X[:,list(sfs1.k_feature_idx_)]
model_lgb.fit(X_train, y_train)

y_pred = model_lgb.predict(x_test)
evaluate = np.sqrt(mean_squared_error(y_test, y_pred))
evaluate2 = r2_score(y_test, y_pred)

print("RMSE {} - R2 {}".format(evaluate, evaluate2))
model_lgb.fit(X_, Y)

preds = model_lgb.predict(X_test)
preds = ss2.inverse_transform(preds.reshape(-1, 1))
preds = preds.reshape(preds.shape[0],)
preds = preds.round(2)
preds
!pip install --force-reinstall pandas==0.25.3

df = pd.DataFrame({'Id':test.Id, 'SalePrice': preds})
df.to_csv('house.csv', index=False)
df.head()