import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline  
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler, RobustScaler
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)
train_ID = train['Id']
test_ID = test['Id']

train.drop("Id", axis = 1, inplace = True)
test.drop("Id", axis = 1, inplace = True)
train.head()
test.head()
ntrain = train.shape[0]
ntest = test.shape[0]
y_train = train.SalePrice.values
all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop(['SalePrice'], axis=1, inplace=True)
print("all_data size is : {}".format(all_data.shape))
all_cols = all_data.columns
num_cols = all_data._get_numeric_data().columns
print("Numerical columns: ", num_cols)
cat_cols = list( set(all_cols) -  set(num_cols) )
print("Categorical columns: ",cat_cols)
df_missing = all_data.isnull().sum(axis=0).reset_index()
df_missing.columns = ['column_name', 'missing_count']
df_missing['missing_ratio'] = df_missing['missing_count'] / all_data.shape[0]
# print(df_missing[df_missing['missing_ratio']>.05])
print(df_missing[df_missing['missing_ratio']>0 ])

df_num  = all_data[num_cols]
missing_num = df_num.isnull().sum(axis=0).reset_index()
missing_num.columns = ['column_name', 'missing_count']
missing_num['missing_ratio'] = missing_num['missing_count'] / train.shape[0]
print(missing_num[missing_num['missing_ratio']>.0])
all_data['GarageYrBlt'] = all_data['GarageYrBlt'].fillna(all_data['GarageYrBlt'].median() )
all_data['MasVnrArea'] = all_data['MasVnrArea'].fillna(all_data['MasVnrArea'].mean() )
all_data['LotFrontage'] = all_data['LotFrontage'].fillna(all_data['LotFrontage'].mean() )

all_data['BsmtFinSF1'] = all_data['BsmtFinSF1'].fillna(all_data['BsmtFinSF1'].median() )
all_data['BsmtFinSF2'] = all_data['BsmtFinSF2'].fillna(all_data['BsmtFinSF2'].mean() )
all_data['BsmtFullBath'] = all_data['BsmtFullBath'].fillna(all_data['BsmtFullBath'].mean() )
all_data['BsmtHalfBath'] = all_data['BsmtHalfBath'].fillna(all_data['BsmtHalfBath'].mean() )
all_data['BsmtUnfSF'] = all_data['BsmtUnfSF'].fillna(all_data['BsmtUnfSF'].mean() )
all_data['GarageArea'] = all_data['GarageArea'].fillna(all_data['GarageArea'].mean() )
all_data['GarageCars'] = all_data['GarageCars'].fillna(all_data['GarageCars'].mean() )
all_data['TotalBsmtSF'] = all_data['TotalBsmtSF'].fillna(all_data['TotalBsmtSF'].mean() )

df_cat  = all_data[cat_cols].copy()
missing_cat = df_cat.isnull().sum(axis=0).reset_index()
missing_cat.columns = ['column_name', 'missing_count']
missing_cat['missing_ratio'] = missing_cat['missing_count'] / train.shape[0]
print(missing_cat[missing_cat['missing_ratio']>.0])
all_data.GarageCond.fillna(train.GarageCond.value_counts().idxmax(), inplace = True)

all_data.GarageCond.fillna(train.GarageCond.value_counts().idxmax(), inplace = True)
all_data.Electrical.fillna(train.Electrical.value_counts().idxmax(), inplace = True)
all_data.BsmtExposure.fillna(train.BsmtExposure.value_counts().idxmax(), inplace = True)
all_data.FireplaceQu.fillna(train.FireplaceQu.value_counts().idxmax(), inplace = True)
all_data.BsmtFinType2.fillna(train.BsmtFinType2.value_counts().idxmax(), inplace = True)
all_data.PoolQC.fillna(train.PoolQC.value_counts().idxmax(), inplace = True)
all_data.Alley.fillna(train.Alley.value_counts().idxmax(), inplace = True)
all_data.BsmtFinType1.fillna(train.BsmtFinType1.value_counts().idxmax(), inplace = True)
all_data.MasVnrType.fillna(train.MasVnrType.value_counts().idxmax(), inplace = True)
all_data.GarageFinish.fillna(train.GarageFinish.value_counts().idxmax(), inplace = True)
all_data.BsmtCond.fillna(train.BsmtCond.value_counts().idxmax(), inplace = True)
all_data.GarageQual.fillna(train.GarageQual.value_counts().idxmax(), inplace = True)
all_data.Fence.fillna(train.Fence.value_counts().idxmax(), inplace = True)
all_data.BsmtQual.fillna(train.BsmtQual.value_counts().idxmax(), inplace = True)
all_data.GarageType.fillna(train.GarageType.value_counts().idxmax(), inplace = True)
all_data.MiscFeature.fillna(train.MiscFeature.value_counts().idxmax(), inplace = True)

all_data.Exterior2nd.fillna(train.Exterior2nd.value_counts().idxmax(), inplace = True)
all_data.MSZoning.fillna(train.MSZoning.value_counts().idxmax(), inplace = True)
all_data.Utilities.fillna(train.Utilities.value_counts().idxmax(), inplace = True)
all_data.Exterior1st.fillna(train.Exterior1st.value_counts().idxmax(), inplace = True)
all_data.SaleType.fillna(train.SaleType.value_counts().idxmax(), inplace = True)
all_data.Functional.fillna(train.Functional.value_counts().idxmax(), inplace = True)
all_data.KitchenQual.fillna(train.KitchenQual.value_counts().idxmax(), inplace = True)

print( "Number of missing values:", all_data.isnull().values.sum() )
le = LabelEncoder()
for i in cat_cols:
    all_data[i] = le.fit_transform(all_data[i])
X_train = all_data[:train.shape[0]]
X_test = all_data[train.shape[0]:]
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test) 
kf = KFold(n_splits = 5, random_state=1001)
for i, (train_index, val_index) in enumerate(kf.split(X_train, y_train)):
#     start_time = timer(None)
    Xtrain, Xval = X_train[train_index], X_train[val_index]
    ytrain, yval = y_train[train_index], y_train[val_index]
    
    model = RandomForestRegressor(n_estimators=100, oob_score=True, random_state=42)
    clf = model.fit(Xtrain, ytrain)
    
    fimp = clf.feature_importances_  
    sfeats = all_data.columns[np.argsort(fimp)[::-1]][0:17]
    sfloc  = [all_data.columns.get_loc(sfeats[i]) for i in range(len(sfeats))]
#     print(sfeats)
    
    Xtrain2 = Xtrain[:,sfloc]
    Xval2 = Xval[:,sfloc]

    model = RandomForestRegressor(n_estimators=100, oob_score=True, random_state=42)
    model.fit(Xtrain2, ytrain)
    scores_val = model.predict(Xval2)
    RMSE = np.sqrt(mean_squared_error(yval, scores_val))
    R2 = r2_score(yval, scores_val)
    
    print('\n Fold %02d RMSE: %.6f' % ((i + 1), RMSE))
    print('R2', R2)
model = RandomForestRegressor(n_estimators=500, oob_score=True, random_state=42)
model.fit(X_train, y_train)
pred = model.predict(X_test)
solution = pd.DataFrame({"id":test_ID, "SalePrice":pred})
solution.to_csv("rf2.csv", index = False)

