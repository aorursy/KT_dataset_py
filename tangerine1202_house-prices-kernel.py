import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

import warnings

def ignore_warn(*args, **kwargs):

    pass

warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)



org_data = pd.read_csv('../input/train.csv').drop(['Id'], axis=1)
org_data.info()
org_data.describe()
lost = org_data.count().copy(deep=True)

lost = 1460-lost[lost<1460]

lost_pd = lost.reset_index(name='count')



plt.figure(figsize=(10, 8))

ax = sns.barplot(x='count', y='index', data=lost_pd)

ax.set(xlabel='Lost')
data = org_data.copy(deep=True)
corr_mat = data.corr()

corr_mat['SalePrice'].sort_values(ascending=False)
plt.figure(figsize=(9,9))

sns.heatmap(corr_mat, cmap='RdBu', center=0, square=True)
plt.figure(figsize=(10,10))

corr_cols = corr_mat.nlargest(10, 'SalePrice')['SalePrice']

sns.heatmap(data[corr_cols.index].corr(), annot=True, square=True)
from scipy import stats

from scipy.stats import norm, skew
sns.distplot(data['SalePrice'], fit=norm)

plt.ylabel('Frequency', fontsize=13)



plt.figure()

res = stats.probplot(data['SalePrice'], plot=plt)
data['log_SalePrice'] = np.log(data['SalePrice'])

sns.distplot(data['log_SalePrice'], fit=norm)



plt.figure()

res = stats.probplot(data['log_SalePrice'], plot=plt)
data['HouseAge'] = data['YrSold'] - data['YearBuilt']

plt.figure(figsize=(20, 20))

sns.jointplot(x='HouseAge' ,y='YearBuilt', data=data)
var = 'MSSubClass'

new_var = 'cate_MSSubClass' 

# data[var].apply(str)

data[new_var] = data[var].astype(str)

print(data[new_var].value_counts().sort_index())

plt.figure(figsize=(15, 9))

sns.boxplot(x=new_var, y='log_SalePrice', data=data)
msz = data['MSZoning'].copy()

msz.value_counts()

sns.boxplot(x='MSZoning', y='log_SalePrice', data=data)
var = 'LotFrontage'

# sns.jointplot(x='LotFrontage', y='log_SalePrice', data=data)

ax = sns.kdeplot(data[var])

ax.set(xlabel='LotFrontage')

print('skew', data[var].skew())

print('kurt', data[var].kurt())

plt.show()



loged = np.log(data[var])

print('log_skew', loged.skew())

print('log_kurt', loged.kurt())

print('median', loged.median())

log_ax = sns.kdeplot(loged)

log_ax.set(xlabel='log LotFrontage')

plt.show()



mean = data[var].mean()

std = data[var].std()

z_score = (data[var]-mean)/std

print('std_skew', z_score.skew())

print('std_kurt', z_score.kurt())

print('median', z_score.median())

std_ax = sns.kdeplot(z_score)

std_ax.set(xlabel='z score LotFrontage')

# z_score
var = 'Street'

print(data[var].value_counts())

sns.boxplot(x=var, y='log_SalePrice', data=data)
var = 'Alley'

data[var].value_counts()

sns.boxplot(x=var, y='log_SalePrice', data=data, order=['Pave', 'Grvl'])
df = data.copy(deep=True)

##### type 0 

print(df['Street'].value_counts())

sns.boxplot(x='Street', y='log_SalePrice', data=df)

plt.show()





####################



##### type 1

df['Alley_filled'] = df['Alley'].fillna(df['Street'])

print(df['Alley'].value_counts())

print(df['Alley_filled'].value_counts())

sns.boxplot(x='Alley_filled', y='log_SalePrice', data=df)

plt.show()



#### type 2

def fn_roadtype1(s, a):

    if s==a:

        return s

    elif s!=a:

        return s+a

    

df['RoadType1'] = df.apply(lambda row: fn_roadtype1(row['Street'], row['Alley_filled']), axis=1)

print(df['RoadType1'].value_counts())

sns.boxplot(x='RoadType1', y='log_SalePrice', data=df)

plt.show()



#### type 3

def isnan(x):

    return x!=x



def fn_roadtype2(s, a):

    if isnan(a):

        return s

    elif a==s:

        return 'both_'+s

    elif a!=s:

        return s+a



df['RoadType2'] = df.apply(lambda row: fn_roadtype2(row['Street'], row['Alley']), axis=1)

print(df['RoadType2'].value_counts())

sns.boxplot(x='RoadType2', y='log_SalePrice', data=df)
var = 'LotShape'

print(data[var].value_counts())

sns.boxplot(x=var, y='log_SalePrice', data=data, order=['Reg', 'IR1', 'IR2', 'IR3'])
var = 'LandContour'

print(data[var].value_counts())

sns.boxplot(x=var, y='log_SalePrice', data=data, order=['Lvl', 'Bnk', 'HLS', 'Low'])
var = 'Utilities'

print(data[var].value_counts())
var = 'LotConfig'

print(data[var].value_counts())

sns.boxplot(x=var, y='log_SalePrice', data=data)
var = 'LandSlope'

print(data[var].value_counts())

sns.boxplot(x=var, y='log_SalePrice', data=data, order=['Gtl', 'Mod', 'Sev'])
var = 'Neighborhood'

print(data[var].value_counts())

plt.figure(figsize=(26,12))

sns.boxplot(x=var, y='log_SalePrice', data=data, )
var = 'Condition1'

print(data[var].value_counts())

plt.figure(figsize=(10,6))

sns.boxplot(x=var, y='log_SalePrice', data=data, order=['Norm', 'Feedr', 'Artery', 'RRAn', 'PosN', 'RRAe', 'PosA', 'RRNn', 'RRNe'])
df = data.copy(deep=True)

def fn_condition(x):

    cond = x['Condition1']

    if cond in ('RRNn', 'RRAn', 'RRNe', 'RRAe'):

        return 'Near_Railroad'

    elif cond in ('PosN', 'PosA'):

        return 'Near_PositiveSite'

    else:

        return cond



df['Condition'] = df.apply(lambda x: fn_condition(x), axis=1)

print(df['Condition'].value_counts())

plt.figure(figsize=(10,6))

sns.boxplot(x='Condition', y='log_SalePrice', data=df)
var = 'Condition2'

print(data[var].value_counts())

plt.figure(figsize=(10,6))

sns.boxplot(x=var, y='log_SalePrice', data=data, order=['Norm', 'Feedr', 'Artery', 'RRAn', 'PosN', 'RRAe', 'PosA', 'RRNn', 'RRNe'])
var = 'BldgType'

print(data[var].value_counts())

sns.boxplot(x=var, y='log_SalePrice', data=data, order=['1Fam', '2fmCon', 'Duplex', 'TwnhsE', 'Twnhs'])
var = 'HouseStyle'

print(data[var].value_counts())

plt.figure(figsize=(10, 12))

sns.boxplot(x=var, y='log_SalePrice', data=data, order=data[var].value_counts().sort_index().index)
df = data.copy(deep=True)

def fn_apply(x):

    cond = x['HouseStyle']

    if cond in ('1Story', '1.5Fin'):

        return '1_fin'

    elif cond in ('2Story', '2.5Fin'):

        return '2_fin'

    elif cond in ('1.5Unf', '2.5Unf'):

        return 'unf'

    else:

        return cond



new_var = 'comb_HouseStyle'

df[new_var] = df.apply(lambda x: fn_apply(x), axis=1)

print(df[new_var].value_counts())

plt.figure(figsize=(6,10))

sns.boxplot(x=new_var, y='log_SalePrice', data=df)
var = 'OverallQual'

print(data[var].value_counts())

plt.figure(figsize=(12, 6))

sns.boxplot(x=var, y='log_SalePrice', data=data, order=data[var].value_counts().sort_index().index)
var = 'OverallCond'

print(data[var].value_counts())

plt.figure(figsize=(12, 6))

sns.boxplot(x=var, y='log_SalePrice', data=data, order=data[var].value_counts().sort_index().index)
var = 'YearBuilt'

# print(data[var].value_counts())

plt.figure(figsize=(20, 10))

plt.xticks(rotation='90')

# sns.jointplot(x=var, y='log_SalePrice', data=data)

sns.boxplot(x=var, y='log_SalePrice', data=data)
var = 'YearRemodAdd'

# print(data[var].value_counts())

plt.figure(figsize=(12, 6))

sns.jointplot(x=var, y='log_SalePrice', data=data, kind='reg')
var = 'BsmtExposure'

print(data[var].value_counts())

sns.boxplot(x=var, y='log_SalePrice', data=data, \

            order=['Gd', 'Av', 'Mn', 'No', 'NA'])
var = 'BsmtFinType1'

print(data[var].value_counts())

plt.figure(figsize=(8,12))

sns.boxplot(x=var, y='log_SalePrice', data=data, \

           order=['GLQ', 'ALQ', 'BLQ', 'Rec', 'LwQ', 'Unf', 'NA'])
var = 'BsmtFinSF1'

print(data[var].value_counts()[0])

plt.figure(figsize=(8,12))

sns.jointplot(x=var, y='log_SalePrice', data=data, kind='reg')
# data[['BsmtFinSF1', 'BsmtFinType1']]

gp = data.groupby('BsmtFinSF1')

gp.get_group(0)['BsmtFinType1'].value_counts(dropna=False)
var = 'BsmtFinType2'

print(data[var].value_counts(dropna=False))

plt.figure(figsize=(8,12))

sns.boxplot(x=var, y='log_SalePrice', data=data, \

           order=['GLQ', 'ALQ', 'BLQ', 'Rec', 'LwQ', 'Unf', 'NA'])
var = 'BsmtFinSF2'

print(data[var].value_counts()[0])

plt.figure(figsize=(8,12))

sns.jointplot(x=var, y='log_SalePrice', data=data, kind='reg')
# data[['BsmtFinSF2', 'BsmtFinType2']]

gp = data.groupby('BsmtFinSF1')

gp.get_group(0)['BsmtFinSF2'].value_counts(dropna=False)
var = 'Electrical'

print(data[var].value_counts())

plt.figure(figsize=(8,12))

sns.boxplot(x=var, y='log_SalePrice', data=data)
var = 'GarageFinish'

print(data[var].value_counts())

sns.boxplot(x=var, y='log_SalePrice', data=data)
var = 'PavedDrive'

print(data[var].value_counts())

sns.boxplot(x=var, y='log_SalePrice', data=data)
var = 'Fence'

print(data[var].value_counts(dropna=False))

sns.boxplot(x=var, y='log_SalePrice', data=data)
var = 'MoSold'

# print(data[var].value_counts().sort_index())

plt.figure(figsize=(12, 8))

# sns.barplot(x=var, y='SalePrice', data=data, order=np.sort(data[var].unique()))

sns.boxplot(x=var, y='log_SalePrice', data=data, order=np.sort(data[var].unique()))
var = 'YrSold'

# print(data[var].value_counts().sort_index())

plt.figure(figsize=(6, 8))

sns.boxplot(x=var, y='log_SalePrice', data=data, order=np.sort(data[var].unique()))
var = 'SaleType'

print(data[var].value_counts(dropna=False))

plt.figure(figsize=(15, 8))

sns.boxplot(x=var, y='log_SalePrice', data=data)
var = 'SaleCondition'

print(data[var].value_counts(dropna=False))

# plt.figure(figsize=(15, 8))

sns.boxplot(x=var, y='log_SalePrice', data=data)
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder, StandardScaler, PowerTransformer, RobustScaler

from sklearn.pipeline import make_pipeline

from sklearn.utils import shuffle
def fill_na(X):



    filled_X = X.copy()



    # LotFrontage

    """

    Fill in missing values by the median LotFrontage of the neighborhood.

    

    https://i.stack.imgur.com/sgCn1.jpg

    

    df.groupby('col_1')['col_2'] -> looks like split section

    df.groupby('col_1')['col_2'].transform(lambda k: ...) -> k get the split_1, 2, 3

    

    """

    filled_X.groupby('Neighborhood')['LotFrontage'].transform(lambda k: k.fillna(k.median()))

#     filled_X['LotFrontage'].fillna(value=filled_X['LotFrontage'].median(), inplace=True)

    

    # Alley

    filled_X['Alley'].fillna(value='NA', inplace=True)

    

    # MasVnrType

    filled_X['MasVnrType'].fillna(value='NA', inplace=True)

    # MasVnrArea

    filled_X['MasVnrArea'].fillna(value=0, inplace=True)

    

    # BsmtQual

    filled_X['BsmtQual'].fillna(value="NA", inplace=True)

    # BsmtCond

    filled_X['BsmtCond'].fillna(value="NA", inplace=True)

    # BsmtExposure

    filled_X["BsmtExposure"].fillna(value="NA", inplace=True)

    # BsmtFinType1

    filled_X["BsmtFinType1"].fillna(value="NA", inplace=True)

    # BsmtFinType2

    filled_X["BsmtFinType2"].fillna(value="NA", inplace=True)

    

    # Electrical

    filled_X["Electrical"].fillna("SBrkr", inplace=True)

    

    # FireplaceQu

    filled_X["FireplaceQu"].fillna(value="NA", inplace=True)

    

    # GargaeYrBlt

    filled_index = filled_X[filled_X["GarageYrBlt"].isna()].index

    YrBlt = filled_X.loc[filled_index, "YearBuilt"]

    filled_X.loc[filled_index, "GarageYrBlt"] = YrBlt

    

    # GarageType

    filled_X["GarageType"].fillna(value="NA", inplace=True)

    # GarageFinish

    filled_X["GarageFinish"].fillna(value="NA", inplace=True)

    # GarageQual

    filled_X["GarageQual"].fillna(value="NA", inplace=True)

    # GarageCond

    filled_X["GarageCond"].fillna(value="NA", inplace=True)

    

    # PoolQC

    filled_X["PoolQC"].fillna(value="NA", inplace=True)

    

    # Fence

    filled_X["Fence"].fillna(value="NA", inplace=True)

    

    # MiscFeature

    filled_X["MiscFeature"].fillna(value="NA", inplace=True)

    

    return filled_X
def process_data(X, y, num_imputer=None, scaler=None, cat_imputer=None, onehoter=None):

    

    # fill missing value

    X = fill_na(X)

    

    ###### Transform

    

    # num -> cat

    X['MSSubClass'] = X['MSSubClass'].astype(str)

    X['MoSold'] = X['MoSold'].astype(str)

    X['YrSold'] = X['YrSold'].astype(str)

#     X['YearBuilt'] = X['YearBuilt'].astype(str)

    

    # cat -> ord  

    cat_to_ord_cols_1 = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', \

                         'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']

    cat_to_ord_dict_1 = {'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'NA':0}

    for col in cat_to_ord_cols_1:

        X[col] = X[col].map(cat_to_ord_dict_1, )

    

    X['BsmtExposure'] = X['BsmtExposure'].map({'Gd':4, 'Av':3, 'Mn':2, 'No':1, 'NA':0})

    X['CentralAir'] = X['CentralAir'].map({'Y':1, 'N':0})

    X['GarageFinish'] = X['GarageFinish'].map({'Fin':3, 'RFn':2, 'Unf':1, 'NA':0})

    X['PavedDrive'] = X['PavedDrive'].map({'Y':1, 'P':0.5, 'N':0})

    X['BsmtFinType1'] = X['BsmtFinType1'].map({'GLQ':5, 'ALQ':4, 'BLQ':3, 'Rec':2, 'LwQ':1, 'Unf':0})

    X['BsmtFinType2'] = X['BsmtFinType2'].map({'GLQ':5, 'ALQ':4, 'BLQ':3, 'Rec':2, 'LwQ':1, 'Unf':0})

    X['Electrical'] = X['Electrical'].map({'SBrkr':4, 'FuseA':3, 'FuseF':2, 'FuseP':1, 'Mix':0})

    X['Fence'] = X['Fence'].map({'GdPrv':4, 'MnPrv':3, 'FdWo':2, 'MnWw':1, 'NA':0})

    

    # add new feature

    X['TotalSF'] = X['TotalBsmtSF'] + X['1stFlrSF'] + X['2ndFlrSF']

    

    ######



    # split num/cat data

    num_X = X.select_dtypes(include=np.number)

    cat_X = X.select_dtypes(exclude=np.number)

#     print(num_X.columns, cat_X.columns, sep='\n')

    

    # impute num data

    if num_imputer==None:

        num_imputer = SimpleImputer(strategy='median')

        imputed_num_X = num_imputer.fit_transform(num_X)

        imputed_num_X = pd.DataFrame(imputed_num_X, index=num_X.index, columns=num_X.columns)

    else:

        imputed_num_X = num_imputer.transform(num_X)

        imputed_num_X = pd.DataFrame(imputed_num_X, index=num_X.index, columns=num_X.columns)

    

    

    imputed_num_X += 1

    # scale num data

    if scaler==None:

        scaler = make_pipeline(PowerTransformer(method='box-cox'),  RobustScaler())

        scaled_num_X = scaler.fit_transform(imputed_num_X)

        scaled_num_X = pd.DataFrame(scaled_num_X, index=num_X.index, columns=num_X.columns)

    else:

        scaled_num_X = scaler.transform(imputed_num_X)

        scaled_num_X = pd.DataFrame(scaled_num_X, index=num_X.index, columns=num_X.columns)

    

    

    # impute cat data

    if cat_imputer==None:

        cat_imputer = SimpleImputer(strategy='most_frequent')

        imputed_cat_X = cat_imputer.fit_transform(cat_X)

    else:

        imputed_cat_X = cat_imputer.transform(cat_X)   

    

    # onehotencode cat data

    if onehoter==None:

        onehoter = OneHotEncoder(sparse=False, handle_unknown='ignore')

        onehot_cat_X = pd.DataFrame(onehoter.fit_transform(imputed_cat_X), index=cat_X.index)

    else:

        onehot_cat_X = pd.DataFrame(onehoter.transform(imputed_cat_X), index=cat_X.index)



    ######

    

    fin_num_X = scaled_num_X

    fin_cat_X = onehot_cat_X

    

    processed_X = pd.concat([fin_num_X, fin_cat_X], axis=1, sort=False)

    processed_X.info()

    

    ######

    

    if y is not None:

        processed_y = np.log(y)

    else:

        processed_y = None

    

    return processed_X, processed_y, num_imputer, scaler, cat_imputer, onehoter
data = org_data.copy(deep=True)

data = shuffle(data, random_state=1).reset_index(drop=True)
X, y, num_imputer, scaler, cat_imputer, onehoter = process_data( X=data.drop(['SalePrice'], axis=1), y=data['SalePrice'])
X.head()
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor

from sklearn.linear_model import Lasso, RidgeCV, ElasticNetCV, ElasticNet

from sklearn.tree import DecisionTreeRegressor

from sklearn.svm import SVR

from xgboost import XGBRegressor

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone



from sklearn.model_selection import cross_val_score

from sklearn.metrics import mean_squared_error



from sklearn.model_selection import KFold

import keras
n_cv = 5

def rmse_cv(model, X, y):

    rmse = np.sqrt(-cross_val_score(model, X, y, cv=n_cv, scoring='neg_mean_squared_error'))

    return rmse
dnn_X = np.array(X)

dnn_y = np.array(y)



train_X, test_X, train_y, test_y = train_test_split(dnn_X, dnn_y)
def get_model():

    dnn = keras.Sequential()

    dnn.add(keras.layers.Dense(units=100, activation='relu'))

    dnn.add(keras.layers.Dense(100, activation='relu'))

    dnn.add(keras.layers.Dense(1))

    dnn.compile(loss='mean_squared_error', optimizer='adam') #, metrics='mean_squared_error'

    return dnn
kf = KFold(n_splits=5, random_state=1)

score = np.array([])



batch_size = 16

epochs = 20



fold_cnt = 1

for train_id, test_id in kf.split(dnn_X, dnn_y):

    print("\nFold {}\n".format(fold_cnt))

    fold_cnt += 1

    

    dnn = get_model()

    dnn.fit(dnn_X[train_id], dnn_y[train_id], batch_size=batch_size, epochs=epochs, validation_data=(dnn_X[test_id], dnn_y[test_id]))

    score = np.append(score, (dnn.evaluate(dnn_X[test_id], dnn_y[test_id])))



print('DNN error: {:.4f} ({:.4f})'.format(score.mean(), score.std()))
xgb = XGBRegressor(learning_rate=.2, max_depth=3, n_estimators=100 , random_state=1)



score = rmse_cv(xgb, X, y)



print('Xgboost error: {:.4f} ({:.4f})'.format(score.mean(), score.std()))
gb = GradientBoostingRegressor(learning_rate=.1, n_estimators=100, random_state=1)



score = rmse_cv(gb, X, y)



print('Gradient Boosting error: {:.4f} ({:.4f})'.format(score.mean(), score.std()))
ridge = RidgeCV()



score = rmse_cv(ridge, X, y)



print('Ridge error: {:.4f} ({:.4f})'.format(score.mean(), score.std()))
lasso = Lasso(alpha=0.0005, random_state=1)



score = rmse_cv(lasso, X, y)



print('Lasso error: {:.4f} ({:.4f})'.format(score.mean(), score.std()))
##### l1_ratio = [0.1, 0.3, 0.5, 0.7, 0.9, 1]

elastic = ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=1)



score = rmse_cv(elastic, X, y)



print('ElasticNet error: {:.4f} ({:.4f})'.format(score.mean(), score.std()))
svm = SVR()



score = rmse_cv(svm, X, y)



print('SVM error: {:.4f} ({:.4f})'.format(score.mean(), score.std()))
class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):

    def __init__(self, model_dict):

        self.model_dict = model_dict

    

    def fit(self, X, y):

        for name, model in self.model_dict.items():

            if name=='dnn':

                model.fit(np.array(X), np.array(y), batch_size=16, epochs=20, verbose=0)

            else:

                model.fit(X, y)

            

        return self

    

    def predict(self, X):

        predictions = np.column_stack([model.predict(X) for name, model in self.model_dict.items()])

        return np.mean(predictions, axis=1)
dnn = get_model()

averaged_models = AveragingModels({'xgb':xgb, 'gb':gb, 'ridge':ridge, 'lasso':lasso, 'elastic':elastic, 'svm':svm, 'dnn':dnn})



score = rmse_cv(averaged_models, X, y)



print('averged models error: {:.4f} ({:.4f})'.format(score.mean(), score.std()))
train_data = org_data.copy()

train_data = shuffle(train_data).reset_index(drop=True)



org_test_data = pd.read_csv('../input/test.csv')

test_data = org_test_data.drop(['Id'], axis=1).copy()
train_X, train_y, num_imputer, scaler, cat_imputer, onehoter = process_data(train_data.drop(['SalePrice'], axis=1), train_data['SalePrice'])

print()

test_X, _, _, _, _, _ = process_data(test_data, None, num_imputer, scaler, cat_imputer, onehoter )
def rmse_error(y, y_hat):

    return np.sqrt(mean_squared_error(y, y_hat))
dnn = get_model()

final_model = AveragingModels({'xgb':xgb, 'gb':gb, 'ridge':ridge, 'lasso':lasso, 'elastic':elastic, 'svm':svm, 'dnn':dnn})

final_model.fit(train_X, train_y)

pred = final_model.predict(train_X)

print(rmse_error(pred, train_y))



final_y_hat = final_model.predict(test_X)

inverse_y_hat = np.exp(final_y_hat)
final_y_hat
inverse_y_hat
# The lines below shows how to save predictions in format used for competition scoring

# Just uncomment them.



output = pd.DataFrame({'Id': org_test_data.Id,

                      'SalePrice': inverse_y_hat})

output.to_csv('submission.csv', index=False)