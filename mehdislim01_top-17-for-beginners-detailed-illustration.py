#Importing what we are going to use

import pandas as pd, numpy as np, seaborn as sns

from scipy.stats import norm

from sklearn.base import clone

from sklearn.kernel_ridge import KernelRidge

from sklearn.preprocessing import RobustScaler,PowerTransformer, OrdinalEncoder

from sklearn.pipeline import make_pipeline

from sklearn.linear_model import ElasticNetCV, Lasso, LinearRegression 

from sklearn.metrics import mean_squared_log_error

from sklearn.model_selection import cross_val_score, KFold

from xgboost import XGBRegressor

from catboost import CatBoostRegressor
#importing our training and testing data

train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
sns.scatterplot(x='GrLivArea', y='SalePrice' , data=train)
train = train[train.loc[:, 'GrLivArea'] < 4000] #deleting the outliers tht we've talked about.

#meringing the two data sets together so we preprocess them together

y  = train['SalePrice']

train.drop('SalePrice', axis=1, inplace=True)

data = train.append(test)
sns.distplot(y, fit=norm) #as you can see the target value is skewed now let's log it 
y = np.log(y) 

sns.distplot(y, fit=norm)#it's so much better now
#Checking how much missing values for each column we have

col_nnan = pd.DataFrame(data.isna().sum()).sort_values(0, ascending=False)

col_nnan.iloc[:34]
data['MasVnrType'].fillna('None+', inplace=True)

data['MasVnrArea'].fillna(data['MasVnrArea'].mean(), inplace=True)

data['BsmtFinType1'].fillna('None', inplace=True)

data['BsmtFinType2'].fillna('None', inplace=True)

data['BsmtCond'].fillna('None', inplace=True)

data['BsmtQual'].fillna('None', inplace=True)

data['BsmtExposure'].fillna('None', inplace=True)

data['Fence'].fillna('None', inplace=True)

data['LotFrontage'].fillna(0, inplace=True)

data['FireplaceQu'].fillna('None', inplace=True)

data['Alley'].fillna('None', inplace=True)

data['PoolQC'].fillna('None', inplace=True)

data['MiscFeature'].fillna('None', inplace=True)

data[['GarageCond', 'GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageType']]

for col in ['GarageCond', 'GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageType']:

    if 'int' in str(data[col].dtype) or 'float' in str(data[col].dtype):

        data[col].fillna(0, inplace=True)

    else:

        data[col].fillna('None', inplace=True)

for col in ['MSZoning', 'BsmtHalfBath', 'BsmtFullBath', 'Functional', 'BsmtUnfSF', 'SaleType', 'BsmtFinSF1', 'KitchenQual', 'BsmtFinSF2', 'GarageCars', 'TotalBsmtSF', 'Electrical', 'Exterior2nd', 'Exterior1st', 'GarageArea']:

    if 'float' in str(data[col].dtype) or 'int' in str(data[col].dtype) and data[col].value_counts().shape[0] > 500:

        data[col].fillna(data[col].mean(), inplace=True)

    elif 'object' in str(data[col].dtype):

        data[col].fillna(data[col].mode()[0], inplace=True)# mode function have in it's output's index 0 the most frequent value
data['Utilities'].value_counts()
data.drop(['Utilities', 'Id'], axis=1, inplace=True)

#it looks like Utilities doesn't contain useful information since all values are the same except for one

#Id either doesn't represent a useful information since it's a non repetitive personal information
print('We have in our Data {} Features, {} Samples, {} NaNs'.format(data.shape[1], data.shape[0], data.isna().sum().sum()))
data['YearBuiltCategorie'] = pd.cut(data['YearBuilt'], bins=[-np.inf, 1900, 1950, 2000, +np.inf], labels=[0, 1, 2, 3])

#generating new column from YearBuilt column as follows if the value in YearBuilt between -inifiti and 1900 then

#it's correspondance in the new column it's 0 

#else if it's between 1900 and 1950 then it's 1 else if it's between 1950 and 2000 then it's 2

#else 3 (between 2000 and +infiniti)
#getting the categorical columns

numeric_cols = ['LotFrontage', 'LotArea', 'BsmtFinSF2', 'LowQualFinSF', 'YrSold', 'YearBuilt', 'BsmtFullBath', 'BsmtHalfBath', 'YearRemodAdd', 'GarageCars',  'Fireplaces', 'TotRmsAbvGrd', 'KitchenAbvGr', 'BedroomAbvGr', 'HalfBath', 'FullBath', 'MasVnrArea', 'BsmtFinSF1', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'GarageYrBlt', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'MiscVal', 'PoolArea']

categorical_cols = []

for col in data.columns:

    if col not in numeric_cols:

        categorical_cols.append(col)
#Finding for the skewed features so we make them normal like as we talked about in the previous cells

skewed_cols = (np.abs(data[numeric_cols].skew()) > 0.5)

skewed_cols = np.array(skewed_cols.keys())[np.array(skewed_cols.values)]
#Normal Like the Skewed features

pws = PowerTransformer()

data.loc[:, skewed_cols] = pws.fit_transform(data.loc[:, skewed_cols])
#O_cols contains all the columns that their values should be ordered because they have information in the order

#Now we encode the categories of these columns manually because OrdinalEncoder/LabelEncoder will miss to code them in order

O_cols =  ['MoSold', 'OverallCond', 'YearBuiltCategorie' ,'OverallQual', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtFinType1','BsmtCond', 'BsmtExposure', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond']

data.replace({

              'ExterQual': {'None': 0, 'Po': 1, 'Fa': 2, 'TA':3, 'Gd': 4,

                            'Ex': 5},

              'ExterCond': {'None': 0, 'Po': 1, 'Fa': 2, 'TA':3, 'Gd': 4,

                            'Ex': 5},

    

              'BsmtQual': {'None': 0, 'Po': 1, 'Fa': 2, 'TA':3, 'Gd': 4, 

                           'Ex': 5},

              'BsmtCond': {'None': 0, 'Po': 1, 'Fa': 2, 'TA':3, 'Gd': 4,

                           'Ex': 5},

              'BsmtExposure': {'None': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4},

              'BsmtFinType1': {'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4,

                               'ALQ': 5, 'GLQ': 6},

              'BsmtFinType2': {'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4,

                               'ALQ': 5, 'GLQ': 6},

    

              'HeatingQC': {'None': 0, 'Po': 1, 'Fa': 2, 'TA':3, 'Gd': 4,

                            'Ex': 5},

    

              'KitchenQual': {'None': 0, 'Po': 1, 'Fa': 2, 'TA':3, 'Gd': 4,

                              'Ex': 5},

    

              'FireplaceQu': {'None': 0, 'Po': 1, 'Fa': 2, 'TA':3, 'Gd': 4,

                              'Ex': 5},

              

              'GarageQual': {'None': 0, 'Po': 1, 'Fa': 2, 'TA':3, 'Gd': 4,

                             'Ex': 5},

              'GarageCond': {'None': 0, 'Po': 1, 'Fa': 2, 'TA':3, 'Gd': 4,

                             'Ex': 5},

              'GarageFinish': {'None': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3},

    

              'PavedDrive': {'None': 0, 'N': 1, 'P': 2, 'Y': 3},

    

              'PoolQC': {'None': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4},

    

              'Fence': {'None': 0, 'MnWw': 1, 'GdWo': 2, 'MnPrv': 3, 'GdPrv': 4},

    

              'Functional': {'None': 0, 'Sal': 1, 'Sev': 2, 'Maj2': 3, 'Maj1': 4,

                             'Mod': 5, 'Min2': 6, 'Min1': 7, 'Typ': 8}

    

             }, inplace=True)





#Finding the categorical columns that their values have no information in ordering them and then Label Encoding them

OB_cols = []

for col in categorical_cols:

    if col not in O_cols:

        OB_cols.append(col)

enc = OrdinalEncoder()

data[OB_cols] = enc.fit_transform(np.c_[data[OB_cols]])
train = data.iloc[:len(y)].copy()

test = data.iloc[len(y):].copy()
RMSLE =  lambda y_true, y_pred : np.sqrt(mean_squared_log_error(np.exp(y_true), np.exp(y_pred)))#This will calculate our RMSLError

def cvoRMSLE(model, X, y, cv=5, shuffle=True): #it performs Cross Validation on the given data with 5-Folds as default

    kfolds = KFold(n_splits=cv, shuffle=shuffle)

    scores = []

    for training_indices, testing_indices in kfolds.split(X, y):

        model.fit(X[training_indices], np.c_[y[training_indices]])

        scores.append(RMSLE(y[testing_indices], model.predict(X[testing_indices])))

    return np.array(scores)
ElasticNet = make_pipeline(RobustScaler(), ElasticNetCV(alphas=[1, 2, 3, 4, 5, 6, 7, 8, .1, .2, .3, .4, .6, .7, .8, .9, 1e-2, 1e-4, 1e-3, 5e-4]))

ElasticNet.fit(train.values, np.c_[y])

best_alpha = ElasticNet[1].alpha_ 

#after finding the best alpha w search arround that alpha for better accuracy



ElasticNet = make_pipeline(RobustScaler(), ElasticNetCV(alphas=[best_alpha, best_alpha*.9, best_alpha*.8, best_alpha*.7, best_alpha*.6, 

                                                                best_alpha*1.05, best_alpha*1.2, best_alpha*1.5, best_alpha*2]))

print(cvoRMSLE(ElasticNet, train.values, np.c_[y], shuffle=False).mean())

best_alpha = ElasticNet[1].alpha_



lasso = make_pipeline(RobustScaler(), Lasso(alpha=best_alpha))

lasso.fit(train.values, np.c_[y])

cvoRMSLE(lasso, train.values, np.c_[y]).mean()
KRR = make_pipeline(RobustScaler(), KernelRidge())

KRR.fit(train.values, np.c_[y])

cvoRMSLE(KRR, train.values, np.c_[y], shuffle=False).mean()
LR = LinearRegression()

cvoRMSLE(LR, train.values, np.c_[y], shuffle=False).mean()#0.1208868807912646

CatReg = CatBoostRegressor(verbose=0, iterations=2500)

cvoRMSLE(CatReg, train.values, np.c_[y], shuffle=False).mean()
class Stack_Models(object):

    def __init__(self, base_models=[]):

        self.base_models = [clone(model) for model in base_models]

        self.k = None

    

    def fit(self, X, y, cv=5):

        self.k = cv

        kfold = KFold(n_splits=self.k, shuffle=True)

        generated_data = np.zeros((len(X), len(self.base_models)))

        for i in range(len(self.base_models)):

            model = self.base_models[i]

            for training_indices, testing_indices in kfold.split(X, y):

                X_train, y_train = X[training_indices], y[training_indices]

                X_test, y_test = X[testing_indices], y[testing_indices]

                model.fit(X_train, y_train) 

                generated_data[testing_indices, i] = model.predict(X_test).reshape(-1)

            model.fit(X, y)

        return np.column_stack(generated_data)

        

    def predict(self, X):

        data = np.column_stack([ model.predict(X) for model in self.base_models])

        return data

        
stacking = Stack_Models(base_models=[KRR, LR, CatReg, lasso, ElasticNet])
new_data = stacking.fit(train.values, np.c_[y]).T
#Now we are going to train the data comming from the first layer onto two models (meta models)

#Model 1

cat = CatBoostRegressor(verbose=0)

cat.fit(new_data, np.c_[y])

print(cvoRMSLE(cat, new_data, np.c_[y]).mean())

#Model 2

xgbr = XGBRegressor(objective='reg:squarederror')

xgbr.fit(new_data, np.c_[y])

print(cvoRMSLE(xgbr, new_data, np.c_[y]).mean())
test_new_data = stacking.predict(test.values)
y_pred = (np.exp(cat.predict(test_new_data)) + np.exp(xgbr.predict(test_new_data)))/2
y_pred.mean()