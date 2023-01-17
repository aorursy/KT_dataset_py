import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer

import warnings

warnings.filterwarnings(action="ignore")



from scipy.stats import skew, norm

from scipy.special import boxcox1p

from scipy.stats import boxcox_normmax



from sklearn.ensemble import RandomForestRegressor



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test= pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

train.head()
train.info()
#dealing with null values in the train and test dataset

null = train.isna().sum().sort_values(ascending = True)

null_2 = test.isna().sum().sort_values(ascending = True)

null_values = pd.concat([null, null_2], keys = ['train null', 'test null'], axis = 1)

null_values.head(40)
#Lot Frontage

data = [train, test]

for dataset in data:

    x = dataset.iloc[:, 3].values

    x = x.reshape(-1,1)

    imputer  = SimpleImputer(strategy = 'mean', missing_values = np.nan)

    imputer = imputer.fit(x)

    x = imputer.transform(x)

    dataset.iloc[:, 3] = x
#Alley

data = [train, test]

for dataset in data:

    x = dataset.iloc[:,6].values

    x = x.reshape(-1,1)

    impute = SimpleImputer(strategy = 'most_frequent', missing_values = np.nan)

    impute = impute.fit(x)

    x = impute.transform(x)

    dataset.iloc[:, 6] = x    
# BsmtCond

data = [train, test]

for dataset in data:

    x = dataset.iloc[:,31].values

    x = x.reshape(-1,1)

    impute = SimpleImputer(strategy = 'most_frequent', missing_values = np.nan)

    impute = impute.fit(x)

    x = impute.transform(x)

    dataset.iloc[:, 31] = x 
#BsmtExposure

data = [train, test]

for dataset in data:

    x = dataset.iloc[:,32].values

    x = x.reshape(-1,1)

    impute = SimpleImputer(strategy = 'most_frequent', missing_values = np.nan)

    impute = impute.fit(x)

    x = impute.transform(x)

    dataset.iloc[:, 32] = x 
#BsmtFinType1

data = [train, test]

for dataset in data:

    x = dataset.iloc[:,33].values

    x = x.reshape(-1,1)

    impute = SimpleImputer(strategy = 'most_frequent', missing_values = np.nan)

    impute = impute.fit(x)

    x = impute.transform(x)

    dataset.iloc[:, 33] = x 
#BsmtFinType2

data = [train, test]

for dataset in data:

    x = dataset.iloc[:,35].values

    x = x.reshape(-1,1)

    impute = SimpleImputer(strategy = 'most_frequent', missing_values = np.nan)

    impute = impute.fit(x)

    x = impute.transform(x)

    dataset.iloc[:, 35] = x 
#BsmtQual

data = [train, test]

for dataset in data:

    x = dataset.iloc[:,30].values

    x = x.reshape(-1,1)

    impute = SimpleImputer(strategy = 'most_frequent', missing_values = np.nan)

    impute = impute.fit(x)

    x = impute.transform(x)

    dataset.iloc[:, 30] = x 
#Fence

data = [train, test]

for dataset in data:

    x = dataset.iloc[:,73].values

    x = x.reshape(-1,1)

    impute = SimpleImputer(strategy = 'most_frequent', missing_values = np.nan)

    impute = impute.fit(x)

    x = impute.transform(x)

    dataset.iloc[:, 73] = x 
#FireplaceQu

data = [train, test]

for dataset in data:

    x = dataset.iloc[:,57].values

    x = x.reshape(-1,1)

    impute = SimpleImputer(strategy = 'most_frequent', missing_values = np.nan)

    impute = impute.fit(x)

    x = impute.transform(x)

    dataset.iloc[:, 57] = x 
#GarageCond

data = [train, test]

for dataset in data:

    x = dataset.iloc[:,64].values

    x = x.reshape(-1,1)

    impute = SimpleImputer(strategy = 'most_frequent', missing_values = np.nan)

    impute = impute.fit(x)

    x = impute.transform(x)

    dataset.iloc[:, 64] = x 
#GarageFinish

data = [train, test]

for dataset in data:

    x = dataset.iloc[:,60].values

    x = x.reshape(-1,1)

    impute = SimpleImputer(strategy = 'most_frequent', missing_values = np.nan)

    impute = impute.fit(x)

    x = impute.transform(x)

    dataset.iloc[:, 60] = x 
#GarageQual

data = [train, test]

for dataset in data:

    x = dataset.iloc[:,63].values

    x = x.reshape(-1,1)

    impute = SimpleImputer(strategy = 'most_frequent', missing_values = np.nan)

    impute = impute.fit(x)

    x = impute.transform(x)

    dataset.iloc[:, 63] = x 
#GarageType

data = [train, test]

for dataset in data:

    x = dataset.iloc[:,58].values

    x = x.reshape(-1,1)

    impute = SimpleImputer(strategy = 'most_frequent', missing_values = np.nan)

    impute = impute.fit(x)

    x = impute.transform(x)

    dataset.iloc[:, 58] = x 
#GarageYrBit

data = [train, test]

for dataset in data:

    x = dataset.iloc[:,59].values

    x = x.reshape(-1,1)

    impute = SimpleImputer(strategy = 'most_frequent', missing_values = np.nan)

    impute = impute.fit(x)

    x = impute.transform(x)

    dataset.iloc[:, 59] = x 
null_values.tail(40)
#MiscFeature

data = [train, test]

for dataset in data:

    x = dataset.iloc[:,74].values

    x = x.reshape(-1,1)

    impute = SimpleImputer(strategy = 'most_frequent', missing_values = np.nan)

    impute = impute.fit(x)

    x = impute.transform(x)

    dataset.iloc[:, 74] = x 
#PoolQC

data = [train, test]

for dataset in data:

    x = dataset.iloc[:,72].values

    x = x.reshape(-1,1)

    impute = SimpleImputer(strategy = 'most_frequent', missing_values = np.nan)

    impute = impute.fit(x)

    x = impute.transform(x)

    dataset.iloc[:, 72] = x 
#MasVnrType

data = [train, test]

for dataset in data:

    x = dataset.iloc[:,25].values

    x = x.reshape(-1,1)

    impute = SimpleImputer(strategy = 'most_frequent', missing_values = np.nan)

    impute = impute.fit(x)

    x = impute.transform(x)

    dataset.iloc[:, 25] = x 
#MasVnrArea

data = [train, test]

for dataset in data:

    x = dataset.iloc[:,26].values

    x = x.reshape(-1,1)

    impute = SimpleImputer(strategy = 'mean', missing_values = np.nan)

    impute = impute.fit(x)

    x = impute.transform(x)

    dataset.iloc[:, 26] = x 
#Remaining null values 

null_3 = train.isna().sum().sort_values(ascending = True)

null_4 = test.isna().sum().sort_values(ascending = True)

null_values_1 = pd.concat([null_3, null_4], keys = ['train null', 'test null'], axis = 1)

null_values_1.head(40)
null_values_1.tail(40)
#Drop remaining null rows in train set

train = train.dropna(axis = 0)

# test = test.fillna(0, axis = 0)
test.columns
#imputing remaining test set null values

x = test.iloc[:,39].values

x = x.reshape(-1,1)

impute = SimpleImputer(strategy = 'most_frequent', missing_values = np.nan)

impute = impute.fit(x)

x = impute.transform(x)

dataset.iloc[:, 39] = x 
# #imputing remaining test set null values

# b = test.iloc[:,80].values

# b = b.reshape(-1,1)

# impute = SimpleImputer(strategy = 'most_frequent', missing_values = np.nan)

# impute = impute.fit(b)

# b = impute.transform(b)

# dataset.iloc[:, 80] = b 
#imputing remaining test set null values

c = test.iloc[:,55].values

c = c.reshape(-1,1)

impute = SimpleImputer(strategy = 'most_frequent', missing_values = np.nan)

impute = impute.fit(c)

c = impute.transform(c)

dataset.iloc[:, 55] = c 
#imputing remaining test set null values

d = test.iloc[:,2].values

d = d.reshape(-1,1)

impute = SimpleImputer(strategy = 'most_frequent', missing_values = np.nan)

impute = impute.fit(d)

d = impute.transform(d)

dataset.iloc[:, 2] = d 
#imputing remaining test set null values

e = test.iloc[:,-18].values

e = e.reshape(-1,1)

impute = SimpleImputer(strategy = 'most_frequent', missing_values = np.nan)

impute = impute.fit(e)

e = impute.transform(e)

dataset.iloc[:, -18] = e 


#imputing remaining test set null values

f = test.iloc[:,-19].values

f = f.reshape(-1,1)

impute = SimpleImputer(strategy = 'most_frequent', missing_values = np.nan)

impute = impute.fit(f)

f = impute.transform(f)

dataset.iloc[:, -19] = f 

#imputing remaining test set null values

g = test.iloc[:,23].values

g = g.reshape(-1,1)

impute = SimpleImputer(strategy = 'most_frequent', missing_values = np.nan)

impute = impute.fit(g)

g = impute.transform(g)

dataset.iloc[:, 23] = g 



#imputing remaining test set null values

h = test.iloc[:,24].values

h = h.reshape(-1,1)

impute = SimpleImputer(strategy = 'most_frequent', missing_values = np.nan)

impute = impute.fit(h)

h = impute.transform(h)

dataset.iloc[:, 24] = h 

#imputing remaining test set null values

i = test.iloc[:,57].values

i = i.reshape(-1,1)

impute = SimpleImputer(strategy = 'most_frequent', missing_values = np.nan)

impute = impute.fit(i)

i = impute.transform(i)

dataset.iloc[:, 57] = i 
#imputing remaining test set null values

j = test.iloc[:,38].values

j = j.reshape(-1,1)

impute = SimpleImputer(strategy = 'most_frequent', missing_values = np.nan)

impute = impute.fit(j)

j = impute.transform(j)

dataset.iloc[:, 38] = j
#imputing remaining test set null values

k = test.iloc[:,37].values

k = k.reshape(-1,1)

impute = SimpleImputer(strategy = 'most_frequent', missing_values = np.nan)

impute = impute.fit(k)

k = impute.transform(k)

dataset.iloc[:, 37] = k 
#imputing remaining test set null values

a = test.iloc[:,9].values

a = a.reshape(-1,1)

impute = SimpleImputer(strategy = 'most_frequent', missing_values = np.nan)

impute = impute.fit(a)

a = impute.transform(a)

dataset.iloc[:, 9] = a 

#imputing remaining test set null values

n = test.iloc[:,35].values

n = n.reshape(-1,1)

impute = SimpleImputer(strategy = 'most_frequent', missing_values = np.nan)

impute = impute.fit(n)

n = impute.transform(n)

dataset.iloc[:, 35] = n 
#imputing remaining test set null values

l = test.iloc[:,49].values

l = l.reshape(-1,1)

impute = SimpleImputer(strategy = 'most_frequent', missing_values = np.nan)

impute = impute.fit(l)

l = impute.transform(l)

dataset.iloc[:, 49] = l 



#imputing remaining test set null values

m = test.iloc[:,48].values

m = m.reshape(-1,1)

impute = SimpleImputer(strategy = 'most_frequent', missing_values = np.nan)

impute = impute.fit(m)

m = impute.transform(m)

dataset.iloc[:, 48] = m 
#imputing remaining test set null values

l = test.iloc[:,-2].values

l = l.reshape(-1,1)

impute = SimpleImputer(strategy = 'most_frequent', missing_values = np.nan)

impute = impute.fit(l)

l = impute.transform(l)

dataset.iloc[:, -2] = l 
#imputing remaining test set null values

l = test.iloc[:,-27].values

l = l.reshape(-1,1)

impute = SimpleImputer(strategy = 'most_frequent', missing_values = np.nan)

impute = impute.fit(l)

l = impute.transform(l)

dataset.iloc[:, -27] = l 
#imputing remaining test set null values

l = test.iloc[:,34].values

l = l.reshape(-1,1)

impute = SimpleImputer(strategy = 'most_frequent', missing_values = np.nan)

impute = impute.fit(l)

l = impute.transform(l)

dataset.iloc[:, 34] = l 
#imputing remaining test set null values

l = test.iloc[:,36].values

l = l.reshape(-1,1)

impute = SimpleImputer(strategy = 'most_frequent', missing_values = np.nan)

impute = impute.fit(l)

l = impute.transform(l)

dataset.iloc[:, 36] = l 
#imputing remaining test set null values

l = test.iloc[:,47].values

l = l.reshape(-1,1)

impute = SimpleImputer(strategy = 'most_frequent', missing_values = np.nan)

impute = impute.fit(l)

l = impute.transform(l)

dataset.iloc[:, 47] = l 
#imputing remaining test set null values

l = test.iloc[:,49].values

l = l.reshape(-1,1)

impute = SimpleImputer(strategy = 'most_frequent', missing_values = np.nan)

impute = impute.fit(l)

l = impute.transform(l)

dataset.iloc[:, 49] = l 
#Remaining null values 

test.isna().any()
#Drop the ID colum

train = train.drop('Id', axis = 1)

test = test.drop('Id', axis = 1)
#correlation plot



# Compute correlations

corr = train.corr()



sns.set_style(style = 'white')

f, ax = plt.subplots(figsize=(16, 11))

sns.heatmap(corr, vmax=0.9, cmap="Reds", square=True)
#check for skewness

f, ax = plt.subplots(figsize=(9, 8))

sns.distplot(train['SalePrice'], bins = 20, color = 'Magenta')

ax.set(ylabel="Frequency")

ax.set(xlabel="SalePrice")

ax.set(title="SalePrice distribution")

# log transformation

train["SalePrice"] = np.log1p(train["SalePrice"])
#check for skewness after transformation

f, ax = plt.subplots(figsize=(9, 8))

sns.distplot(train['SalePrice'], bins = 20, color = 'Magenta')

ax.set(ylabel="Frequency")

ax.set(xlabel="SalePrice")

ax.set(title="SalePrice distribution")

#Extracting the independent variables

train_features = train.drop('SalePrice', axis = 1)

train_dependent = train['SalePrice'].reset_index(drop=True)

test_features = test
test_features.shape
#Joining the features tables

all_variables = pd.concat([train_features, test_features]).reset_index(drop=True)
# Fetch all numeric features

numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

numeric = []

for i in all_variables.columns:

    if all_variables[i].dtype in numeric_dtypes:

        numeric.append(i)
# Find skewed numerical features

skew_features = all_variables[numeric].apply(lambda x: skew(x)).sort_values(ascending=False)



high_skew = skew_features[skew_features > 0.5]

skew_index = high_skew.index



print("There are {} numerical features with Skew > 0.5 :".format(high_skew.shape[0]))

skewness = pd.DataFrame({'Skew' :high_skew})

skew_features

# Normalize skewed features

for i in skew_index:

    all_variables[i] = boxcox1p(all_variables[i], boxcox_normmax(all_variables[i] + 1))
def logs(res, ls):

    m = res.shape[1]

    for l in ls:

        res = res.assign(newcol=pd.Series(np.log(1.01+res[l])).values)   

        res.columns.values[m] = l + '_log'

        m += 1

    return res



log_features = ['LotFrontage','LotArea','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF',

                 'TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea',

                 'BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr',

                 'TotRmsAbvGrd','Fireplaces','GarageCars','GarageArea','WoodDeckSF','OpenPorchSF',

                 'EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal','YearRemodAdd']



all_variables = logs(all_variables, log_features)
def squares(res, ls):

    m = res.shape[1]

    for l in ls:

        res = res.assign(newcol=pd.Series(res[l]*res[l]).values)   

        res.columns.values[m] = l + '_sq'

        m += 1

    return res 



squared_features = ['YearRemodAdd', 'LotFrontage_log', 

              'TotalBsmtSF_log', '1stFlrSF_log', '2ndFlrSF_log', 'GrLivArea_log',

              'GarageCars_log', 'GarageArea_log']

all_variables = squares(all_variables, squared_features)

all_variables = pd.get_dummies(all_variables).reset_index(drop=True)

all_variables.shape

# one_hot_encoded_test_predictors = pd.get_dummies(test)

# final_train, final_test = one_hot_encoded_training_predictors.align(one_hot_encoded_test_predictors,join='left',axis=1)
all_variables.head()
#Restracture back to original 2 datasets

X = all_variables.iloc[:len(train_dependent), :]

X_test = all_variables.iloc[len(train_dependent):, :]

X.shape, train_dependent.shape, X_test.shape
# Fitting Random Forest Regressor

regressor = RandomForestRegressor(n_estimators=300, random_state=0)

regressor.fit(X, train_dependent.ravel())
#cross validation score

from sklearn.model_selection import cross_val_score

regressor = RandomForestRegressor(n_estimators=300, random_state=0)

# Multiply by -1 since sklearn calculates *negative* MAE

scores = -1 * cross_val_score(regressor, X, train_dependent,

                              cv=5,

                              scoring='neg_mean_absolute_error')



print("Average MAE score:", scores.mean())
# XGBRegressor

from xgboost import XGBRegressor

regressor_2 = XGBRegressor(n_estimators = 500, learning_rate = 0.05)

# Fit the model

regressor_2 = regressor_2.fit(X, train_dependent, verbose = False)



# Get predictions

# pred_2 = regressor_2.predict(X_test) 
#cross validation score

from xgboost import XGBRegressor

from sklearn.model_selection import cross_val_score

regressor_2 = XGBRegressor(n_estimators = 500, learning_rate = 0.05)

scores = -1 * cross_val_score(regressor_2, X, train_dependent,

                              cv=5,

                              scoring='neg_mean_absolute_error')



print("Average MAE score:", scores.mean())
# Predicting results (with XGB)

regressor_2 = regressor_2.fit(X, train_dependent, verbose = False)

Y_pred = regressor_2.predict(X_test)
Y_pred.shape
submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")

submission.iloc[:,1] = np.floor(np.expm1(Y_pred))
# Fix outlier predictions

q1 = submission['SalePrice'].quantile(0.05)

q2 = submission['SalePrice'].quantile(0.95)

submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x > q1 else x*0.77)

submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x < q2 else x*1.1)

submission.to_csv("submission_1.csv", index=False)
# Scale predictions

submission['SalePrice'] *= 1.001619

submission.to_csv("submission_2.csv", index=False)