import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_extraction import FeatureHasher
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')

train_id = df_train['Id']
test_id = df_test['Id']

df_train.drop(columns=['Id'], inplace=True)
df_test.drop(columns=['Id'], inplace=True)

ntrain = df_train.shape[0]
ntest = df_test.shape[0]

all_data = pd.concat((df_train, df_test)).reset_index(drop=True)
print(df_test.shape, df_train.shape, all_data.shape)
print(df_test.shape[0] + df_train.shape[0])
# see the decoration
df_train.columns
# do this in order to view all the columns, otherwise pandas just shows a summary
pd.set_option('display.max_columns', None) 

# Lets see what the data looks like
df_train.head(5)
df_train['SalePrice'].describe()
# Separate out all the numeric features in the data
num_columns = df_train._get_numeric_data().columns
print(num_columns)
print(len(num_columns))

# Separate out all the non-numeric features in the data
categ_columns = pd.Index(list(set(df_train.columns) - set(num_columns)))
print(categ_columns)
print(len(categ_columns))
# moving features from num_columns to categ_columns
categ_columns = categ_columns.append(pd.Index(['MSSubClass', 'OverallQual', 'OverallCond']))
num_columns = num_columns.drop(['MSSubClass', 'OverallQual', 'OverallCond'])
print(len(num_columns), len(categ_columns))
df_train[num_columns].head(5)
def barplot_with_anotate(feature_list, y_values):
    x_pos = np.arange(len(feature_list))

    plt.bar(x_pos, y_values);
    plt.xticks(x_pos, feature_list, rotation=270);
    for i in range(len(feature_list)):
        plt.text(x=x_pos[i]-0.3, y=y_values[i]+1.0, s=y_values[i])
feature_lengths_sorted = sorted([len(df_train[feature].unique()) for feature in num_columns])
barplot_with_anotate(num_columns, feature_lengths_sorted)
plt.rcParams["figure.figsize"] = [20, 12]
num_discrete_columns = []

for feature in num_columns:
    feature_len = len(df_train[feature].unique())
    if feature_len < 30:
        num_discrete_columns.append(feature)
num_discrete_columns = pd.Index(num_discrete_columns)
num_cont_columns = pd.Index(list(set(num_columns) - set(num_discrete_columns)))
# print the details of those discrete valued features
for feature in num_discrete_columns:
    feature_len = len(df_train[feature].unique())
    print(feature, feature_len)
    print(df_train[feature].unique())
# The features LowQualFinSF, 3SsnPorch, PoolArea, MiscVal 
# belong to the list of continous features as they take on values from a continous distribution.
num_cont_columns = num_cont_columns.append(pd.Index(['LowQualFinSF', '3SsnPorch', 'PoolArea', 'MiscVal']))
num_discrete_columns = num_discrete_columns.drop(['LowQualFinSF', '3SsnPorch', 'PoolArea', 'MiscVal'])
# print the details of the continuous valued features
for feature in sorted(num_cont_columns, key=lambda feature: len(df_train[feature].unique()), reverse=True):
    feature_len = len(df_train[feature].unique())
    print(feature, feature_len)
# The 3 year related features - YearBuilt, GarageYrBlt and YearRemodAdd belong to the list of discrete features.
num_discrete_columns = num_discrete_columns.append(pd.Index(['YearBuilt', 'GarageYrBlt', 'YearRemodAdd']))
num_cont_columns = num_cont_columns.drop(['YearBuilt', 'GarageYrBlt', 'YearRemodAdd'])
df_train[categ_columns].head(5)
# lets look at what the categorical features represent
for feature in sorted(categ_columns, key=lambda feature: len(df_train[feature].unique()), reverse=True):
    feature_len = len(df_train[feature].unique())
    print(feature, feature_len)
    print(df_train[feature].unique())
# remove duplicate columns
num_cont_columns = num_cont_columns.drop_duplicates()
num_discrete_columns = num_discrete_columns.drop_duplicates()
categ_columns = categ_columns.drop_duplicates()
print("Categorical columns: ", len(categ_columns))
print("Continuous-valued numeric columns: ", len(num_cont_columns))
print("Discrete-valued numeric columns: ", len(num_discrete_columns))
print("-"*10)
print("Total columns: ", df_train.shape[1])
df_train.drop(df_train[df_train["GrLivArea"] > 4000].index, inplace=True)
df_train.shape

ntrain = df_train.shape[0]
ntest = df_test.shape[0]

all_data = pd.concat((df_train, df_test)).reset_index(drop=True)
df_train[num_cont_columns].describe()
def missing_features(data, column_set):
    incomplete_features = {feature: data.shape[0]-sum(data[feature].value_counts())
                                   for feature in column_set
                                   if not sum(data[feature].value_counts()) == data.shape[0]}
    incomplete_features_sorted = sorted(incomplete_features, key=lambda feature: incomplete_features[feature], reverse=True)
    incompleteness = [round((incomplete_features[feature]/data.shape[0])*100, 2) for feature in incomplete_features_sorted]
    barplot_with_anotate(incomplete_features_sorted, incompleteness)
    plt.ylabel("Percentage (%) of values that are missing")
    plt.rcParams["figure.figsize"] = [16, 8]
    
    for feature, percentage in zip(incomplete_features_sorted, incompleteness):
        print(feature, incomplete_features[feature], "(", percentage, ")")
missing_features(all_data, num_cont_columns)
all_data.loc[:, 'LotFrontage'].fillna(all_data['LotFrontage'].mean(), inplace=True)

for feature in ['GarageArea', 'MasVnrArea', 'TotalBsmtSF', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF']:
    all_data.loc[:, feature].fillna(0, inplace=True)
# first we view a scatter plot of each feature vs. SalePrice

num_cont_columns_list = list(num_cont_columns)

# show max of 6 features in each row
max_in_row = 6

print(len(num_cont_columns_list))
for i in range(0, len(num_cont_columns_list), max_in_row):
    sns.pairplot(df_train, x_vars=num_cont_columns_list[i:i+max_in_row], y_vars=['SalePrice'])
# then we see the correlations
print('Correlation of each feature with SalePrice:')
corr = df_train[num_cont_columns].corr()
corr_sorted = corr.sort_values(["SalePrice"], ascending = False)
print(corr_sorted['SalePrice'])

sns.set(font_scale=1.10)
plt.figure(figsize=(8, 8))

sns.heatmap(corr)
all_data['TotPorchSF'] = all_data['OpenPorchSF'] + all_data['ScreenPorch'] + \
                         all_data['3SsnPorch'] + all_data['EnclosedPorch']
num_cont_columns = num_cont_columns.append(pd.Index(['TotPorchSF']))
(df_train['PoolArea'] == 0).value_counts()
all_data.drop(columns=['PoolArea'], inplace=True)
num_cont_columns = num_cont_columns.drop(['PoolArea'])
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
num_cont_columns = num_cont_columns.append(pd.Index(['TotalSF']))
all_data.drop(columns=['TotalBsmtSF', '1stFlrSF', '2ndFlrSF'], inplace=True)
num_cont_columns = num_cont_columns.drop(['TotalBsmtSF', '1stFlrSF', '2ndFlrSF'])
df_train = all_data[:ntrain][:]
sns.distplot(df_train['SalePrice'])
SalePriceLog = np.log1p(df_train['SalePrice'])
sns.distplot(SalePriceLog)
df_train[num_discrete_columns].describe()
missing_features(all_data, num_discrete_columns)
all_data.loc[:, 'GarageYrBlt'].fillna(all_data['YearBuilt'], inplace=True)
all_data.loc[:, 'BsmtFullBath'].fillna(0, inplace=True)
all_data.loc[:, 'BsmtHalfBath'].fillna(0, inplace=True)
all_data.loc[:, 'GarageCars'].fillna(0, inplace=True)
# print the details of those discrete valued features
for feature in num_discrete_columns:
    feature_len = len(df_train[feature].unique())
    print(feature, feature_len)
    print(df_train[feature].unique())
print('Total: ', len(num_discrete_columns))
corr = df_train[list(num_discrete_columns) + ['SalePrice']].corr()
corr.sort_values(["SalePrice"], ascending = False, inplace = True)
print(corr['SalePrice'])
all_data.drop(columns=['GarageArea'], inplace=True)
num_cont_columns = num_cont_columns.drop(['GarageArea'])
df_train[['YearBuilt', 'YearRemodAdd', 'GarageYrBlt']].describe()
# Since the min of all 3 is 1872, we could replace their values with something like `year % 1872`

all_data['YearBuilt'] = all_data['YearBuilt'] % 1872
all_data['YearRemodAdd'] = all_data['YearRemodAdd'] % 1872
all_data['GarageYrBlt'] = all_data['GarageYrBlt'] % 1872
all_data['Bath'] = all_data['FullBath'] + 0.5*all_data['HalfBath']
all_data.drop(columns=['FullBath', 'HalfBath'], inplace=True)
num_discrete_columns = num_discrete_columns.drop(['FullBath', 'HalfBath'])
all_data['BsmtBath'] = all_data['BsmtFullBath'] + 0.5*all_data['BsmtHalfBath']
all_data.drop(columns=['BsmtFullBath', 'BsmtHalfBath'], inplace=True)
num_discrete_columns = num_discrete_columns.drop(['BsmtFullBath', 'BsmtHalfBath'])
missing_features(all_data, categ_columns)
for feature in ['PoolQC', 'MiscFeature', 'Alley', 'Fence']:
    print(df_train[feature].value_counts())
# drop PoolQC
all_data.drop(columns=['PoolQC'], inplace=True)
categ_columns = categ_columns.drop(['PoolQC'])

# filling NA
all_data.fillna(value= {'MiscFeature': 'NA',
                        'Fence': 'NA',
                        'Alley': 'NA'}, inplace=True)

# Shed
all_data['Shed'] = all_data['MiscFeature'] == 'Shed'
all_data.drop(columns=['MiscFeature'], inplace=True)
categ_columns = categ_columns.drop(['MiscFeature'])
categ_columns = categ_columns.append(pd.Index(['Shed']))
filling_dict = {'FireplaceQu': 'NA',
                'GarageFinish': 'NA',
                'GarageQual': 'NA',
                'GarageType': 'NA',
                'GarageCond': 'NA',
                'BsmtExposure': 'NA',
                'BsmtFinType2': 'NA',
                'BsmtFinType1': 'NA',
                'BsmtCond': 'NA',
                'BsmtQual': 'NA',
                'MasVnrType': 'None',
                'Exterior1st': 'Other',
                'Exterior2nd': 'Other',
                'SaleType': 'Oth'}

for feature in ['Electrical', 'MSZoning', 'Functional', 'Utilities', 'KitchenQual']:
    filling_dict[feature] = all_data[feature].mode().item()
# Now, handle the rest of the incomplete features
all_data.fillna(value=filling_dict, inplace=True)
# Neighborhood is an ordinal feature but we can't decide on the order responsibly unless we know the Ames city
# Hence we drop it
all_data.drop(columns=['Neighborhood'], inplace=True)
categ_columns = categ_columns.drop(['Neighborhood'])
# GarageQual and GarageCond measure the same thing
all_data.drop(columns=['GarageQual'], inplace=True)
categ_columns = categ_columns.drop(['GarageQual'])
# OverallQual and OverallCond seem to measure the same thing
# yet they have a very different correlation coefficient with SalePrice
df_train[['SalePrice', 'OverallQual', 'OverallCond']].corr()
# drop OverallCond
all_data.drop(columns=['OverallCond'], inplace=True)
categ_columns = categ_columns.drop(['OverallCond'])
# new feature extraction
all_data["NewerDwelling"] = all_data["MSSubClass"].isin([20, 60, 120, 160])
# NewerDwelling, BldgType and HouseStyle together capture everything that MSSubClass can tell
# Hence we drop MSSubClass
all_data.drop(columns=['MSSubClass'], inplace=True)
categ_columns = categ_columns.drop(['MSSubClass'])
# simplity MSZoning as Residential or Non-residential
all_data['Residential'] = all_data['MSZoning'].isin(['RH', 'RL', 'RP', 'RM'])

# and drop MSZoning
all_data.drop(columns=['MSZoning'], inplace=True)
categ_columns = categ_columns.drop(['MSZoning'])
# simplify Exterior1st and Exterior2nd 
print(df_train['Exterior1st'].value_counts())
# We keep only the top 4 and move the rest to `Other`
all_data['Exterior1st'][-all_data['Exterior1st'].isin(all_data['Exterior1st'].value_counts().index[0:4])] \
    = 'Other'
all_data['Exterior2nd'][-all_data['Exterior2nd'].isin(all_data['Exterior1st'].value_counts().index[0:4])] \
    = 'Other'
# simplify SaleType
all_data['SaleType'].value_counts()
# keep only the top 3 and move rest to `Oth`
all_data['SaleType'][-all_data['SaleType'].isin(all_data['SaleType'].value_counts().index[0:3])] = 'Oth'
# simplify Functional
all_data['Functional'][all_data['Functional'].isin(['Min1', 'Min2'])] = 'Min'
all_data['Functional'][all_data['Functional'].isin(['Maj1', 'Maj2'])] = 'Maj'
# GarageType can either be Attchd or Detchd
# new feature - GarageDetchd
all_data['GarageDetchd'] = all_data['GarageType'].replace({'BuiltIn': 'Attchd', 
    'Basment': 'Attchd', 'CarPort': 'Attchd', '2Types': 'Detchd'})

# drop GarageType
all_data.drop(columns=['GarageType'], inplace=True)
categ_columns = categ_columns.drop(['GarageType'])
# Step 1: Convert ordinal features to numbers
replace_dict =   {'LotShape': {'IR3': 0, 'IR2': 1, 'IR1': 2, 'Reg': 3},
                  'Utilities': {'ELO': 0, 'NoSeWa': 1, 'NoSewr': 2, 'AllPub': 3},
                  'LandSlope': {'Sev': 0, 'Mod': 1, 'Gtl': 2},
                  'ExterQual': {'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4},
                  'ExterCond': {'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4},
                  'BsmtQual': {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
                  'BsmtCond': {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
                  'BsmtExposure': {'NA': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4},
                  'BsmtFinType1': {'NA': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6},
                  'BsmtFinType2': {'NA': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6},
                  'HeatingQC': {'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4},
                  'Electrical': {'Mix': 0, 'FuseP': 1, 'FuseF': 2, 'FuseA': 3, 'SBrkr': 4},
                  'KitchenQual': {'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4},
                  'Functional': {'Sal': 0, 'Sev': 1, 'Maj2': 2, 'Maj1': 3, 'Mod': 4, 'Min2': 5, 'Min1': 6,
                                 'Typ': 7},
                  'FireplaceQu': {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
                  'GarageFinish': {'NA': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3},
                  'GarageCond': {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
                  'PavedDrive': {'N': 0, 'P': 1, 'Y': 2},
                  'Fence': {'NA': 0, 'MnWw': 1, 'GdWo': 2, 'MnPrv': 3, 'GdPrv': 4},
                  'Condition1': {'Artery': 0, 'Feedr': 0, 'RRAn': 1, 'RRAe': 1, 'RRNn': 2, 'RRNe': 2, 
                                 'Norm': 5, 'PosA': 10, 'PosN': 11},
                  'Condition2': {'Artery': 0, 'Feedr': 0, 'RRAn': 1, 'RRAe': 1, 'RRNn': 2, 'RRNe': 2, 
                                 'Norm': 5, 'PosA': 10, 'PosN': 11}}
all_data = all_data.replace(replace_dict)

# all the other features of categ_columns are nominal
nominal_columns = categ_columns.drop(list(replace_dict.keys()))
nominal_columns = nominal_columns.drop(['OverallQual'])
ordinal_columns = pd.Series(list(replace_dict.keys()).append('OverallQual'))
# Step 2: One-hot encoding for nominal features
all_data = pd.get_dummies(all_data)
all_data.shape
# BsmtFinType1 * BsmtFinSF1
all_data['BsmtFin1_Type*SF'] = all_data['BsmtFinType1'] * all_data['BsmtFinSF1']

# BsmtFinType2 * BsmtFinSF2
all_data['BsmtFin2_Type*SF'] = all_data['BsmtFinType2'] * all_data['BsmtFinSF2']

# ExterQual * ExterCond
all_data['Exter_Qual*Cond'] = all_data['ExterQual'] * all_data['ExterCond']

# KitchenQual * no. of kitchens
all_data['Kitchen_no*Qual'] = all_data['KitchenAbvGr'] * all_data['KitchenQual']

# Condition1 * Condition2
all_data['Condition1*Contition2'] = all_data['Condition1'] * all_data['Condition2']

# OverallQual * TotalSF
all_data['OverallQual*TotalSF'] = all_data['OverallQual'] * all_data['TotalSF']

# BsmtQual * BsmtCond * BsmtExposure
all_data['BsmtQual*Cond*Expo'] = all_data['BsmtQual'] * all_data['BsmtCond'] * all_data['BsmtExposure']
# get a list of the top 10 most correlated features
corr = all_data[all_data.columns].corr()
corr_sorted = corr.sort_values(["SalePrice"], ascending = False)
top10_features = list(corr_sorted['SalePrice'][1:11].keys())
# generate polynomial features and add to existing DataFrame
poly_features = PolynomialFeatures(degree=2, include_bias=False)
poly_features.fit_transform(all_data[top10_features])
poly_features.get_feature_names(top10_features)

poly_df = pd.DataFrame(poly_features.transform(all_data[top10_features]), 
                                     columns=poly_features.get_feature_names(top10_features))

all_data = pd.concat([all_data, poly_df], axis=1)

print("Final shape of data: ", all_data.shape)
poly_df.head(5)
# then we see the correlations
corr = all_data[all_data.columns].corr()
corr_sorted = corr.sort_values(["SalePrice"], ascending = False)

#corr_sorted['SalePrice'][0:50]
best_corr = corr_sorted[corr_sorted['SalePrice']>0.1]['SalePrice']
worst_corr = corr_sorted[corr_sorted['SalePrice']<-0.1]['SalePrice']
selected_corr = pd.concat([best_corr, worst_corr])

print(len(list(selected_corr)))
print(len(list(best_corr)), len(list(worst_corr)))
black_list = [elem for elem in list(all_data.columns) if elem not in list(selected_corr.keys())]
all_data.drop(columns=black_list, inplace=True)

all_data.shape
attrs = corr.drop('SalePrice').drop('SalePrice', axis=1)

threshold = 0.98
important_corrs = (attrs[abs(attrs) > threshold][attrs != 1.0]) \
    .unstack().dropna().to_dict()
    
print("There are", len(important_corrs), "pairs of non-target features with correlation coefficient >", threshold)

important_corrs
to_drop = []
for (feature1, feature2) in important_corrs.keys():
    if feature1 not in to_drop and feature2 not in to_drop:
        to_drop.append(feature1)
        
to_drop
all_data.drop(columns=to_drop, inplace=True)
all_data = all_data.loc[:,~all_data.columns.duplicated()]
nvalidate = int(0.2 * ntrain)
ntrain = int(ntrain - nvalidate)

df_train = all_data[:ntrain][:]
df_validate = all_data[ntrain:ntrain+nvalidate][:]
df_test = all_data[ntrain+nvalidate:][:]
df_test = df_test.drop('SalePrice', axis=1)
print(df_train.shape)
print(df_validate.shape)
print(df_test.shape)
print(ntrain, nvalidate, ntest)
X_train= df_train.drop('SalePrice', axis= 1)
Y_train= df_train['SalePrice']
Y_train_log = SalePriceLog[:ntrain]

X_validate = df_validate.drop('SalePrice', axis=1)
Y_validate = df_validate['SalePrice']
Y_validate_log = SalePriceLog[ntrain:ntrain+nvalidate]

X_test= df_test
# make sure that there are no NaNs
missing_features(X_train, X_train.columns)
missing_features(X_test, X_test.columns)
missing_features(X_validate, X_validate.columns)
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
def rmse_score(Y_true, Y_pred):
    return round(np.sqrt(mean_squared_error(np.log(Y_true), np.log(Y_pred))), 5)
linreg = LinearRegression(normalize=True)
linreg.fit(X_train, Y_train)

Y_train_pred = linreg.predict(X_train)
Y_train_pred[Y_train_pred < 1] = 1
Y_validate_pred = linreg.predict(X_validate)
Y_validate_pred[Y_validate_pred<1] = 1

acc_lin_train = rmse_score(Y_train.values, Y_train_pred)
acc_lin_validate = rmse_score(Y_validate.values, Y_validate_pred)

print("RMSE on train: ", acc_lin_train)
print("RMSE on validation: ", acc_lin_validate)
linreg_on_log = LinearRegression(normalize=True)
linreg_on_log.fit(X_train, Y_train_log)

Y_train_pred = np.exp(linreg_on_log.predict(X_train))
Y_train_pred[Y_train_pred < 0] = 0
Y_validate_pred = np.exp(linreg_on_log.predict(X_validate))
Y_validate_pred[Y_validate_pred<0] = 0

acc_lin_train = rmse_score(Y_train.values, Y_train_pred)
acc_lin_validate = rmse_score(Y_validate.values, Y_validate_pred)

print("RMSE on train: ", acc_lin_train)
print("RMSE on validation: ", acc_lin_validate)
lasso = Lasso()
lasso.fit(X_train, Y_train)

acc_lasso_train = rmse_score(Y_train.values, lasso.predict(X_train))
acc_lasso_validate = rmse_score(Y_validate.values, lasso.predict(X_validate))

print("RMSE on train: ", acc_lasso_train)
print("RMSE on validation: ", acc_lasso_validate)
lasso_on_log = Lasso()
lasso_on_log.fit(X_train, Y_train_log)

acc_lasso_train = rmse_score(Y_train.values, np.exp(lasso_on_log.predict(X_train)))
acc_lasso_validate = rmse_score(Y_validate.values, np.exp(lasso_on_log.predict(X_validate)))

print("RMSE on train: ", acc_lasso_train)
print("RMSE on validation: ", acc_lasso_validate)
ridge = Ridge()
ridge.fit(X_train, Y_train)

acc_ridge_train = rmse_score(Y_train.values, ridge.predict(X_train))
acc_ridge_validate = rmse_score(Y_validate.values, ridge.predict(X_validate))

print("RMSE on train: ", acc_ridge_train)
print("RMSE on validation: ", acc_ridge_validate)
ridge_on_log = Ridge()
ridge_on_log.fit(X_train, Y_train_log)

acc_ridge_train = rmse_score(Y_train.values, np.exp(ridge_on_log.predict(X_train)))
acc_ridge_validate = rmse_score(Y_validate.values, np.exp(ridge_on_log.predict(X_validate)))

print("RMSE on train: ", acc_ridge_train)
print("RMSE on validation: ", acc_ridge_validate)
elastic_net = ElasticNet()
elastic_net.fit(X_train, Y_train)

acc_en_train = rmse_score(Y_train.values, elastic_net.predict(X_train))
acc_en_validate = rmse_score(Y_validate.values, elastic_net.predict(X_validate))

print("RMSE on train: ", acc_en_train)
print("RMSE on validation: ", acc_en_validate)
elastic_net_on_log = ElasticNet()
elastic_net_on_log.fit(X_train, Y_train_log)

acc_en_train = rmse_score(Y_train.values, np.exp(elastic_net_on_log.predict(X_train)))
acc_en_validate = rmse_score(Y_validate.values, np.exp(elastic_net_on_log.predict(X_validate)))

print("RMSE on train: ", acc_en_train)
print("RMSE on validation: ", acc_en_validate)
decision_tree = DecisionTreeRegressor()
decision_tree.fit(X_train, Y_train)

acc_decision_tree_train = rmse_score(Y_train.values, decision_tree.predict(X_train))
acc_decision_tree_validate = rmse_score(Y_validate.values, decision_tree.predict(X_validate))

print("RMSE on train: ", acc_decision_tree_train)
print("RMSE on validation: ", acc_decision_tree_validate)
decision_tree_on_log = DecisionTreeRegressor()
decision_tree_on_log.fit(X_train, Y_train_log)

acc_decision_tree_train = rmse_score(Y_train.values, np.exp(decision_tree_on_log.predict(X_train)))
acc_decision_tree_validate = rmse_score(Y_validate.values, np.exp(decision_tree_on_log.predict(X_validate)))

print("RMSE on train: ", acc_decision_tree_train)
print("RMSE on validation: ", acc_decision_tree_validate)
random_forest = RandomForestRegressor(n_estimators=100)
random_forest.fit(X_train, Y_train)

acc_random_forest_train = rmse_score(Y_train.values, random_forest.predict(X_train))
acc_random_forest_validate = rmse_score(Y_validate.values, random_forest.predict(X_validate))

print("RMSE on train: ", acc_random_forest_train)
print("RMSE on validation: ", acc_random_forest_validate)
random_forest_on_log = RandomForestRegressor(n_estimators=100)
random_forest_on_log.fit(X_train, Y_train_log)

acc_random_forest_train = rmse_score(Y_train.values, np.exp(random_forest_on_log.predict(X_train)))
acc_random_forest_validate = rmse_score(Y_validate.values, np.exp(random_forest_on_log.predict(X_validate)))

print("RMSE on train: ", acc_random_forest_train)
print("RMSE on validation: ", acc_random_forest_validate)
xgboost = XGBRegressor()
xgboost.fit(X_train, Y_train)

acc_xgboost_train = rmse_score(Y_train.values, xgboost.predict(X_train))
acc_xgboost_validate = rmse_score(Y_validate.values, xgboost.predict(X_validate))

print("RMSE on train: ", acc_xgboost_train)
print("RMSE on validation: ", acc_xgboost_validate)
xgboost_on_log = XGBRegressor()
xgboost_on_log.fit(X_train, Y_train_log)

acc_xgboost_train = rmse_score(Y_train.values, np.exp(xgboost_on_log.predict(X_train)))
acc_xgboost_validate = rmse_score(Y_validate.values, np.exp(xgboost_on_log.predict(X_validate)))

print("RMSE on train: ", acc_xgboost_train)
print("RMSE on validation: ", acc_xgboost_validate)
models = pd.DataFrame({
    'Model': ['Linear Regression',
              'Lasso Regression',
              'Ridge Regression',
              'ElasticNet Regression',
              'Random Forest', 
              'Decision Tree',
              'XGBoost'],
    'Train Score': [acc_lin_train,
                    acc_lasso_train,
                    acc_ridge_train,
                    acc_en_train,
                    acc_random_forest_train, 
                    acc_decision_tree_train,
                    acc_xgboost_train],
    'Cross-Validation Score': [acc_lin_validate,
                    acc_lasso_validate,
                    acc_ridge_validate,
                    acc_en_validate,
                    acc_random_forest_validate, 
                    acc_decision_tree_validate,
                    acc_xgboost_validate]})
models.sort_values(by='Cross-Validation Score', ascending=True)
import operator

def find_best_alpha(alphas, model):
    rmse = [rmse_score(Y_validate.values, np.exp(model(alpha=alpha).fit(X_train, Y_train_log).predict(X_validate))) 
            for alpha in alphas]
    cv_ridge = pd.Series(rmse, index = alphas)
    cv_ridge.plot(title = "On Validation set")
    plt.xlabel("alpha")
    plt.ylabel("rmse")

    min_index, min_value = min(enumerate(rmse), key=operator.itemgetter(1))

    print("Minimum RMSE =", min_value, "found at alpha =", alphas[min_index])

    return (alphas[min_index], min_value)
(ridge_alpha, _) = find_best_alpha(alphas=[0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300],
                                   model=Ridge)
(elasticNet_alpha, _) = find_best_alpha(alphas=[0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10, 30, 100],
                                        model=ElasticNet)
(lasso_alpha, _) = find_best_alpha(alphas=[0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10, 30, 100, 300, 1000],
                                   model=Lasso)
from sklearn.model_selection import RandomizedSearchCV
ridge_random_grid = {'alpha': [int(x) for x in np.linspace(start=0.5*ridge_alpha, stop=1.5*ridge_alpha, num=10)],
                     'fit_intercept': [True],
                     'normalize': [True, False],
                     'copy_X': [True],
                     'max_iter': [10, 30, 100, 300, 1000, 3000, 10000],
                     'tol': [0.0001],
                     'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
                     'random_state': [42]}
ridge = Ridge()
ridge_random = RandomizedSearchCV(estimator = ridge, 
                                  param_distributions = ridge_random_grid, 
                                  n_iter = 100, 
                                  verbose=1, 
                                  random_state=42, 
                                  n_jobs = -1)

ridge_random.fit(X_train, Y_train_log)
rmse_score(Y_validate.values, np.exp(ridge_random.best_estimator_.predict(X_validate)))
ridge_random.best_params_
elasticNet_random_grid = {'alpha': [int(x) for x in np.linspace(start=0.5*elasticNet_alpha, 
                                                                stop=1.5*elasticNet_alpha, 
                                                                num=10)],
                          'l1_ratio': [0.01, 0.03, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
                          'fit_intercept': [True],
                          'normalize': [True, False],
                          'precompute': [True, False],
                          'copy_X': [True],
                          'max_iter': [10, 30, 100, 300, 1000, 3000, 10000],
                          'tol': [0.0001],
                          'warm_start': [True, False],
                          'positive': [True, False],
                          'selection': ['cyclic', 'random'],
                          'random_state': [42]}

elastic_net = ElasticNet()
elastic_net_random = RandomizedSearchCV(estimator = elastic_net, 
                                  param_distributions = elasticNet_random_grid, 
                                  n_iter = 100, 
                                  verbose=1, 
                                  random_state=42, 
                                  n_jobs = -1)

elastic_net_random.fit(X_train, Y_train_log)
print(rmse_score(Y_validate.values, np.exp(elastic_net_random.best_estimator_.predict(X_validate))))
elastic_net_random.best_params_
lasso_random_grid = {'alpha': [int(x) for x in np.linspace(start=0.5*lasso_alpha, 
                                                                stop=1.5*lasso_alpha, 
                                                                num=10)],
                     'fit_intercept': [True],
                     'normalize': [True, False],
                     'precompute': [True, False],
                     'copy_X': [True],
                     'max_iter': [10, 30, 100, 300, 1000, 3000, 10000],
                     'tol': [0.0001],
                     'warm_start': [True, False],
                     'positive': [True, False],
                     'selection': ['cyclic', 'random'],
                     'random_state': [42]}

lasso = Lasso()
lasso_random = RandomizedSearchCV(estimator = lasso, 
                                  param_distributions = lasso_random_grid, 
                                  n_iter = 100, 
                                  verbose=1, 
                                  random_state=42, 
                                  n_jobs = -1)

lasso_random.fit(X_train, Y_train_log)
print(rmse_score(Y_validate.values, np.exp(lasso_random.best_estimator_.predict(X_validate))))
lasso_random.best_params_
feature_importances = pd.DataFrame({
    'Features': X_train.columns,
    'Importances': random_forest.feature_importances_
})
feature_importances = feature_importances.sort_values(by='Importances', ascending=False)
best = {}

for feature_count in range(10, 210, 10):
    best[feature_count] = {'feature_list': list(feature_importances['Features'][:feature_count])}
for feature_set_size, features in best.items():
    train_accuracy_sum = 0
    validation_accuracy_sum = 0
    
    nattempts = 3
    for attempt in range(nattempts):
        
        X_train_best = X_train[features['feature_list']]
        X_validate_best = X_validate[features['feature_list']]
        X_test_best = X_test[features['feature_list']]

        random_forest = RandomForestRegressor(n_estimators=100)
        random_forest.fit(X_train_best, Y_train)

        acc_random_forest_train = rmse_score(Y_train.values, random_forest.predict(X_train_best))
        acc_random_forest_validate = rmse_score(Y_validate.values, random_forest.predict(X_validate_best))
        
        train_accuracy_sum += acc_random_forest_train
        validation_accuracy_sum += acc_random_forest_validate

    features['train_accuracy_avg'] = train_accuracy_sum/nattempts
    features['validation_accuracy_avg'] = validation_accuracy_sum/nattempts
    
    print("No. of features:", feature_set_size)
    print("Train accuracy: ", features['train_accuracy_avg'])
    print("Cross-Validation accuracy: ", features['validation_accuracy_avg'])
feature_sizes_sorted = sorted(best.keys(), 
                              key=lambda feature_count: best[feature_count]['validation_accuracy_avg'])
accuracies = [best[nfeatures]['validation_accuracy_avg'] for nfeatures in feature_sizes_sorted]

for nfeatures, accuracies in zip(feature_sizes_sorted, accuracies):
    print(nfeatures, ": ", round(accuracies, 5))
final_train = all_data[:ntrain+nvalidate][:]
final_test = all_data[ntrain+nvalidate:][:]

final_train_X = final_train.drop('SalePrice', axis= 1)
final_train_Y = final_train['SalePrice']
final_train_Y_log = SalePriceLog
final_test_X = final_test.drop('SalePrice', axis=1)
# make sure there are no missing features
missing_features(final_train_X, final_train_X.columns)
ridge_random_grid = {'alpha': [int(x) for x in np.linspace(start=0.5*ridge_alpha, stop=1.5*ridge_alpha, num=10)],
                     'fit_intercept': [True],
                     'normalize': [True, False],
                     'copy_X': [True],
                     'max_iter': [10, 30, 100, 300, 1000, 3000, 10000],
                     'tol': [0.0001],
                     'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
                     'random_state': [42]}
ridge = Ridge()
ridge_random = RandomizedSearchCV(estimator = ridge, 
                                  param_distributions = ridge_random_grid, 
                                  n_iter = 100, 
                                  verbose=1, 
                                  random_state=42, 
                                  n_jobs = -1)

ridge_random.fit(final_train_X, final_train_Y_log)
elasticNet_random_grid = {'alpha': [int(x) for x in np.linspace(start=0.5*elasticNet_alpha, 
                                                                stop=1.5*elasticNet_alpha, 
                                                                num=10)],
                          'l1_ratio': [0.01, 0.03, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
                          'fit_intercept': [True],
                          'normalize': [True, False],
                          'precompute': [True, False],
                          'copy_X': [True],
                          'max_iter': [10, 30, 100, 300, 1000, 3000, 10000],
                          'tol': [0.0001],
                          'warm_start': [True, False],
                          'positive': [True, False],
                          'selection': ['cyclic', 'random'],
                          'random_state': [42]}

elastic_net = ElasticNet()
elastic_net_random = RandomizedSearchCV(estimator = elastic_net, 
                                  param_distributions = elasticNet_random_grid, 
                                  n_iter = 100, 
                                  verbose=1, 
                                  random_state=42, 
                                  n_jobs = -1)

elastic_net_random.fit(final_train_X, final_train_Y_log)
xgboost = XGBRegressor()
xgboost.fit(final_train_X, final_train_Y_log)
rmse_score(final_train_Y.values, np.exp(ridge_random.best_estimator_.predict(final_train_X)))
rmse_score(final_train_Y.values, np.exp(elastic_net_random.best_estimator_.predict(final_train_X)))
rmse_score(final_train_Y.values, np.exp(xgboost.predict(final_train_X)))
final_Y_pred = np.exp((ridge_random.best_estimator_.predict(final_test_X) +
                   elastic_net_random.best_estimator_.predict(final_test_X) +
                   xgboost.predict(final_test_X)) / 3)
submission = pd.DataFrame({'Id': test_id,
                           'SalePrice': final_Y_pred})
submission.to_csv('submission.csv', index=False)