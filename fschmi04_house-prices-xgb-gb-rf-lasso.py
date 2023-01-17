import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from pandas.plotting import register_matplotlib_converters

%matplotlib inline



from sklearn.model_selection import train_test_split, cross_val_score, learning_curve

from sklearn.linear_model import LassoCV

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, LabelEncoder, OneHotEncoder

from sklearn.metrics import mean_squared_error

import xgboost

import warnings



register_matplotlib_converters()

warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', 2500, 'display.max_rows', 2500, 'display.width', None)
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

train.name = 'Training set'



test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

test.name = 'Test set'



df = pd.concat([train, test]).reset_index(drop=True)

df.name = 'Total database'
def correlation(value):

    correlation = df.drop(['Id'], axis=1).apply(lambda x: x.factorize()[0]).corr().abs().unstack().sort_values(kind='quicksort', ascending=False).drop_duplicates(keep='first')

    correlation = correlation.reset_index().rename(columns={0: 'Correlation'})

    correlation = correlation[correlation['level_0'].str.contains(value) |

                              correlation['level_1'].str.contains(value)]

    return correlation[:5]



def countplot(feature, data):

    sns.set(style='darkgrid')

    sns.countplot(data[feature])

    plt.title('{0} ({1})'.format(feature, data.name))

    if len(df.groupby(['Exterior1st']).sum()) > 6:

        plt.xticks(rotation=45, ha='right')

    plt.show()



def catplot(feature, data):

    sns.catplot(x=feature, y='SalePrice', data=data, kind="bar")

    plt.title('Effect of {} on SalePrice'.format(feature)), plt.show()
print('Training set:', train.shape)

print('Test set:', test.shape)

print('\nColumns:\n', list(df.columns))



# Data types

print('\nData types:\n{}'.format(df.dtypes))



# Descriptive statistics

df.describe()
for dataset in (train, test, df):

    dataset['MSSubClass'] = dataset['MSSubClass'].astype('str')
print('Missing values in training set: {}'.format(train['SalePrice'].isna().sum()))



train['SalePrice'].describe()
sns.set(style='darkgrid')

sns.distplot(df['SalePrice'], 20),

plt.xticks(rotation=45, ha='right')

plt.show()
sns.set(style='darkgrid')

sns.boxplot(x=train['SalePrice'])

plt.title('Boxplot SalePrice', fontsize=12), plt.xlabel('SalePrice', fontsize=10), plt.xticks(fontsize=10, rotation=90)

plt.show()
df[df['SalePrice'] > 700000]
highest_correlation_target = df.drop(['Id'], axis=1).corr().abs().unstack().sort_values(kind='quicksort', ascending=False)#.drop_duplicates(keep='first')

highest_correlation_target = highest_correlation_target.reset_index().rename(columns={0: 'Correlation'})

highest_correlation_target = highest_correlation_target[highest_correlation_target['level_0'].str.contains('SalePrice') |

                                                        highest_correlation_target['level_1'].str.contains('SalePrice')]

highest_correlation_target = highest_correlation_target[highest_correlation_target['Correlation'] < 1]

highest_correlation_target.drop_duplicates(subset='Correlation')[:10]
correlation_matrix = df.drop(['Id'], axis=1).corr()#.drop_duplicates(keep='first')



plt.figure(figsize=(12,12))

sns.set(font_scale=0.75)

ax = sns.heatmap(correlation_matrix, vmin=-1, vmax=1, center=0, linewidths=0.5, cmap='coolwarm', square=True, annot=False)
catplot('OverallQual', train)
sns.lmplot(x="GrLivArea", y="SalePrice", data=train)

plt.show()
train = train.drop(train[(train['GrLivArea'] > 4000) & (train['SalePrice'] < 300000)].index)#.reset_index(drop=False)

df = df.drop(df[(df['GrLivArea'] > 4000) & (df['SalePrice'] < 300000)].index)#.reset_index(drop=False)
# Relplot

sns.relplot(x='YearBuilt', y='SalePrice', hue='TotRmsAbvGrd', data=train)

plt.show()
mean_sale_price_neighborhood = train.groupby('Neighborhood')['SalePrice'].mean().sort_values()



sns.pointplot(x =mean_sale_price_neighborhood.index, y =mean_sale_price_neighborhood.values, data=train,

              order=mean_sale_price_neighborhood.index)

plt.xticks(rotation=45)

plt.show()
central_air = train.groupby(['CentralAir'])['SalePrice'].mean()

central_air = central_air.sort_index(ascending=False)



plt.figure()

sns.barplot(x=central_air.index, y=central_air.values)

plt.title('Effect of CentralAir on SalePrice')

plt.show()



sns.set(style='darkgrid')

sns.countplot(x=pd.qcut(train['SalePrice'], 5), hue='CentralAir', data=train)

plt.xticks(ha='right', rotation=45)

plt.show()
missing = df.isna().sum()

missing = missing[missing.values != 0].sort_values(ascending=False)



plt.figure(figsize=(14,8))

sns.barplot(missing.index, missing.values)

plt.xticks(rotation=90), plt.ylabel('Missing values'), plt.title('Missing values by feature\n(total dataset: {} observations)'.format(len(df)))

for i, v in enumerate(np.around(np.array(missing.values), 4)):

    plt.text(i, v+20, str('%.0f' % v), ha='center', fontsize=8)

plt.show()
for dataset in (train, test, df):

    # Categorial features

    for column in ['Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature']:

        dataset[column] = dataset[column].fillna('Not available')

    

    # Numerical features

    for column in ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF']:

        dataset[column] = dataset[column].fillna(0)
sns.set(style='darkgrid')

sns.scatterplot(x='GarageYrBlt', y='YearBuilt', data=df)

plt.show()



correlation('GarageYrBlt')
df.loc[:, ('Id', 'GarageYrBlt', 'YearBuilt', 'YearRemodAdd')][df['GarageYrBlt'] == 2207]
df['GarageYrBlt'] = df['GarageYrBlt'].replace({2207: 2007})



sns.set(style='darkgrid')

sns.scatterplot(x='GarageYrBlt', y='YearBuilt', data=df)

plt.show()
print('Missing values:', len(df[df['GarageYrBlt'].isna()]))
for dataset in (train, test, df):

    dataset['GarageYrBlt'] = dataset['GarageYrBlt'].fillna(0)
df.loc[:, ('Id', 'GarageCars', 'GarageQual', 'GarageCond')][df['GarageCars'].isna()]
for dataset in (train, test, df):

    dataset['GarageCars'] = dataset['GarageCars'].fillna(0)
df.loc[:, ('Id', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond')][df['GarageArea'].isna()]
for dataset in (train, test, df):

    dataset['GarageArea'] = dataset['GarageArea'].fillna(0)
for feature in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

    print('{0}: {1}'.format(feature, len(df[df[feature] == 'Not available'])))
basement = df

basement['Count'] = basement['BsmtQual'].str.count('Not available') + basement['BsmtCond'].str.count('Not available') + basement['BsmtExposure'].str.count('Not available') + basement['BsmtFinType1'].str.count('Not available') + basement['BsmtFinType2'].str.count('Not available')

basement.loc[:, ('Id', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'TotalBsmtSF')][(basement['Count'] < 5) & (basement['Count'] > 0)]
correlation('BsmtQual')
df.loc[2217:2218, ('BsmtQual', 'OverallQual')]
print('Mean OverallQual: {}'.format(round(df['OverallQual'].mean(), 2)))



for dataset in (train, test, df):

    for value in (2218, 2219):

        dataset.loc[dataset['Id'] == value, 'BsmtQual'] = 'TA'
correlation('BsmtCond')
for dataset in (train, test, df):

    dataset['BsmtCond'] = np.where((dataset['BsmtCond'] == 'Not available') &

                                   (dataset['BsmtQual'] != 'Not available'), dataset['BsmtQual'], dataset['BsmtCond'])

        

df.loc[(2040, 2185, 2524), ('BsmtQual', 'BsmtCond')]
correlation('BsmtExposure')
df.loc[(948, 1487, 2348), ('BsmtExposure', 'HouseStyle')]
correlation('BsmtExposure')

sns.catplot(x='HouseStyle', hue='BsmtExposure', data=df, kind='count')

plt.show()
for dataset in (train, test, df):

    for value in (949, 1488, 2349):

        dataset.loc[dataset['Id'] == value, 'BsmtExposure'] = 'No'
correlation('BsmtFinType2')
df.loc[[332], ('Id', 'BsmtFinType2', 'BsmtFinSF2')]
df['BsmtFinSF2Grouped'] = pd.cut(df['BsmtFinSF2'], 5)



sns.catplot(x='BsmtFinSF2Grouped', data=df, kind='count')

plt.xticks(rotation=45, ha='right')

plt.title('Number of houses with BsmtFinSF2 (305.2, 610.4)')

for i, v in enumerate(np.array(df.groupby('BsmtFinSF2Grouped')['BsmtFinType2'].count())):

    plt.text(i, v+20, v, ha='center', fontsize=10) 

plt.show()
df_new = df[df['BsmtFinSF2Grouped'] == pd.Interval(305.2, 610.4)]



percentage = 100 * df_new['BsmtFinType2'].value_counts() / df_new['BsmtFinType2'].value_counts().sum()

plt.figure(figsize=(10,6))

plt.pie(percentage)

plt.legend(['{0} - {1:1.2f} %'.format(i, j) for i, j in zip(percentage.index, percentage)], loc='best', fontsize=10, frameon=True)

plt.title('Distribution of BsmtFinType2 in the BsmtFinSF2 group (305.2, 610.4)')

plt.show()



for dataset in (train, test, df):

    dataset.loc[dataset['Id'] == 333, 'BsmtFinType2'] = 'Rec'

for feature in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

    print('{0}: {1}'.format(feature, len(df[df[feature] == 'Not available'])))
df.name = 'Total database'

countplot('MSZoning', df)



print('Missing values in MSZoning: {}'.format(df['MSZoning'].isna().sum()))



correlation('MSZoning')
df.loc[:, ('Id', 'MSZoning', 'Alley')][df['MSZoning'].isna()]
sns.catplot(x='Alley', hue='MSZoning', data=df, kind='count')

plt.title('MSZoning for Alley values')

plt.show()



for dataset in (train, test, df):

    dataset['MSZoning'] = dataset['MSZoning'].fillna('RL')
print('Missing values in Utilities: {}'.format(df['Utilities'].isna().sum()))



countplot('Utilities', df)



print(df['Utilities'].value_counts())



for dataset in (train, test, df):

    dataset['Utilities'] = dataset['Utilities'].fillna('AllPub')
print('Missing values in Exterior1st: {}'.format(df['Exterior1st'].isna().sum()))

print('Missing values in Exterior2nd: {}'.format(df['Exterior2nd'].isna().sum()))



fig, ax  = plt.subplots(nrows=1, ncols=2, figsize=(12,5))

sns.countplot(df['Exterior1st'], ax=ax[0])

sns.countplot(df['Exterior2nd'], ax=ax[1])

for ax in fig.axes:

    plt.sca(ax)

    plt.xticks(rotation=45)

fig.show()
correlation('Exterior1st|Exterior2nd')
df.loc[:, ('Id', 'Exterior1st', 'Exterior2nd', 'Foundation')][df['Exterior1st'].isnull()]
sns.catplot(x='Foundation', data=df, hue='Exterior2nd', kind='count')

plt.xticks(rotation=45, ha='right')

plt.show()



for dataset in (train, test, df):

    dataset['Exterior1st'] = dataset['Exterior1st'].fillna('VinylSd')

    dataset['Exterior2nd'] = dataset['Exterior2nd'].fillna('VinylSd')
print('Missing values in MasVnrType: {}'.format(df['MasVnrType'].isna().sum()))

print('Missing values in MasVnrArea: {}'.format(df['MasVnrArea'].isna().sum()))



sns.set(style='darkgrid')

sns.countplot(df['MasVnrType'])

plt.title('{0} ({1})'.format('MasVnrType', df.name))

plt.show()
df['YearBuiltGrouped'] = pd.cut(df['YearBuilt'], 10)

countplot('YearBuiltGrouped', df)
sns.set(style='darkgrid')

sns.catplot(x='YearBuiltGrouped', data=df, hue='MasVnrType', kind='count')

plt.xticks(rotation=45, ha='right')

plt.title('MasVnrType per YearBuiltGrouped')

plt.show()



missing_MasVnrType = df[df["MasVnrType"].isnull()]

sns.set(style='darkgrid')

sns.countplot(missing_MasVnrType['YearBuiltGrouped'])

plt.title('YearBuiltGrouped for missing values in MasVnrType')

plt.xticks(rotation=45, ha='right')

plt.show()
correlation('MasVnrArea')
sns.catplot(x='Fireplaces', data=df, hue='MasVnrType', kind='count')

plt.xticks(rotation=45, ha='right')

plt.show()



df.loc[:, ('Id', 'MasVnrType', 'Fireplaces')][df['MasVnrType'].isnull()]
train = train.drop(['MasVnrType', 'MasVnrArea'], axis=1)

test = test.drop(['MasVnrType', 'MasVnrArea'], axis=1)

df = df.drop(['MasVnrType', 'MasVnrArea'], axis=1)
print('Missing values in LotFrontage: {}'.format(df['LotFrontage'].isna().sum()))



df['LotFrontageGrouped'] = pd.cut(df['LotFrontage'], 10)



sns.countplot(df['LotFrontageGrouped'])

plt.xticks(rotation=45, ha='right')

plt.show()



correlation('LotFrontage')
missing_LotFrontage = df.loc[:, ('Id', 'LotFrontage', 'BldgType')][df['LotFrontage'].isna()]

sns.countplot(missing_LotFrontage['BldgType'])

plt.title('Missing values in LotFrontage grouped by BldgType')

plt.show()



for dataset in (train, test, df):

    dataset['LotFrontage'] = dataset['LotFrontage'].fillna(df.groupby('BldgType')['LotFrontage'].transform('mean'))
print('Missing values in Electrical: {}'.format(df['Electrical'].isna().sum()))



sns.countplot(df['Electrical'])

plt.xticks(rotation=45, ha='right')

plt.show()



correlation('Electrical')
df.loc[:, ('Id', 'Electrical', 'CentralAir')][df['Electrical'].isna()]
sns.catplot(x='CentralAir', data=df, hue='Electrical', kind='count')

plt.xticks(rotation=45, ha='right')

plt.show()



distribution = df.groupby('CentralAir')['Electrical'].value_counts()[4:]

legend = ['SBrkr', 'FuseA', 'FuseF', 'FuseP', 'Mix']    

percentage = 100 * distribution.values / distribution.values.sum()



plt.pie(distribution.values / distribution.values.sum(), wedgeprops=dict(edgecolor='black', linewidth=0.25))

plt.legend(['{0} - {1:1.2f} %'.format(i, j) for i, j in zip(legend, percentage)], loc='best', fontsize=10, frameon=True)

plt.title('Distribution of Electrical for CentralAir=Y', fontsize=12)

plt.show()



for dataset in (train, test, df):

    dataset['Electrical'] = dataset['Electrical'].fillna('Sbrkr')
print('Missing values in BsmtFullBath: {}'.format(df['BsmtFullBath'].isna().sum()))

print('Missing values in BsmtHalfBath: {}'.format(df['BsmtHalfBath'].isna().sum()))
df.loc[:, ('Id', 'BsmtQual', 'BsmtFullBath', 'BsmtHalfBath')][df['BsmtFullBath'].isna()|df['BsmtHalfBath'].isna()]
for dataset in (train, test, df):

    for feature in ('BsmtFullBath', 'BsmtHalfBath'):

        dataset[feature] = dataset[feature].fillna(0)
print('Missing values in KitchenQual: {}'.format(df['KitchenQual'].isna().sum()))



correlation('KitchenQual')
df.loc[:, ('KitchenQual','ExterQual', 'BsmtQual', 'OverallQual')][df['KitchenQual'].isna()]
for dataset in (train, test, df):

    dataset['KitchenQual'] = dataset['KitchenQual'].fillna(dataset['ExterQual'])



df.loc[[1555], ('KitchenQual','ExterQual', 'OverallQual')]
print('Missing values in Functional: {}'.format(df['Functional'].isna().sum()))



plt.figure(figsize=(8, 6))

plt.pie(df['Functional'].value_counts() / df['Functional'].value_counts().sum(),

        wedgeprops=dict(edgecolor='black', linewidth=0.25))

percentage = 100. * df['Functional'].value_counts() / df['Functional'].value_counts().sum()

plt.legend(['{0} - {1:1.2f} %'.format(i, j) for i, j in zip(percentage.index, percentage)],

           loc='best', fontsize=10, frameon=True)

plt.title('Distribution of Functional', fontsize=14)

plt.show()



# Replace NaN with 'Typ'

for dataset in (train, test, df):

    dataset['Functional'] = dataset['Functional'].fillna('Typ')
print('Missing values in SaleType: {}'.format(df['SaleType'].isna().sum()))



df.loc[:, ('SalePrice', 'SaleType')].apply(lambda x: x.factorize()[0]).corr()



train = train.drop('SaleType', axis=1)

test = test.drop('SaleType', axis=1)

df = df.drop('SaleType', axis=1)
for column in ['Count', 'BsmtFinSF2Grouped', 'YearBuiltGrouped', 'LotFrontageGrouped']:

    df = df.drop(column, axis=1)



for dataset in (train, test,df):

    for column in ['MoSold', 'SaleCondition']:

        dataset = dataset.drop(column, axis=1)
df.columns[df.isnull().any()].tolist()
for dataset in (train, test, df):

    dataset['TotalSF'] = dataset['TotalBsmtSF'] + dataset['1stFlrSF'] + dataset['2ndFlrSF']
catplot('FullBath', train)

catplot('HalfBath', train)
for dataset in (train, test, df):

    dataset['Bath'] = dataset['FullBath'] + dataset['HalfBath']



catplot('Bath', train)
df.loc[:, ('YearBuilt', 'YearRemodAdd')].corr()
fig, ax  = plt.subplots(nrows=1, ncols=2, figsize=(14,4))

sns.scatterplot(x='YearBuilt', y='SalePrice', data=train, ax=ax[0])

sns.scatterplot(x='YearRemodAdd', y='SalePrice', data=train, ax=ax[1])



for ax in fig.axes:

    plt.sca(ax)

    plt.xticks(fontsize=8), plt.yticks(fontsize=8)

fig.show()
train = train.drop(['YearRemodAdd'], axis=1)

test = test.drop(['YearRemodAdd'], axis=1)

df = df.drop(['YearRemodAdd'], axis=1)
X = df.drop(['SalePrice'], axis=1)



print('Columns before one-hot encoding:', X.shape[1])

        

for column in X:

    if X[column].dtypes == 'object':

        one_hot_encoding = pd.get_dummies(X[column])

        one_hot_encoding.columns = column + '_' + one_hot_encoding.columns.astype('str')

        X = pd.concat([X, one_hot_encoding], ignore_index=False, axis=1, sort=False)

        X = X.drop(column, axis=1)



print('Columns after one-hot encoding:', X.shape[1])
highest_correlation_target = df.drop(['Id'], axis=1).corr().abs().unstack().sort_values(kind='quicksort', ascending=False)#.drop_duplicates(keep='first')

highest_correlation_target = highest_correlation_target.reset_index().rename(columns={0: 'Correlation'})

highest_correlation_target = highest_correlation_target[highest_correlation_target['level_0'].str.contains('SalePrice') |

                                                        highest_correlation_target['level_1'].str.contains('SalePrice')]

highest_correlation_target = highest_correlation_target[highest_correlation_target['Correlation'] < 1]

highest_correlation_target.drop_duplicates(subset='Correlation')[:10]
poly = PolynomialFeatures(degree=3)

polynomial = pd.DataFrame(poly.fit_transform(X.loc[:, ('TotalSF', 'OverallQual', 'GrLivArea', 'TotalBsmtSF', 'GarageCars', '1stFlrSF', 'GarageArea', 'Bath', 'FullBath', 'TotRmsAbvGrd')]),

                          columns=poly.get_feature_names(X.loc[:, ('TotalSF', 'OverallQual', 'GrLivArea', 'TotalBsmtSF', 'GarageCars', '1stFlrSF', 'GarageArea', 'Bath', 'FullBath','TotRmsAbvGrd')].columns))



X = pd.concat([X, polynomial.loc[:, 'TotalSF^2':'TotRmsAbvGrd^3']], ignore_index=False, axis=1, sort=False)
for column in X:

    if column != 'Id':

        X[column] = MinMaxScaler().fit_transform(X[[column]])
X_trainval = X.loc[X['Id'] < 1461]

X_trainval = X_trainval.drop(['Id'], axis=1)

X_trainval.name = 'X_trainval'



X_test = X.loc[X['Id'] >= 1461]  

X_test = X_test.drop(['Id'], axis=1)

X_test.name = 'X_test'
X_training, X_validation = train_test_split(X_trainval, test_size=0.2, shuffle=False)

X_training.name = 'X_training'

X_validation.name = 'X_validation'
print('Datasets for estimating generalization performance:')

print('X_training: {}'.format(X_training.shape))

print('X_validation: {}\n'.format(X_validation.shape))



print('Datasets for predicting test set:')

print('X_trainval: {}'.format(X_trainval.shape))

print('X_test: {}'.format(X_test.shape))
X_trainval[:5]
y_trainval = np.ravel(train[['SalePrice']])

y_training = y_trainval[:len(X_training)]

y_validation = y_trainval[len(X_training):]
model_comparison = pd.DataFrame({'Model': [], 'RMSE': []})



xg_boost = xgboost.XGBRegressor(n_estimators=500, max_depth=3, learning_rate=0.1)

xg_boost.name ='XGBoost'

gradient_boosting = GradientBoostingRegressor(learning_rate=0.1, n_estimators=500, max_depth=3, alpha=0.9)

gradient_boosting.name = 'Gradient boosting'

random_forest = RandomForestRegressor(n_estimators=200, max_features=40, max_depth=40)

random_forest.name = 'Random forest'

lasso = LassoCV(alphas=None, n_alphas=50, cv=10)

lasso.name = 'LASSO'



for model in [xg_boost, gradient_boosting, random_forest, lasso]:

    # Train the model with polynomials

    model.fit(X_training, y_training)

    rmse_poly = round(mean_squared_error(np.log(y_validation), np.log(model.predict(X_validation))) ** (1/2), 5)



    model_results = pd.DataFrame({'Model': [model.name + ' (polynomials)'], 'RMSE': [rmse_poly]})

    model_comparison = model_comparison.append(model_results, ignore_index=True)



    # Train the model without polynomials

    X_training_no_poly = X_training.loc[:, :'SaleCondition_Partial']

    X_validation_no_poly = X_validation.loc[:, :'SaleCondition_Partial']



    model.fit(X_training_no_poly, y_training)



    rmse_no_poly = round(mean_squared_error(np.log(y_validation), np.log(model.predict(X_validation_no_poly))) ** (1/2), 5)



    model_results = pd.DataFrame({'Model': [model.name + ' (no polynomials)'], 'RMSE': [rmse_no_poly]})

    model_comparison = model_comparison.append(model_results, ignore_index=True)

    

model_comparison.sort_values(by='RMSE', ascending=True).reset_index(drop=True)
def learning(model):

    train_sizes, train_scores, test_scores = learning_curve(model, X_trainval, y_trainval, cv=5)

    train_mean = np.mean(train_scores, axis=1)

    train_std = np.std(train_scores, axis=1)

    test_mean = np.mean(test_scores, axis=1)

    test_std = np.std(test_scores, axis=1)



    plt.style.use('seaborn-darkgrid')

    plt.plot(train_sizes, train_mean, color='#1f77b4', label='Training set', linewidth=2)  # Draw lines train

    plt.plot(train_sizes, test_mean, color='#d62728', label='Validation set', linewidth=2)  # Draw lines validation

    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color='#1f77b4', alpha=0.25)  # Draw band train

    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color='#d62728', alpha=0.25)  # Draw band validation

    plt.title('Learning curve {}'.format(model.name)), plt.xlabel('Training set size'), plt.ylabel('Score'), plt.ylim(0, 1)

    plt.legend(loc='best')

    plt.show()
X_training_no_poly = X_training.loc[:, :'SaleCondition_Partial']

X_validation_no_poly = X_validation.loc[:, :'SaleCondition_Partial']



xg_boost.fit(X_training_no_poly, y_training)



feature_importance = pd.DataFrame({'Feature': X_training_no_poly.columns, 'Relative Importance': xg_boost.feature_importances_})

feature_importance = feature_importance.iloc[feature_importance['Relative Importance'].abs().argsort()[::-1]].reset_index(drop=True)

feature_importance[:10]
learning(xg_boost)
gradient_boosting.fit(X_training_no_poly, y_training)



feature_importance = pd.DataFrame({'Feature': X_training_no_poly.columns, 'Relative Importance': gradient_boosting.feature_importances_})

feature_importance = feature_importance.iloc[feature_importance['Relative Importance'].abs().argsort()[::-1]].reset_index(drop=True)

feature_importance[:10]
learning(gradient_boosting)
random_forest.fit(X_training_no_poly, y_training)



feature_importance = pd.DataFrame({'Feature': X_training_no_poly.columns, 'Relative Importance': random_forest.feature_importances_})

feature_importance = feature_importance.iloc[feature_importance['Relative Importance'].abs().argsort()[::-1]].reset_index(drop=True)

feature_importance[:10]
learning(random_forest)
lasso.fit(X_training_no_poly, y_training)



alphas = pd.DataFrame(list(lasso.alphas_), columns=['Alpha'])

coefficient_path = lasso.path(X_training_no_poly, y_training, alphas=alphas)

coefficients = pd.DataFrame(coefficient_path[1], index=X_training_no_poly.columns).T.iloc[::-1].reset_index(drop=True)

result = pd.concat([alphas, coefficients], axis=1, sort=False)

best_alpha = result[result['Alpha'] == lasso.alpha_].tail(1)



used_features = best_alpha.columns[(best_alpha != 0).iloc[0]].tolist()

used_features.remove('Alpha')



best_coefficients = best_alpha[best_alpha['Alpha'] == lasso.alpha_].reset_index(drop=True)

best_coefficients = best_coefficients.drop(['Alpha'], axis=1)

best_coefficients = best_coefficients.iloc[0]

best_coefficients = pd.DataFrame({'Feature': best_coefficients.index, 'Coefficient': best_coefficients.values},

                                columns=['Feature', 'Coefficient'])



best_coefficients = best_coefficients.iloc[(-best_coefficients['Coefficient'].abs()).argsort()].reset_index(drop=True)



print('At the best alpha {0}, LASSO set {1} of the total {2} features equal to zero.'.format(

    round(lasso.alpha_, 5),  sum(best_coefficients['Coefficient'] == 0), len(best_coefficients['Coefficient'])))



best_coefficients[:10]
learning(lasso)
prediction_validation_xg_boost = pd.DataFrame({'Id': X['Id'][:len(X_validation_no_poly)], 'Actual': y_validation, 'SalePrice': xg_boost.predict(X_validation_no_poly)})

prediction_validation_gradient_boosting = pd.DataFrame({'Id': X['Id'][:len(X_validation_no_poly)], 'Actual': y_validation, 'SalePrice': gradient_boosting.predict(X_validation_no_poly)})

prediction_validation_random_forest = pd.DataFrame({'Id': X['Id'][:len(X_validation_no_poly)], 'Actual': y_validation, 'SalePrice': random_forest.predict(X_validation_no_poly)})

prediction_validation_lasso = pd.DataFrame({'Id': X['Id'][:len(X_validation_no_poly)], 'Actual': y_validation, 'SalePrice': lasso.predict(X_validation_no_poly)})
plt.figure(figsize=(16,10))

plt.plot(prediction_validation_gradient_boosting['Id'][:50], prediction_validation_gradient_boosting['Actual'].sort_values()[:50])

plt.plot(prediction_validation_gradient_boosting['Id'][:50], prediction_validation_gradient_boosting['SalePrice'].sort_values()[:50])

plt.plot(prediction_validation_xg_boost['Id'][:50], prediction_validation_xg_boost['SalePrice'].sort_values()[:50])

plt.plot(prediction_validation_lasso['Id'][:50], prediction_validation_lasso['SalePrice'].sort_values()[:50])

plt.plot(prediction_validation_random_forest['Id'][:50], prediction_validation_random_forest['SalePrice'].sort_values()[:50])

plt.legend(['True SalePrice', 'XG boost', 'Gradient boosting', 'Random forest', 'LASSO']), plt.ylabel('SalePrice'), plt.xlabel('Observation')

plt.title('True SalePrice vs. predictions (cheapest houses)')

plt.show()
plt.figure(figsize=(16,10))

plt.plot(prediction_validation_gradient_boosting['Id'][240:], prediction_validation_gradient_boosting['Actual'].sort_values()[240:])

plt.plot(prediction_validation_gradient_boosting['Id'][240:], prediction_validation_gradient_boosting['SalePrice'].sort_values()[240:])

plt.plot(prediction_validation_xg_boost['Id'][240:], prediction_validation_xg_boost['SalePrice'].sort_values()[240:])

plt.plot(prediction_validation_random_forest['Id'][240:], prediction_validation_random_forest['SalePrice'].sort_values()[240:])

plt.plot(prediction_validation_lasso['Id'][240:], prediction_validation_lasso['SalePrice'].sort_values()[240:])

plt.legend(['True SalePrice', 'Gradient boosting', 'XG boost', 'Random forest', 'LASSO']), plt.ylabel('SalePrice'), plt.xlabel('Observation')

plt.title('True SalePrice vs. predictions (most expensive houses)')

plt.show()
prediction_validation_average = ((gradient_boosting.predict(X_validation_no_poly) +

                                  xg_boost.predict(X_validation_no_poly) +

                                  random_forest.predict(X_validation_no_poly) +

                                  lasso.predict(X_validation_no_poly)) / 4)



rmse_average = round(mean_squared_error(np.log(y_validation), (np.log(prediction_validation_average))) ** (1/2), 5)

print('XGBoost, Gradient Boosting, Random Forest & LASSO:')

print('RMSE =', rmse_average)
xg_boost.fit(X_training_no_poly, y_training)

gradient_boosting.fit(X_training_no_poly, y_training)

random_forest.fit(X_training_no_poly, y_training)

lasso.fit(X_training_no_poly, y_training)



# Index has to be reset to match indices of X_validation_no_poly and prediction_validation_average (since observations have been dropped earlier)

X_validation_no_poly.set_index([list(range(len(X_training)+1, len(X_training)+1+len(X_validation_no_poly)))], inplace=True)



for index, value in enumerate(prediction_validation_average):

    if value > 400000:

        print('Index:', index)

        print(value)

        prediction_validation_average[index] = (gradient_boosting.predict(X_validation_no_poly.loc[[index+1167], :]) +

                                                xg_boost.predict(X_validation_no_poly.loc[[index+1167], :])) / 2

        print(prediction_validation_average[index])
rmse_average = round(mean_squared_error(np.log(y_validation), (np.log(prediction_validation_average))) ** (1/2), 5)

print('Outliers only XGBoost, Gradient Boosting, Random Forest & LASSO:')

print('RMSE =', rmse_average)
X_trainval_no_poly = X_trainval.loc[:, :'SaleCondition_Partial']

X_test = X_test.loc[:, :'SaleCondition_Partial']



xg_boost.fit(X_trainval_no_poly, y_trainval)

gradient_boosting.fit(X_trainval_no_poly, y_trainval)

random_forest.fit(X_trainval_no_poly, y_trainval)

lasso.fit(X_trainval_no_poly, y_trainval)



prediction_test = pd.DataFrame({'Id': test['Id'], 'SalePrice': ((xg_boost.predict(X_test) +

                                                                 gradient_boosting.predict(X_test) +

                                                                 random_forest.predict(X_test) +

                                                                 lasso.predict(X_test)) / 4)})



# Make XGBoost and gradient boosting predict houses with predicted SalePrice >400K

for index, column in prediction_test.iterrows():

    if column['SalePrice'] > 400000:

        print('Index:', index)

        print(column['SalePrice'])

        prediction_test.loc[index, 'SalePrice'] = ((xg_boost.predict(X_test.loc[[1460+index], :]) + 

                                                   gradient_boosting.predict(X_test.loc[[1460+index], :]))/2)

        print(prediction_test.loc[index, 'SalePrice'])
prediction_test.to_csv('my_submission.csv', index=False)