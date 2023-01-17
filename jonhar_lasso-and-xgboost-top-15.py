from IPython.display import HTML

# This code bit is directly taken from this notebook: https://www.kaggle.com/agodwinp/stacking-house-prices-walkthrough-to-top-5/notebook
HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
<form action="javascript:code_toggle()"><input type="submit" value="Click here to toggle on/off the raw code."></form>''')
# First imports
import pandas as pd
import seaborn as sns
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')
print('First imports loaded.')

#Turn off some annoying deprecation warnings
import warnings
warnings.warn = lambda *args, **kwargs: None
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
print('Deprecation warnings turned off.')

# Load the data
train_data = pd.read_csv('../input/train.csv', index_col='Id')
test_data = pd.read_csv('../input/test.csv', index_col='Id')
print('Data loaded.')
print('Training observations: {}'.format(train_data.shape[0]))
print('Test observations: {}'.format(test_data.shape[0]))
train_data['SalePrice'].describe()
fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(10,5))
sns.distplot(train_data['SalePrice'], fit=stats.norm, ax=ax[0])
sns.distplot(np.log(train_data['SalePrice']), fit=stats.norm, ax=ax[1])
ax[0].legend(['Closest Normal'])
ax[0].text(x=3.7e5, y=0.0000055, 
           s='Data Statistics\n--------------------\n' + 
           ' mean = {:.2f} \n std     = {:.2f}'.format(
               train_data['SalePrice'].mean(), train_data['SalePrice'].std()) + 
           '\n skew  = {:.2f} \n e.kurt = {:.2f}'.format(
               train_data['SalePrice'].skew(), train_data['SalePrice'].kurt())
          )
ax[1].legend(['Closest Normal'])
ax[1].text(x=12.5, y=0.8, 
           s='Data Statistics\n--------------------\n' + 
           ' mean = {:.2f} \n std     = {:.2f}'.format(
               np.log(train_data['SalePrice']).mean(), np.log(train_data['SalePrice']).std()) + 
           '\n skew  = {:.2f} \n e.kurt = {:.2f}'.format(
               np.log(train_data['SalePrice']).skew(), np.log(train_data['SalePrice']).kurt())
          )
ax[0].set_xlabel('SalePrice')
ax[0].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
ax[1].set_xlabel('log-SalePrice')
plt.show()
outlier_ID = train_data[(train_data['GrLivArea'] > 4000) & (train_data['SalePrice'] < 300000)].index
unusual_ID = train_data[(train_data['GrLivArea'] > 4000) & (train_data['SalePrice'] > 300000)].index
sns.lmplot('GrLivArea', 'SalePrice', hue='Outliers', legend=False, fit_reg=False,
           data=pd.concat([train_data, (train_data['GrLivArea'] > 4000).rename('Outliers')], axis=1)
           )
plt.axvline(4000, color='black', linewidth=1)

print('Number of training observations with more than 4000 sqft: {}'.format(
    (train_data['GrLivArea'] > 4000).sum()))
print('Number of observations with more than 4000 sqft in the test set: {}'.format(
    (test_data['GrLivArea'] > 4000).sum()))
avg_large_partial_price = train_data.loc[outlier_ID, 'SalePrice'].mean()
print('The average sale price of the two unusual partial sales is ${:.0f}.'.format(avg_large_partial_price))
train_data.drop(outlier_ID, inplace=True)
train_data.drop(unusual_ID, inplace=True)
print('Outliers dropped.')
y_train = np.log(train_data['SalePrice'])
print('y_train defined as log-SalePrice.')
train_data.drop('SalePrice', axis=1, inplace=True)
print('SalePrice dropped from remaining data.')
X = pd.concat([train_data, test_data])
print('Train and test data for explanatory variables merged.')

train_ID = train_data.index
test_ID = test_data.index
print('Training and test set ID stored.')
X_na_means_none = ['Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 
                   'BsmtFinType2', 'FireplaceQu', 'GarageType', 'GarageFinish',
                   'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature']

X.loc[:, X_na_means_none] = X.loc[:, X_na_means_none].fillna('None')
print('False positives fixed.')
X_missing_perc = X.isnull().mean()[X.isnull().any()]
X_missing_perc.sort_values(ascending=False).plot.bar(figsize=(6,3))
plt.title('Missing values')
plt.ylabel('Share')
plt.show()
tmp = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 
       'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']

for var in tmp:
    print('Missing observations for {}: {}'.format(var, X[var].isnull().sum()))
X.loc[X[tmp].isnull().any(axis=1), tmp]
X.loc[2189,['BsmtFullBath', 'BsmtHalfBath']] = 0
print('ID 2819 fixed.')
X.loc[2121, [col for col in X.columns if 'Bsmt' in col]]
X.loc[((X['Neighborhood'] == X.loc[2121, 'Neighborhood'])
       & (X['MSSubClass'] == X.loc[2121, 'MSSubClass'])
       & (X['YearBuilt'] >= X.loc[2121, 'YearBuilt'] - 5)
       & (X['YearBuilt'] <= X.loc[2121, 'YearBuilt'] + 5)), 
      ['GrLivArea', 'BsmtCond', 'BsmtFinType1', 'BsmtExposure', 
       'BsmtFinSF1', 'BsmtFinSF2', 'TotalBsmtSF', 'BsmtFullBath', 
       'BsmtHalfBath']]
X.loc[2121,['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 
            'BsmtFullBath', 'BsmtHalfBath']] = 0.
print('ID 2121 fixed.')
print('Basement missing values filled.')
print('Missing observations for {}: {}'.format('SaleType', X['SaleType'].isnull().sum()))
X.loc[X['SaleType'].isnull(), ['SaleType', 'YrSold', 'SaleCondition']]
(X['SaleType'].value_counts()/X['SaleType'].count()).plot.bar()
plt.title('SaleType')
plt.ylabel('Share')
plt.show()
tmp = X.loc[((X['Neighborhood'] == X.loc[2490, 'Neighborhood'])
             & (X['MSSubClass'] == X.loc[2490, 'MSSubClass'])
             & (X['YrSold'] == X.loc[2490, 'YrSold'])
             & (X['SaleCondition'] == X.loc[2490, 'SaleCondition']))]['SaleType']

print('There are {} neighboring observations.'.format(len(tmp)))
tmp.value_counts(dropna=False).plot.bar()
plt.title('SaleType for neighboring observations.')
plt.show()
X.loc[2490, 'SaleType'] = 'WD'
print('SaleType missing values filled.')
print('Missing observations for {}: {}'.format('KitchenQual', X['KitchenQual'].isnull().sum()))
X.loc[X['KitchenQual'].isnull(), ['KitchenAbvGr', 'KitchenQual']]
X.loc[((X['Neighborhood'] == X.loc[1556, 'Neighborhood'])
       & (X['MSSubClass'] == X.loc[1556, 'MSSubClass'])), 
      ['GrLivArea', 'OverallQual', 'OverallCond', 'KitchenQual', 'KitchenAbvGr']]
X.loc[X['KitchenQual'].isnull(), 'KitchenQual'] = 'TA'
print('KitchenQual missing values filled.')
tmp = ['GarageCars', 'GarageArea', 'GarageYrBlt']

for var in tmp:
    print('Missing observations for {}: {}'.format(var, X[var].isnull().sum()))
X.loc[X[['GarageCars', 'GarageArea']].isnull().any(axis=1),
     [col for col in X.columns if 'Garage' in col]]
tmp = X.loc[((X['Neighborhood'] == X.loc[2577, 'Neighborhood'])
       & (X['MSSubClass'] == X.loc[2577, 'MSSubClass'])
       & (X['GarageType'] == X.loc[2577, 'GarageType'])
       & (X['YearBuilt'] >= X.loc[2577, 'YearBuilt'] - 10)
       & (X['YearBuilt'] <= X.loc[2577, 'YearBuilt'] + 10)), 
      ['LotArea', 'GarageArea', 'GarageCars', 'GarageCond', 'GarageFinish',
      'GarageQual', 'GarageType', 'GarageYrBlt', 'YearBuilt', 'YearRemodAdd']]

tmp
X.loc[2577, 'GarageArea'] = tmp['GarageArea'].mean()
X.loc[2577, 'GarageCars'] = tmp['GarageCars'].mean()
X.loc[2577, 'GarageCond'] = tmp['GarageCond'].mode()[0]
X.loc[2577, 'GarageFinish'] = tmp['GarageFinish'].mode()[0]
X.loc[2577, 'GarageQual'] = 'TA'
X.loc[2577, 'GarageYrBlt'] = X.loc[2577, 'YearBuilt'] 
print('Missing observations for {}: {}'.format('GarageYrBlt', X['GarageYrBlt'].isnull().sum()))
print('of which have no garage: {}'.format(X['GarageYrBlt'].isnull()[X['GarageArea'] == 0].sum()))
X.loc[X['GarageYrBlt'].isnull() & (X['GarageArea'] == 0), 'GarageYrBlt'] = 0.
X.loc[X['GarageYrBlt'].isnull(), 
      ['LotArea', 'GarageArea', 'GarageCars', 'GarageCond', 'GarageFinish',
      'GarageQual', 'GarageType', 'GarageYrBlt', 'YearBuilt', 'YearRemodAdd']]
tmp = X[X['GarageYrBlt'].isnull()].index[0]

X.loc[((X['Neighborhood'] == X.loc[tmp, 'Neighborhood'])
       & (X['MSSubClass'] == X.loc[tmp, 'MSSubClass'])
       & (X['GarageType'] == X.loc[tmp, 'GarageType'])), 
      ['LotArea', 'GarageArea', 'GarageCars', 'GarageCond', 'GarageFinish',
      'GarageQual', 'GarageType', 'GarageYrBlt', 'YearBuilt', 'YearRemodAdd']]
X.loc[2127, 'GarageCond'] = 'TA'
X.loc[2127, 'GarageQual'] = 'TA'
X.loc[2127, 'GarageFinish'] = 'Unf'
X.loc[2127, 'GarageYrBlt'] = X.loc[2127, 'YearBuilt']
print('Largest GarageYrBlt: {:.0f}'.format(X['GarageYrBlt'].max()))
X.loc[X['GarageYrBlt'].idxmax(), ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt',
                                  'YrSold']]
X.loc[2593, 'GarageYrBlt'] = 2007
print('Garage missing values filled.')
print('Missing observations for {}: {}'.format('Electrical', X['Electrical'].isnull().sum()))
X.loc[((X['Neighborhood'] == X.loc[1380, 'Neighborhood'])
       & (X['MSSubClass'] == X.loc[1380, 'MSSubClass'])), 
      ['GrLivArea','LotArea', 'YearBuilt', 'YearRemodAdd', 
       'Electrical', 'Heating', 'CentralAir']]
X.loc[1380, 'Electrical'] = 'SBrkr'
print('Electrical missing values filled.')
print('Missing observations for {}: {}'.format('Exterior1st', X['Exterior1st'].isnull().sum()))
print('Missing observations for {}: {}'.format('Exterior2nd', X['Exterior2nd'].isnull().sum()))
X.loc[X['Exterior1st'].isnull(), 
      ['GrLivArea', 'Exterior1st', 'Exterior2nd', 'ExterQual',
       'ExterCond', 'Foundation', 'RoofMatl', 'RoofStyle', 'YearBuilt',
       'YearRemodAdd']]
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12,10))
X['Exterior1st'].value_counts().plot.bar(ax=ax[0,0])
ax[0,0].set_title('Exterior1st')
X['Exterior2nd'].value_counts().plot.bar(ax=ax[0,1])
ax[0,1].set_title('Exterior2nd')
X['RoofMatl'].value_counts().plot.bar(ax=ax[1,0])
ax[1,0].set_title('RoofmMatl')

tmp = pd.get_dummies(X[['Exterior1st', 'RoofMatl']]).corr()
sns.heatmap(tmp.loc[[var for var in tmp.index if 'Exterior' in var], [var for var in tmp.columns if 'Roof' in var]], ax=ax[1,1])
ax[1,1].set_title('Correlation between Exterior1st and RoofMatl')

fig.tight_layout()
tmp1 = pd.DataFrame(X.groupby('RoofMatl')['Exterior1st'].value_counts(dropna=False))
tmp1['Exterior'] = '1st'
tmp1.columns = ['Count', 'Exterior']
tmp1.reset_index(inplace=True)
tmp1.columns = ['RoofMatl', 'ExteriorMatl', 'Count', 'Exterior']
tmp2 = pd.DataFrame(X.groupby('RoofMatl')['Exterior2nd'].value_counts())
tmp2['Exterior'] = '2nd'
tmp2.columns = ['Count', 'Exterior']
tmp2.reset_index(inplace=True)
tmp2.columns = ['RoofMatl', 'ExteriorMatl', 'Count', 'Exterior']
tmp = pd.concat([tmp1, tmp2])
tmp = tmp[tmp['RoofMatl'].isin(['Tar&Grv'])]
tmp

sns.factorplot(x="RoofMatl", y="Count",
                    hue="ExteriorMatl", col="Exterior",
                    data=tmp, kind="bar")
plt.show()
X.loc[2152, ['Exterior1st', 'Exterior2nd']] = 'Plywood'
print('Exterior missing values filled.')
print('Missing observations for {}: {}'.format('Functional', X['Functional'].isnull().sum()))
X.loc[X['Functional'].isnull(), 
      ['Neighborhood', 'MSSubClass', 'YearBuilt', 'YearRemodAdd',
       'OverallQual', 'OverallCond', 'Functional']]
X.loc[((X['Neighborhood'] == 'IDOTRR')
       & (X['OverallQual'] == 1) | (X['OverallCond'] == 1)), 
      ['Functional', 'OverallQual', 'OverallCond']]
X.loc[X['Functional'].isnull(), 'Functional'] = 'Maj1'
print('Functional missing values filled.')
print('Missing observations for {}: {}'.format('Utilities', X['Utilities'].isnull().sum()))
print('Number of occurences of Utilities categories')
print('--------------------------------------------')
X['Utilities'].value_counts()
X.loc[X['Utilities'].isnull(), 'Utilities'] = 'AllPub'
print('Utilities missing values filled.')
print('Missing observations for {}: {}'.format('MSZoning', X['MSZoning'].isnull().sum()))
X.loc[X['MSZoning'].isnull(), ['Neighborhood', 'MSSubClass', 'MSZoning']]
gridsize = (2, 3)
fig = plt.figure(figsize=(12, 5))
ax1 = plt.subplot2grid(gridsize, (0, 0), colspan=1, rowspan=2)
ax2 = plt.subplot2grid(gridsize, (0, 1))
ax3 = plt.subplot2grid(gridsize, (0, 2))
ax4 = plt.subplot2grid(gridsize, (1, 1))
ax5 = plt.subplot2grid(gridsize, (1, 2))

X['MSZoning'].value_counts().plot.bar(rot=45, ax=ax1)
ax1.set_title('Bachart of MSZoning')

ax = [ax2, ax3, ax4, ax5]

for i, comb in enumerate([('IDOTRR', 30), ('IDOTRR', 20),
                          ('IDOTRR', 70), ('Mitchel', 20)]):
    X.loc[(X['Neighborhood'] == comb[0]) & (X['MSSubClass'] == comb[1]), 'MSZoning'].value_counts().plot.bar(ax = ax[i])
    ax[i].set_title('MSZoning \n(Neighborhood: {}, MSSubClass: {})'.format(comb[0], str(comb[1]))) 
    
fig.tight_layout()
X.loc[1916, 'MSZoning'] = 'RM'
X.loc[2217, 'MSZoning'] = 'C (all)'
X.loc[2251, 'MSZoning'] = 'RM'
X.loc[2905, 'MSZoning'] = 'RL'
print('MSSubClass missing values filled.')
X.groupby('MasVnrType')['MasVnrArea'].describe()
X.loc[(((X['MasVnrArea'] > 0) & (X['MasVnrType'] == 'None'))
       |((X['MasVnrArea'] == 0) & (X['MasVnrType'] != 'None'))), 
      ['MasVnrArea', 'MasVnrType']]
X.loc[((X['MasVnrArea'] == 0) & (X['MasVnrType'] != 'None')), 'MasVnrType'] = 'None'
X.loc[((X['MasVnrArea'] == 1) & (X['MasVnrType'] == 'None')), 'MasVnrArea'] = 0.
X.loc[((X['MasVnrArea'] > 1) & (X['MasVnrType'] == 'None')), 'MasVnrType'] = np.nan
print('Missing observations for {}: {}'.format('MasVnrArea', X['MasVnrArea'].isnull().sum()))
print('Missing observations for {}: {}'.format('MasVnrType', X['MasVnrType'].isnull().sum()))
tmp = X.loc[X[['MasVnrArea', 'MasVnrType']].isnull().any(axis=1), ['MasVnrArea', 'MasVnrType']]
print('Total number of missing observations in either MasVnrArea or MasVnrType: {}'.format(tmp.shape[0]))
for i in tmp.index:
    tmp2 = X.loc[((X['Neighborhood'] == X.loc[i, 'Neighborhood'])
                  & (X['MSSubClass'] == X.loc[i, 'MSSubClass'])), 
                 ['MasVnrType', 'MasVnrArea']]
    mode = tmp2['MasVnrType'].mode()[0]
    X.loc[i, 'MasVnrType'] = mode
    if np.isnan(X.loc[i, 'MasVnrArea']):
        if mode == 'None':
            X.loc[i, 'MasVnrArea'] = 0.
        else:
            X.loc[i, 'MasVnrArea'] = tmp2.loc[tmp2['MasVnrType'] == mode].median()[0]

print('Masonry Veneer missing values filled.')
print('Missing observations for {}: {}'.format('LotFrontage', X['LotFrontage'].isnull().sum()))
fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(12,4))

sns.distplot(X.loc[X['LotFrontage'].notnull(), 'LotFrontage'], ax=ax[0])
ax[0].set_title('Histogram of LotFrontage')

sns.distplot(X.loc[X['LotFrontage'].notnull(), 'LotFrontage'].apply(np.log), ax=ax[1])
ax[1].set_title('Histogram of log-LotFrontage')
ax[1].set_xlabel('log-LotFrontage')

sns.regplot(x='LotFrontage', y='LotArea', data=X.loc[X['LotFrontage'].notnull(), ['LotArea', 'LotFrontage']].apply(np.log), ax=ax[2])
ax[2].set_title('Scatterplot')
ax[2].set_xlabel('log-LotFrontage')
ax[2].set_ylabel('log-LotArea')

fig.tight_layout()
from sklearn.linear_model import LinearRegression

tmp = pd.get_dummies(X[['LotFrontage', 'LotArea', 'LotConfig', 'Neighborhood']])
tmp.loc[:,['LotFrontage', 'LotArea']] = np.log(tmp.loc[:,['LotFrontage', 'LotArea']])
tmp2 = LinearRegression()
tmp2.fit(tmp.dropna().drop('LotFrontage', axis=1), tmp.dropna()['LotFrontage'])
print('R2 of the regression: {:.2f}'.format(
    tmp2.score(tmp.dropna().drop('LotFrontage', axis=1), tmp.dropna()['LotFrontage']))
     )

tmp3 = tmp2.predict(tmp.loc[tmp.isnull().any(axis=1),:].drop('LotFrontage', axis=1))

X.loc[tmp.isnull().any(axis=1), 'LotFrontage'] = np.exp(tmp3)

print('LotFrontage missing values filled.')
print('Number of remaining missing values: {}'.format(X.isnull().sum().sum()))
nominal = ['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 
           'LandContour', 'LotConfig', 'LandSlope', 'Neighborhood',
           'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 
           'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 
           'MasVnrType', 'Foundation', 'Heating', 'GarageType', 
           'PavedDrive', 'Fence', 'MiscFeature', 'SaleType',
           'SaleCondition']
print('Nominal variables allocated.')

ordinal = ['Utilities', 'OverallQual', 'OverallCond', 'ExterQual',
           'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
           'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'CentralAir',
           'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu',
           'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC']
print('Ordinal variables allocated.')

card_cont = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1',
             'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF',
             '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'GarageArea',
             'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
             'ScreenPorch', 'PoolArea', 'MiscVal']
print('Continuous cardinal variables allocated.')

card_disc = ['BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 
             'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 
             'GarageCars']
print('Discrete cardinal variables allocated.')

time = ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'MoSold', 'YrSold']
print('Time variables allocated.')

tmp = [var for l in [card_cont, card_disc, ordinal, nominal, time] for var in l]
if (pd.Series(tmp).sort_values() 
    == pd.Series(X.columns).sort_values().values).all():
    print('All variables allocated.')
X['YrBlt_to_sold'] = X['YrSold'] - X['YearBuilt']
X['YrRemod_to_sold'] = X['YrSold'] - X['YearRemodAdd']
X['GrgYrBlt_to_sold'] = X['YrSold'] - X['GarageYrBlt']
time.append('YrBlt_to_sold')
time.append('YrRemod_to_sold')
time.append('GrgYrBlt_to_sold')

X['TotalSqFt'] = X['GrLivArea'] + X['TotalBsmtSF']
card_cont.append('TotalSqFt')
X_transformed_lin = []
X_nom_dummied = pd.get_dummies(X[nominal].astype(object), drop_first=True)
X_transformed_lin.append(X_nom_dummied)

X_transformed_nonlin = []
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X_nom_labeled = X[nominal].astype(object).apply(le.fit_transform)
X_transformed_nonlin.append(X_nom_labeled)

print('Nominal variables transformed.')
X_ord_label = X[ordinal].copy()

X_ord_label['Utilities'].replace({'AllPub': 3, 'NoSeWr': 2, 'NoSeWa': 1, 'ELO': 0}, inplace=True)

X_ord_label['ExterQual'].replace({'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'Po': 0}, inplace=True)
X_ord_label['ExterCond'].replace({'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'Po': 0}, inplace=True)

X_ord_label['BsmtExposure'].replace({'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, 'None': 0}, inplace=True)
X_ord_label['BsmtFinType1'].replace({'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1, 'None': 0}, inplace=True)
X_ord_label['BsmtFinType2'].replace({'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1, 'None': 0}, inplace=True)

X_ord_label['HeatingQC'].replace({'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'Po': 0}, inplace=True)

X_ord_label['CentralAir'].replace({'N': 0, 'Y': 1}, inplace=True)

X_ord_label['Electrical'].replace({'SBrkr': 4, 'FuseA': 3, 'FuseF': 2, 'FuseP': 1, 'Mix': 0}, inplace=True)

X_ord_label['KitchenQual'].replace({'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'Po': 0}, inplace=True)

X_ord_label['Functional'].replace({'Typ': 7, 'Min1': 6, 'Min2': 5, 'Mod': 4, 'Maj1': 3, 'Maj2': 2, 'Sev': 1, 'Sal': 0}, inplace=True)

X_ord_label['FireplaceQu'].replace({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0}, inplace=True)

X_ord_label['GarageFinish'].replace({'Fin': 3, 'RFn': 2, 'Unf': 1, 'None': 0}, inplace=True)

X_ord_label['GarageQual'].replace({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0}, inplace=True)
X_ord_label['GarageCond'].replace({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0}, inplace=True)

X_ord_label['BsmtQual'].replace({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0}, inplace=True)
X_ord_label['BsmtCond'].replace({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0}, inplace=True)

X_ord_label['PoolQC'].replace({'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'None': 0}, inplace=True)

X_ord_label[['OverallQual', 'OverallCond']] -= 5

X_transformed_lin.append(X_ord_label)
X_transformed_nonlin.append(X_ord_label)

X_ord_label_sq = np.sign(X_ord_label) * (X_ord_label)
X_ord_label_sq.columns = X_ord_label.columns + '_sq'

X_transformed_lin.append(X_ord_label_sq)

X_ord_label_cu = X_ord_label ** 3
X_ord_label_cu.columns = X_ord_label.columns + '_cu'

X_transformed_lin.append(X_ord_label_cu)

X_transformed_lin.append(pd.get_dummies(X_ord_label, drop_first=True))

print('Ordinal variables transformed.')
(X[card_cont] == 0).mean().sort_values(ascending=False).plot.bar()
plt.title('Percentage zeros')
plt.show()
print('Variables with less than 100 non-zero observations in training sample:\n')
print((X.loc[train_ID, card_cont] > 0).sum()[(X.loc[train_ID, card_cont] > 0).sum() < 100].sort_values())
X_card_dummied = (X[['PoolArea', '3SsnPorch', 'LowQualFinSF', 'MiscVal']] > 0).rename(
    columns={'PoolArea': 'Pool_dum', '3SsnPorch': 'SsnPorch_dum', 'LowQualFinSF': 'LowQualFin_dum', 'MiscVal': 'MiscVal_dum'}).astype(int)

X_transformed_lin.append(X_card_dummied)
fig, ax = plt.subplots(nrows=4, ncols=5, figsize=(15,10))

for i, var in enumerate(card_cont):
    active_ax = ax.flatten()[i]
    sns.distplot(X.loc[X[var] > 0, var], ax=active_ax)

fig.tight_layout()
fig, ax = plt.subplots(nrows=4, ncols=5, figsize=(15,10))

for i, var in enumerate(card_cont):
    active_ax = ax.flatten()[i]
    sns.distplot(X.loc[X[var] > 0, var].apply(np.log), ax=active_ax)

fig.tight_layout()
print('Smallest value given non-zero across continuous conditional variables: {}'.format(
X[card_cont].where(X[card_cont] > 0).min().min()))
X_card_log = X[card_cont].applymap(lambda x: np.log(x) if x > 0 else 0)

X_transformed_lin.append(X_card_log)
X_transformed_nonlin.append(X_card_log)

X_card_log_sq = X_card_log ** 2
X_card_log_sq.columns = X_card_log.columns + '_sq'

X_transformed_lin.append(X_card_log_sq)

X_card_log_cu = X_card_log ** 3
X_card_log_cu.columns = X_card_log.columns + '_cu'

X_transformed_lin.append(X_card_log_cu)

print('Continuous cardinal variables transformed.')
X_card_disc_transformed = X[card_disc]

X_transformed_lin.append(X_card_disc_transformed)
X_transformed_nonlin.append(X_card_disc_transformed)

X_card_disc_transformed_sq = X_card_disc_transformed**2
X_card_disc_transformed_sq.columns = X_card_disc_transformed.columns + '_sq'

X_transformed_lin.append(X_card_disc_transformed_sq)

X_card_disc_transformed_cu = X_card_disc_transformed**3
X_card_disc_transformed_cu.columns = X_card_disc_transformed.columns + '_cu'

X_transformed_lin.append(X_card_disc_transformed_cu)

print('Discrete cardinal variables transformed.')
import calendar
tmp = X['MoSold'].value_counts().sort_index().reset_index()
tmp['index'] = tmp['index'].map(lambda x: calendar.month_name[x])
tmp.columns = ['MoSold', 'Number of Sales']
sns.barplot(x='MoSold', y='Number of Sales', data=tmp)
plt.xticks(rotation=90)
plt.title('Barchart for MoSold')
plt.show()
def le_mosold(x):
    if x >=7:
        x = 12-x
    else:
        x -= 1
    return x

X_time_transformed = pd.concat([X[['YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'YrSold']]-2010, 
                                X[['YrBlt_to_sold', 'YrRemod_to_sold', 'GrgYrBlt_to_sold']], 
                                X['MoSold'].apply(le_mosold)], axis=1)

X_transformed_lin.append(X_time_transformed)
X_transformed_nonlin.append(X_time_transformed)                                

X_time_transformed_sq = X_time_transformed**2
X_time_transformed_sq.columns = X_time_transformed.columns + '_sq'

X_transformed_lin.append(X_time_transformed_sq)

X_time_transformed_cu = X_time_transformed**3
X_time_transformed_cu.columns = X_time_transformed.columns + '_cu'

X_transformed_lin.append(X_time_transformed_cu)

print('Time variables transformed.')
X_transformed_lin = pd.concat(X_transformed_lin, axis=1)
X_transformed_nonlin = pd.concat(X_transformed_nonlin, axis=1)
X_transformed_lin = X_transformed_lin.T.drop_duplicates().T
X_transformed_nonlin = X_transformed_nonlin.T.drop_duplicates().T
print('Duplicates dropped.')
def check_full_col_rank(X, silent=False):
    rank_def = X.shape[1] - np.linalg.matrix_rank(X)
    if rank_def == 0:
        return True
    else: 
        return False
        if silent == False:
            print('Rank deficient by {}'.format(rank_def))

def find_collinearity(X, fix=True, show_comb=False):
    offset = 0
    collinear_columns = []
    
    tmp_X = X.copy()
    
    for i in range(1,tmp_X.shape[1]):
        i += offset
        if not check_full_col_rank(tmp_X.iloc[:,:i], silent=True):
            print('Collinear column: {}'.format(tmp_X.columns[i-1]))
            collinear_columns.append(tmp_X.columns[i-1])
            
            # find what causes collinearity
            tmp_coef = np.linalg.lstsq(tmp_X.iloc[:,:i-1], tmp_X.iloc[:,i-1])[0]
            tmp_coef[np.isclose(tmp_coef, 0)] = 0
            if show_comb:
                print('Linear Combination of:')
                print(pd.Series(tmp_coef, index=X.iloc[:,:i-1].columns)[tmp_coef != 0])
            
            tmp_X.drop(tmp_X.columns[i-1], inplace=True, axis=1)
            offset -=1
            
    if fix:
        print('Collinear columns dropped.')
        X.drop(collinear_columns, inplace=True, axis=1)

find_collinearity(X_transformed_lin, fix=True)
tmp = (X_transformed_lin.replace(0, np.nan).std() > 0)
X_transformed_lin.loc[:, tmp] /= X_transformed_lin.loc[:, tmp].replace(0, np.nan).std()
print('Linear regressors scaled.')
X_train_transformed_lin = X_transformed_lin.loc[train_ID, :]
X_test_transformed_lin = X_transformed_lin.loc[test_ID, :]

X_train_transformed_nonlin = X_transformed_nonlin.loc[train_ID, :]
X_test_transformed_nonlin = X_transformed_nonlin.loc[test_ID, :]
print('Data split into training and test observations.')
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.base import clone
from sklearn.linear_model import LassoCV, Lasso, RidgeCV, Ridge, ElasticNetCV, ElasticNet
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, make_scorer
print('Packages imported')

rmse_scorer = make_scorer(lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)), 
                     greater_is_better=False)

def get_rmse(y_pred, y_true):
    return np.sqrt(np.mean((y_pred - y_true)**2))

kfold = KFold(n_splits=5, random_state=1, shuffle=True)
print('5-fold cross validation strategy defined.')

def rmse_oof_cv(model, X, y):
    # This function returns RMSEs and out-of-fold predictions across the cross validation folds.
    rmses = []
    oof_preds = []
    
    model = clone(model)
    
    for train_idx, val_idx in kfold.split(X, y):
        X_train_fold, y_train_fold = X.iloc[train_idx, :], y.iloc[train_idx]
        X_val_fold, y_val_fold = X.iloc[val_idx, :], y.iloc[val_idx]
        
        model.fit(X_train_fold, y_train_fold)
        
        y_pred = model.predict(X_val_fold)
        rmses.append(get_rmse(y_pred, y_val_fold))
        oof_preds.append(pd.Series(y_pred, index=X_val_fold.index))
    
    rmses = np.array(rmses)
    
    return rmses, oof_preds

models = {}
lasso = LassoCV(normalize=False, random_state=2, cv=5)
print('Lasso model defined.')
lasso_rmses_cv, lasso_oof_fitted = rmse_oof_cv(lasso, X_train_transformed_lin, y_train)
print('RMSE across folds')
print('-----------------\n')
print(pd.Series(lasso_rmses_cv).describe()[1:])
lasso_fitted = pd.concat(lasso_oof_fitted).sort_index().rename('Fitted')
lasso_residuals = (y_train - lasso_fitted).rename('Residual')

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12,4))

sns.distplot(lasso_residuals, fit=stats.norm, ax=ax[0])
ax[0].legend(['Closest Normal'])
ax[0].text(x=-0.99, y=3.1, 
           s='Data Statistics\n--------------------\n' + 
           ' mean = {:.2f} \n std     = {:.2f}'.format(
               lasso_residuals.mean(), lasso_residuals.std()) + 
           '\n skew  = {:.2f} \n e.kurt = {:.2f}'.format(
               lasso_residuals.skew(), lasso_residuals.kurt())
          )
ax[0].set_title('Histogram of Residuals (Lasso)')

sns.regplot(x='Fitted', y='True', ci=None, fit_reg=False,
              data=pd.concat([y_train.rename('True'), lasso_fitted], axis=1),
              ax=ax[1])
ax[1].set_title('True vs Fitted (Lasso)')
ax[1].plot(np.linspace(*ax[1].get_xlim()), np.linspace(*ax[1].get_xlim()), color='black', lw=0.75, ls='--')
ax[1].legend(['45-degree line'])
ax[1].set_xlim(ax[1].get_ylim())

sns.regplot(y='Residual', x='Fitted', ci=None, fit_reg=False,
              data=pd.concat([lasso_fitted, lasso_residuals], axis=1),
              ax=ax[2])
ax[2].set_title('Residual plot (Lasso)')
ax[2].axhline(y=lasso_residuals.mean(), xmin=0, xmax=1, color='black', lw=1.5, ls='--')

q1_fitted = lasso_fitted[lasso_fitted.rank(pct=0.01).sort_values()[:14].index]
q1_resid_mean = lasso_residuals.loc[q1_fitted.index].mean()
rel_q1 = (q1_fitted.max() - ax[2].get_xlim()[0])/(ax[2].get_xlim()[1] - ax[2].get_xlim()[0])
ax[2].axhline(y=q1_resid_mean, xmin=0, xmax=rel_q1, color='royalblue', lw=1.5, ls='-')

q99_fitted = lasso_fitted[lasso_fitted.rank(pct=0.99).sort_values()[-14:].index]
q99_resid_mean = lasso_residuals.loc[q99_fitted.index].mean()
rel_q99 = (q99_fitted.min() - ax[2].get_xlim()[0])/(ax[2].get_xlim()[1] - ax[2].get_xlim()[0])
ax[2].axhline(y=q99_resid_mean, xmin=rel_q99, xmax=1, color='darkgreen', lw=1.5, ls='-')


ax[2].legend(['Overall mean', '<1st fitted quantile mean', '>99th fitted quantile mean'])
ax[2].set_ylim((-0.99, 0.59))

fig.tight_layout()
fig.show()
lasso_small_residuals = lasso_residuals.rank(pct=True) < 0.01

sns.lmplot(x='Fitted', y='True', fit_reg=True, ci=None,  hue='small', legend=False,
              data=pd.concat([y_train.rename('True'), lasso_fitted, 
                              lasso_small_residuals.rename('small')], axis=1))
plt.title('True vs Fitted (Lasso)')
plt.plot(np.linspace(*ax[1].get_xlim()), np.linspace(*ax[1].get_xlim()), color='black', lw=0.75, ls='--')
plt.legend(['$>$ 1st residuals quantile', '$\leq$ 1st residuals quantile', '45-degree line'])
plt.xlim(ax[1].get_xlim())
plt.ylim(ax[1].get_xlim())

c_x = 0.05
c_y = 0
for row in pd.concat([y_train.rename('True'), lasso_fitted], axis=1)[lasso_small_residuals].iterrows():
    if row[0] == 875:
        c_x, c_y = -0.05, 0.05
    elif row[0] == 813:
        c_x, c_y = -0.05, -0.125
    elif row[0] == 411:
        c_y = -0.1
    elif row[0] == 1433:
        c_y = 0.03
    elif row[0] in (469, 463):
        c_y = -0.05
    elif row[0] in (1454, 1325):
        c_y = 0.05
    elif row[0] == 589:
        c_y = -0.05
    plt.annotate(row[0], xy=(row[1]['Fitted'], row[1]['True']), 
                 xytext=(row[1]['Fitted'] + c_x, row[1]['True'] + c_y), 
                 arrowprops=dict(arrowstyle = '->'))
    c_x = 0.05
    c_y = 0
    
plt.show()
lasso_rmses_drop_cv = np.array([get_rmse(j.mask(lasso_small_residuals), y_train[j.index].mask(lasso_small_residuals)) for j in lasso_oof_fitted])
print('With all observations, the average RMSE across folds is {:.4f} ({:.4f})'.format(lasso_rmses_cv.mean(), lasso_rmses_cv.std(ddof=1)))
print('Dropping observations relating to the smallest 1% residuals, the average RMSE across folds drops to {:.4f} ({:.4f})'.format(lasso_rmses_drop_cv.mean(), lasso_rmses_drop_cv.std(ddof=1)))
lasso_large_residuals = lasso_residuals.rank(pct=True) >0.99
tmp = np.array([get_rmse(j.mask(lasso_large_residuals), y_train[j.index].mask(lasso_large_residuals)) for j in lasso_oof_fitted])
print('Without the largest 1% of residual observations, the average RMSE across folds is {:.4f} ({:.4f})'.format(tmp.mean(), tmp.std(ddof=1)))
def double_cv(model, X, y):
    
    oof_predictions = []
    val_fold_predictions = []

    for train_idx, val_idx in kfold.split(X, y):
        X_train_fold, y_train_fold = X.iloc[train_idx, :], y.iloc[train_idx]
        X_val_fold, y_val_fold = X.iloc[val_idx, :], y.iloc[val_idx]

        oof_preds = pd.concat(rmse_oof_cv(model, X_train_fold, y_train_fold)[1])

        oof_predictions.append(oof_preds)

        tmp = clone(model).fit(X_train_fold, y_train_fold)

        val_fold_preds = pd.Series(tmp.predict(X_val_fold), index=X_val_fold.index)

        val_fold_predictions.append(val_fold_preds) 
    
    return oof_predictions, val_fold_predictions

def offset_tails(oof, val_f):
    tmp_oof_pred = []
    tmp_val_pred = []
    
    offsets_low = []
    offsets_high = []
    
    for i in range(len(oof)):
        tmp_oof = oof[i].copy()
        
        high_q = tmp_oof.quantile(0.99)
        low_q = tmp_oof.quantile(0.01)
        
        offset_low = (y_train[tmp_oof.index] - tmp_oof)[tmp_oof < low_q].mean()
        offset_high = (y_train[tmp_oof.index] - tmp_oof)[tmp_oof > high_q].mean()
    
        offsets_low.append(offset_low)
        offsets_high.append(offset_high)
        
        tmp_oof.loc[tmp_oof > high_q] += offset_high
        tmp_oof.loc[tmp_oof < low_q] += offset_low
        
        tmp_oof_pred.append(tmp_oof)
        
        tmp_val_f = val_f[i].copy()
        tmp_val_f.loc[tmp_val_f > high_q] += offset_high
        tmp_val_f.loc[tmp_val_f < low_q] += offset_low
        tmp_val_pred.append(tmp_val_f)
    
    return tmp_oof_pred, tmp_val_pred, np.array(offsets_low), np.array(offsets_high)

def rmse_from_val_preds(val_preds, y):
    return np.array([get_rmse(y[p.index], p) for p in val_preds])
lasso_oof_predictions, lasso_val_f_predictions = double_cv(lasso, X_train_transformed_lin, y_train)

lasso_oof_offset, lasso_val_offset, lasso_offsets_low, lasso_offsets_high = offset_tails(lasso_oof_predictions, lasso_val_f_predictions)

lasso_offset_results = pd.DataFrame([lasso_rmses_cv, rmse_from_val_preds(lasso_val_offset, y_train), 
                                     lasso_offsets_low, lasso_offsets_high], 
                                    index=['RMSE', 'RMSE/Offset', 'Offset 1st', 'Offset 99th']).T

lasso_offset_results.describe().iloc[1:, :]
lasso.fit(X_train_transformed_lin, y_train)
print('The chosen alpha for the Lasso is {:.5f}.'.format(lasso.alpha_))
print('The number of retained regressors is is {} out of {}.\n'.format(np.sum(lasso.coef_ != 0), X_train_transformed_lin.shape[1]))
lasso_coefs = pd.Series(lasso.coef_, X_train_transformed_lin.columns)[lasso.coef_ != 0].sort_values()
lasso_coefs.plot.bar(figsize=(20,5))
plt.title('Non-zero coefficients in the Lasso model')
plt.ylabel('Coefficient')
plt.show()
models['lasso'] = lasso
print('Lasso added to model collection.')
class XGBRegressorCV(object):
    
    def __init__(self, random_state=0, n_jobs=-1, cv=None, scoring=None):
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.cv = cv
        self.scoring = scoring
    
    def fit(self, X, y):
        xgb = XGBRegressor(random_state=self.random_state, n_jobs=self.n_jobs, 
                           max_depth=5, min_child_weight=1, gamma=0, 
                           subsample=0.8, colsample_bytree=0.8)

        # n_estimators
        grid = GridSearchCV(xgb, param_grid={'n_estimators': np.arange(25, 500, 25)},
                           cv=self.cv, scoring=self.scoring)

        xgb = grid.fit(X, y).best_estimator_

        # max_depth and min_child_weight
        grid = GridSearchCV(xgb, param_grid={
            'max_depth': np.arange(1, 10, 1), 'min_child_weight': np.arange(1,10, 1)
        }, cv=self.cv, scoring=self.scoring)

        xgb = grid.fit(X, y).best_estimator_

        # gamma
        grid = GridSearchCV(xgb, param_grid={'gamma': np.arange(0,0.5, 0.1)}, 
                           cv=self.cv, scoring=self.scoring)

        xgb = grid.fit(X, y).best_estimator_

        ## n_estimators again
        #grid = GridSearchCV(xgb, param_grid={
        #    'n_estimators': np.arange(xgb.n_estimators-150, xgb.n_estimators+150, 10)},
        #                   cv=self.cv, scoring=self.scoring)
        #
        #xgb = grid.fit(X, y).best_estimator_

        # subsample and colsample_bytree
        grid = GridSearchCV(xgb, param_grid={
            'subsample': np.arange(0.1, 1, 0.1), 'colsample_bytree': np.arange(0.1, 1, 0.1)
        }, cv=self.cv, scoring=self.scoring)

        xgb = grid.fit(X, y).best_estimator_

        # reg_alpha and reg_lambda
        grid = GridSearchCV(xgb, param_grid={
            'reg_alpha': [0, 0.001, 0.005, 0.01, 0.05], 'reg_lambda': [1e-5, 1e-2, 0.1, 1, 100]
        }, cv=self.cv, scoring=self.scoring)

        xgb = grid.fit(X, y).best_estimator_
        
        # increase n_estimators and lower learning rate
        grid = GridSearchCV(xgb, param_grid={
            'learning_rate': [xgb.learning_rate/2], 'n_estimators': np.arange(2*xgb.n_estimators-150, 2*xgb.n_estimators+150, 25)
        }, cv=self.cv, scoring=self.scoring)

        final_grid = grid.fit(X, y)
        
        self.best_estimator_ = final_grid.best_estimator_
        
        return final_grid
    
    def predict(self, X):
        return self.best_estimator_.predict(X)
    
print('XGBoost tuning class defined.')
# Since this optimization takes quite a while, we skip this and simply
# use the hyperparameters we found by running this ourselves.
# Thus we wo not run these four lines:
#
#xgb_cv = XGBRegressorCV(random_state=3, cv=kfold, scoring=rmse_scorer)
#xgb_cv.fit(X_train_transformed_nonlin, y_train)
#print('XGBoost hyperparameters optimized.\n')
#xgb = xgb_cv.best_estimator_
#
# but rather just this:
xgb = XGBRegressor(random_state=3, n_estimators=1075, learning_rate=0.05,
                   gamma=0.0, max_depth=3, min_child_weight=2, reg_alpha=0,
                   reg_lambda=1, subsample=0.8, colsample_bytree=0.8)
xgb.fit(X_train_transformed_nonlin, y_train)


tmp = xgb.get_xgb_params()

print('Hyperparameters')
print('---------------')
for para in ('n_estimators', 'learning_rate', 'gamma', 'max_depth', 'min_child_weight',
             'reg_alpha', 'reg_lambda', 'subsample', 'colsample_bytree'):
    print('{}: {}'.format(para, tmp[para]))
xgb_rmses_cv, xgb_oof_fitted = rmse_oof_cv(xgb, X_train_transformed_nonlin, y_train)
print('RMSE across folds')
print('-----------------\n')
print(pd.Series(xgb_rmses_cv).describe()[1:])
xgb_fitted = pd.concat(xgb_oof_fitted).sort_index().rename('Fitted')
xgb_residuals = (y_train - xgb_fitted).rename('Residual')

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12,4))

sns.distplot(xgb_residuals, fit=stats.norm, ax=ax[0])
ax[0].legend(['Closest Normal'])
ax[0].text(x=-0.9, y=3.1, 
           s='Data Statistics\n--------------------\n' + 
           ' mean = {:.2f} \n std     = {:.2f}'.format(
               xgb_residuals.mean(), xgb_residuals.std()) + 
           '\n skew  = {:.2f} \n e.kurt = {:.2f}'.format(
               xgb_residuals.skew(), xgb_residuals.kurt())
          )
ax[0].set_title('Histogram of Residuals (XGBoost)')

sns.regplot(x='Fitted', y='True', ci=None, fit_reg=False,
              data=pd.concat([y_train.rename('True'), xgb_fitted], axis=1),
              ax=ax[1])
ax[1].set_title('True vs Fitted (XGBoost)')
ax[1].plot(np.linspace(*ax[1].get_xlim()), np.linspace(*ax[1].get_xlim()), color='black', lw=0.75, ls='--')
ax[1].legend(['45-degree line'])
ax[1].set_xlim(ax[1].get_ylim())

sns.regplot(y='Residual', x='Fitted', ci=None, fit_reg=False,
              data=pd.concat([xgb_fitted, xgb_residuals], axis=1),
              ax=ax[2])
ax[2].set_title('Residual plot (XGBoost)')
ax[2].axhline(y=xgb_residuals.mean(), xmin=0, xmax=1, color='black', lw=1.5, ls='--')

q1_fitted = xgb_fitted[xgb_fitted.rank(pct=0.01).sort_values()[:14].index]
q1_resid_mean = xgb_residuals.loc[q1_fitted.index].mean()
rel_q1 = (q1_fitted.max() - ax[2].get_xlim()[0])/(ax[2].get_xlim()[1] - ax[2].get_xlim()[0])
ax[2].axhline(y=q1_resid_mean, xmin=0, xmax=rel_q1, color='royalblue', lw=1.5, ls='-')

q99_fitted = xgb_fitted[xgb_fitted.rank(pct=0.99).sort_values()[-14:].index]
q99_resid_mean = xgb_residuals.loc[q99_fitted.index].mean()
rel_q99 = (q99_fitted.min() - ax[2].get_xlim()[0])/(ax[2].get_xlim()[1] - ax[2].get_xlim()[0])
ax[2].axhline(y=q99_resid_mean, xmin=rel_q99, xmax=1, color='darkgreen', lw=1.5, ls='-')


ax[2].legend(['Overall mean', '<1st fitted quantile mean', '>99th fitted quantile mean'])
ax[2].set_ylim((-0.84, 0.63))

fig.tight_layout()
fig.show()
xgb_small_residuals = xgb_residuals.rank(pct=True) < 0.01

tmp = pd.concat([lasso_small_residuals, xgb_small_residuals], axis=1).all(axis=1).sum()
print('Number of overlapping outliers between Lasso and XGBoost: {}'.format(tmp))
print('------------------------------------------------------------\n')


sns.lmplot(x='Fitted', y='True', fit_reg=True, ci=None,  hue='small', legend=False,
              data=pd.concat([y_train.rename('True'), xgb_fitted, 
                              xgb_small_residuals.rename('small')], axis=1))
plt.title('True vs Fitted (XGBoost)')
plt.plot(np.linspace(*ax[1].get_xlim()), np.linspace(*ax[1].get_xlim()), color='black', lw=0.75, ls='--')
plt.legend(['$>$ 1st residuals quantile', '$\leq$ 1st residuals quantile', '45-degree line'])
plt.xlim(ax[1].get_xlim())
plt.ylim(ax[1].get_xlim())

c_x = 0.05
c_y = 0
for row in pd.concat([y_train.rename('True'), xgb_fitted], axis=1)[xgb_small_residuals].iterrows():
    if row[0] == 534:
        c_x, c_y = -0.05, 0.05
    elif row[0] == 589:
        c_y = -0.1
    elif row[0] in (411, 633):
        c_y = -0.1
    elif row[0] == 1063:
        c_y = 0.07
    elif row[0] in (969, 463):
        c_y = -0.05
    elif row[0] in (1454, 1325):
        c_y = 0.02
    elif row[0] == 496:
        c_x, c_y = -0.05, -0.125
    plt.annotate(row[0], xy=(row[1]['Fitted'], row[1]['True']), 
                 xytext=(row[1]['Fitted'] + c_x, row[1]['True'] + c_y), 
                 arrowprops=dict(arrowstyle = '->'))
    c_x = 0.05
    c_y = 0
    
plt.show()
xgb_oof_predictions, xgb_val_f_predictions = double_cv(xgb, X_train_transformed_nonlin, y_train)

xgb_oof_offset, xgb_val_offset, xgb_offsets_low, xgb_offsets_high = offset_tails(xgb_oof_predictions, xgb_val_f_predictions)

xgb_offset_results = pd.DataFrame([xgb_rmses_cv, rmse_from_val_preds(xgb_val_offset, y_train), 
                                     xgb_offsets_low, xgb_offsets_high], 
                                    index=['RMSE', 'RMSE/Offset', 'Offset 1st', 'Offset 99th']).T

xgb_offset_results.describe().iloc[1:, :]
pd.Series(xgb.feature_importances_, index=X_train_transformed_nonlin.columns).sort_values().plot.bar(figsize=(20,5))
plt.title('Feature importance in XGBoost')
plt.show()
models['xgb'] = xgb
print('XGBoost added to model collection.')
model_residuals = pd.concat([lasso_residuals.rename('Lasso'), xgb_residuals.rename('XGBoost')], axis=1)
sns.jointplot(x='Lasso', y='XGBoost', data=model_residuals)
plt.subplots_adjust(top=0.925)
plt.suptitle('Lasso vs XGBoost residuals')
plt.show()
def fit_meta_model(meta_model, oof1, oof2, val1, val2):
    def get_regressors(y1, y2):
        return [pd.concat([p, y2[i]], axis=1) for i, p in enumerate(y1)]
    
    meta_rmse = []
    meta_oof_predictions = []
    meta_val_fold_predictions = []
    meta_weights = []

    for i,p in enumerate(get_regressors(oof1, oof2)):
        meta_model.fit(p, y_train[p.index])
        try:
            meta_w = pd.Series(meta_model.coef_, index=p.columns)
        except AttributeError:
            meta_w = pd.Series(meta_model.feature_importances_, index=p.columns)
        meta_weights.append(meta_w)

        meta_oof_predictions.append(pd.Series(meta_model.predict(p), index=p.index))

        q = get_regressors(val1, val2)
        meta_pred = pd.Series(meta_model.predict(q[i]), index=q[i].index)
        try:
            meta_rmse.append(get_rmse(meta_pred, y_train[q[i].index]))
        except KeyError:
            pass
        meta_val_fold_predictions.append(pd.Series(meta_pred, index=q[i].index))

    return meta_rmse, meta_weights, meta_oof_predictions, meta_val_fold_predictions
average_val_fold_pred = [pd.concat([p, xgb_val_f_predictions[i]], axis=1).mean(axis=1) for i, p in enumerate(lasso_val_f_predictions)]
average_val_fold_pred_offset = [pd.concat([p, xgb_val_offset[i]], axis=1).mean(axis=1) for i, p in enumerate(lasso_val_offset)]
average_val_fold_pred_offset_lasso = [pd.concat([p, xgb_val_f_predictions[i]], axis=1).mean(axis=1) for i, p in enumerate(lasso_val_offset)]
average_val_fold_pred_offset_xgb = [pd.concat([p, xgb_val_offset[i]], axis=1).mean(axis=1) for i, p in enumerate(lasso_val_f_predictions)]

pd.DataFrame(
    {'Lasso': [get_rmse(p, y_train[p.index]) for p in lasso_val_f_predictions],
     'XGBoost': [get_rmse(p, y_train[p.index]) for p in xgb_val_f_predictions],
     'Lasso/Offset': [get_rmse(p, y_train[p.index]) for p in lasso_val_offset],
     'XGBoost/Offset': [get_rmse(p, y_train[p.index]) for p in xgb_val_offset],
     'Avg': [get_rmse(p, y_train[p.index]) for p in average_val_fold_pred],
     'Avg/Offset Lasso': [get_rmse(p, y_train[p.index]) for p in average_val_fold_pred_offset_lasso],
     'Avg/Offset XGBoost': [get_rmse(p, y_train[p.index]) for p in average_val_fold_pred_offset_xgb],
     'Avg/Offset': [get_rmse(p, y_train[p.index]) for p in average_val_fold_pred_offset],
    }
).describe().iloc[1:, :]
meta_ols_rmse = fit_meta_model(LinearRegression(fit_intercept=False), lasso_oof_predictions, xgb_oof_predictions, lasso_val_f_predictions, xgb_val_f_predictions)[0]
meta_ols_rmse_offset, meta_ols_weights_offset, _, meta_ols_val_fold_offset = fit_meta_model(LinearRegression(fit_intercept=False), lasso_oof_offset, xgb_oof_offset, lasso_val_offset, xgb_val_offset)
meta_ols_rmse_offset_xgb = fit_meta_model(LinearRegression(fit_intercept=False), lasso_oof_predictions, xgb_oof_offset, lasso_val_f_predictions, xgb_val_offset)[0]
meta_ols_rmse_offset_lasso= fit_meta_model(LinearRegression(fit_intercept=False), lasso_oof_offset, xgb_oof_predictions, lasso_val_offset, xgb_val_f_predictions)[0]

pd.DataFrame([meta_ols_rmse, meta_ols_rmse_offset_lasso, meta_ols_rmse_offset_xgb, meta_ols_rmse_offset],
             index=['Meta OLS', 'Meta OLS/Offset Lasso', 'Meta OLS/Offset XGBoost', 'Meta OLS/Offset']).T.describe().iloc[1:, :]
print('Weights')
print('-------\n')
pd.DataFrame(meta_ols_weights_offset).rename(columns={0: 'Lasso/Offset', 1: 'XGBoost/Offset'}).describe().iloc[1:, :]
meta_comb_rmse = []
meta_comb_val_fold_predictions = []

for i in range(len(average_val_fold_pred_offset_lasso)):
    meta_comb_pred = pd.concat([average_val_fold_pred_offset_lasso[i],
                                meta_ols_val_fold_offset[i]
                               ], axis=1).mean(axis=1)
    meta_comb_rmse.append(get_rmse(y_train[meta_comb_pred.index], meta_comb_pred))
    meta_comb_val_fold_predictions.append(meta_comb_pred)


meta_comb_rmse = np.array(meta_comb_rmse)

print('RMSE across folds')
print('-----------------\n')
print(pd.Series(meta_comb_rmse).describe()[1:])
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12,4))

sns.regplot(x='Fitted', y='True', ci=None, fit_reg=False,
              data=pd.concat([y_train.rename('True'), lasso_fitted], axis=1),
              ax=ax[0])
ax[0].set_title('True vs Fitted (Lasso)')
ax[0].plot(np.linspace(*ax[0].get_xlim()), np.linspace(*ax[0].get_xlim()), color='black', lw=0.75, ls='--')
ax[0].legend(['45-degree line'])
ax[0].set_xlim(ax[0].get_ylim())

sns.regplot(x='Fitted', y='True', ci=None, fit_reg=False,
              data=pd.concat([y_train.rename('True'), xgb_fitted], axis=1),
              ax=ax[1])
ax[1].set_title('True vs Fitted (XGBoost)')
ax[1].plot(np.linspace(*ax[1].get_xlim()), np.linspace(*ax[1].get_xlim()), color='black', lw=0.75, ls='--')
ax[1].legend(['45-degree line'])
ax[1].set_xlim(ax[1].get_ylim())

meta_comb_fitted = pd.concat(meta_comb_val_fold_predictions).rename('Fitted')

sns.regplot(x='Fitted', y='True', ci=None, fit_reg=False,
              data=pd.concat([y_train.rename('True'), meta_comb_fitted], axis=1),
              ax=ax[2])
ax[2].set_title('True vs Fitted (Combined Meta)')
ax[2].plot(np.linspace(*ax[2].get_xlim()), np.linspace(*ax[2].get_xlim()), color='black', lw=0.75, ls='--')
ax[2].legend(['45-degree line'])
ax[2].set_xlim(ax[2].get_ylim())

fig.tight_layout()
final_oof_preds = {}
final_oof_preds['lasso'] = pd.concat(rmse_oof_cv(models['lasso'], X_train_transformed_lin, y_train)[1])
final_oof_preds['xgb'] = pd.concat(rmse_oof_cv(models['xgb'], X_train_transformed_nonlin, y_train)[1])

final_oof_preds = pd.DataFrame(final_oof_preds)
print('Final out of fold predictions generated.')

test_pred = {}
test_pred['lasso'] = lasso.predict(X_test_transformed_lin)
test_pred['xgb'] = xgb.predict(X_test_transformed_nonlin)

test_pred = pd.DataFrame(test_pred, index=X_test_transformed_lin.index)
print('Test set prediction of Lasso and XGBoost generated.')

final_lasso_oof_offset, test_pred_lasso_offset, _, _ = offset_tails([final_oof_preds['lasso']], [test_pred['lasso']])
final_xgb_oof_offset, test_pred_xgb_offset, _, _ = offset_tails([final_oof_preds['xgb']], [test_pred['xgb']])

final_oof_offset = pd.concat([final_lasso_oof_offset[0], final_xgb_oof_offset[0]], axis=1)
test_pred_offset = pd.concat([test_pred_lasso_offset[0], test_pred_xgb_offset[0]], axis=1)
print('Offsets for Lasso and XGBoost predictions computed.')

test_pred_average = pd.concat([test_pred_offset['lasso'], test_pred['xgb']], axis=1).mean(axis=1)
print('Model average test set predictions computed.')

_, final_ols_w, _, test_pred_meta_ols = fit_meta_model(LinearRegression(fit_intercept=False), 
                                                       [final_oof_offset['lasso']], [final_oof_offset['xgb']], 
                                                       [test_pred_offset['lasso']], [test_pred_offset['xgb']])
print('Meta-OLS predictions computed.\n')

print('Weights of Meta-OLS')
print('-------------------')
final_ols_w = final_ols_w[0]
final_ols_w.index = ['Lasso/Offset', 'XGBoost/Offset']
print(final_ols_w)
print('\n')

test_preds = pd.concat([test_pred_average, test_pred_meta_ols[0]], axis=1).mean(axis=1)
print('\nMeta-model predictions averaged')

test_predictions = np.exp(test_preds).rename('SalePrice')
print('\nPredictions transformed to original scale.')
print('The prediction for the large home is ${:.0f}'.format(test_predictions[2550]))
test_predictions[2550] = avg_large_partial_price
print('Large home prediction adjusted to ${:.0f}.'.format(avg_large_partial_price))
test_predictions.to_csv('predictions.csv', header=True)
print('Predictions saved.')