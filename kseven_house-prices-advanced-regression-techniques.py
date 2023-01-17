import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.pipeline import make_pipeline

from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestRegressor

from sklearn.tree import DecisionTreeRegressor

from xgboost import XGBRegressor



from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.metrics import mean_absolute_error
iowa_file_path = '../input/train.csv'

home_data = pd.read_csv(iowa_file_path) # import training data



test_file_path = '../input/test.csv'

test_data = pd.read_csv(test_file_path) # import test data
len(home_data.index) # number of rows



def missing_vals(training, test):

    train_missing_vals_count = training.isnull().sum()

    print("Missing Values from Training Data:")

    print(train_missing_vals_count[train_missing_vals_count > 0]) # print features that are missing data

    print('-'*30)

    print("Missing Values from Test Data:")

    test_missing_vals_count = test.isnull().sum()

    print(test_missing_vals_count[test_missing_vals_count > 0]) # print features that are missing data



missing_vals(home_data, test_data)
corr=home_data.drop(['Id'], axis=1).corr()

plt.figure(figsize=(15, 15))

plt.title('Correlation between features')

sns.heatmap(corr, vmax=.8, linewidths=0.01,square=True,cmap='coolwarm',linecolor="white")
fig, axes = plt.subplots(2, 2, figsize=(15,10))

axes[0, 0].scatter(home_data['GrLivArea'], home_data['SalePrice'])

axes[0, 0].set_xlabel('GrLivArea', fontsize=13)

axes[0, 1].scatter(home_data['OverallQual'], home_data['SalePrice'])

axes[0, 1].set_xlabel('OverallQual', fontsize=13)

axes[1, 0].scatter(home_data['GarageCars'], home_data['SalePrice'])

axes[1, 0].set_xlabel('GarageCars', fontsize=13)

axes[1, 1].scatter(home_data['TotalBsmtSF'], home_data['SalePrice'])

axes[1, 1].set_xlabel('TotalBsmtSF', fontsize=13)

plt.show()
def plot_categorical_counts(categorical_columns, grid_x, grid_y):

    fig, axes = plt.subplots(grid_x, grid_y, figsize=(15,10))

    axes = axes.ravel()

    for idx, x in enumerate(categorical_columns):

        home_data[x].value_counts().plot(kind='bar',ax=axes[idx], title=x)

    plt.show()



plot_categorical_counts(['MSSubClass', 'MSZoning','Street', 'Alley'], 2, 2)

plot_categorical_counts(['LotShape', 'LandContour', 'Utilities', 'LotConfig'], 2, 2)

plot_categorical_counts(['Condition1', 'Condition2', 'LandSlope', 'Neighborhood'], 2, 2)

plot_categorical_counts(['BldgType', 'HouseStyle', 'OverallQual', 'OverallCond'], 2, 2)



plot_categorical_counts(['RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd'], 2, 2) 

plot_categorical_counts(['MasVnrType','ExterQual', 'ExterCond', 'Foundation'], 2, 2)

plot_categorical_counts(['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1'], 2, 2) 

plot_categorical_counts(['BsmtFinType2', 'Heating','HeatingQC', 'CentralAir'], 2, 2) 

plot_categorical_counts(['Electrical', 'KitchenQual','Functional', 'FireplaceQu'], 2, 2) 

plot_categorical_counts(['GarageType', 'GarageFinish','GarageQual','GarageCond'], 2, 2) 

plot_categorical_counts(['PavedDrive', 'PoolQC','Fence', 'MiscFeature'], 2, 2) 

plot_categorical_counts(['SaleType', 'SaleCondition'], 1, 2) 
home_dropped_na = home_data.dropna(axis=1)

print("Columns in original dataset: %d" % home_data.shape[1])

print("Columns with na's dropped: %d" % home_dropped_na.shape[1])

print("Total loss of columns (percentage): %.2f%%" % (100*(home_data.shape[1]-home_dropped_na.shape[1])/home_data.shape[1]))
home_data_imputed = home_data.copy()

# Remove outliers

home_data_imputed = home_data_imputed.drop(home_data_imputed[(home_data_imputed.GrLivArea > 4000) & (home_data_imputed.SalePrice < 300000)].index)

home_data_imputed = home_data_imputed.drop(home_data_imputed[(home_data_imputed.TotalBsmtSF > 5000)].index)

home_data_imputed = home_data_imputed.drop(home_data_imputed[(home_data_imputed.GarageCars > 3)].index)

home_data_imputed = home_data_imputed.drop(home_data_imputed[(home_data_imputed.OverallQual == 10) & (home_data_imputed.SalePrice < 250000)].index)

print("Rows removed from training data set: %d" % (home_data.shape[0]-home_data_imputed.shape[0]))

print("Percentage loss: %.2f%%" % (100*(home_data.shape[0]-home_data_imputed.shape[0])/home_data.shape[0]))
# Combine data sets together for fixing missing values / imputation

combined = pd.concat([home_data_imputed.drop(['SalePrice'], axis=1), test_data], keys=['home', 'test'])

# Fix numerical data



# If the property has a miscellaneous feature then the value should not be 0, set to null for imputation

combined.loc[((combined.MiscVal == 0) & (combined.MiscFeature.notnull())), 'MiscVal'] = None 



# Fix categorical data

# Convert MSSubClass to Categorical --> commented out because this makes it less accurate after one hot encoding

#combined['MSSubClass'] = combined['MSSubClass'].astype(str)

#combined['OverallQual'] = combined['OverallQual'].astype(str)

#combined['OverallCond'] = combined['OverallCond'].astype(str)

# Relabel basement categorical data if it is not actually missing

combined.loc[((combined.BsmtQual.isnull()) & (combined.BsmtCond.isnull()) & 

                       (combined.BsmtExposure.isnull()) & (combined.BsmtFinType1.isnull()) & 

                       (combined.BsmtFinType2.isnull()) & (combined.BsmtFinSF1 == 0) & 

                       (combined.BsmtFinSF2 == 0) & (combined.BsmtUnfSF == 0) & (combined.TotalBsmtSF == 0)), 

                      ['BsmtQual','BsmtCond','BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']] = 'NA'

# Relabel garage categorical data if it is not actually missing

combined.loc[((combined.GarageType.isnull()) & (combined.GarageYrBlt.isnull()) & 

                       (combined.GarageFinish.isnull()) & (combined.GarageQual.isnull()) & 

                       (combined.GarageCond.isnull()) & (combined.GarageCars == 0) & 

                       (combined.GarageArea == 0)), 

                      ['GarageYrBlt']] = 0

combined.loc[((combined.GarageType.isnull()) & (combined.GarageYrBlt == 0) & 

                       (combined.GarageFinish.isnull()) & (combined.GarageQual.isnull()) & 

                       (combined.GarageCond.isnull()) & (combined.GarageCars == 0) & 

                       (combined.GarageArea == 0)), 

                      ['GarageType','GarageFinish', 'GarageQual', 'GarageCond']] = 'NA'

# Relabel alley, data description says NA means no alley access so we have to assume it means that for all values

combined.Alley.fillna('NA', inplace=True)

# Relabel fireplace quality if it is not actually missing

combined.loc[((combined.Fireplaces == 0) & (combined.FireplaceQu.isnull())), 'FireplaceQu'] = 'NA'



# Relabel pool quality if it is not actually missing

combined.loc[((combined.PoolArea == 0) & (combined.PoolQC.isnull())), 'PoolQC'] = 'NA'



# Relabel fence, data description says NA means no fence so we have to assume it means that for all values

combined.Fence.fillna('NF', inplace=True)



# Relabel miscellaneous feature if it is not actually missing

combined.loc[((combined.MiscVal == 0) & (combined.MiscFeature.isnull())), 'MiscFeature'] = 'NA'



# Fix error in test data, build year of garage cannot be 2207. 2007 is a safe assumption

combined.loc[(combined.GarageYrBlt == 2207), 'GarageYrBlt'] = 2007



# Get mean lotfrontage based on neighborhood

#combined.LotFrontage = combined.groupby('Neighborhood')['LotFrontage'].apply(lambda x: x.fillna(x.median()))



# Impute missing values

# Fill missing categorical values, mode is the value occuring the most

columns = combined.select_dtypes(exclude=['int64', 'float64']).columns

for column in columns:

    combined[column] = combined[column].fillna(combined[column].mode()[0])



# Fill missing numerical values, use the mean 

columns = combined.select_dtypes(exclude=['object']).columns

for column in columns:

    combined[column] = combined[column].fillna(combined[column].mean())

# Check for missing values

missing_vals(combined.loc['home'], combined.loc['test'])
# Ordinal features, mapping to strings to numbers

utilities_map = {'Utilities': {'ELO': 1, 'NoSeWa': 2, 'NoSewr': 3, 'AllPub': 4}}

landslope_map = {'LandSlope': {'Sev': 1, 'Mod': 2, 'Gtl': 3} }

bsmtexpo_map = {'BsmtExposure': {'NA': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4}}

centralair_map = {'CentralAir': {'N': 0, 'Y': 1}}

functional_map = {'Functional': {'Sal': 1, 'Sev': 2, 'Maj2': 3, 'Maj1': 4, 'Mod': 5, 'Min2': 6, 'Min1': 7, 'Typ': 8}}

garagefinish_map = {'GarageFinish': {'NA': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3}}

bsmtfintype1_map = {'BsmtFinType1': {'NA': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}}

bsmtfintype2_map = {'BsmtFinType2': {'NA': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}}



external_qual_map = {'ExterQual': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}}

external_cond_map = {'ExterCond': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}}

bsmtqual_map = {'BsmtQual': {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}}

bsmtcond_map = {'BsmtCond': {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}}

heatingqc_map = {'HeatingQC': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}}

kitchenqual_map = {'KitchenQual': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}}

fireplacequ_map = {'FireplaceQu': {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}}

garagequal_map = {'GarageQual': {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}}

garagecond_map = {'GarageCond': {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}}

poolqc_map = {'PoolQC': {'NA': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4}}





combined.replace(utilities_map, inplace=True)

combined.replace(landslope_map, inplace=True)

combined.replace(external_qual_map, inplace=True)

combined.replace(external_cond_map, inplace=True)

combined.replace(bsmtqual_map, inplace=True)

combined.replace(bsmtcond_map, inplace=True)

combined.replace(bsmtexpo_map, inplace=True)

combined.replace(bsmtfintype1_map, inplace=True)

combined.replace(bsmtfintype2_map, inplace=True)

combined.replace(heatingqc_map, inplace=True)

combined.replace(centralair_map, inplace=True)

combined.replace(kitchenqual_map, inplace=True)

combined.replace(functional_map, inplace=True)

combined.replace(fireplacequ_map, inplace=True)

combined.replace(garagefinish_map, inplace=True)

combined.replace(garagequal_map, inplace=True)

combined.replace(garagecond_map, inplace=True)

combined.replace(poolqc_map, inplace=True)

#Add all the bathrooms together

#Condense neighbourhood?

#Total Living Area

combined['TotalBath'] = combined.BsmtFullBath + combined.FullBath + 0.5*(combined.BsmtHalfBath  + combined.HalfBath)

combined['TotalLivingArea'] = combined.TotalBsmtSF + combined.GrLivArea

combined['Remodelled'] = 1

combined.loc[((combined.YearRemodAdd != combined.YearBuilt)), 'Remodelled'] = 0

combined['IsNew'] = 0

combined.loc[((combined.YrSold == (combined.YearBuilt))), 'IsNew'] = 1

combined['Age'] = combined.YrSold - combined.YearRemodAdd





y = home_data_imputed['SalePrice']

y = np.log1p(y)

features = ['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea',

        'LotShape', 'LandContour', 'LotConfig',

       'LandSlope', 'Neighborhood', 'Condition1', 'BldgType',

       'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'Remodelled',

       'RoofStyle', 'Exterior1st', 'MasVnrType',

       'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',

       'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',

       'BsmtFinSF2', 'BsmtUnfSF', 'Heating',

       'HeatingQC', 'CentralAir', '2ndFlrSF',

       'LowQualFinSF', 'TotalLivingArea', 'TotalBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',

       'TotRmsAbvGrd', 'Fireplaces', 'FireplaceQu', 'GarageType',

       'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual',

       'GarageCond', 'PavedDrive', 'OpenPorchSF',

       'EnclosedPorch', 'ScreenPorch', 'PoolArea', 'PoolQC',

       'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'IsNew', 'Age']

X = pd.get_dummies(combined[features])

X_home = X.loc['home'].copy()



train_X, test_X, train_y, test_y = train_test_split(X_home, y, random_state=1)



xgbmodel = XGBRegressor(n_estimators=20000, learning_rate=0.01,random_state=1, n_jobs=4, min_child_weight=4)

# Add silent=True to avoid printing out updates with each cycle

xgbmodel.fit(train_X, train_y, early_stopping_rounds=1000,eval_set=[(test_X, test_y)], verbose=False)

xgb_pred = xgbmodel.predict(test_X)

xgb_pred = np.expm1(xgb_pred)

xgb_mae = mean_absolute_error(xgb_pred, np.expm1(test_y))

print(xgb_mae)
X_test = X.loc['test'].copy()

test_pred = xgbmodel.predict(X_test)

test_pred = np.exp(test_pred)

output = pd.DataFrame({'Id': test_data.Id, 'SalePrice': test_pred})

output.to_csv('submission.csv', index=False)