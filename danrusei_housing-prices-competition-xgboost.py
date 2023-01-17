import pandas as pd
# *XGBoost** is the leading model for working with standard tabular data (the type of data you store in Pandas DataFrames)
from xgboost import XGBRegressor
# Mean absolute error regression loss
from sklearn.metrics import mean_absolute_error
# Split arrays or matrices into random train and test subsets
from sklearn.model_selection import train_test_split
# Imputation transformer for completing missing values.
from sklearn.impute import SimpleImputer
# path to the file to read
iowa_file_path = '../input/train.csv'
iowa_file_path_test ='../input/test.csv'
# read into a PD DataFrame
home_data = pd.read_csv(iowa_file_path)
home_data_test = pd.read_csv(iowa_file_path_test)

#Save the 'Id' column
train_ID = home_data['Id']
test_ID = home_data_test['Id']

#droping ID column
home_data.drop("Id", axis = 1, inplace = True)
home_data_test.drop("Id", axis = 1, inplace = True)

# keep original shape
ntrain = home_data.shape[0]
ntest = home_data_test.shape[0]

# save target
home_target = home_data.SalePrice.values

# Concat data to prepare for one_hot_encoding
all_data = pd.concat((home_data, home_data_test)).reset_index(drop=True)

#drop SalePrice column
all_data.drop(['SalePrice'], axis=1, inplace=True)
print("all_data size is : {}".format(all_data.shape))
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
missing_data.head(20)
all_data["PoolQC"] = all_data["PoolQC"].fillna("None")
all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")
all_data["Alley"] = all_data["Alley"].fillna("None")
all_data["Fence"] = all_data["Fence"].fillna("None")
all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")
#Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    all_data[col] = all_data[col].fillna('None')
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    all_data[col] = all_data[col].fillna(0)
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    all_data[col] = all_data[col].fillna(0)
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    all_data[col] = all_data[col].fillna('None')
all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)
all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])
all_data = all_data.drop(['Utilities'], axis=1)
all_data["Functional"] = all_data["Functional"].fillna("Typ")
all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])
all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])
all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])
all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")
#Check remaining missing values if any 
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
missing_data.head()
#MSSubClass=The building class
all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)


#Changing OverallCond into a categorical variable
all_data['OverallCond'] = all_data['OverallCond'].astype(str)


#Year and month sold are transformed into categorical features.
all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)
# "cardinality" means the number of unique values in a column.
# We use it as our only way to select categorical columns here. 

low_cardinality_cols = [cname for cname in all_data.columns if 
                                all_data[cname].nunique() < 10 and
                                all_data[cname].dtype == "object"]
numeric_cols = [cname for cname in all_data.columns if 
                                all_data[cname].dtype in ['int64', 'float64']]
my_cols = low_cardinality_cols + numeric_cols
all_data = all_data[my_cols]
all_data_encoded = pd.get_dummies(all_data)
print(all_data_encoded.shape)
home_data = all_data_encoded[:ntrain]
home_data_test = all_data_encoded[ntrain:]
X_train, X_test, y_train, y_test = train_test_split(home_data, home_target, train_size=0.7, test_size=0.3, random_state=0) 
# Shape after droping categorical columns with nunique >= 10
print(X_train.shape)
print(X_test.shape)
# check out the header after encoding
X_train.head()
## use Imputer to fill in missing data
#my_imputer = SimpleImputer()
#imputed_X_train = my_imputer.fit_transform(X_train)
#imputed_X_test = my_imputer.transform(X_test)
my_model = XGBRegressor(n_estimator=200)
my_model.fit(X_train, y_train, early_stopping_rounds=5, eval_set=[(X_test, y_test)],verbose=False)
predictions = my_model.predict(X_test)
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, y_test)))

# predict on test data and submit
pred_test = my_model.predict(home_data_test)
print(pred_test)
#Submission:
sub = pd.DataFrame()
sub['Id'] = test_ID
sub['SalePrice'] = pred_test
sub.to_csv('submission.csv', index=False)