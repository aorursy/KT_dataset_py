from shutil import copyfile
copyfile("../input/jarvis/jarvis.py", "./jarvis.py")
import jarvis as j
train = j.read("../input/house-prices-advanced-regression-techniques/train.csv")
test = j.read("../input/house-prices-advanced-regression-techniques/test.csv")
j.peek(train)
j.peek(test)
#Save the 'Id' column
train_ID = train['Id']
test_ID = test['Id']

#Now drop the  'Id' colum since it's unnecessary for the prediction process.
train.drop("Id", axis = 1, inplace = True)
test.drop("Id", axis = 1, inplace = True)

#check again the data size after dropping the 'Id' variable
print("\nThe train data size after dropping Id feature is : {} ".format(train.shape)) 
print("The test data size after dropping Id feature is : {} ".format(test.shape))
j.scatter(train, "GrLivArea", "SalePrice")
#Deleting outliers
train = train.drop(train[(train['GrLivArea']>3000) & (train['SalePrice']>100000)].index)

#Check the graphic again
j.scatter(train, "GrLivArea", "SalePrice")
j.log_transform(train, "SalePrice")
j.dist(train, "SalePrice")
ntrain = train.shape[0]
ntest = test.shape[0]
y_train = train.SalePrice.values

all_data = j.concat(train, test, "SalePrice")
j.check_missing(all_data)
j.corr(train)
all_data = all_data.drop("PoolQC", axis=1)
all_data = all_data.drop("MiscFeature", axis=1)
all_data = all_data.drop("Alley", axis=1)
all_data = all_data.drop("Fence", axis=1)
j.fillna_group_mean(all_data, "Neighborhood", "LotFrontage")
def quality_num_fill(name):
    j.label_encode(all_data, name)

for col in ('GarageType', 'GarageFinish'):
    all_data[col] = all_data[col].fillna('None')
quality_num_fill("GarageQual")
quality_num_fill("GarageCond")
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    all_data[col] = all_data[col].fillna(0)
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    all_data[col] = all_data[col].fillna(0)
for col in ('BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    all_data[col] = all_data[col].fillna('None')
quality_num_fill("BsmtQual")
quality_num_fill("BsmtCond")
all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)
all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])
all_data = all_data.drop(['Utilities'], axis=1)
all_data["Functional"] = all_data["Functional"].fillna("Typ")
all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])
quality_num_fill("KitchenQual")
all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])
all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")
all_data.drop('FireplaceQu', axis=1, inplace=True)
j.check_missing(all_data)
#MSSubClass=The building class
all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)

#Year and month sold are transformed into categorical features.
all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)
cols = ('HeatingQC', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'CentralAir', 'MSSubClass',
        'YrSold', 'MoSold')
# process columns, apply LabelEncoder to categorical features
for c in cols:
    j.label_encode(all_data, c)

# shape        
print('Shape all_data: {}'.format(all_data.shape))
# Adding total sqfootage feature 
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
skewness = j.get_skewness(all_data)
j.remove_skew_coxbox(all_data, skewness)
import pandas as pd
all_data = pd.get_dummies(all_data)
print(all_data.shape)
train = all_data[:ntrain]
test = all_data[ntrain:]
from jarvis import ENet, GBoost, KRR, lasso, model_lgb
averaged_models = j.AveragingModels(models = (ENet, GBoost, KRR, lasso))

score = j.rmsle_cv(averaged_models, train, y_train)
print(" Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
stacked_averaged_models = j.StackingAveragedModels(base_models = (ENet, GBoost, KRR),
                                                   meta_model = lasso)

score = j.rmsle_cv(stacked_averaged_models, train, y_train)
print("Stacking Averaged models score: {:.4f} ({:.4f})".format(score.mean(), score.std()))
import numpy as np
stacked_averaged_models.fit(train.values, y_train)
stacked_train_pred = stacked_averaged_models.predict(train.values)
stacked_pred = np.expm1(stacked_averaged_models.predict(test.values))
print(j.rmsle(y_train, stacked_train_pred))
lgb = model_lgb(train, y_train, n_learning_rate=0.05, n_estimators=720)
lgb_train_pred = lgb.predict(train)
lgb_pred = np.expm1(lgb.predict(test.values))
print(j.rmsle(y_train, lgb_train_pred))
ensemble = stacked_pred * 0.75 + lgb_pred * 0.25
sub = pd.DataFrame()
sub['Id'] = test_ID
sub['SalePrice'] = ensemble
sub.to_csv('submission.csv',index=False)
train