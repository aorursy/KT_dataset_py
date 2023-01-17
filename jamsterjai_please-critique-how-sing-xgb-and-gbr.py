import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



import warnings

warnings.filterwarnings('ignore')



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

#I/O



train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")

testid = test_df['Id']

trainid = train_df['Id']



test_df = test_df.drop(['Id'], axis = 1)

train_df = train_df.drop(['Id'], axis = 1);
corrheatmap = train_df.corr()

f, ax = plt.subplots(figsize=(10, 8))

sns.heatmap(corrheatmap, vmax=.8, square=True);
#https://seaborn.pydata.org/generated/seaborn.pairplot.html 

plt.figure()

columns = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageArea',

           '1stFlrSF', 'FullBath', 'YearBuilt']

sns.pairplot(train_df[columns])

plt.show()
combined_df = train_df.drop(["SalePrice"], axis=1)

combined_df = pd.concat([combined_df, test_df])
#More than 600 Missing Values:

combined_df = combined_df.drop(['MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'PoolQC'], axis=1)



#Filling Missing Data with Mean Value

combined_df['LotFrontage'].fillna(combined_df['LotFrontage'].mean(), inplace=True)



# Filling Garage Features with "None" and 0

for x in ('GarageType', 'GarageFinish', 'GarageCond', 'GarageQual',

          'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',

          'BsmtFinType2', 'MasVnrType', 'Utilities'):

    combined_df[x] = combined_df[x].fillna('None')

    

#Filling with 0

for x in ('GarageArea', 'GarageCars', 'MasVnrArea', 'TotalBsmtSF',

          'BsmtUnfSF', 'BsmtHalfBath', 'BsmtFullBath', 'BsmtFinSF1',

          'BsmtFinSF2'):

    combined_df[x] = combined_df[x].fillna(0)

    

#Filling with Mode

combined_df['YearBuilt'] = combined_df['YearBuilt'].fillna(combined_df['YearBuilt'].mode()[0])

combined_df['GarageYrBlt'] = combined_df['GarageYrBlt'].fillna(combined_df['GarageYrBlt'].mode()[0])

combined_df['MSZoning'] = combined_df['MSZoning'].fillna(combined_df['MSZoning'].mode()[0])

combined_df['KitchenQual'] = combined_df['KitchenQual'].fillna(combined_df['KitchenQual'].mode()[0])

combined_df['Electrical'] = combined_df['Electrical'].fillna(combined_df['Electrical'].mode()[0])

combined_df['Exterior1st'] = combined_df['Exterior1st'].fillna(combined_df['Exterior1st'].mode()[0])

combined_df['Exterior2nd'] = combined_df['Exterior2nd'].fillna(combined_df['Exterior2nd'].mode()[0])

combined_df['SaleType'] = combined_df['SaleType'].fillna(combined_df['SaleType'].mode()[0])

#Misc

combined_df['Functional'] = combined_df['Functional'].fillna('Typ'); #Notes Say that NA == Typical
missing_df = combined_df.isnull().sum().sort_values(ascending=False)

print(missing_df.head(1))

#2919 Entries Total x 75 Features
combined_df['CombinedSF'] = (combined_df['TotalBsmtSF'] + combined_df['GarageArea'] + combined_df['1stFlrSF'] + combined_df['2ndFlrSF'])



combined_df.drop(['TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GarageArea'], inplace=True, axis=1)
categoricaldata = ('Street', 'BsmtCond', 'GarageCond', 'GarageQual', 

                  'BsmtQual', 'CentralAir', 'ExterQual', 'ExterCond', 'HeatingQC', 

                  'KitchenQual', 'BsmtFinType1','BsmtFinType2', 'Functional',

                  'BsmtExposure', 'GarageFinish','LandSlope', 'LotShape', 'MSZoning',

                  'LandContour', 'Utilities', 'LotConfig', 'Neighborhood', 'Condition1',

                  'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',

                  'Exterior2nd', 'MasVnrType', 'Foundation', 'Heating', 'Electrical', 'GarageType',

                  'PavedDrive','SaleType', 'SaleCondition')

from sklearn.preprocessing import LabelEncoder



lec = LabelEncoder()

for x in categoricaldata:

    lec.fit(list(combined_df[x].values))

    combined_df[x] = lec.transform(list(combined_df[x].values))
skewcheck = combined_df.dtypes[combined_df.dtypes != "object"].index

skewed = combined_df[skewcheck].skew().sort_values(ascending=False)

skew_df = pd.DataFrame(skewed)

skew_df = abs(skew_df)

skew_df.shape
skew_df = skew_df[skew_df > 0.75]

skew_df = skew_df.dropna()

skew_df.shape
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.boxcox1p.html

#Using Serigne's general code for this part: https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard



from scipy.special import boxcox1p

needs_fixing = skew_df.index

lm = 0.25

for x in needs_fixing:

    combined_df[x] = boxcox1p(combined_df[x], lm)

    combined_df[x] += 1
train_df['SalePrice'] = np.log1p(train_df['SalePrice'])



combined_df = pd.get_dummies(combined_df, columns=list(categoricaldata))

combined_df.shape
X_train = combined_df[:1460]

X_test = combined_df[1460:]

Y_train = train_df['SalePrice']



print(X_train.shape)

print(X_test.shape)

print(Y_train.shape)
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor

from sklearn.model_selection import KFold, cross_val_score, train_test_split, GridSearchCV

from sklearn import metrics

import xgboost as xgb

from sklearn.metrics import mean_squared_error
#Info from:

#http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html

#https://www.kaggle.com/wiki/RootMeanSquaredError

#https://www.kaggle.com/apapiu/regularized-linear-models



kf = KFold(n_splits=10, shuffle=True, random_state=42)

kf = kf.get_n_splits(X_train)



def rmsecv(model):

    rmse = np.sqrt(-cross_val_score(model, X_train, Y_train, scoring="neg_mean_squared_error", cv = kf))

    return (rmse)
gbr_clf = GradientBoostingRegressor(n_estimators=8000, learning_rate=0.005, subsample=0.8,

                                   random_state=42, max_features='sqrt', max_depth=5, )



xgb_clf = xgb.XGBRegressor(n_estimators=15000, colsample_bytree=0.8, gamma=0.0, 

                             learning_rate=0.005, max_depth=3, 

                             min_child_weight=1,reg_alpha=0.9, reg_lambda=0.6,

                             subsample=0.2,seed=0, silent=1)



#I got these with gridsearch on my own system and some intuition...took a while 
#gbr_score = rmsecv(gbr_clf)

#print(gbr_score.mean())

## it comes out to ~0.11626
#xgb_score = rmsecv(xgb_clf)

#print(xgb_score.mean())

# it comes out to ~0.119 but final score is 0.1265....
xgb_clf.fit(X_train,Y_train)

gbr_clf.fit(X_train,Y_train)



xx = np.expm1(xgb_clf.predict(X_test))

gg = np.expm1(gbr_clf.predict(X_test))



final = pd.read_csv("../input/sample_submission.csv")



final['SalePrice'] = (xx*0.6 + gg*0.4)



final.to_csv('housing_pred.csv', index=False)
xgb.plot_importance(xgb_clf, max_num_features=20)

plt.show()