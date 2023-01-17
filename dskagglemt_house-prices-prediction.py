# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

# Here we are just importing which are important for starting with.. and will add-on once I need more when reaching towards Modeling and Prdiction.
train_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
train_data.head()
print(train_data.shape)
train_data.info()
# Lets get the % of each null values.

total = train_data.isnull().sum().sort_values(ascending=False)

percent_1 = train_data.isnull().sum()/train_data.isnull().count()*100

percent_2 = (round(percent_1, 1)).sort_values(ascending=False)

missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])

missing_data.head(25)
# Visualizing the NULL data using Seaborn HeatMap.

sns.heatmap(train_data.isnull(), yticklabels = False, cbar = False)
#Using Pearson Correlation



plt.figure(figsize=(20,10))

cor = train_data.corr()

sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)

plt.show()
#Correlation with output variable

cor_target = abs(cor["SalePrice"])

#Selecting highly correlated features

relevant_features = cor_target[cor_target>0.5]

relevant_features

# From the output of this, we can see, only the features displayed below are highly correlated with the output variable SalePrice. Other variables has less impact.
# One of the assumptions of linear regression is that the independent variables need to be uncorrelated with each other. If these variables are correlated with each other, then we need to keep only one of them and drop the rest.

print(train_data[["OverallQual","YearBuilt"]].corr())

# We do have 10 Features.. and checking the correlation between is tedius. So ignoring for now.
train_data.describe(include=['O'])
print(train_data['FireplaceQu'].describe())

print('-'*20)

print(train_data.groupby('FireplaceQu').size())

print('-'*20)

print(train_data['FireplaceQu'].isnull().sum(axis=0))
# From above its clear that the FireplaceQu is a Categorical value, and Gd is the most common value, so will fill the NaN with the same.

train_data['FireplaceQu'] = train_data['FireplaceQu'].fillna(train_data['FireplaceQu'].mode()[0])

print(train_data['FireplaceQu'].isnull().sum(axis=0))
print(train_data['LotFrontage'].describe())

#print('-'*20)

#print(train_data.groupby('LotFrontage').size())

print('-'*20)

print(train_data['LotFrontage'].isnull().sum(axis=0))
# Check Mean; Median; Mode 

print('MEAN : ', train_data['LotFrontage'].mean())

print('MEDIAN : ', train_data['LotFrontage'].median())

print('MODE : ', train_data['LotFrontage'].mode())
# Since LotFrontage is float type, and Mean/Median are nearly same.. we can use Mean value to fill NaN.

train_data['LotFrontage'] = train_data['LotFrontage'].fillna(train_data['LotFrontage'].mean())

print(train_data['LotFrontage'].isnull().sum(axis=0))
print(train_data['GarageCond'].describe())

print('-'*20)

print(train_data.groupby('GarageCond').size())

print('-'*20)

print(train_data['GarageCond'].isnull().sum(axis=0))
# From above its clear that the GarageCond is a Categorical value, and TA is the most common value, so will fill the NaN with the same.

train_data['GarageCond'] = train_data['GarageCond'].fillna(train_data['GarageCond'].mode()[0])

print(train_data['GarageCond'].isnull().sum(axis=0))

print(train_data.groupby('GarageCond').size())
print(train_data['GarageType'].describe())

print('-'*20)

print(train_data.groupby('GarageType').size())

print('-'*20)

print(train_data['GarageType'].isnull().sum(axis=0))
# From above its clear that the GarageType is a Categorical value, and Attchd is the most common value, so will fill the NaN with the same.

train_data['GarageType'] = train_data['GarageType'].fillna(train_data['GarageType'].mode()[0])

print(train_data['GarageType'].isnull().sum(axis=0))

print(train_data.groupby('GarageType').size())
print(train_data['GarageYrBlt'].describe())

print('-'*20)

print(train_data.groupby('GarageYrBlt').size())

print('-'*20)

print(train_data['GarageYrBlt'].isnull().sum(axis=0))
# GarageYrBlt : Year build on will not provide much input, instead will add new feature aith Age of Garage, and as the Age grows the value goes down (assuming : depreciate in the construction).

train_data['GarageAge'] = 2019 - train_data['GarageYrBlt']

# Lets calculate the construction age from current year.
train_data.head()
# Will drop the feature / column : GarageYrBlt.

train_data.drop('GarageYrBlt', inplace = True, axis = 1)

train_data.head()
print(train_data['GarageAge'].describe())

print('-'*20)

print(train_data.groupby('GarageAge').size())

print('-'*20)

print(train_data['GarageAge'].isnull().sum(axis=0))
# Check Mean; Median; Mode 

print('MEAN : ', train_data['GarageAge'].mean())

print('MEDIAN : ', train_data['GarageAge'].median())

print('MODE : ', train_data['GarageAge'].mode())
# Will replace the NaN by mean value, but will do round of.

train_data['GarageAge'] = train_data['GarageAge'].fillna(40)

print(train_data['GarageAge'].isnull().sum(axis=0))
# Will just check the correlation of GarageAge with SalePrice

train_data[["GarageAge","SalePrice"]].corr()

# Not bad.
sns.lineplot(x="GarageAge", y="SalePrice", data=train_data)
sns.catplot(x="GarageFinish", y="SalePrice", data=train_data);

# From visualization it seems for Furnished flats price is high.
sns.barplot(x='GarageFinish',y='SalePrice',data=train_data);
print(train_data['GarageFinish'].describe())

print('-'*20)

print(train_data.groupby('GarageFinish').size())

print('-'*20)

print(train_data['GarageFinish'].isnull().sum(axis=0))
# From above its clear that the GarageFinish is a Categorical value, and Unf is the most common value, so will fill the NaN with the same.

train_data['GarageFinish'] = train_data['GarageFinish'].fillna(train_data['GarageFinish'].mode()[0])

print(train_data['GarageFinish'].isnull().sum(axis=0))

print(train_data.groupby('GarageFinish').size())
sns.barplot(x='GarageQual',y='SalePrice',data=train_data);
sns.catplot(x="GarageQual", y="SalePrice", data=train_data);
print(train_data['GarageQual'].describe())

print('-'*20)

print(train_data.groupby('GarageQual').size())

print('-'*20)

print(train_data['GarageQual'].isnull().sum(axis=0))
# From above its clear that the GarageQual is a Categorical value, and TA is the most common value, so will fill the NaN with the same.

train_data['GarageQual'] = train_data['GarageQual'].fillna(train_data['GarageQual'].mode()[0])

print(train_data['GarageQual'].isnull().sum(axis=0))

print(train_data.groupby('GarageQual').size()) 

sns.catplot(x="BsmtExposure", y="SalePrice", data=train_data);
sns.barplot(x='BsmtExposure',y='SalePrice',data=train_data);
print(train_data['BsmtExposure'].describe())

print('-'*20)

print(train_data.groupby('BsmtExposure').size())

print('-'*20)

print(train_data['BsmtExposure'].isnull().sum(axis=0))

# From above its clear that the BsmtExposure is a Categorical value, and No is the most common value, so will fill the NaN with the same.

train_data['BsmtExposure'] = train_data['BsmtExposure'].fillna(train_data['BsmtExposure'].mode()[0])

print(train_data['BsmtExposure'].isnull().sum(axis=0))

print(train_data.groupby('BsmtExposure').size()) 

sns.catplot(x="BsmtFinType2", y="SalePrice", data=train_data);
sns.barplot(x='BsmtFinType2',y='SalePrice',data=train_data);
print(train_data['BsmtFinType2'].describe())

print('-'*20)

print(train_data.groupby('BsmtFinType2').size())

print('-'*20)

print(train_data['BsmtFinType2'].isnull().sum(axis=0))



# From above its clear that the BsmtFinType2 is a Categorical value, and Unf is the most common value, so will fill the NaN with the same.

train_data['BsmtFinType2'] = train_data['BsmtFinType2'].fillna(train_data['BsmtFinType2'].mode()[0])

print(train_data['BsmtFinType2'].isnull().sum(axis=0))

print(train_data.groupby('BsmtFinType2').size()) 
train_data['BsmtFinType1'] = train_data['BsmtFinType1'].fillna(train_data['BsmtFinType1'].mode()[0])

train_data['BsmtCond'] = train_data['BsmtCond'].fillna(train_data['BsmtCond'].mode()[0])

train_data['BsmtQual'] = train_data['BsmtQual'].fillna(train_data['BsmtQual'].mode()[0])

train_data['MasVnrType'] = train_data['MasVnrType'].fillna(train_data['MasVnrType'].mode()[0])

train_data['Electrical'] = train_data['Electrical'].fillna(train_data['Electrical'].mode()[0])
print(train_data['MasVnrArea'].describe())

print('-'*20)

print(train_data['MasVnrArea'].isnull().sum(axis=0))
# Check Mean; Median; Mode 

print('MEAN : ', train_data['MasVnrArea'].mean())

print('MEDIAN : ', train_data['MasVnrArea'].median())

print('MODE : ', train_data['MasVnrArea'].mode())
# Will replace the NaN by mean value, but will do round of.

train_data['MasVnrArea'] = train_data['MasVnrArea'].fillna(103)

print(train_data['MasVnrArea'].isnull().sum(axis=0))
# Dropping off not-used features.

train_data.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence'], axis = 1, inplace = True)
sns.heatmap(train_data.isnull(), yticklabels = False, cbar = False, cmap = 'YlGnBu')

# Cool no more NaN
test_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

test_data.head()
print(test_data.shape)
# Lets get the % of each null values.

total = test_data.isnull().sum().sort_values(ascending=False)

percent_1 = test_data.isnull().sum()/test_data.isnull().count()*100

percent_2 = (round(percent_1, 1)).sort_values(ascending=False)

missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'], sort=False)

missing_data.head(25)
sns.catplot(x="MSZoning", y="SalePrice", data=train_data);

sns.barplot(x='MSZoning',y='SalePrice',data=train_data);
test_data['LotFrontage'] = test_data['LotFrontage'].fillna(test_data['LotFrontage'].mean())

test_data['MasVnrArea'] = test_data['MasVnrArea'].fillna(test_data['MasVnrArea'].mean())



test_data['GarageAge'] = 2019 - test_data['GarageYrBlt']

test_data.drop('GarageYrBlt', inplace = True, axis = 1)

test_data['GarageAge'] = test_data['GarageAge'].fillna(test_data['GarageAge'].mean())
test_data['MSZoning'] = test_data['MSZoning'].fillna(test_data['MSZoning'].mode()[0])

test_data['FireplaceQu'] = test_data['FireplaceQu'].fillna(test_data['FireplaceQu'].mode()[0])

test_data['GarageCond'] = test_data['GarageCond'].fillna(test_data['GarageCond'].mode()[0])

test_data['GarageType'] = test_data['GarageType'].fillna(test_data['GarageType'].mode()[0])

test_data['GarageFinish'] = test_data['GarageFinish'].fillna(test_data['GarageFinish'].mode()[0])

test_data['GarageQual'] = test_data['GarageQual'].fillna(test_data['GarageQual'].mode()[0])

test_data['BsmtExposure'] = test_data['BsmtExposure'].fillna(test_data['BsmtExposure'].mode()[0])

test_data['BsmtFinType2'] = test_data['BsmtFinType2'].fillna(test_data['BsmtFinType2'].mode()[0])

test_data['BsmtFinType1'] = test_data['BsmtFinType1'].fillna(test_data['BsmtFinType1'].mode()[0])

test_data['BsmtCond'] = test_data['BsmtCond'].fillna(test_data['BsmtCond'].mode()[0])

test_data['BsmtQual'] = test_data['BsmtQual'].fillna(test_data['BsmtQual'].mode()[0])

test_data['MasVnrType'] = test_data['MasVnrType'].fillna(test_data['MasVnrType'].mode()[0])

test_data['Electrical'] = test_data['Electrical'].fillna(test_data['Electrical'].mode()[0])

test_data.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence'], axis = 1, inplace = True)
sns.heatmap(test_data.isnull(), yticklabels = False, cbar = False, cmap = 'YlGnBu')

# Cool we have just few handful of NaN.. we can live with it.

# Seems we should not have any NaN while doing prediction, as none of the model expect NaN... as i got the error "ValueError: Input contains NaN, infinity or a value too large for dtype('float64')."

# Will now handle all NaN in test_data.
total = test_data.isnull().sum().sort_values(ascending=False)

percent_1 = test_data.isnull().sum()/test_data.isnull().count()*100

percent_2 = (round(percent_1, 1)).sort_values(ascending=False)

missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'], sort=False)

missing_data.head(15)
missing_data.index 
missing_col = ['Functional','BsmtFullBath','BsmtHalfBath','Utilities','GarageArea','BsmtFinSF2','BsmtUnfSF','Exterior2nd','Exterior1st','KitchenQual','GarageCars','TotalBsmtSF','BsmtFinSF1','SaleType']

missing_col

test_data[missing_col].head()

# From here we see that Functional; Utilities; Exterior2nd; Exterior1st; KitchenQual; SaleType are Categorical

# and rest are Numeric. So will replace the values with Mode and Mean respectively.
test_data['Functional'] = test_data['Functional'].fillna(test_data['Functional'].mode()[0])

test_data['Utilities'] = test_data['Utilities'].fillna(test_data['Utilities'].mode()[0])

test_data['Exterior2nd'] = test_data['Exterior2nd'].fillna(test_data['Exterior2nd'].mode()[0])

test_data['Exterior1st'] = test_data['Exterior1st'].fillna(test_data['Exterior1st'].mode()[0])

test_data['KitchenQual'] = test_data['KitchenQual'].fillna(test_data['KitchenQual'].mode()[0])

test_data['SaleType'] = test_data['SaleType'].fillna(test_data['SaleType'].mode()[0])



test_data['BsmtFullBath'] = test_data['BsmtFullBath'].fillna(test_data['BsmtFullBath'].mean())

test_data['BsmtHalfBath'] = test_data['BsmtHalfBath'].fillna(test_data['BsmtHalfBath'].mean())

test_data['GarageArea'] = test_data['GarageArea'].fillna(test_data['GarageArea'].mean())

test_data['BsmtFinSF2'] = test_data['BsmtFinSF2'].fillna(test_data['BsmtFinSF2'].mean())

test_data['BsmtUnfSF'] = test_data['BsmtUnfSF'].fillna(test_data['BsmtUnfSF'].mean())

test_data['GarageCars'] = test_data['GarageCars'].fillna(test_data['GarageCars'].mean())

test_data['TotalBsmtSF'] = test_data['TotalBsmtSF'].fillna(test_data['TotalBsmtSF'].mean())

test_data['BsmtFinSF1'] = test_data['BsmtFinSF1'].fillna(test_data['BsmtFinSF1'].mean())
print(train_data.shape, test_data.shape)
cat_features = train_data.select_dtypes(include='object').columns

cat_features
len(cat_features)
# Lets transform these Categorical Features into Number using get_dummies function (One Hot Encoding)

final_train_data = pd.get_dummies(train_data, columns=cat_features, drop_first=True)

print(final_train_data.shape)

#final_train_data.drop(cat_features, axis = 1, inplace = True)

final_train_data.head()
# Check for Duplicate Columns

final_train_data.columns.duplicated()
print(final_train_data.columns.values)
cat_features_test = test_data.select_dtypes(include='object').columns

#print("Categorical Feature Columns from Test DataFrame", cat_features_test)

print("length of categoricl features", len(cat_features_test))



# Lets transform these Categorical Features into Number using get_dummies function (One Hot Encoding)

final_test_data = pd.get_dummies(test_data, columns=cat_features_test, drop_first=True)



print(final_test_data.shape)



#final_test_data.drop(cat_features_test, axis = 1, inplace = True)

final_test_data.head()



# Check for Duplicate Columns

final_test_data.columns.duplicated()
print(final_test_data.columns.values)
merge_data = pd.concat([train_data, test_data], axis = 0)

merge_data.shape
cat_features_merge_data = merge_data.select_dtypes(include='object').columns

print("length of categoricl features", len(cat_features_merge_data))
# Now transform these Categorical Features into Number using get_dummies function (One Hot Encoding)

final_merge_data = pd.get_dummies(merge_data, columns=cat_features_merge_data, drop_first=True)

final_merge_data.shape
# Drop duplicate columns, in-case we have. But we do not have any.

final_merge_data = final_merge_data.loc[:, ~final_merge_data.columns.duplicated()]

final_merge_data.shape
final_merge_data.head()
final_merge_data.tail()
# splitting back.

final_train_data = final_merge_data.iloc[:1460, :]

final_test_data = final_merge_data.iloc[1460:, :]

final_test_data.drop(['SalePrice'], axis = 1, inplace = True)
final_train_data.shape, final_test_data.shape
# Defining Feature and Target.

#print (final_train_data.columns)

#features = final_train_data.drop(['Id', 'SalePrice']).columns

target = final_train_data["SalePrice"]

#print ("Features", features)

print ("Target", target.head())
# split the train_data into 2 DF's aka X_train, X_test, y_train, y_test.

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(final_train_data.drop(['SalePrice'], axis = 1), target, test_size=0.2)



print (X_train.shape, y_train.shape)

print (X_test.shape, y_test.shape)
# test_data 

X_test_df  = final_test_data.copy()

print (X_test_df.shape)
# machine learning

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier
lm = LinearRegression()

lm.fit(X_train,y_train)
Y_pred_lm = lm.predict(X_test)

#print(Y_pred_lr)

acc_lm = round(lm.score(X_train, y_train) * 100, 2)

print("Accuracy (LinearRegression)", acc_lm)
# Predicting on test_data

Y_pred_test_df = lm.predict(X_test_df)

Y_pred_test_df 
# Decision Tree



decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, y_train)

Y_pred_dt = decision_tree.predict(X_test)

acc_decision_tree = round(decision_tree.score(X_train, y_train) * 100, 2)

print("Accuracy (Decision Tree)", acc_decision_tree)
# Random Forest



random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, y_train)

Y_pred_rf = random_forest.predict(X_test)

acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)

print("Accuracy (random Forest)", acc_random_forest)
modelling_score = pd.DataFrame({

    'Model': ['Linear Regression','Random Forest','Decision Tree'],

    'Score': [acc_lm, acc_random_forest, acc_decision_tree]})
modelling_score.sort_values(by='Score', ascending=False)
X_test_df.shape
# Predicting on actual test_data

Y_pred_test_df = random_forest.predict(X_test_df)

Y_pred_test_df 
X_test_df.head()
X_test_df.index
submission_example_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')

submission_example_data.shape
submission_example_data.head()
#submission = pd.DataFrame( { 'Id': X_test_df.index , 'SalePrice': Y_pred_test_df } )

submission = pd.DataFrame( { 'Id': submission_example_data['Id'] , 'SalePrice': Y_pred_test_df } )
print("Submission File Shape ",submission.shape)

submission.head()
submission.to_csv( '/kaggle/working/house_prices_submission.csv' , index = False )