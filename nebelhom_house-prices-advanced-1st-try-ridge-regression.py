# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.model_selection import train_test_split

from sklearn.linear_model import Ridge

from sklearn import metrics



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



def rmsle(predicted,real):

    s = 0.0

    i = 0

    for x in range(len(predicted)):

        ps = np.sqrt(predicted[x]**2)

        p = np.log(ps+1)

        r = np.log(real[x]+1)

        pr_sq = (p - r)**2

        s += pr_sq

    return (s/len(predicted))**0.5



def impute_quality(item):

    if item == 'Ex':

        return 5

    elif item == 'Gd':

        return 4

    elif item == 'TA':

        return 3

    elif item == 'Fa':

        return 2

    elif item == 'Po':

        return 1

    else:

        return 0



def impute_paved(item):

    if item == 'Y':

        return 1

    elif item == 'P':

        return 0.5

    else:

        return 0



def neighMeanPriceRatio(nghd):

    return neighs[nghd.values[0]] / maximum

    

# Any results you write to the current directory are saved as output.

train = pd.read_csv('../input/train.csv')



# **Drop unnecessary rows**

train.drop('Id', axis=1, inplace=True)

# Remove Fence, no real difference

train.drop('Fence', axis=1, inplace=True)

# GarageType and GarageYrBuilt should already be covered by GarageArea

train.drop(['GarageType', 'GarageYrBlt'], axis=1, inplace=True)

# MsnVnrArea NaN only one or two --> drop row

train = train[np.isfinite(train['MasVnrArea'])]

# MiscFeature Value is already given, can drop the row

train.drop('MiscFeature', axis=1, inplace=True)



# **Trying to remove N/As from columns**<br>

# *LotFrontage fill n/As with mean*

# Fill up lot frontage with mean number

mean = train['LotFrontage'].mean()

train['LotFrontage'].fillna(value=mean, inplace=True)



# **Convert all Rankings into Integers**

qualis = ['BldgType','Utilities','ExterQual','ExterCond','BsmtQual',

          'BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',

          'KitchenQual','Functional','FireplaceQu','GarageFinish',

          'GarageQual','GarageCond','HeatingQC','PoolQC']



for q in qualis:

    train[q] = train[q].apply(impute_quality)



# **Create Dummy Groups for each binary or more option**

# Special Type (N/A means no alley, so no drop_first)

alley = pd.get_dummies(train['Alley'], prefix='Alley')

train = pd.concat([train, alley], axis=1)

train.drop('Alley', axis=1, inplace=True)



train['PavedDrive'] = train['PavedDrive'].map(impute_paved)



dummies = ['Street', 'LotShape', 'LandContour',

           'LotConfig', 'LandSlope', 'RoofStyle', 'CentralAir',

           'MSSubClass','MSZoning','Condition1','Condition2','HouseStyle',

           'RoofMatl','Exterior1st','Exterior2nd','MasVnrType',

           'Foundation','Heating','Electrical','SaleType','SaleCondition']



for dummy in dummies:

    d = pd.get_dummies(train[dummy], drop_first=True, prefix='{}'.format(dummy))

    train = pd.concat([train, d], axis=1)

    train.drop(dummy, axis=1, inplace=True)



# **Address specific case - Neighborhood**

neighs = {}

for n in train['Neighborhood'].unique():

    # Mean SalesPrice by neighborhood

    neighs[n] = train[train['Neighborhood'] == n]['SalePrice'].mean()

maximum = max(neighs.values())



train['neighMeanPriceRatio'] = train[['Neighborhood']].apply(neighMeanPriceRatio, axis=1)

train.drop('Neighborhood', axis=1, inplace=True)



# Regression Model

# **Split train sample**

X = train.drop('SalePrice', axis=1)

y = train['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)



ridge = Ridge(alpha=1.0)

ridge.fit(X=X_train, y=y_train)

pred_ridge = ridge.predict(X_test)



print('Mean Absolute Error: ', metrics.mean_absolute_error(y_test, pred_ridge))

print('Mean Squared Error: ', metrics.mean_squared_error(y_test, pred_ridge))

print('Root Mean Squared Error: ', np.sqrt(metrics.mean_squared_error(y_test, pred_ridge)))

print('R^2: ', metrics.r2_score(y_test, pred_ridge))

print('Root Mean Squared Log Error: ', rmsle(pred_ridge, y_test.values))

print('Training set score: {:.2f}'.format(ridge.score(X_train, y_train)))

print('Test set score: {:.2f}'.format(ridge.score(X_test, y_test)))
