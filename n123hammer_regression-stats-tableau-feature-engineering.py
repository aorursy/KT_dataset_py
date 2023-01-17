import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
train = pd.read_csv('../input/train.csv')

train.head()
test = pd.read_csv('../input/test.csv')

test.head()
train.info()
from IPython.display import IFrame

IFrame('https://public.tableau.com/profile/nitin2520#!/vizhome/FindingOutliersHouseRegression/Sheet1?publish=yes', width=1000, height=925)
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)

train.reset_index(drop=True, inplace=True)

train.info()
sns.heatmap(train.isnull(),yticklabels=False, cmap = 'RdYlGn',cbar=False)
# for LotFrontage it seems good to fill null values using mean

train['LotFrontage'].fillna(train['LotFrontage'].mean(), inplace = True)

test['LotFrontage'].fillna(test['LotFrontage'].mean(), inplace = True)
# those not having MasVnrType we can take them as None Type which is also the mode

# Having None MasVnrType will also have 0 area which is also the mode so filling both of them by mode

train['MasVnrType'].fillna(train['MasVnrType'].mode()[0], inplace = True)

train['MasVnrArea'].fillna(train['MasVnrArea'].mode()[0], inplace = True)

test['MasVnrType'].fillna(test['MasVnrType'].mode()[0], inplace = True)

test['MasVnrArea'].fillna(test['MasVnrArea'].mode()[0], inplace = True)
print('Pearson: ')

train[['BsmtFinType1', 'BsmtFinSF1']].corr(method='pearson')
# Tableau visualization sheet 2  and sheet 3 needed
train['BsmtFinType1'].fillna('Unf', inplace = True)

train['BsmtFinType2'].fillna('Unf', inplace = True)

test['BsmtFinType1'].fillna('Unf', inplace = True)

test['BsmtFinType2'].fillna('Unf', inplace = True)
# sheet 4
# all those points having NA in BsmtQual lies on BsmtFullBath as 0 and in that the maximum nuber of time TA occurs so we will fill this 

# with the mode which is TA in this the same is with the test set

train['BsmtQual'].fillna('TA', inplace = True)

test['BsmtQual'].fillna('TA', inplace = True)
# Similarly the BsmtCond also depends according to BsmtFullBath on 0 so we will also fill null values with TA

train['BsmtCond'].fillna('TA', inplace = True)

test['BsmtCond'].fillna('TA', inplace = True)
# Similarly we will fill BsmtExposure with No

train['BsmtExposure'].fillna('No', inplace = True)

test['BsmtExposure'].fillna('No', inplace = True)
# all those data having fireplace 0 have FireplaceQu as null so we will fill those with new value as No

train['FireplaceQu'].fillna('No', inplace = True)

test['FireplaceQu'].fillna('No', inplace = True)
# We will drop these features as thse have maximum as null values

train.drop(['MiscFeature', 'Fence', 'PoolQC', 'Alley'], axis = 1, inplace = True)

test.drop(['MiscFeature', 'Fence', 'PoolQC', 'Alley'], axis = 1, inplace = True)
train.info()

test.info()
# As can be easily seen in graph there is only one data point which has bot garagecars and garagearea value as null

# So as there are also 76 other records in which both are 0 so will fill this record with also 0

test['GarageCars'].fillna( 0 , inplace = True)

test['GarageArea'].fillna( 0 , inplace = True)
# sheet 7

# In Train Set all records having GarageQual as NA has GarageCars as 0 so will replace all NA with new value as NO

# -----------/

# while in Test Set there are all record having GarageQual as NA has GarageCars also 0 but there is 1 record which have GarageCars as 1

# So of that special record we will fill with 'TA' while of others with new value as 'NO'

train['GarageQual'].fillna('No', inplace = True)

test['GarageQual'].fillna('No', inplace = True)

def compute_garageQual(cols):

    cars = cols[0]

    qual = cols[1]

    

    if str(qual) == 'No' :

        if cars == 1 :

            return 'TA'

        else:

            return 'No'

    else:

        return qual
# replacing No with TA for particular datapoint

test['GarageQual'] = test[['GarageCars', 'GarageQual']].apply(compute_garageQual, axis = 1)

test.info()
#  Sheet 8

# In GargageType where values are null all have GarageCars as 0 so will create a new value as No

train['GarageType'].fillna('No', inplace = True)

test['GarageType'].fillna('No', inplace = True)

# train.info()

# test.info()
# Sheet 9 and Sheet 2

# The case co GarageYrBlt is same to GarageQual 

# Similarly to that there is 1 point which has GarageCars as 1 and GarageYrBlt as null 

#  We will fill taht special value with the median as expressed in the graph

train['GarageYrBlt'].fillna( 0, inplace = True)

test['GarageYrBlt'].fillna( 0, inplace = True)

# train.info()
def compute_yrblt(cols):

    cars = cols[0]

    yrblt = cols[1]

    

    if yrblt == 0 :

        if cars == 1 :

            return 1956.5

        else:

            return 0

    else:

        return yrblt
# Replacing in test set with median for special case

test['GarageYrBlt'] = test[['GarageCars', 'GarageYrBlt']].apply(compute_yrblt, axis = 1)

# test.info()
#  Sheet 3

# Similarly is the case with GaregeCond

train['GarageCond'].fillna( 'No', inplace = True)

test['GarageCond'].fillna( 'No', inplace = True)
def compute_garagecond(cols):

    cars = cols[0]

    cond = cols[1]

    

    if str(cond) == 'No' :

        if cars == 1 :

            return 'TA'

        else:

            return 'No'

    else:

        return cond
# Replacing the special Value with 'TA'

test['GarageCond'] = test[['GarageCars', 'GarageCond']].apply(compute_garagecond, axis = 1)

# test.info()
# The Same is the case with GarageFinish foe special Value

train['GarageFinish'].fillna( 'No', inplace = True)

test['GarageFinish'].fillna( 'No', inplace = True)
def compute_garagefinish(cols):

    cars = cols[0]

    finish = cols[1]

    

    if str(finish) == 'No' :

        if cars == 1 :

            return 'RFn'

        else:

            return 'No'

    else:

        return finish
# Replacing with special value

test['GarageFinish'] = test[['GarageCars', 'GarageFinish']].apply(compute_garagefinish, axis = 1)

# train.info()

# test.info()
# There is 1 null value left in the training set of Electrical Column so dropping the row

# As dropping single row will not affect our model

train.dropna(inplace = True)

# train.info()

# test.info()
# Comparing on Different MS Sub Classes MSZoning as RL is always Maximum so we will fill its null values with Rl

test['MSZoning'].fillna( 'RL', inplace = True)

# test.info()
# Replacing the utilities with mode as its optimum for filling those

test['Utilities'].fillna( test['Utilities'].mode()[0], inplace = True)

# test.info()
# Replacing null values of Exterior1st and 2nd with mode

test['Exterior1st'].fillna( test['Exterior1st'].mode()[0], inplace = True)

test['Exterior2nd'].fillna( test['Exterior2nd'].mode()[0], inplace = True)

# test.info()
# Replacing null values of SF1, SF2, SF with zero

test['BsmtFinSF1'].fillna( 0, inplace = True)

test['BsmtFinSF2'].fillna( 0, inplace = True)

test['BsmtUnfSF'].fillna( 0, inplace = True)

test['TotalBsmtSF'] = test['BsmtFinSF1'] + test['BsmtFinSF2'] + test['BsmtUnfSF']

# test.info()
# Those whose BsmtFullBath and HalfBath are not given it would be best to fill them with 0

test['BsmtFullBath'].fillna( 0, inplace = True)

test['BsmtHalfBath'].fillna( 0, inplace = True)

# test.info()
# replacing functional with mode

test['Functional'].fillna( test['Functional'].mode()[0], inplace = True)

# test.info()
# sheet 8 of train

# As when we compare kitchenqual with kitchenabvGr always 'TA' are the most so replacing them with 'TA'

test['KitchenQual'].fillna( 'TA', inplace = True)

# test.info()
# Thwrw is 1 null in saletype and the sale condition of that is normal 

# so will replace it with 'WD'

test['SaleType'].fillna( 'WD', inplace = True)

# train.info()

# test.info()
train.drop(['Id'], axis = 1, inplace = True)

test.drop(['Id'], axis = 1, inplace = True)
#  just copyiing to a new dataframe

train2 = train.copy()

test2 = test.copy()
Y = train2['SalePrice'].values

train2.drop(['SalePrice'], axis = 1, inplace = True)

train2.info()
#Acquiring mu and sigma for normal distribution plot of 'SalePrice'

from scipy import stats

from scipy.stats import norm,skew

(mu, sigma) = norm.fit(Y)



#Plotting distribution plot of 'SalePrice' and trying to fit the normal distribution corve on that

plt.figure(figsize=(8,8))

ax = sns.distplot(train['SalePrice'] , fit=norm);

plt.ylabel('Frequency')

plt.title('SalePrice distribution')

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.show()
fig, ax = plt.subplots(figsize=(8,8))



sns.distplot(Y, kde=False,color = 'green', fit=stats.lognorm)



ax.set_title("Log Normal",fontsize=24)

plt.show()
Y
y
# Applying the log transformation now

y = np.log(Y)
plt.figure(figsize=(8,8))

ax = sns.distplot(y , fit=stats.norm);

plt.ylabel('Frequency')

plt.title('SalePrice distribution')

plt.legend(['Normal Distribution'],

            loc='best')

plt.show()
import category_encoders as ce

ohe = ce.OneHotEncoder(handle_unknown='ignore', use_cat_names=True)

train2 = ohe.fit_transform(train2)

test2 = ohe.transform(test2)

train2.info()
train_final = train2.values

test_final = test2.values
from sklearn.preprocessing import MinMaxScaler

SC = MinMaxScaler(feature_range = (0,1))

train_final = SC.fit_transform(train_final)

test_final = SC.transform(test_final)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train_final, 

                                                    y, test_size=0.2, 

                                                    random_state=42)
from sklearn.svm import SVR

regressor = SVR(kernel = 'poly', degree = 4, epsilon = 0.01, gamma = 0.5 )

regressor.fit(X_train, y_train)
y_pred = (regressor.predict(X_test))

from sklearn.metrics import mean_squared_error, r2_score

print("R2 score : %.2f" % r2_score(y_test,y_pred))

print("Mean squared error: %.2f" % mean_squared_error(y_test,y_pred))
from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree = 2)

X_poly = poly_reg.fit_transform(X_train)

X_test = poly_reg.transform(X_test)
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression(n_jobs = -1)

lin_reg.fit(X_poly,y_train)
y_pred = (lin_reg.predict(X_test))





from sklearn.metrics import mean_squared_error, r2_score

print("R2 score : %.2f" % r2_score(y_test,y_pred))

print("Mean squared error: %.2f" % mean_squared_error(y_test,y_pred))

from sklearn import ensemble

from sklearn import datasets

from sklearn.utils import shuffle

from sklearn.metrics import mean_squared_error
params = {'n_estimators': 2000, 'max_depth': 6, 'min_samples_split': 30, 'min_samples_leaf': 1, 'max_features': 50,

          'learning_rate': 0.01, 'loss': 'huber', 'subsample': 0.8 , 'validation_fraction': 0.01}

clf = ensemble.GradientBoostingRegressor(**params)



clf.fit(X_train, y_train)

#mse = mean_squared_error(y_test, clf.predict(X_test))

#print("MSE: %.4f" % mse)
# y_pred = clf.predict(X_test)

# from sklearn.metrics import mean_squared_error, r2_score

# print("R2 score : %.2f" % r2_score(y_test,y_pred))

# print("Mean squared error: %.2f" % mean_squared_error(y_test,y_pred))

y_pred = regressor.predict(test_final)

#from sklearn.metrics import confusion_matrix

#cm = confusion_matrix(y_test, y_pred)

#cm

y_pred = np.exp(y_pred)

y_pred = list(y_pred)

print(len(y_pred))





test4 = pd.read_csv('../input/test.csv')

#test4.head()

passengerid = list(test4['Id'])

dictionary = {'Id':passengerid, 'SalePrice':y_pred}

df = pd.DataFrame(dictionary)

df.head()

df.to_csv('gradient.csv',index = False)