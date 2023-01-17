# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn as sk
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#Read in our data set
train_original = pd.read_csv('../input/train.csv')
real_test = pd.read_csv('../input/test.csv')
sample = pd.read_csv('../input/sample_submission.csv')
#Let's take a look at it
print(train_original.head())
train_original.info()

#A lot of these fields are mainly null. Let's go ahead and remove them

train = train_original.drop(['MiscFeature', 'Fence', 'Alley'], axis = 1)
#Looking at the mainly empty fields also gives us some insight into related fields.
#PoolQC only has 7 non-null entries, if we check how many observations have non-zero pools, we see, that in fact it is just those 7.
print(sum(train.PoolArea >0))

#So, we can remove both PoolQC and PoolArea, as each is only relevant to 7 out of 1460 houses in the dataset.

train = train.drop(['PoolQC', 'PoolArea'], axis = 1)
#We see the same with the 2 fields of fireplace data: The missing entries in the Fireplace Quality field are just those that don't have any fireplaces. 
sum(train.Fireplaces >0)


#Since over half the houses in our data set *do* have fireplaces, it's not as obvious that we want to just drop all the data we have about fireplaces. 
#Let's look at what kind of entries we have in Fireplace Quality
train.FireplaceQu.unique()
#Fireplace Quality is already an object-data field. We know the null entries just represent not having a fireplace. 
#We'll just go ahead and replace the missing entries with 'None'

train.FireplaceQu =train.FireplaceQu.fillna('None')
train.FireplaceQu.describe()
#Take another look at the data we have to see what's still missing
train.info()
train.loc[pd.isnull(train.LotFrontage)]
#Instead of just using the mean of LotFrontage, first let's take a look and see if there's any reasonable relation we can detect between LotArea and LotFrontage.
#We have all the LotArea data, so maybe we can use that to make better guesses about the missing frontage.

plt.scatter(train.LotArea, train.LotFrontage)
plt.show()

#Let's zoom in to get a better sense of what's happening away from the outliers
plt.scatter(train.LotArea, train.LotFrontage)
plt.xlim(0,30000)
plt.ylim(0,200)
plt.show()

#There's clearly a general trend upwards, but it's obviously still very noisy. It's probably not worth it at this point to try anything fancier than just using the mean
#Let's look at a summary of the LotFrontage data
train.LotFrontage.describe()
#Looks like the mean is 70 and the median is 69. A pretty nice distribution. Let's go ahead and replace the missing values with the mean (removing NaNs from the computation)

train.LotFrontage= train.LotFrontage.fillna(np.nanmean(train.LotFrontage))

#Now let's do a quick check for the other missing values, to see whether the NaNs just correspond to houses without the given features,
#so we can replace them with some definite 'none' category

#Again, we're missing
#* 8 entries about MasVnr,
#* about 40 entires about basements,
#* 1 entry in Electrical, which we'll handle on its own
#* around 80 entries about garages

#Count the houses with MasVnr
np.array((sum(train.MasVnrArea >0), sum(train.TotalBsmtSF>0), sum(train.GarageArea >0)))

train.loc[pd.isnull(train.MasVnrType)]
train.MasVnrType.unique()
#It looks like the MasVnrType field already inclues a 'None' category for those houses that don't have Masonry Veneer
#and the few house that are missing entries are also the houses missing entries in the MasVnrArea field.

#Since only 591 houses actually have Masonry Veneer, let's just assume the houses that don't have anything listed also don't have it,
#and we'll go ahead and fill out the MasVnrArea column with 0 and the MasVnrType column with 'None'

train.MasVnrArea = train.MasVnrArea.fillna(0)
train.MasVnrType = train.MasVnrType.fillna('None')
#Let's take a look at the house without Electrical data
train.loc[pd.isnull(train.Electrical)]

#Seems like a reasonable house: Built in 2006, remodeled in 2007, sold in 2008 for $167500. It has central air and all public Utilities. 
#It's doubtful that it just doesn't have electricity. We can just fill in the Electrical value with whatever the most common kind of Electrical is. 
#Take a look at our Electrical field
train.Electrical.describe()

#The overwhelming majority of houses just have SBrkr - Standard Circuit Breakers & Romex, so let's go ahead and say this probably has the same. 
#In any case, if the Electrical data is so uniform it's quite doubtful that it will have an effect on the model we build. 

train.Electrical = train.Electrical.fillna('SBrkr')
#The basement fields that have NaNs are:
#BsmtQual
#BsmtCond
#BsmtExposure
#BsmtFinType1
#BsmtFinType2

#One thing we have to note is that 2 of the fields, BsmtExposure and BsmtFinType2 are missing 1 more entry than all the others 
#(and in fact, there are 1423 entries with non-zero basements, so at least one of these is missing these fields)
#For the other fields, we'll just fill the missing values in with 'None'
np.array((train.BsmtQual.unique(), train.BsmtCond.unique(), train.BsmtExposure.unique(), train.BsmtFinType1.unique(), train.BsmtFinType2.unique()))

train.BsmtQual = train.BsmtQual.fillna('None')
train.BsmtCond = train.BsmtCond.fillna('None')
train.BsmtFinType1 = train.BsmtFinType1.fillna('None')
#Let's see the house that missing BsmtFinType2, but not BsmtFinType 1
train.loc[pd.isnull(train.BsmtFinType2) & (train.TotalBsmtSF>0)]

#It seems this house actually has almost all the basement information listed (it even an entry for BsmtExposure, so there's a different house that's missing that)

#Let's take a look at BsmtFinType2 to see what makes sense to fill in here
train.BsmtFinType2.describe()
#It looks like the vast majority of houses with basements have BsmtFinType2 as 'Unfinished'
#That's a little weird, given that BsmtFinType2 is supposed to be the rating of basement *finished* area,
#but maybe it's just how they record unfinished sections of basement
#Let's check
sum((train.BsmtFinType2=='Unf') & (train.BsmtFinSF2<train.BsmtUnfSF))

#Ok, so actually most of the houses have 'Unfinished' BsmtFinType2, and and that doesn't actually count all the unfinished basement square footage.
#It seems like a weird way to do it, but that's fine.
#In this case, it probably makes the most sense to go ahead and set the missing BsmtFinType2 to 'Unfinished' as well
train.loc[332,'BsmtFinType2'] = 'Unf'

#Let's take a look at the last missing basement data point
train.loc[pd.isnull(train.BsmtExposure) & (train.TotalBsmtSF>0)]
#It look like this house has a totally unfinished basement, so let's go ahead and assume it also doesn't have basement exposure

train.loc[948, 'BsmtExposure'] = 'No'
#Now the rest of the BsmtFinType2 and BsmtExposure NaNs are just houses without basements, so we can go ahead and fill those in with the category 'None'

train.BsmtFinType2 = train.BsmtFinType2.fillna('None')
train.BsmtExposure = train.BsmtExposure.fillna('None')

#Let's take a last look at our data, and make sure the only remaining null entries are in our data on garages
train.info()

#Recall that we had 1379 houses with GarageArea greater than 0, so the missing values are just houses without garages
#So, let's just fill in all the missing data with 'None'
#or, for GarageYrBlt, which is a number, lets use '0', as something out of the acceptable range (years in the Common Era) for garages to have been built.
#Presumably, more modern garages are worth more, so not having a garage should map onto roughly the same idea as having a very very very old one. 

#We'll fill the missing entries in the Year Built field with zeroes first
train.GarageYrBlt = train.GarageYrBlt.fillna(0)

#Now the only missing entries in the table at all are those in the garage fields under consideration, so we can just fill the rest of the table
train = train.fillna('None')
#Let's just repeat the steps we performed for cleaning our training data

#Drop the features that are mainly missing
real_test = real_test.drop(['MiscFeature', 'Fence', 'Alley','PoolQC', 'PoolArea'], axis = 1)

#Replace missing LotFrontage with the mean
real_test.LotFrontage= real_test.LotFrontage.fillna(np.nanmean(real_test.LotFrontage))
#Replace missing Fireplace data with new category 'None'
real_test.FireplaceQu =real_test.FireplaceQu.fillna('None')

#Replace missing Masonry Veneer data with 0 for Area dn 'None' for type
real_test.MasVnrArea = real_test.MasVnrArea.fillna(0)
real_test.MasVnrType = real_test.MasVnrType.fillna('None')

real_test.loc[pd.isnull(real_test.BsmtCond)&pd.notnull(real_test.BsmtQual)]

#It turns out we have 3 houses with Basements, but no BsmtCond given. Let's see what the most common basement conditions are 
#(given what we know about the condition of the rest of their basements) and fill in the missing values with that condition
#Check most common basement conditions for the situations of the missing houses
print(real_test.loc[(real_test.BsmtQual == 'Gd')&(real_test.BsmtFinType1=='GLQ')].BsmtCond.describe())
print(real_test.loc[(real_test.BsmtQual == 'TA')&(real_test.BsmtFinType1=='BLQ')].BsmtCond.describe())
print(real_test.loc[(real_test.BsmtQual == 'Gd')&(real_test.BsmtFinType1=='ALQ')].BsmtCond.describe())
#In all cases, 'TA' is by far the most common Basement Condition. We'll fill in these missing values with 'TA'

real_test.loc[[580, 725, 1064],'BsmtCond'] = 'TA'
#We also have a house where no basement info is recorded
real_test.loc[pd.isnull(real_test.TotalBsmtSF)]

#Let's assume it doesn't have a basement, and set its basement square footage to 0
real_test.loc[660,'TotalBsmtSF'] =0
real_test.loc[660,'BsmtFinSF1'] =0
real_test.loc[660,'BsmtFinSF2'] =0
real_test.loc[660,'BsmtUnfSF'] =0
#For the houses without basements (i.e. TotalBsmtSF = 0), we'll set all the basement features to 'None'
real_test.loc[real_test.TotalBsmtSF==0,'BsmtQual']= real_test.loc[real_test.TotalBsmtSF==0].BsmtQual.fillna('None')
real_test.loc[real_test.TotalBsmtSF==0,'BsmtCond']= real_test.loc[real_test.TotalBsmtSF==0].BsmtCond.fillna('None')
real_test.loc[real_test.TotalBsmtSF==0,'BsmtExposure']= real_test.loc[real_test.TotalBsmtSF==0].BsmtExposure.fillna('None')
real_test.loc[real_test.TotalBsmtSF==0,'BsmtFinType1']= real_test.loc[real_test.TotalBsmtSF==0].BsmtFinType1.fillna('None')
real_test.loc[real_test.TotalBsmtSF==0,'BsmtFinType2']= real_test.loc[real_test.TotalBsmtSF==0].BsmtFinType2.fillna('None')
#Check what we're still missing
real_test.info()
#Looks like there's still 2 missing BsmtQual and 2 missing BsmtExposure. Let's take a look

real_test.loc[pd.isnull(real_test.BsmtExposure) | pd.isnull(real_test.BsmtQual)]

#The two houses missing BsmtExposure *do* have basements (albeit unfinished).
#The 'None' category is meant for house without basments, so let's set these entries to 'No' (i.e. a basement with no exposure)
real_test.BsmtExposure = real_test.BsmtExposure.fillna('No')
#For the houses missing BsmtQual, let's fill them in with the most common Basement Quality for their basement condition (BsmtCond + totally unfinished)
print(real_test.loc[(real_test.BsmtCond=='Fa')&(real_test.BsmtUnfSF==real_test.TotalBsmtSF)].BsmtQual.describe())
print(real_test.loc[(real_test.BsmtCond=='TA')&(real_test.BsmtUnfSF==real_test.TotalBsmtSF)].BsmtQual.describe())

real_test.loc[757,'BsmtQual'] = 'TA'
real_test.loc[758,'BsmtQual'] = 'Gd'
#We're also missing 2 entries for Basement bathrooms
real_test.loc[pd.isnull(real_test.BsmtHalfBath)]
#These houses have no basements, so presumably they have no basement bathroom either. We'll set the values to 0
real_test.BsmtHalfBath = real_test.BsmtHalfBath.fillna(0)
real_test.BsmtFullBath = real_test.BsmtHalfBath.fillna(0)
#Find the houses with extra GarageType data
real_test.loc[pd.notnull(real_test.GarageType)&pd.isnull(real_test.GarageCond)]

#It looks like one of them just doesn't have a garage. It must be an error in the collection that the type is Detached
#The other house has 360 sqft of garage, enough to fit 1 car, but we have no other information about the quality of the garage
# For the house without a garage, we'll just set the GarageArea to 0, and GarageType to 'None',
# and then when we sort out the whole batch of housese without garages we'll get the rest of the missing fields
real_test.loc[1116,'GarageArea'] = 0
real_test.loc[1116, 'GarageType'] ='None'

#For the other house, we'll just fill in the missing data with the most common values for those fields.
#For the year built, we'll assume the garage was built when the house was, since this is pretty common
print(real_test.GarageFinish.describe(), real_test.GarageQual.describe(), real_test.GarageCond.describe())

real_test.loc[666, 'GarageFinish'] = 'Unf'
real_test.loc[666, 'GarageQual'] = 'TA'
real_test.loc[666, 'GarageCond'] = 'TA'
real_test.loc[666, 'GarageYrBlt'] = real_test.loc[666,'YearBuilt']
#Now for the houses without garages (i.e. GarageArea = 0), we'll set all the garage features to 'None'
real_test.loc[real_test.GarageArea==0,'GarageType']= real_test.loc[real_test.GarageArea==0].GarageType.fillna('None')
real_test.loc[real_test.GarageArea==0,'GarageFinish']= real_test.loc[real_test.GarageArea==0].GarageFinish.fillna('None')
real_test.loc[real_test.GarageArea==0,'GarageYrBlt']= real_test.loc[real_test.GarageArea==0].GarageYrBlt.fillna(0)
real_test.loc[real_test.GarageArea==0,'GarageQual']= real_test.loc[real_test.GarageArea==0].GarageQual.fillna('None')
real_test.loc[real_test.GarageArea==0,'GarageCond']= real_test.loc[real_test.GarageArea==0].GarageCond.fillna('None')
#There's one house with GarageCars missing. If it has no garage area, presumably it also has no garage space for cars.
real_test.loc[real_test.GarageArea==0,'GarageCars']= real_test.loc[real_test.GarageArea==0].GarageCars.fillna(0)
real_test.info()
#We still have a few missing values here and there. We'll just impute that the missing values are the most common for their columns
real_test.MSZoning.describe(), real_test.Utilities.describe(), real_test.Exterior1st.describe(), real_test.Exterior2nd.describe(), real_test.KitchenQual.describe(),
real_test.Functional.describe(), real_test.SaleType.describe()

real_test.loc[pd.isnull(real_test.MSZoning), 'MSZoning'] = real_test.loc[:,"MSZoning"].mode()[0]
real_test.loc[pd.isnull(real_test.Utilities), 'Utilities'] = real_test.loc[:,"Utilities"].mode()[0]
real_test.loc[pd.isnull(real_test.Exterior1st), 'Exterior1st'] = real_test.loc[:,"Exterior1st"].mode()[0]
real_test.loc[pd.isnull(real_test.Exterior2nd), 'Exterior2nd'] = real_test.loc[:,"Exterior2nd"].mode()[0]
real_test.loc[pd.isnull(real_test.KitchenQual), 'KitchenQual'] = real_test.loc[:,"KitchenQual"].mode()[0]
real_test.loc[pd.isnull(real_test.Functional), 'Functional'] = real_test.loc[:,"Functional"].mode()[0]
real_test.loc[pd.isnull(real_test.SaleType), 'SaleType'] = real_test.loc[:,"SaleType"].mode()[0]
#First, let's split off the target variable from the training data
target = train.loc[:,'SalePrice']
train = train.drop(['SalePrice'], axis=1)

#Second, concatenate the training and test sets
train_objs_num = len(train)
alldata = pd.concat(objs=[train, real_test], axis=0)

#Then, get dummy variables for the concatenations
alldata = pd.get_dummies(alldata, drop_first =1)

#Then, split the train and test parts back out
train_onehot = alldata[:train_objs_num]
test_onehot = alldata[train_objs_num:]
train_onehot = train_onehot.drop('Id', axis = 1)
test_onehot = test_onehot.drop('Id', axis = 1)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_onehot, target, test_size=0.3, random_state=200)
from sklearn.linear_model import LinearRegression

linreg = LinearRegression()
linreg.fit(X_train, y_train)

y_pred = linreg.predict(X_test)
print("The training score is", linreg.score(X_train, y_train))
print("The test score is", linreg.score(X_test, y_test))
#We can actually do reasonably better if we first scalse our data (which has wildly different distributions for the different variables)
#and then use a regression method with some regularization
scaler = sk.preprocessing.MinMaxScaler()
scaled_X = scaler.fit_transform(train_onehot)
X_train, X_test, y_train, y_test = train_test_split(scaled_X, target, test_size=0.3, random_state=200)

from sklearn.linear_model import Lasso
lasso = Lasso(alpha = 20, max_iter=1000, normalize = 0)
lasso.fit(X_train, y_train)

y_lasso_pred = lasso.predict(X_test)
print("The training score is", lasso.score(X_train, y_train))
print("The test score is", lasso.score(X_test, y_test))
#Plot the coefficient of the Ridge model
plt.plot(lasso.coef_)
plt.ylabel('Size of Coefficients')
#plt.yscale('log')
plt.show()
#There are some clear outliers at the top end. Let's see which variables they correspond to.
print(np.nonzero(lasso.coef_>30000))
print(train_onehot.columns.values[(np.nonzero(lasso.coef_>30000))])
#Let's try building a model using the most important variables
important_coeffs = np.reshape(np.nonzero((lasso.coef_>5000)+(lasso.coef_<-5000) ),-1)
X_train2, X_test2, y_train2, y_test2 = train_test_split(scaled_X[:,important_coeffs], target, test_size=0.3, random_state=200)
X_train2.shape
lasso_small = Lasso(alpha = 20, max_iter = 1000)
lasso_small.fit(X_train2, y_train2)

y_rid_pred = lasso_small.predict(X_test2)
print("The training score is", lasso_small.score(X_train2, y_train2))
print("The test score is", lasso_small.score(X_test2, y_test2))
#We're doing alright so far with these lasso models, but alpha = 20 was just a stab in the dark. To choose the parameters more efficiently,
#let's use a grid search algorithm to test multiple options at once

from sklearn.model_selection import GridSearchCV

#We set up the hyperparameter grid
alpha_space = np.arange(100, 500, 25)
#normalizers = np.array((0,1))
#max_iter = np.array((1000,10000))

param_grid = {'alpha': alpha_space}

# Use a Lasso regressor
lassoreg = Lasso()

# Build the GridSearchCV 
lassoreg_cv = GridSearchCV(lassoreg, param_grid, cv=5)

# Fit it to the data. GridSearchCV does the k-fold cross validation itself, so we don't need to use our train-test split data,
#instead we can give it all our training data
lassoreg_cv.fit(scaled_X, target)

# Print the tuned parameters and score
print("Tuned Regression Parameters: {}".format(lassoreg_cv.best_params_)) 
print("Best score is {}".format(lassoreg_cv.best_score_))
#Looks like the best alpha value we found is 150, which is much bigger than 20. It seems to produce a much better fit to the data as well.
#Here we don't have a cross-validation set kept separate for getting results, but the models buuilt by the grid search are cross-validated against data
#that is withheld from each one. The best model that is eventually chosen then reports its score on the whole data set, but because it wasn't trained on 
#the whole set we shouldn't worry too much about overfitting


#Let's see what the distribution of coefficients looks like. This tells us which variables the model assigns more weight to. 
best_coeffs = lassoreg_cv.best_estimator_.coef_

#Plot the coefficient of the Ridge model
plt.plot(best_coeffs)
plt.ylabel('Size of Coefficients')
#plt.yscale('log')
plt.show()
print(sum(best_coeffs !=0))

print(train_onehot.columns.values[best_coeffs !=0])
#Let's try training a model to only focus of those coefficients

#First, split the data up
important_coeffs = np.reshape(np.nonzero(best_coeffs !=0),-1)
X_train3, X_test3, y_train3, y_test3 = train_test_split(scaled_X[:,important_coeffs], target, test_size=0.3, random_state=200)
#Train a lasso model on the reduced feature collection

lasso_smaller = Lasso(alpha = 150, max_iter = 1000)
lasso_smaller.fit(X_train3, y_train3)

y_rid_pred = lasso_smaller.predict(X_test3)
print("The training score is", lasso_smaller.score(X_train3, y_train3))
print("The test score is", lasso_smaller.score(X_test3, y_test3))
#Submit our predictions with the GridSearched model
scaled_test = scaler.fit_transform(test_onehot)
test_submission = lassoreg_cv.predict(scaled_test)
d = {'Id': real_test.Id, 'SalePrice': test_submission}
submission = pd.DataFrame(data=d)
submission.to_csv('submission2.csv', index = 0)
#Submit our predictions with the reduced-variable model
scaled_test = scaler.fit_transform(test_onehot)
test_submission = lasso_smaller.predict(scaled_test[:,important_coeffs])
d = {'Id': real_test.Id, 'SalePrice': test_submission}
submission = pd.DataFrame(data=d)
submission.to_csv('submission3.csv', index = 0)
#Let's try running the parameter search on the reduced set of features
# Setup the hyperparameter grid
alpha_space = np.arange(10, 100, 5)
#normalizers = np.array((0,1))
#max_iter = np.array((1000,10000))

param_grid = {'alpha': alpha_space}

#Set up the grid search
lassoreg = Lasso()
lassoreg_cv = GridSearchCV(lassoreg, param_grid, cv=5)

# Fit it to the data
lassoreg_cv.fit(scaled_X[:,important_coeffs], target)

# Print the tuned parameters and score
print("Tuned Regression Parameters: {}".format(lassoreg_cv.best_params_)) 
print("Best score is {}".format(lassoreg_cv.best_score_))
#Submit our predictions for the new grid search on reduced feature model
scaled_test = scaler.fit_transform(test_onehot)
test_submission = lassoreg_cv.predict(scaled_test[:,important_coeffs])
d = {'Id': real_test.Id, 'SalePrice': test_submission}
submission = pd.DataFrame(data=d)
submission.to_csv('submission4.csv', index = 0)
import xgboost
xgb = xgboost.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,
                           colsample_bytree=1, max_depth=7)
#And let's run the XGBregressor on our reduced feature set, X_train3 and y_train3

xgb.fit(X_train3, y_train3)

y_xgb_pred = xgb.predict(X_test3)
print("The training score is", xgb.score(X_train3, y_train3))
print("The test score is", xgb.score(X_test3, y_test3))



#Submit our predictions for the XGBregressor model
scaled_test = scaler.fit_transform(test_onehot)
test_submission = xgb.predict(scaled_test[:,important_coeffs])
d = {'Id': real_test.Id, 'SalePrice': test_submission}
submission = pd.DataFrame(data=d)
submission.to_csv('submission5.csv', index = 0)
#xgb = xgboost.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,
                           #colsample_bytree=1, max_depth=7)


# Setup the hyperparameter grid
estimator_range = np.arange(100,200,50)
max_depth_range = np.arange(3,5,1)
learning_rate_range = (0.3,0.5,0.8)


#'n_estimators': estimator_range, 'learning_rate': learning_rate_range, 'max_depth':max_depth_range, 
param_grid = {'n_estimators':estimator_range,'max_depth':max_depth_range, 'learning_rate' : learning_rate_range}

#Set up the grid search
xgb = xgboost.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75, colsample_bytree=1, max_depth=7)
xgb_cv = GridSearchCV(xgb, param_grid, cv=5)

# Fit it to the data
xgb_cv.fit(scaled_X[:,important_coeffs], target)

# Print the tuned parameters and score
print("Tuned Regression Parameters: {}".format(xgb_cv.best_params_)) 
print("Best score is {}".format(xgb_cv.best_score_))
scaled_test = scaler.fit_transform(test_onehot)
test_submission = xgb_cv.predict(scaled_test[:,important_coeffs])
d = {'Id': real_test.Id, 'SalePrice': test_submission}
submission = pd.DataFrame(data=d)
submission.to_csv('submission7.csv', index = 0)
#For example, let's take only those features which large (in absolute value) coefficients in our lasso model. 

important_coeffs_2 = np.reshape(np.nonzero(np.abs(best_coeffs) >1000),-1)
X_train4, X_test4, y_train4, y_test4 = train_test_split(scaled_X[:,important_coeffs_2], target, test_size=0.3, random_state=200)
#xgb = xgboost.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,
                           #colsample_bytree=1, max_depth=7)


# Setup the hyperparameter grid
estimator_range = np.arange(100,200,25)
max_depth_range = np.arange(2,5,1)
learning_rate_range = (0.2,0.3,0.5)


#'n_estimators': estimator_range, 'learning_rate': learning_rate_range, 'max_depth':max_depth_range, 
param_grid = {'n_estimators':estimator_range,'max_depth':max_depth_range, 'learning_rate' : learning_rate_range}

#Set up the grid search
xgb = xgboost.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75, colsample_bytree=1, max_depth=7)
xgb_cv = GridSearchCV(xgb, param_grid, cv=5)

# Fit it to the data
xgb_cv.fit(scaled_X[:,important_coeffs_2], target)

# Print the tuned parameters and score
print("Tuned Regression Parameters: {}".format(xgb_cv.best_params_)) 
print("Best score is {}".format(xgb_cv.best_score_))
print("The test score is", xgb_cv.score(X_test4, y_test4))
scaled_test = scaler.fit_transform(test_onehot)
test_submission = xgb_cv.predict(scaled_test[:,important_coeffs_2])
d = {'Id': real_test.Id, 'SalePrice': test_submission}
submission = pd.DataFrame(data=d)
submission.to_csv('submission8.csv', index = 0)
#Let's limit our model to those variables with coefficients larger than 2000 in absolute value.

important_coeffs_3 = np.reshape(np.nonzero(np.abs(best_coeffs) >2000),-1)
X_train5, X_test5, y_train5, y_test5 = train_test_split(scaled_X[:,important_coeffs_3], target, test_size=0.3, random_state=200)

print(X_train5.shape)
#And now let's try searching for hyperparameters that work well on this reduced set. 

# Setup the hyperparameter grid
estimator_range = np.arange(100,125,5)
max_depth_range = np.arange(2,5,1)
learning_rate_range = (0.1,0.2,0.3)


#'n_estimators': estimator_range, 'learning_rate': learning_rate_range, 'max_depth':max_depth_range, 
param_grid = {'n_estimators':estimator_range,'max_depth':max_depth_range, 'learning_rate' : learning_rate_range}

#Set up the grid search
xgb = xgboost.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75, colsample_bytree=1)
xgb_cv = GridSearchCV(xgb, param_grid, cv=5)

# Fit it to the data
xgb_cv.fit(scaled_X[:,important_coeffs_3], target)

# Print the tuned parameters and score
print("Tuned Regression Parameters: {}".format(xgb_cv.best_params_)) 
print("Best score is {}".format(xgb_cv.best_score_))
print("The test score is", xgb_cv.score(X_test5, y_test5))
scaled_test = scaler.fit_transform(test_onehot)
test_submission = xgb_cv.predict(scaled_test[:,important_coeffs_3])
d = {'Id': real_test.Id, 'SalePrice': test_submission}
submission = pd.DataFrame(data=d)
submission.to_csv('submission9.csv', index = 0)