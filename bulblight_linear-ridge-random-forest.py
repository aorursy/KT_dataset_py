import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



train = pd.read_csv('../input/train.csv') #bring in the train dataset

test = pd.read_csv('../input/test.csv') #bring in the test dataset
train.SalePrice.describe() #what does SalePrice look like?
# we'll want to use the log of Sale Price to minimize the effects of outliers

target = np.log(train.SalePrice)
target.skew()
# I always like to evaluate what variables are correlated with the dependent variable.

# In this case let's look at the top 5 highest correlated variables.



numeric_features = train.select_dtypes(include=[np.number])

corr = numeric_features.corr()

print(corr['SalePrice'].sort_values(ascending=False)[:5])
train = train[train['GrLivArea'] <= 4000]
# In looking at the data preview there are a lot of fields with missing values.  Let's look at the top 30

nans = pd.DataFrame(train.isnull().sum().sort_values(ascending=False)[:30])

nans
#first feature creation.  Transforming categorical street type into 1 for paved and 0 for gravel

train['street_type']=pd.get_dummies(train.Street, drop_first=True)

test['street_type']=pd.get_dummies(test.Street, drop_first=True)

train.street_type.unique()
# In creating this report I did each categorical variable one by one and submitted the results to Kaggle

# This allowed me to monitor the results and benefits that each variable added to the modeling.

# The 3 fields update one categorical variable each.

train['LotShape']=pd.get_dummies(train.LotShape, drop_first=True)

test['LotShape']=pd.get_dummies(test.LotShape, drop_first=True)
train['LandContour']=pd.get_dummies(train.LandContour, drop_first=True)

test['LandContour']=pd.get_dummies(test.LandContour, drop_first=True)
train['MSZoning']=pd.get_dummies(train.MSZoning, drop_first=True)

test['MSZoning']=pd.get_dummies(test.MSZoning, drop_first=True)
#dealing with null values

data = train.select_dtypes(include=[np.number]).interpolate().dropna()
#let's confirm that we don't have any more null values:

sum(data.isnull().sum() !=0)
# finally! Let's get set up for modeling.  Don't forget that we decided to use the log of SalePrice because

# of the skewed ness in the data.

y = np.log(train.SalePrice)

X = data.drop(['SalePrice','Id'], axis=1)
#Separate our data into 66% training and 34% test

from sklearn.model_selection import train_test_split

train_X, test_X, train_y, test_y = train_test_split(X, y, 

                                                    train_size=0.66,

                                                    test_size=0.34,

                                                    random_state=123)
#super simple linear regression

from sklearn import linear_model

linearmodel=linear_model.LinearRegression() #instantiate the model

model = linearmodel.fit(train_X,train_y) #fit the model

model.score(test_X,test_y)

#so far this model actually performs the best but I'd like to rerun the ridge regression after some

# additional data enhancements (ie more catagorical variables changed to numeric)
#a little more complex but very common Ridge Regression. We'll run through alpha = 0.01, 0.1, 1, 10, 100

for i in range (-2, 3):

    alpha = 10**i

    rm = linear_model.Ridge(alpha=alpha)

    ridge_model = rm.fit(train_X, train_y)

    preds_ridge = ridge_model.predict(test_X)

    print(ridge_model.score(test_X,test_y))



# notice the results for each alpha, this helps me to pick the best alpha to use in the model.
rm = linear_model.Ridge(alpha=100)

ridge_model = rm.fit(train_X, train_y)

preds_ridge = ridge_model.predict(test_X)

print(ridge_model.score(test_X,test_y))
#setup for creating our submission file.  We need the ID field from the test data set.

submission = pd.DataFrame()

submission['Id']=test.Id

feats = test.select_dtypes(include=[np.number]).drop(['Id'],axis=1).interpolate()
# for the random forest model we'll need to import another package.

from sklearn.ensemble import RandomForestRegressor



forest_model = RandomForestRegressor()

forest_model.fit(train_X, train_y)

melb_preds = forest_model.predict(test_X)

print(forest_model.score(test_X,test_y))
#creating the submission file to be created as an output csv file on Kaggle.

predictions = ridge_model.predict(feats)

final_predictions=np.exp(predictions) #reverse log

submission['SalePrice']=final_predictions

submission.to_csv('submissionLR.csv',index=False)