import os



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
!ls '../input'

!ls '../input/ipricristik20'
#STEP 1: IMPORTING LIBRARIES AND DATASET



#import some necessary librairies

from scipy import stats

from scipy.stats import norm, skew

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import KFold, cross_val_score, train_test_split, cross_validate, GridSearchCV

import xgboost as xgb

from xgboost.sklearn import XGBClassifier, XGBRegressor

from sklearn import metrics

from sklearn.metrics import mean_absolute_error

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

%matplotlib inline

import matplotlib.pyplot as plt  # Matlab-style plotting

import seaborn as sns



color = sns.color_palette()

sns.set_style('darkgrid')

import warnings

def ignore_warn(*args, **kwargs):

    pass

warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)



pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) #Limiting floats output to 3 decimal points
# Importing dataset from kaggle



train = pd.read_csv('../input/ipricristik20/train.csv')

test = pd.read_csv('../input/ipricristik20/test.csv')
train['price'].describe()
train.head(5)
test.head(5)
print("The train data size before dropping Id feature is : {} ".format(train.shape))

print("The test data size before dropping Id feature is : {} ".format(test.shape))



# Save the 'Id' column

train_ID = train['id']

test_ID = test['id']



# Now drop the  'Id' colum since it's unnecessary for  the prediction process.

train.drop("id", axis = 1, inplace = True)

test.drop("id", axis = 1, inplace = True)



#check again the data size after dropping the 'id' variable

print("\nThe train data size after dropping Id feature is : {} ".format(train.shape)) 

print("The test data size after dropping Id feature is : {} ".format(test.shape))
sns.distplot(train['price'] , fit=norm);



#Now plot the distribution

plt.ylabel('Frequency')

plt.title('price distribution')



#Get also the QQ-plot

fig = plt.figure()

res = stats.probplot(train['price'], plot=plt)

plt.show()
#Log transformation



#We use the numpy fuction log1p which  applies log(1+x) to all elements of the column

train["price"] = np.log1p(train["price"])



#Check the new distribution 

sns.distplot(train['price'] , fit=norm);



plt.ylabel('Frequency')

plt.title('price distribution')



#Get also the QQ-plot

fig = plt.figure()

res = stats.probplot(train['price'], plot=plt)

plt.show()
#Correlation map to see how features are correlated with price

corrmat = train.corr()

plt.subplots(figsize=(12,12))

sns.heatmap(corrmat, vmax=0.9, square=True)
data = train.corr()["price"].sort_values()[::-1]

plt.figure(figsize=(12, 8))

sns.barplot(x=data.values, y=data.index)

plt.title("Correlation with price")

plt.xlim(-0.2, 1)

plt.show()
#deleting outliers points by index --> room_size

var = 'room_size'

temp = pd.concat([train[var], train['price']], axis=1)

temp.plot.scatter(x=var, y='price')

temp.sort_values(by = var, ascending = True)



train = train.drop(train[train['price'] < 12].index, axis=0)
#deleting outliers points by index --> room_size

var = 'distance_poi_A2'

temp = pd.concat([train[var], train['price']], axis=1)

temp.plot.scatter(x=var, y='price')

temp.sort_values(by = var, ascending = True)



# train = train.drop(train[(train['price'] > 14.5) & train['distance_poi_A2'] > 10000].index, axis=0)

# train = train.drop(train[(train['price'] > 14.0) & train['distance_poi_A2'] > 25000].index, axis=0)



train = train.drop(train[train['distance_poi_A2'] > 12500].index, axis=0)
#deleting outliers points by index --> room_size

var = 'distance_poi_A1'

temp = pd.concat([train[var], train['price']], axis=1)

temp.plot.scatter(x=var, y='price')

temp.sort_values(by = var, ascending = True)



train = train.drop(train[train['distance_poi_A1'] > 16250].index, axis=0)
#deleting outliers points by index --> room_size

var = 'distance_poi_A3'

temp = pd.concat([train[var], train['price']], axis=1)

temp.plot.scatter(x=var, y='price')

temp.sort_values(by = var, ascending = True)



# train = train.drop(train[(train['price'] > 14.5) & train['distance_poi_A3'] > 10000].index, axis=0)

# train = train.drop(train[(train['price'] > 14.0) & train['distance_poi_A3'] > 30000].index, axis=0)



train = train.drop(train[(train['price'] > 14.0) & train['distance_poi_A3'] > 10000].index, axis=0)

train = train.drop(train[train['distance_poi_A3'] > 15000].index, axis=0)
#deleting outliers points by index --> room_size

var = 'distance_poi_B3'

temp = pd.concat([train[var], train['price']], axis=1)

temp.plot.scatter(x=var, y='price')

temp.sort_values(by = var, ascending = True)



# train = train.drop(train[(train['price'] > 14.0) & train['distance_poi_B3'] > 15000].index, axis=0)

# # train = train.drop(train[(train['price'] > 14.0) & train['distance_poi_A2'] > 30000].index, axis=0)



train = train.drop(train[train['distance_poi_B3'] > 13000].index, axis=0)
#deleting outliers points by index --> room_size

var = 'distance_poi_B4'

temp = pd.concat([train[var], train['price']], axis=1)

temp.plot.scatter(x=var, y='price')

temp.sort_values(by = var, ascending = True)



# train = train.drop(train[(train['price'] > 14.0) & train['distance_poi_B4'] > 15000].index, axis=0)

# train = train.drop(train[(train['price'] > 14.0) & train['distance_poi_A2'] > 30000].index, axis=0)



train = train.drop(train[train['distance_poi_B4'] > 14000].index, axis=0)
#deleting outliers points by index --> room_size

var = 'longitude'

temp = pd.concat([train[var], train['price']], axis=1)

temp.plot.scatter(x=var, y='price')

temp.sort_values(by = var, ascending = True)



# train = train.drop(train[(train['price'] > 14.0) & train['distance_poi_B4'] > 15000].index, axis=0)

# train = train.drop(train[(train['price'] > 14.0) & train['distance_poi_A2'] > 30000].index, axis=0)



train = train.drop(train[train['longitude'] < 110.28].index, axis=0)
# #deleting outliers points by index --> room_size

# var = 'latitude'

# temp = pd.concat([train[var], train['price']], axis=1)

# temp.plot.scatter(x=var, y='price')

# temp.sort_values(by = var, ascending = True)



# # train = train.drop(train[(train['price'] > 14.0) & train['distance_poi_B4'] > 15000].index, axis=0)

# # train = train.drop(train[(train['price'] > 14.0) & train['distance_poi_A2'] > 30000].index, axis=0)



# train = train.drop(train[train['longitude'] < -7.88].index, axis=0)
#STEP 3: DATA PRE-RPOCESSING AND FEATURE ENGINEERING ON COMBINED DATASET



ntrain = train.shape[0]

ntest = test.shape[0]

y_train = train.price.values

all_data = pd.concat((train, test)).reset_index(drop=True)

# all_data.drop(['price'], axis=1, inplace=True)

print("all_data size is : {}".format(all_data.shape))
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100

all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]

missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})

missing_data.head(20)
f, ax = plt.subplots(figsize=(15, 12))

plt.xticks(rotation='90')

sns.barplot(x=all_data_na.index, y=all_data_na)

plt.xlabel('Features', fontsize=15)

plt.ylabel('Percent of missing values', fontsize=15)

plt.title('Percent missing data by feature', fontsize=15)
#karena secret_2 dan secret_9 semuanya 1 dan 0. Lebih baik di drop saja.

all_data = all_data.drop(['secret_2', 'secret_9'],axis=1)
all_data = all_data.drop(['item_4'],axis=1)
all_data = all_data.drop(['secret_7'],axis=1)
all_data = all_data.drop(['secret_10'],axis=1)
all_data = all_data.drop(['distance_poi_A4', 'distance_poi_A6', 'distance_poi_A5', 'distance_poi_B2', 'distance_poi_B1'],axis=1)
for col in ('distance_poi_A1', 'distance_poi_A2', 'distance_poi_A3', 'distance_poi_B3', 'distance_poi_B4','room_size'):

    all_data[col] = all_data[col].fillna((all_data[col].mean()))
all_data['room_size'] = all_data['room_size'].fillna((all_data['room_size'].mean()))
for col in ('facility_1', 'facility_2', 'facility_3','facility_4', 'facility_5', 'female', 'male', 'item_1', 'item_2', 'item_3', 'item_5','secret_1','secret_3','secret_4','secret_5','secret_6','secret_8'):

    all_data[col] = all_data[col].fillna((all_data[col].mode()[0]))
all_data['latitude'] = all_data['latitude'].abs()



#change the latitude value to positive.
for col in ('longitude', 'latitude'):

    all_data[col] = all_data[col].fillna((all_data[col].interpolate(method='linear')))
#Check remaining missing values if any 

all_data_na = (all_data.isnull().sum() / len(all_data)) * 100

all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)

missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})

missing_data.head()


# Analysing and normalising target variable

var = 'room_size'

sns.distplot(all_data[var], fit=norm);

fig = plt.figure()

res = stats.probplot(all_data[var], plot=plt)



# Applying log transformation to resolve skewness

all_data[var] = np.log1p(all_data[var])

sns.distplot(all_data[var], fit=norm);

fig = plt.figure()

res = stats.probplot(all_data[var], plot=plt)



# Analysing and normalising target variable

var = 'distance_poi_A1'

sns.distplot(all_data[var], fit=norm);

fig = plt.figure()

res = stats.probplot(all_data[var], plot=plt)



# Applying log transformation to resolve skewness

all_data[var] = np.log1p(all_data[var])

sns.distplot(all_data[var], fit=norm);

fig = plt.figure()

res = stats.probplot(all_data[var], plot=plt)



# Analysing and normalising target variable

var = 'distance_poi_A2'

sns.distplot(all_data[var], fit=norm);

fig = plt.figure()

res = stats.probplot(all_data[var], plot=plt)



# Applying log transformation to resolve skewness

all_data[var] = np.log1p(all_data[var])

sns.distplot(all_data[var], fit=norm);

fig = plt.figure()

res = stats.probplot(all_data[var], plot=plt)



# Analysing and normalising target variable

var = 'distance_poi_A3'

sns.distplot(all_data[var], fit=norm);

fig = plt.figure()

res = stats.probplot(all_data[var], plot=plt)



# Applying log transformation to resolve skewness

all_data[var] = np.log1p(all_data[var])

sns.distplot(all_data[var], fit=norm);

fig = plt.figure()

res = stats.probplot(all_data[var], plot=plt)



# Analysing and normalising target variable

var = 'distance_poi_B3'

sns.distplot(all_data[var], fit=norm);

fig = plt.figure()

res = stats.probplot(all_data[var], plot=plt)



# Applying log transformation to resolve skewness

all_data[var] = np.log1p(all_data[var])

sns.distplot(all_data[var], fit=norm);

fig = plt.figure()

res = stats.probplot(all_data[var], plot=plt)



# Analysing and normalising target variable

var = 'distance_poi_B4'

sns.distplot(all_data[var], fit=norm);

fig = plt.figure()

res = stats.probplot(all_data[var], plot=plt)



# Applying log transformation to resolve skewness

all_data[var] = np.log1p(all_data[var])

sns.distplot(all_data[var], fit=norm);

fig = plt.figure()

res = stats.probplot(all_data[var], plot=plt)



# Analysing and normalising target variable

var = 'longitude'

sns.distplot(all_data[var], fit=norm);

fig = plt.figure()

res = stats.probplot(all_data[var], plot=plt)



# Applying log transformation to resolve skewness

all_data[var] = np.log1p(all_data[var])

sns.distplot(all_data[var], fit=norm);

fig = plt.figure()

res = stats.probplot(all_data[var], plot=plt)



# Analysing and normalising target variable

var = 'latitude'

sns.distplot(all_data[var], fit=norm);

fig = plt.figure()

res = stats.probplot(all_data[var], plot=plt)



# Applying log transformation to resolve skewness

all_data[var] = np.log1p(all_data[var])

sns.distplot(all_data[var], fit=norm);

fig = plt.figure()

res = stats.probplot(all_data[var], plot=plt)
#STEP 4: XGBOOST MODELING WITH PARAMETER TUNING



#Creating train_test_split for cross validation

X = all_data.loc[all_data['price']>0]

X = X.drop(['price'], axis=1)

y = all_data[['price']]

y = y.drop(y.loc[y['price'].isnull()].index, axis=0)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.2, random_state=8)



#Creating DMatrices for XGBoost

dtrain = xgb.DMatrix(X_train, label=y_train)

dtest = xgb.DMatrix(X_test, label=y_test)
#Setting initial parameters

params = {

    # Parameters that we are going to tune.

    'max_depth':5,

    'min_child_weight': 1,

    'eta':0.3,

    'subsample': 0.80,

    'colsample_bytree': 0.80,

    'reg_alpha': 0,

    'reg_lambda': 0,

    # Other parameters

    'objective':'reg:squarederror',

}





#Setting evaluation metrics - MAE from sklearn.metrics

params['eval_metric'] = "mae"



num_boost_round = 5000



#Begin training of XGB model

model = xgb.train(

    params,

    dtrain,

    num_boost_round=num_boost_round,

    evals=[(dtest, "Test")],

    early_stopping_rounds=50

)



print("Best MAE: {:.2f} with {} rounds".format(

                 model.best_score,

                 model.best_iteration+1))



#replace num_boost_round with best iteration + 1

num_boost_round = model.best_iteration+1



#Establishing baseline MAE

cv_results = xgb.cv(

    params,

    dtrain,

    num_boost_round=num_boost_round,

    seed=8,

    nfold=5,

    metrics={'mae'},

    early_stopping_rounds=50

)

cv_results

cv_results['test-mae-mean'].min()
#Parameter-tuning for max_depth & min_child_weight (First round)

gridsearch_params = [

    (max_depth, min_child_weight)

    for max_depth in range(0,12,2)

    for min_child_weight in range(0,12,2)

]



min_mae = float("Inf")

best_params = None

for max_depth, min_child_weight in gridsearch_params:

    print("CV with max_depth={}, min_child_weight={}".format(

                             max_depth,

                             min_child_weight))

    # Update our parameters

    params['max_depth'] = max_depth

    params['min_child_weight'] = min_child_weight

    # Run CV

    cv_results = xgb.cv(

        params,

        dtrain,

        num_boost_round=num_boost_round,

        seed=8,

        nfold=5,

        metrics={'mae'},

        early_stopping_rounds=50

    )

    # Update best MAE

    mean_mae = cv_results['test-mae-mean'].min()

    boost_rounds = cv_results['test-mae-mean'].idxmin()

    print("\tMAE {} for {} rounds".format(mean_mae, boost_rounds))

    if mean_mae < min_mae:

        min_mae = mean_mae

        best_params = (max_depth,min_child_weight)

print("Best params: {}, {}, MAE: {}".format(best_params[0], best_params[1], min_mae))





#Parameter-tuning for max_depth & min_child_weight (Second round)

gridsearch_params = [

    (max_depth, min_child_weight)

    for max_depth in [3,4,5]

    for min_child_weight in [3,4,5]

]



min_mae = float("Inf")

best_params = None

for max_depth, min_child_weight in gridsearch_params:

    print("CV with max_depth={}, min_child_weight={}".format(

                             max_depth,

                             min_child_weight))

    # Update our parameters

    params['max_depth'] = max_depth

    params['min_child_weight'] = min_child_weight

    # Run CV

    cv_results = xgb.cv(

        params,

        dtrain,

        num_boost_round=num_boost_round,

        seed=8,

        nfold=5,

        metrics={'mae'},

        early_stopping_rounds=50

    )

    # Update best MAE

    mean_mae = cv_results['test-mae-mean'].min()

    boost_rounds = cv_results['test-mae-mean'].idxmin()

    print("\tMAE {} for {} rounds".format(mean_mae, boost_rounds))

    if mean_mae < min_mae:

        min_mae = mean_mae

        best_params = (max_depth,min_child_weight)

print("Best params: {}, {}, MAE: {}".format(best_params[0], best_params[1], min_mae))





#Parameter-tuning for max_depth & min_child_weight (Third round)

gridsearch_params = [

    (max_depth, min_child_weight)

    for max_depth in [3]

    for min_child_weight in [i/10. for i in range(30,50,2)]

]



min_mae = float("Inf")

best_params = None

for max_depth, min_child_weight in gridsearch_params:

    print("CV with max_depth={}, min_child_weight={}".format(

                             max_depth,

                             min_child_weight))

    # Update our parameters

    params['max_depth'] = max_depth

    params['min_child_weight'] = min_child_weight

    # Run CV

    cv_results = xgb.cv(

        params,

        dtrain,

        num_boost_round=num_boost_round,

        seed=8,

        nfold=5,

        metrics={'mae'},

        early_stopping_rounds=50

    )

    # Update best MAE

    mean_mae = cv_results['test-mae-mean'].min()

    boost_rounds = cv_results['test-mae-mean'].idxmin()

    print("\tMAE {} for {} rounds".format(mean_mae, boost_rounds))

    if mean_mae < min_mae:

        min_mae = mean_mae

        best_params = (max_depth,min_child_weight)

print("Best params: {}, {}, MAE: {}".format(best_params[0], best_params[1], min_mae))
#Updating max_depth and mind_child_weight parameters

params['max_depth'] = 3

params['min_child_weight'] = 3.0
#Recalibrating num_boost_round after parameter updates

num_boost_round = 5000



model = xgb.train(

    params,

    dtrain,

    num_boost_round=num_boost_round,

    evals=[(dtest, "Test")],

    early_stopping_rounds=50

)



print("Best MAE: {:.2f} with {} rounds".format(

                 model.best_score,

                 model.best_iteration+1))



#replace num_boost_round with best iteration + 1

num_boost_round = model.best_iteration+1
#Parameter-tuning for subsample & colsample (First round)

gridsearch_params = [

    (subsample, colsample)

    for subsample in [i/10. for i in range(3,11)]

    for colsample in [i/10. for i in range(3,11)]

]



min_mae = float("Inf")

best_params = None

# We start by the largest values and go down to the smallest

for subsample, colsample in reversed(gridsearch_params):

    print("CV with subsample={}, colsample={}".format(

                             subsample,

                             colsample))

    # We update our parameters

    params['subsample'] = subsample

    params['colsample_bytree'] = colsample

    # Run CV

    cv_results = xgb.cv(

        params,

        dtrain,

        num_boost_round=num_boost_round,

        seed=8,

        nfold=5,

        metrics={'mae'},

        early_stopping_rounds=50

    )

    # Update best score

    mean_mae = cv_results['test-mae-mean'].min()

    boost_rounds = cv_results['test-mae-mean'].idxmin()

    print("\tMAE {} for {} rounds".format(mean_mae, boost_rounds))

    if mean_mae < min_mae:

        min_mae = mean_mae

        best_params = (subsample,colsample)

print("Best params: {}, {}, MAE: {}".format(best_params[0], best_params[1], min_mae))



#Parameter-tuning for subsample & colsample (Second round)

gridsearch_params = [

    (subsample, colsample)

    for subsample in [i/100. for i in range(80,100)]

    for colsample in [i/100. for i in range(70,90)]

]



min_mae = float("Inf")

best_params = None

# We start by the largest values and go down to the smallest

for subsample, colsample in reversed(gridsearch_params):

    print("CV with subsample={}, colsample={}".format(

                             subsample,

                             colsample))

    # We update our parameters

    params['subsample'] = subsample

    params['colsample_bytree'] = colsample

    # Run CV

    cv_results = xgb.cv(

        params,

        dtrain,

        num_boost_round=num_boost_round,

        seed=8,

        nfold=5,

        metrics={'mae'},

        early_stopping_rounds=50

    )

    # Update best score

    mean_mae = cv_results['test-mae-mean'].min()

    boost_rounds = cv_results['test-mae-mean'].idxmin()

    print("\tMAE {} for {} rounds".format(mean_mae, boost_rounds))

    if mean_mae < min_mae:

        min_mae = mean_mae

        best_params = (subsample,colsample)

print("Best params: {}, {}, MAE: {}".format(best_params[0], best_params[1], min_mae))

#Updating subsample and colsample parameters

params['subsample'] = 0.85

params['colsample'] = 0.88
#Recalibrating num_boost_round after parameter updates

num_boost_round = 5000



model = xgb.train(

    params,

    dtrain,

    num_boost_round=num_boost_round,

    evals=[(dtest, "Test")],

    early_stopping_rounds=50

)



print("Best MAE: {:.2f} with {} rounds".format(

                 model.best_score,

                 model.best_iteration+1))



#replace num_boost_round with best iteration + 1

num_boost_round = model.best_iteration+1
#Parameter-tuning for reg_alpha & reg_lambda

gridsearch_params = [

    (reg_alpha, reg_lambda)

    for reg_alpha in [1e-5, 1e-4, 1e-3, 1e-2, 0.1]

    for reg_lambda in [1e-5, 1e-4, 1e-3, 1e-2, 0.1]

]



min_mae = float("Inf")

best_params = None



for reg_alpha, reg_lambda in gridsearch_params:

    print("CV with reg_alpha={}, reg_lambda={}".format(

                             reg_alpha,

                             reg_lambda))

    # We update our parameters

    params['reg_alpha'] = reg_alpha

    params['reg_lambda'] = reg_lambda

    # Run CV

    cv_results = xgb.cv(

        params,

        dtrain,

        num_boost_round=num_boost_round,

        seed=8,

        nfold=5,

        metrics={'mae'},

        early_stopping_rounds=50

    )

    # Update best score

    mean_mae = cv_results['test-mae-mean'].min()

    boost_rounds = cv_results['test-mae-mean'].idxmin()

    print("\tMAE {} for {} rounds".format(mean_mae, boost_rounds))

    if mean_mae < min_mae:

        min_mae = mean_mae

        best_params = (reg_alpha,reg_lambda)

print("Best params: {}, {}, MAE: {}".format(best_params[0], best_params[1], min_mae))
#Updating reg_alpha and reg_lambda parameters

params['reg_alpha'] = 0.0001

params['reg_lambda'] = 1e-05
#Resetting num_boost_round to 5000

num_boost_round = 5000



#Parameter-tuning for eta

min_mae = float("Inf")

best_params = None

for eta in [0.3, 0.2, 0.1, 0.05, 0.01, 0.005]:

    print("CV with eta={}".format(eta))



    params['eta'] = eta

    cv_results = xgb.cv(

            params,

            dtrain,

            num_boost_round=num_boost_round,

            seed=8,

            nfold=5,

            metrics=['mae'],

            early_stopping_rounds=50

          )

    # Update best score

    mean_mae = cv_results['test-mae-mean'].min()

    boost_rounds = cv_results['test-mae-mean'].idxmin()

    print("\tMAE {} for {} rounds\n".format(mean_mae, boost_rounds))

    if mean_mae < min_mae:

        min_mae = mean_mae

        best_params = eta

print("Best params: {}, MAE: {}".format(best_params, min_mae))
params['eta'] = 0.01
model = xgb.train(

    params,

    dtrain,

    num_boost_round=5000,

    evals=[(dtest, "Test")],

    early_stopping_rounds=50

)



num_boost_round = model.best_iteration + 1

best_model = xgb.train(

    params,

    dtrain,

    num_boost_round=num_boost_round,

    evals=[(dtest, "Test")]

)



mean_absolute_error(best_model.predict(dtest), y_test)
testdf = all_data.loc[all_data['price'].isnull()]

testdf = testdf.drop(['price'],axis=1)

sub = pd.DataFrame()

sub['id'] = test_ID

testdf = xgb.DMatrix(testdf)



y_pred = np.expm1(best_model.predict(testdf))

sub['price'] = y_pred



sub.to_csv('submission.csv', index=False)