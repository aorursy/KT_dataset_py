#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import xticks
import seaborn as sns

from sklearn import metrics
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV

# hide warnings
import warnings
warnings.filterwarnings('ignore')
#Importing train dataset
house = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

print(house.shape)
pd.set_option("display.max_columns", 500)
house.head()
house.info()
# divide all variables to numerical and categorical
house_cols = house.columns
house_num= [x for x in house_cols if house[x].dtype in ['float64','int64']]
house_cat= [x for x in house_cols if house[x].dtype=='object']

# 'MSSubClass','OverallQual','OverallCond' are categorical variable but values are numeric so move to categorical list
house_num.remove('MSSubClass')
house_cat.append('MSSubClass')

house_num.remove('OverallQual')
house_cat.append('OverallQual')

house_num.remove('OverallCond')
house_cat.append('OverallCond')

print(house_num)
print(house_cat)
# check missing values in Numerical columns
NA_house = house[house_num].isnull().sum()
NA_house = NA_house[NA_house.values >0]
NA_house = NA_house.sort_values(ascending=False)
plt.figure(figsize=(20,4))
NA_house.plot(kind='bar')
plt.title('List of Columns & NA counts')
plt.show()
# check missing values in Categorical columns
NA_house = house[house_cat].isnull().sum()
NA_house = NA_house[NA_house.values >0]
NA_house = NA_house.sort_values(ascending=False)
plt.figure(figsize=(20,4))
NA_house.plot(kind='bar')
plt.title('List of Columns & NA counts')
plt.show()
def missing_value_correction(df,df_cat,df_num):
    for col in df_cat:
        df[col]=df[col].fillna('unknown')
    for col in df_num:
        df[col]=df[col].fillna(0)
    return df
# apply missing value correction on the dataset
house = missing_value_correction(house, house_cat, house_num)
house.head()
### GarageFinish GarageQual GarageCond GarageType if NA then no Garage

fig, ax=plt.subplots(nrows =1,ncols=4,figsize=(20,8))
sns.boxplot(y=house['GarageArea'], x=house['GarageFinish'],ax=ax[0])
sns.boxplot(y=house['GarageArea'], x=house['GarageQual'],ax=ax[1])
sns.boxplot(y=house['GarageArea'], x=house['GarageCond'],ax=ax[2])
sns.boxplot(y=house['GarageArea'], x=house['GarageType'],ax=ax[3])
xticks(rotation = 90)
plt.show
### BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1 and BsmtFinSF1, BsmtFinType2 and BsmtFinSF2

fig, ax=plt.subplots(nrows =2,ncols=3,figsize=(20,8))
sns.boxplot(y=house['TotalBsmtSF'], x=house['BsmtQual'],ax=ax[0][0])
sns.boxplot(y=house['TotalBsmtSF'], x=house['BsmtCond'],ax=ax[0][1])
sns.boxplot(y=house['TotalBsmtSF'], x=house['BsmtExposure'],ax=ax[0][2])
sns.boxplot(y=house['BsmtFinSF1'], x=house['BsmtFinType1'],ax=ax[1][0])
sns.boxplot(y=house['BsmtFinSF2'], x=house['BsmtFinType2'],ax=ax[1][1])
house[(house['BsmtExposure']=='unknown')& (house['TotalBsmtSF']>0)]
house[(house['BsmtFinType2']=='unknown')& (house['BsmtFinSF2']>0)]
## MasVnrArea MasVnrType

plt.figure(figsize=(20, 10))
sns.boxplot(y=house['MasVnrArea'], x=house['MasVnrType'])
# create new columns 'BuiltAge', 'RemodelAge','GarageAge'
def date_variable_toAge(house, house_cat, house_num):
    import datetime
    today = datetime.datetime.now()

    house.loc[np.isnan(house['YearBuilt']), ['YearBuilt']] = today.year
    house.loc[np.isnan(house['YearRemodAdd']), ['YearRemodAdd']] = today.year
    house.loc[np.isnan(house['GarageYrBlt']), ['GarageYrBlt']] = today.year
    house.loc[np.isnan(house['YrSold']), ['YrSold']] = today.year

    house['BuiltAge'] = house['YearBuilt'].apply(lambda x : today.year-x )
    house['RemodelAge'] = house['YearRemodAdd'].apply(lambda x: today.year-x )
    house['GarageAge'] = house['GarageYrBlt'].apply(lambda x: today.year-x)
    house['SoldAge'] = house['YrSold'].apply(lambda x: today.year-x)

    house = house.drop(['YearBuilt','YearRemodAdd','GarageYrBlt','YrSold'],axis=1)
    
    house_num.remove('YearBuilt')
    house_num.remove('YearRemodAdd')
    house_num.remove('GarageYrBlt')
    house_num.remove('YrSold')
    
    house_num.append('BuiltAge')
    house_num.append('RemodelAge')
    house_num.append('GarageAge')
    house_num.append('SoldAge')
    
    return house, house_cat, house_num
# change year values to Age of the field as of today
house, house_cat, house_num = date_variable_toAge(house, house_cat, house_num)
house.head()
# Dummy variables for catrgorical variables 
def create_dummy_for_cat(house,house_cat):
    house_dummies=house['Id']

    for col in house_cat:
        house_dummy = pd.get_dummies(house[col], prefix=col ,drop_first=True) 
        house_dummies = pd.concat([house_dummies , house_dummy] ,axis=1) 
    
    house_dummies.drop(columns='Id',inplace=True)

    return house_dummies
house_dummies = create_dummy_for_cat(house, house_cat)
house = pd.concat([house , house_dummies] ,axis=1) 
house.drop(columns =house_cat, inplace = True)
house.shape
house.head()
# Checking outliers at 25%,50%,75%,90%,95% and 99%
house[house_num].describe(percentiles=[.25,.5,.75,.90,.95,.99])
# box plot all numerical variable
plt.figure(figsize=(20, 50))
j=0
for i in ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1',  'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea']:
    j=j+1
    plt.subplot(7,4,j)
    sns.boxplot(data=house, y=i)
    
plt.show()
house[(house['LotFrontage']>300)
        |(house['LotArea']>200000)
        |(house['MasVnrArea']>1200)
        |(house['BsmtFinSF1']>4000)
        |(house['TotalBsmtSF']>5000)
        |(house['1stFlrSF']>4000)
        |(house['2ndFlrSF']>2000)
        |(house['GrLivArea']>5000)]
house=house[~house['Id'].isin([298,314,1299])]
house.shape
house=house.set_index('Id')
house_num.remove('Id')
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

# Apply scaler() to all the columns except the categorical variables and target variable
house_scale_columns = house_num
house_scale_columns.remove('SalePrice')
house[house_scale_columns] = scaler.fit_transform(house[house_scale_columns])
house.head()
X= house.drop('SalePrice',axis=1)
y= house['SalePrice']
# split into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    train_size=0.7,
                                                    test_size = 0.3, random_state=100)
# list of alphas to tune
params = {'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1, 
 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 
 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 20, 50 ]}


ridge = Ridge()

# cross validation
folds = 5
model_cv = GridSearchCV(estimator = ridge, 
                        param_grid = params, 
                        scoring= 'neg_mean_absolute_error', 
                        cv = folds, 
                        return_train_score=True,
                        verbose = 1)            
model_cv.fit(X_train, y_train)
cv_results = pd.DataFrame(model_cv.cv_results_)
cv_results = cv_results[cv_results['param_alpha']>=1]
cv_results.head()
# plotting mean test and train scoes with alpha 
cv_results['param_alpha'] = cv_results['param_alpha'].astype('int32')

# plotting
plt.figure(figsize=(20, 10))
plt.plot(cv_results['param_alpha'], cv_results['mean_train_score'])
plt.plot(cv_results['param_alpha'], cv_results['mean_test_score'])
plt.vlines(x=2.5,ymax=-5000, ymin=-17500, colors="r", linestyles="--")
plt.xlabel('alpha')
plt.ylabel('Negative Mean Absolute Error')
plt.title("Negative Mean Absolute Error and alpha")
plt.legend(['train score', 'test score'], loc='upper left')
plt.show()
alpha = 2.5
ridge = Ridge(alpha=alpha)

ridge.fit(X_train, y_train)
y_pred = ridge.predict(X_test)
# Plotting y_test and y_pred to understand the spread.
fig = plt.figure()
plt.scatter(y_test,y_pred)
fig.suptitle('y_test vs y_pred', fontsize=20)              # Plot heading 
plt.xlabel('y_test', fontsize=18)                          # X-label
plt.ylabel('y_pred', fontsize=16)                          # Y-label
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)
lasso = Lasso()

# list of alphas to tune
params = {'alpha': [1, 10, 25, 35, 40, 45, 50]}


# cross validation
folds = 5

# cross validation
model_cv = GridSearchCV(estimator = lasso, 
                        param_grid = params, 
                        scoring= 'neg_mean_absolute_error', 
                        cv = folds, 
                        return_train_score=True,
                        verbose = 1)            

model_cv.fit(X_train, y_train) 
cv_results = pd.DataFrame(model_cv.cv_results_)
cv_results.head()
# plotting mean test and train scoes with alpha 
cv_results['param_alpha'] = cv_results['param_alpha'].astype('float32')

# plotting
plt.figure(figsize=(20, 10))
plt.plot(cv_results['param_alpha'], cv_results['mean_train_score'])
plt.plot(cv_results['param_alpha'], cv_results['mean_test_score'])
plt.vlines(x=25, ymax=-10000, ymin=-17500, colors="r", linestyles="--")
plt.xlabel('alpha')
plt.ylabel('Negative Mean Absolute Error')
plt.title("Negative Mean Absolute Error and alpha")
plt.legend(['train score', 'test score'], loc='upper left')
plt.show()
alpha =25

lasso = Lasso(alpha=alpha)
        
lasso.fit(X_train, y_train) 
# feature selection using Lasso
features = pd.DataFrame(lasso.coef_,X_train.columns)
features = features.reset_index()
features.columns = ['feature','coeficients']
selected_features = features[features['coeficients']>0]
selected_features=selected_features.sort_values(by='coeficients', ascending =False)
selected_features.columns = ['feature','coeficients']
selected_features
y_pred = lasso.predict(X_test)
# Plotting y_test and y_pred to understand the spread.
fig = plt.figure()
plt.scatter(y_test,y_pred)
fig.suptitle('y_test vs y_pred', fontsize=20)              # Plot heading 
plt.xlabel('y_test', fontsize=18)                          # X-label
plt.ylabel('y_pred', fontsize=16)                          # Y-label
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)
import xgboost as xgb

model_xgb = xgb.XGBRegressor(n_estimators=200)

# list of alphas to tune
params = {'learning_rate':[0.03,0.05,0.1],
         'subsample':[0.2,0.5,0.6]}


# cross validation
folds = 3

# cross validation
model_cv = GridSearchCV(estimator = model_xgb, 
                        param_grid = params, 
                        scoring= 'neg_mean_absolute_error', 
                        cv = folds, 
                        return_train_score=True,
                        verbose = 1)            

model_cv.fit(X_train, y_train) 
cv_results = pd.DataFrame(model_cv.cv_results_)
cv_results[cv_results['mean_test_score']== max(cv_results['mean_test_score'])]
model_xgb = xgb.XGBRegressor(n_estimators=200,learning_rate= 0.05, subsample= 0.6 )
model_xgb.fit(X_train,y_train)

xgb_pred = model_xgb.predict(X_test)
r2_score(y_test, xgb_pred)
model_ridge = Ridge(alpha = 2.5)
model_ridge.fit(X_train, y_train)

model_lasso = Lasso(alpha = 25)
model_lasso.fit(X_train, y_train)

model_xgb = xgb.XGBRegressor(n_estimators=200,learning_rate= 0.05, subsample= 0.6 )
model_xgb.fit(X_train,y_train)

r_pred = model_ridge.predict(X_test)
l_pred = model_lasso.predict(X_test)
x_pred = model_xgb.predict(X_test)

y_pred= r_pred*0.7 + x_pred *0.25 + l_pred* 0.05
r2_score(y_test, y_pred)
test=pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

test_cols = test.columns
test_num= [x for x in test_cols if test[x].dtype in ['float64','int64']]
test_cat= [x for x in test_cols if test[x].dtype=='object']

# 'MSSubClass','OverallQual','OverallCond' are categorical variable but values are numeric so move to categorical list
test_num.remove('MSSubClass')
test_cat.append('MSSubClass')

test_num.remove('OverallQual')
test_cat.append('OverallQual')

test_num.remove('OverallCond')
test_cat.append('OverallCond')

print(test_num)
print(test_cat)
# Pre-Processing test set similarily to training

test = missing_value_correction(test,test_cat,test_num)
test,test_cat,test_num=  date_variable_toAge(test,test_cat,test_num)
# Create Dummies from cateogrical features in test set
test_dummies = create_dummy_for_cat(test,test_cat)
test = pd.concat([test , test_dummies] ,axis=1) 
test.drop(columns =test_cat, inplace = True)
test.shape
test=test.set_index('Id')
test_num.remove('Id')

# Apply scaler() as in training data to all the columns except the categorical variables and target variable
test_scale_columns = test_num
test[test_scale_columns] = scaler.fit_transform(test[test_scale_columns])
# compare and print features of test Not in the Lasso set of selected features
# Add a Dummy column with all zero for such features

for col in list(X_train.columns):
    if col not in list(test.columns):
        print(col)
        test[col]=0
test = test[X_train.columns]
r_pred = model_ridge.predict(test)
l_pred = model_lasso.predict(test)
x_pred = model_xgb.predict(test)

y_test_pred= r_pred*0.7 + x_pred *0.25 + l_pred* 0.05

submission= pd.DataFrame({'Id': test.index, 'SalePrice': y_test_pred})
submission.head()
submission.to_csv('HPP_submission.csv', index=False)