import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

import seaborn as sns

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.pipeline import make_pipeline

from sklearn.linear_model import (

    LinearRegression,

    Ridge,

    Lasso

)

%matplotlib inline
# import data

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
# merge data

train['DataOrigin'] = 'Train'

test['DataOrigin'] = 'Test'

test['SalePrice'] = 9999999999

alldata = pd.concat([train,test],axis=0,sort=True).reset_index(drop=True)

print('The size of train is : ' + str(train.shape))

print('The size of test is : ' + str(test.shape))
alldata.describe()
# missing data in train

train.isnull().sum()[train.isnull().sum()>0].sort_values(ascending=False)
# missing data in test

test.isnull().sum()[test.isnull().sum()>0].sort_values(ascending=False)
# Data type of the column containing the deletion

na_col_list = alldata.isnull().sum()[alldata.isnull().sum()>0].index.tolist()

alldata[na_col_list].dtypes.sort_values()
# float to 0

na_float_cols = alldata[na_col_list].dtypes[alldata[na_col_list].dtypes=='float64'].index.tolist()

for na_float_col in na_float_cols:

    alldata.loc[alldata[na_float_col].isnull(),na_float_col] = 0.0

    

# object to 'NA'

na_obj_cols = alldata[na_col_list].dtypes[alldata[na_col_list].dtypes=='object'].index.tolist()

for na_obj_col in na_obj_cols:

    alldata.loc[alldata[na_obj_col].isnull(),na_obj_col] = 'NA'
# Confirm that all deficits have been compensated

alldata.isnull().sum()[alldata.isnull().sum()>0].sort_values(ascending=False)
# listing number and categorical values.

cat_cols = alldata.dtypes[alldata.dtypes=='object'].index.tolist()

num_cols = alldata.dtypes[alldata.dtypes!='object'].index.tolist()



# divide special values

other_cols = ['Id','DataOrigin']

cat_cols.remove('DataOrigin')

num_cols.remove('Id')



# get_dummies

alldata_cat = pd.get_dummies(alldata[cat_cols])

# merge dummy data

all_data = pd.concat([alldata[other_cols],alldata[num_cols],alldata_cat],axis=1)



all_data.head(3)
sns.distplot(train['SalePrice'])
	

sns.distplot(np.log(train['SalePrice']))
# divide data to train and test

train_ = all_data[all_data['DataOrigin']=='Train'].drop(['DataOrigin','Id'], axis=1).reset_index(drop=True)

test_ = all_data[all_data['DataOrigin']=='Test'].drop(['DataOrigin','SalePrice'], axis=1).reset_index(drop=True)

# divide train data to objective value and explanatory values

train_x = train_.drop('SalePrice',axis=1)

train_y = np.log(train_['SalePrice'])

# divide test data to id and other values

test_id = test_['Id']

test_data = test_.drop('Id',axis=1)
scaler = StandardScaler()  # Apply coefficients to the data set

param_grid = [0.001, 0.01, 0.1, 1.0, 10.0,100.0,1000.0]

cnt = 0

# Predict optimal results with several parameters

for alpha in param_grid:

    ls = Lasso(alpha=alpha) # Lasso regression model

    pipeline = make_pipeline(scaler, ls) # pipeline. StandardScaler, Lasso

    X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.3, random_state=0)

    pipeline.fit(X_train,y_train)

    # root mean squared error

    train_rmse = np.sqrt(mean_squared_error(y_train, pipeline.predict(X_train)))

    test_rmse = np.sqrt(mean_squared_error(y_test, pipeline.predict(X_test)))

    if cnt == 0:

        best_score = test_rmse

        best_estimator = pipeline

        best_param = alpha

    elif best_score > test_rmse:

        best_score = test_rmse

        best_estimator = pipeline

        best_param = alpha

    else:

        pass

    cnt = cnt + 1

    

print('alpha : ' + str(best_param))

print('test score is : ' +str(best_score))
plt.subplots_adjust(wspace=0.4)

plt.subplot(121)

plt.scatter(np.exp(y_train),np.exp(best_estimator.predict(X_train)))

plt.subplot(122)

plt.scatter(np.exp(y_test),np.exp(best_estimator.predict(X_test)))
# make csv to submit

ls = Lasso(alpha = best_param)

pipeline = make_pipeline(scaler, ls)

pipeline.fit(train_x,train_y)

test_SalePrice = pd.DataFrame(np.exp(pipeline.predict(test_data)),columns=['SalePrice'])

test_Id = pd.DataFrame(test_id,columns=['Id'])

pd.concat([test_Id, test_SalePrice],axis=1).to_csv('csv_to_submit.csv',index=False)