### Load all the needed libraries 



import warnings

warnings.filterwarnings('ignore')

import pandas as pd

import numpy as np 

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
from glob import glob

glob('../input/home-data-for-ml-course/*')
train = pd.read_csv('../input/home-data-for-ml-course/train.csv')

test = pd.read_csv('../input/home-data-for-ml-course/test.csv')

print('Training Data Shape: {}'.format(train.shape))

print('Test Data Shape: {}'.format(test.shape))

train.head()
train.info()
train_null = 100*train.isnull().sum()/len(train)          ## percentage of nan values for each columns in training data

train_null = train_null[train_null>0]                      ## consider columns with null values for further analysis

train_null.sort_values(ascending=False).plot(kind = 'bar', 

                                             rot=90, title = 'Percentage Missing data for each column')

train_null_50 = train_null[train_null>45].index.tolist()      ## get columns with more than 50% null data 



test_null = 100*test.isnull().sum()/len(test)

test_null = test_null[test_null>0]

test_null_50 = test_null[test_null>45].index.tolist()



null_to_drop = list(set(train_null_50 + test_null_50))



## percentage of nan values for each columns in training data



zero_data = train.isin([0]).sum().sort_values(ascending=False).head(20)/len(train)*100      

zero_cols_80 = zero_data[zero_data>80].index.tolist()

drop_cols = null_to_drop + zero_cols_80



for col in zero_data[(zero_data<80) & (zero_data>40) ].index.tolist():

    x = train[col].value_counts()

    zero_ppercent = zero_data.loc[col]

    if len(x)>6 and zero_ppercent>50:

        drop_cols.append(col)



drop_cols = list(set(drop_cols))



train.drop(drop_cols, axis=1, inplace=True)

test.drop(drop_cols, axis=1, inplace=True)

print(train.shape , '\n', test.shape)
null_data = train.isnull().sum().sort_values(ascending=False)/len(train)*100

null_columns = null_data[null_data>0].index.tolist()

print(null_columns)



train_des = train[null_columns].describe(include='all')   ## Null data columns statistic description 
fill_mode = train_des.loc['freq'][train_des.loc['freq']>900].index

train[fill_mode]=train[fill_mode].fillna(train.mode().iloc[0])
remain_cols = list(set(null_columns)-set(fill_mode))

train_des = train[remain_cols].describe(include='all')

print(remain_cols)

train_des
sns.violinplot(x ='LotFrontage', data=train)

sns.violinplot(x = train[train['LotFrontage']<110]['LotFrontage'], data=train, color='r');



### We can see from overlapping figure that values greater than ~ 130 are all outliers and most of the values are lying near to 

#its mean so lets fill the LotFrontage variable's nan data with its mean values
train['LotFrontage'].fillna(train['LotFrontage'].mean(), inplace=True)

test['LotFrontage'].fillna(test['LotFrontage'].mean(), inplace=True)

remain_cols.remove('LotFrontage')
num_correlation = train.select_dtypes(exclude='object').corr()

plt.figure(figsize=(10,10))

plt.title('High Correlation')

sns.heatmap(num_correlation>.75, annot=True, square=True)
from scipy import stats

slope, intercept, _ , __, ___ = stats.linregress(train['YearBuilt'],train['GarageYrBlt'].fillna(train['YearBuilt']))

print('Slope: {} , Intercept: {}'.format(slope, intercept))

sns.regplot(x='YearBuilt', y= 'GarageYrBlt', data=train)
train['GarageYrBlt'] = train['GarageYrBlt'].fillna((train['YearBuilt']*slope+intercept).astype(int))

test['GarageYrBlt'] = test['GarageYrBlt'].fillna((test['YearBuilt']*slope+intercept).astype(int))

remain_cols.remove('GarageYrBlt')

print(remain_cols)
train[remain_cols] = train[remain_cols].fillna(train.mode().iloc[0])

test[remain_cols] = test[remain_cols].fillna(test.mode().iloc[0])
test_null = test.isnull().sum().sort_values(ascending=False)

test_null = test_null[test_null>0]

print(test_null.head(10))

test_data_null_cols = test_null.index.tolist()
test[test_data_null_cols] = test[test_data_null_cols].fillna(test.mode().iloc[0])
from sklearn.model_selection import train_test_split

X = train.drop(['SalePrice'], axis=1)

y = np.log1p(train['SalePrice'])

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=2)
categorical_cols = [cname for cname in X.columns if

                    X[cname].nunique() <= 30 and

                    X[cname].dtype == "object"] 

                





numerical_cols = [cname for cname in X.columns if

                 X[cname].dtype in ['int64','float64']]





my_cols = numerical_cols + categorical_cols



X_train = X_train[my_cols].copy()

X_valid = X_valid[my_cols].copy()

X_test = test[my_cols].copy()
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder



num_transformer = Pipeline(steps=[

    ('num_imputer', SimpleImputer(strategy='constant'))

    ])



cat_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

    ])



preprocessor = ColumnTransformer(

    transformers=[

        ('num', num_transformer, numerical_cols),       

        ('cat',cat_transformer,categorical_cols),

        ])
def inv_y(transformed_y):

    return np.exp(transformed_y)
# from xgboost import XGBRegressor

# from sklearn.metrics import mean_absolute_error

# from sklearn.model_selection import GridSearchCV





# model = XGBRegressor()

# params = {'learning_rate':[0.01, 0.05, 0.1],

#           'n_estimators' : [100,500, 3000],

#           'max_depth' : [3, 6, 8],

#           'subsample' : [0.9, 0.7, 0.5], 

#           'colsample_bytree':[0.7],

#           'objective':['reg:squarederror'],

#           'nthread':[-1, 1, 4],

#           'min_child_weight':[0,4]

# }



# grid = GridSearchCV(model, param_grid=params, cv=2, n_jobs = 5, verbose=True)





# clf = Pipeline(steps=[('preprocessor', preprocessor),

#                           ('model', grid)])

# clf.fit(X_train, y_train)

# print(clf.best_params_)



# predict = clf.predict(X_valid)

# print('XGBoost: ' + str(mean_absolute_error(inv_y(predict), inv_y(y_valid))))

from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.linear_model import LassoCV

from sklearn.ensemble import GradientBoostingRegressor



def inv_y(transformed_y):

    return np.exp(transformed_y)



n_folds = 10



# XGBoost

model = XGBRegressor(learning_rate=0.01, n_estimators=3460, max_depth=3, min_child_weight=0,

                     gamma=0, subsample=0.7,colsample_bytree=0.7,objective='reg:squarederror', 

                     nthread=-1,scale_pos_weight=1, seed=27, reg_alpha=0.00006)

clf = Pipeline(steps=[('preprocessor', preprocessor),

                          ('model', model)])

clf.fit(X_train, y_train)

predict = clf.predict(X_valid)

print('XGBoost: ' + str(mean_absolute_error(inv_y(predict), inv_y(y_valid))))



      

# Lasso   

model = LassoCV(max_iter=1e7,  random_state=14, cv=n_folds)

clf = Pipeline(steps=[('preprocessor', preprocessor),

                          ('model', model)])

clf.fit(X_train, y_train)

predict = clf.predict(X_valid)

print('Lasso: ' + str(mean_absolute_error(inv_y(predict), inv_y(y_valid))))



      

# GradientBoosting   

model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=5)

clf = Pipeline(steps=[('preprocessor', preprocessor),

                          ('model', model)])

clf.fit(X_train, y_train)

predict = clf.predict(X_valid)

print('Gradient: ' + str(mean_absolute_error(inv_y(predict), inv_y(y_valid))))
model = XGBRegressor(learning_rate=0.01, n_estimators=3460, max_depth=3, min_child_weight=0,gamma=0, 

                     subsample=0.7,colsample_bytree=0.7,objective='reg:squarederror', nthread=-1,scale_pos_weight=1, 

                     seed=27, reg_alpha=0.00006)

clf = Pipeline(steps=[('preprocessor', preprocessor),

                          ('model', model)])

clf.fit(X, y)

predict = clf.predict(X_test)

# print('XGBoost: ' + str(mean_absolute_error(inv_y(predict), inv_y(y_valid))))

predict = inv_y(predict)
output = pd.DataFrame({'Id': X_test['Id'],

                       'SalePrice': predict})



output.to_csv('submission.csv', index=False)