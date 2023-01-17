import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
train = pd.read_csv('/kaggle/input/bike-sharing-demand/train.csv')

test = pd.read_csv('/kaggle/input/bike-sharing-demand/test.csv')

# We need timestamps from test to create submission file

time = test['datetime']
# Make datetime column date time format

train['datetime'] = pd.to_datetime(train['datetime'])
# Extrac day of the week, hout and month



train['day_of_week'] = train.datetime.dt.day_name()

train['hour'] = train.datetime.dt.hour

train['month'] = train.datetime.dt.month
# Since temp and atemp  have high correlation 

# It make sense to dervice a single feature

# Let`s use average value between two



train['temp'] = (train['temp']+train['atemp'])/2

train = train.drop('atemp',axis=1)
# here we choose columns that later on will be transformed 

# By OneHotEncoder

# By OrdinalEncoder

# Let`s also try to use bins

# And scale numeric values



columns_ohe = ['season','holiday','workingday','day_of_week','month']

columns_bin = ['humidity','windspeed']

columns_num = ['temp','humidity','windspeed']
# Let`s use log(1+x) of target values

# Because there is postitive skewnees

train['count']=np.log1p(train['count'])



def remove_outliers(df_in, col_name):

    q1 = df_in[col_name].quantile(0.25)

    q3 = df_in[col_name].quantile(0.75)

    iqr = q3-q1 #Interquartile range

    fence_low  = q1-1.5*iqr

    fence_high = q3+1.5*iqr

    df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]

    print('Removed: ',(len(df_in)-len(df_out))/len(df_in)*100,' % of initial dataset')

    return df_out



train = remove_outliers(train,'count')

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

from sklearn.preprocessing import KBinsDiscretizer

from sklearn.pipeline import make_pipeline

from sklearn.compose import make_column_transformer

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.svm import SVR

from sklearn.metrics import mean_squared_error,mean_squared_log_error



# Column tranformer is usefull function from sklearn library

# It help to make tranformation of pandas dataframes and put in pipeline if necessary



trans = make_column_transformer(

    (OrdinalEncoder(),['weather']),

    (OneHotEncoder(),columns_ohe),

    (RobustScaler(),columns_num),

    (KBinsDiscretizer(n_bins = 4,encode='ordinal'), columns_bin),

    remainder = 'passthrough'

)





# We use RandomForesrRegressor with already adjusted values by GridSearchCV

rf_reg = RandomForestRegressor(n_estimators = 3000,

                               max_depth = 40,

                               random_state = 42)



# We use GradientBoosting Regressor with already adjusted values by GridSearchCV

gb_reg = GradientBoostingRegressor(n_estimators=1000, 

                                   min_samples_leaf=6, 

                                   random_state=42)





# Create training and validation datasets

X = train.drop(['count','datetime','casual','registered'],axis=1)

y = train['count']



# transform X values

X = trans.fit_transform(X)





from sklearn.model_selection import train_test_split, GridSearchCV

# Split set on traing and test

X_train,X_val,y_train,y_val = train_test_split(X,y,

                                               random_state=0,

                                               test_size = 0.25,

                                               shuffle=True)



# Fit by two regressors

rf_reg.fit(X_train,y_train)



gb_reg.fit(X_train,y_train)





print('(RFR) RMSE: ',np.sqrt(mean_squared_error(y_val,rf_reg.predict(X_val))))

print('(GBR) RMSE: ',np.sqrt(mean_squared_error(y_val,gb_reg.predict(X_val))))
# we can blend results from different regressors to improve the score

# weight parameters are adjusted manualy

# As you can see score has improved slightly



def blend_pred(X):

    pred = 0.3*rf_reg.predict(X) + 0.7*gb_reg.predict(X)

    return pred



print('Blended model RMSE: ',np.sqrt(mean_squared_error(y_val,blend_pred(X_val))))
# We must prepare test file in the same way as train



test['datetime'] = pd.to_datetime(test['datetime'])

test['day_of_week'] = test.datetime.dt.day_name()

test['hour'] = test.datetime.dt.hour

test['month'] = test.datetime.dt.month

test['temp'] = (test['temp']+test['atemp'])/2

test = test.drop(['datetime','atemp'],axis=1)

# then we use the same transformer and predict values

X_test = trans.transform(test)



# dont forget to transform predicted values with np.exp

pred = np.expm1(rf_reg.predict(X_test)).round()

sub = pd.DataFrame({'datetime':time,'count':pred})

sub.to_csv('sub.csv',index = False)