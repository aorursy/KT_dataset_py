import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Additional Libraries

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

from xgboost.sklearn import XGBRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import accuracy_score

from keras.callbacks import ModelCheckpoint

from keras.models import Sequential

from keras.layers import Dense, Activation, Flatten

from keras.layers import Dropout

import lightgbm as lgb

from sklearn.metrics import mean_squared_error

from math import sqrt
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
def define_data(data , info=True ,shape = True, percentage =True,describe = True , sample=True , columns = False):

    if columns == True:

        print('\nColumns of Data...')

        print(data.columns)

        return 

    if shape ==True:

        print('Shape of Data is...')

        print(data.shape)

    if info==True:

        print('\nInfo of Data...')

        print(data.info())

    if percentage ==True:

        print('\nPercentage of Data Missing ...')

        print((data.isnull().sum()/data.shape[0])*100)

    if describe == True:

        print('\nDescription of data...')

        display(data.describe())

    if sample == True:

        print('\nSample of Data...')

        display(data.sample(10).T)

    



define_data(train)
define_data(train  , columns = True)

define_data(test  , columns = True)
# Divide DateTime Column to various Columns

def add_dates(data , column , suffix='time_' , year = True , month = True , day = False ,dayofweek = True, hour = True , minute = False  , second = False , date = False , time = False):

    data['add_date_date_time'] = pd.to_datetime(data[column])

    if year == True:

        data[suffix+'year']=data['add_date_date_time'].dt.year

    if month == True:

        data[suffix+'month']=data['add_date_date_time'].dt.month

    if day == True:

        data[suffix+'day']=data['add_date_date_time'].dt.day

    if hour == True:

        data[suffix+'hour']=data['add_date_date_time'].dt.hour

    if minute == True: 

        data[suffix+'minute']=data['add_date_date_time'].dt.minute

    if date == True:

        data[suffix+'date']=data['add_date_date_time'].dt.date

    if time == True:

        data[suffix+'time']=data['add_date_date_time'].dt.time

    if second == True:

        data[suffix+'second']=data['add_date_date_time'].dt.second

    if dayofweek == True:

        data[suffix+'dayofweek']=data['add_date_date_time'].dt.dayofweek

    data = data.drop(columns = ['add_date_date_time'] , axis =1)

    return data

train = add_dates(train , column = 'datetime') 

define_data(train , columns = True)
def unique_count(data , columns = []):

    for col in columns :

        print('Unique Data Percentage in ',col)

        print((data[col].value_counts()/data.shape[0])*100)

        print('\n')

unique_count(train , columns = ['season','weather','time_year', 'time_dayofweek'])
def display_unique_data(data):

    for i in data.columns:

        unique_cols_data = data[i].unique()

        if len(unique_cols_data)<20:

            print('Correct Type on Column -> ',i)

            print('Unique data in this Column is -> ',unique_cols_data)

            print('\n')

display_unique_data(train)
display(train.corr().style.format("{:.2%}").highlight_min())

# f,ax = plt.subplots(figsize=(15, 15))

# sns.heatmap(train.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
def joint_plots(data , col,columns  = []):

    plt.figure(figsize=(16,16))

    for i in columns:

        sns.jointplot( x=col , y=i , data=data , height=10, ratio=3 , color='g')

        plt.show()

joint_plots(train , columns = ['temp' , 'atemp'  ,'humidity' , 'windspeed' ] , col = 'count')
def plot_bar(data, col ,  feature=[]):

    length = len(feature)*4

    plt.figure(figsize=(20,length))

    for i,j in zip(feature,range(1,len(feature)*2-1,2)):

        plt.subplot(10,2,j)

        #fig = plt.figure(figsize=(9,8))

        sns.barplot(x=i, y=col, data=data, palette='Set2',orient='v')

        plt.plot()

        plt.subplot(10,2,j+1)

        sns.boxplot(x=i, y=col, data=data, palette='Set2'  , width=.4)

        plt.plot()
plot_bar(train, col = 'count',feature =['time_hour','time_month','time_dayofweek','time_year','weather', 'holiday' , 'workingday' , 'season' ])
def hour_group(s):

    if((0<=s) & (s<=6)):

        return 1

    elif((s==7) | (s==9)):

        return 2

    elif((s==8) | (s==16) | (s==19)):

        return 3

    elif((10<=s) & (s<=15)):

        return 4

    elif((s==17) | (s==18)):

        return 5

    elif(20<=s):

        return 6

train['time_hour_group'] = train['time_hour'].apply(hour_group).astype(str)

def new_col_categorical(data , columns = [] , remove_original = True):

    for i in columns:

        unique_cols = data[i].unique()

        if len(unique_cols) < 20:

            print('\nCorrect Type on Column -> ',i)

            print('Unique data in this Column is -> ',unique_cols)

        else:

            return data

    if remove_original == False:

        original_data = data[columns]

    data = pd.get_dummies(data , columns = columns)

    if remove_original == False:

        data = pd.concat([data,original_data] , axis=1)

    return data

        
train = new_col_categorical(train,columns=['season','weather','time_year', 'time_dayofweek' , 'time_month','time_hour_group'] , remove_original = False)
# train[train.holiday == train.workingday].sample(10)

# on Saterday and Sunday there is holiday and thats why both are 0

train['weekend'] = train['time_dayofweek_5']+train['time_dayofweek_6'] 

define_data(train, columns = True )

# train_x_new = train.drop(columns =['datetime','count', 'season_1','casual','registered',

#        'season_2', 'season_3', 'season_4', 'weather_1', 'weather_2',

#        'weather_3', 'weather_4', 'time_year_2011', 'time_year_2012',

#        'time_dayofweek_0', 'time_dayofweek_1', 'time_dayofweek_2',

#        'time_dayofweek_3', 'time_dayofweek_4', 'time_dayofweek_5',

#        'time_dayofweek_6', 'time_month_1', 'time_month_2', 'time_month_3',

#        'time_month_4', 'time_month_5', 'time_month_6', 'time_month_7',

#        'time_month_8', 'time_month_9', 'time_month_10', 'time_month_11',

#        'time_month_12'] , axis = 1)

train_x_new = train.drop(columns =['datetime','count', 'casual','registered',

       'season', 'weather', 'time_year',

       'time_dayofweek', 'time_month','time_hour_group'] , axis = 1)

train_y_new = train['count']

define_data(train_x_new, columns = True )
# Processing Test Data

test = add_dates(test , column = 'datetime') 

test['time_hour_group'] = test['time_hour'].apply(hour_group).astype(str)

test = new_col_categorical(test,columns=['season','weather','time_year', 'time_dayofweek' , 'time_month','time_hour_group'] , remove_original = False)

test['weekend'] = test['time_dayofweek_5']+test['time_dayofweek_6'] 

# test_x_new = test.drop(columns =['datetime', 'season_1',

#        'season_2', 'season_3', 'season_4', 'weather_1', 'weather_2',

#        'weather_3', 'weather_4', 'time_year_2011', 'time_year_2012',

#        'time_dayofweek_0', 'time_dayofweek_1', 'time_dayofweek_2',

#        'time_dayofweek_3', 'time_dayofweek_4', 'time_dayofweek_5',

#        'time_dayofweek_6', 'time_month_1', 'time_month_2', 'time_month_3',

#        'time_month_4', 'time_month_5', 'time_month_6', 'time_month_7',

#        'time_month_8', 'time_month_9', 'time_month_10', 'time_month_11',

#        'time_month_12'] , axis = 1)



test_x_new = test.drop(columns =['datetime',

       'season', 'weather', 'time_year',

       'time_dayofweek', 'time_month','time_hour_group'] , axis = 1)
print('For Train Data .. ')

define_data(train_x_new, columns = True )

print('For Test Data .. ')

define_data(test_x_new , columns = True )
scaler = MinMaxScaler()

train_x_new = scaler.fit_transform(train_x_new)

train_y_new = np.log1p(train_y_new)
test_x_new_1 = scaler.transform(test_x_new)

test_x_new_2 = scaler.fit_transform(test_x_new)
X_train , X_test , Y_train , Y_test = train_test_split(train_x_new , train_y_new , test_size = .15 , random_state = 65 )
valid_0_error =0

valid_1_error =0
def score_diff(valid_0_error , valid_1_error , valid_0_error_new , valid_1_error_new):

    if valid_0_error == 0:

        print('First Observaton')

        print('Train Error is : ',valid_0_error_new)

        print('Test Error is : ',valid_1_error_new)

        print('Diffence Between Train and Test is : ',((valid_1_error_new-valid_0_error_new)/valid_0_error_new)*100 ,' %')

    else:

        if valid_0_error_new > valid_0_error:

            print('Train Error is : ',valid_0_error_new)

            print('Test Error is : ',valid_1_error_new)

            print('Train Error have Gone up by  : ',((valid_0_error_new-valid_0_error)/valid_0_error)*100,'%')

            print('Test Error have Gone up by  : ',((valid_1_error_new-valid_1_error)/valid_1_error)*100,'%')

            print('Diffence Between Train and Test is : ',((valid_1_error_new-valid_0_error_new)/valid_0_error_new)*100 ,' %')

            print('Earlier Diffence Between Train and Test is : ',((valid_1_error-valid_0_error)/valid_0_error)*100 ,' %')

        if valid_0_error_new < valid_0_error:

            print('Train Error is : ',valid_0_error_new)

            print('Test Error is : ',valid_1_error_new)

            print('Train Error have Down up by  : ',((valid_0_error-valid_0_error_new)/valid_0_error_new)*100,'%') 

            print('Test Error have Down up by  : ',((valid_1_error-valid_1_error_new)/valid_1_error_new)*100,'%') 

            print('Diffence Between Train and Test is : ',((valid_1_error_new-valid_0_error_new)/valid_0_error_new)*100 ,' %')

            print('Earlier Diffence Between Train and Test is : ',((valid_1_error-valid_0_error)/valid_0_error)*100 ,' %')

        if valid_0_error_new == valid_0_error:

            print('No Differnce in new Obseravtion')

            print('Train Error is : ',valid_0_error_new)

            print('Test Error is : ',valid_1_error_new)

            print('Diffence Between Train and Test is : ',((valid_1_error_new-valid_0_error_new)/valid_0_error_new)*100 ,' %')

train_set = lgb.Dataset(X_train , label = Y_train)

val_set = lgb.Dataset( X_test, label = Y_test)
%%time

params = {

        "objective" : "regression", 

        "metric" : "mae", 

        "num_leaves" : 60, 

        "learning_rate" : 0.01, 

        "bagging_fraction" : 0.9,

        "bagging_seed" : 0, 

        "num_threads" : 4,

        "colsample_bytree" : 0.5, 

        'lambda_l2':9

}



model = lgb.train(  params, 

                    train_set = train_set,

                    num_boost_round=10000,

                    early_stopping_rounds=200,

                    verbose_eval=100, 

                    valid_sets=[train_set,val_set]

                  )



%%time

lgb_pred_test = model.predict(X_test, num_iteration=model.best_iteration)

lgb_pred_train = model.predict(X_train, num_iteration=model.best_iteration)

lgb_pred_normal = model.predict(test_x_new_1, num_iteration=model.best_iteration)

lgb_pred_fit = model.predict(test_x_new_2, num_iteration=model.best_iteration)
# print(lgb_pred)

# print(np.array(Y_test))

valid_0_error_new = sqrt(mean_squared_error(np.array(Y_train),lgb_pred_train))

valid_1_error_new = sqrt(mean_squared_error(np.array(Y_test),lgb_pred_test))

score_diff(valid_0_error , valid_1_error , valid_0_error_new , valid_1_error_new)

valid_0_error = valid_0_error_new

valid_1_error = valid_1_error_new
lgb.plot_importance(model)
%%time

n_estimators=100

xgb = XGBRegressor(n_estimators=n_estimators,max_depth=4,learning_rate =0.01 , booster = 'gbtree')

xgb.fit(X_train ,Y_train ,eval_set=[(X_train, Y_train), (X_test, Y_test)] , verbose = False)

score = xgb.evals_result()

valid_0_error_new = np.amin(score['validation_0']['rmse'])

valid_1_error_new = np.amin(score['validation_1']['rmse'])

score_diff(valid_0_error , valid_1_error , valid_0_error_new , valid_1_error_new)

valid_0_error = valid_0_error_new

valid_1_error = valid_1_error_new
%%time

model = RandomForestRegressor(random_state=65, n_estimators=200, min_samples_split=4)

result = model.fit(X_train, Y_train)
model.score(X_test, Y_test)
%%time

start_point = n_estimators-100

r = range(start_point,n_estimators)

plt.figure(figsize=(16,8))

plt.plot(r , score['validation_0']['rmse'][start_point:]  ,'r' ,label ='Train')

plt.plot(r , score['validation_1']['rmse'][start_point:]  , 'g' , label = 'Test' )

plt.legend(fontsize='x-large')

%%time

n_estimators=3000

xgb = XGBRegressor(n_estimators=n_estimators,max_depth=4,learning_rate =0.01 , booster = 'gbtree')

xgb.fit(train_x_new , train_y_new ,eval_set=[(X_train, Y_train), (X_test, Y_test)] , verbose = False)

pred_normal = xgb.predict(test_x_new_1)

pred_fit = xgb.predict(test_x_new_2)
%%time

model = RandomForestRegressor(random_state=65, n_estimators=n_estimators-2000)

model.fit(train_x_new , train_y_new)

rfr_pred_normal = model.predict(test_x_new_1)

rfr_pred_fit = model.predict(test_x_new_2)
%%time

pred_normal = np.expm1(pred_normal)

pred_fit = np.expm1(pred_fit)

rfr_pred_normal = np.expm1(rfr_pred_normal)

rfr_pred_fit = np.expm1(rfr_pred_fit)
%%time

NN_model = Sequential()



# The Input Layer :

NN_model.add(Dense(128, kernel_initializer='normal',input_dim = train_x_new.shape[1], activation='relu'))



# The Hidden Layers :

NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))

NN_model.add(Dropout(0.3))

NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))

NN_model.add(Dropout(0.3))

NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))

NN_model.add(Dropout(0.3))

# The Output Layer :

NN_model.add(Dense(1, kernel_initializer='normal',activation='linear'))



# Compile the network :

NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])

NN_model.summary()



history = NN_model.fit(train_x_new,train_y_new, epochs=50, batch_size=64,  verbose=1, validation_split=0.2)

start_point = 150-100

r = range(start_point,150)

plt.figure(figsize=(16,8))

plt.plot( history.history['loss']  ,'r' ,label ='Train')

plt.plot( history.history['val_loss']  , 'g' , label = 'Test' )

plt.legend(fontsize='x-large')

valid_0_error_new = history.history['loss'][-1]

valid_1_error_new = history.history['val_loss'][-1]

score_diff(valid_0_error , valid_1_error , valid_0_error_new , valid_1_error_new)

valid_0_error = valid_0_error_new

valid_1_error = valid_1_error_new

ANN_pred_normal = NN_model.predict(test_x_new_1)

ANN_pred_fit = NN_model.predict(test_x_new_2)

ANN_pred_normal = np.expm1(ANN_pred_normal)

ANN_pred_fit = np.expm1(ANN_pred_fit)

ANN_pred_fit = ANN_pred_fit.reshape(6493)

ANN_pred_normal = ANN_pred_normal.reshape(6493)
output = pd.DataFrame({'datetime': test.datetime,'count': pred_normal})

output.to_csv('xgb_pred_normal.csv', index=False)

output = pd.DataFrame({'datetime': test.datetime,'count': pred_fit})

output.to_csv('xgb_pred_fit.csv', index=False)

output = pd.DataFrame({'datetime': test.datetime,'count': rfr_pred_normal})

output.to_csv('rfr_pred_normal.csv', index=False)

output = pd.DataFrame({'datetime': test.datetime,'count': rfr_pred_fit})

output.to_csv('rfr_pred_fit.csv', index=False)

output = pd.DataFrame({'datetime': test.datetime,'count': ANN_pred_normal})

output.to_csv('ANN_pred_normal.csv', index=False)

output = pd.DataFrame({'datetime': test.datetime,'count': ANN_pred_fit})

output.to_csv('ANN_pred_fit.csv', index=False)

output = pd.DataFrame({'datetime': test.datetime,'count': lgb_pred_normal})

output.to_csv('lgb_pred_normal.csv', index=False)

output = pd.DataFrame({'datetime': test.datetime,'count': lgb_pred_fit})

output.to_csv('lgb_pred_fit.csv', index=False)