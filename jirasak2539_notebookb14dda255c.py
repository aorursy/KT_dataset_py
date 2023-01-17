# import library

import numpy as np

import pandas as pd

from sklearn.preprocessing import LabelEncoder
# import dataset



# From plant 1

# Generation

df1_1 = pd.read_csv(r'../input/solar-power-generation-data/Plant_1_Generation_Data.csv')

# Weather

df1_2 = pd.read_csv(r'../input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv')



# From plant 2

# Generation

df2_1 = pd.read_csv(r'../input/solar-power-generation-data/Plant_2_Generation_Data.csv')

# Weather

df2_2 = pd.read_csv(r'../input/solar-power-generation-data/Plant_2_Weather_Sensor_Data.csv')
# Convert the date "DATE_TIME" column to proper datetime object

# match to the proper format for each data

df1_1['DATE_TIME'] = pd.to_datetime(df1_1['DATE_TIME'],format='%d-%m-%Y %H:%M')



df2_1['DATE_TIME'] = pd.to_datetime(df2_1['DATE_TIME'],format='%Y-%m-%d %H:%M:%S')

df1_2['DATE_TIME'] = pd.to_datetime(df1_2['DATE_TIME'],format='%Y-%m-%d %H:%M:%S')

df2_2['DATE_TIME'] = pd.to_datetime(df2_2['DATE_TIME'],format='%Y-%m-%d %H:%M:%S')
# merge df1_1 and df1_2 together

df1 = pd.merge(df1_1,df1_2,left_on = 'DATE_TIME',right_on = 'DATE_TIME',how = 'outer')

# merge df2_1 and df2_2 together

df2 = pd.merge(df2_1,df2_2,left_on = 'DATE_TIME',right_on = 'DATE_TIME',how = 'outer')



df = pd.concat([df1,df2])

# remove an unnecessary and duplicate column

df = df.drop(['PLANT_ID_y','SOURCE_KEY_y'],axis = 1)



# rename some column back

df = df.rename(columns = {'PLANT_ID_x':'PLANT_ID',

                    'SOURCE_KEY_x':'SOURCE_KEY'})



# reset index back

df = df.reset_index().drop(['index'],axis = 1)
# remove the hour and minute from the data

df['DATE'] = df['DATE_TIME'].dt.strftime('%Y-%m-%d')



# groupby the DataFrame, summarise the feature to be day by day

groupby_df = df.groupby(['DATE','SOURCE_KEY']).agg({'DAILY_YIELD':'max',

                                       'DC_POWER':'sum',

                                       'AC_POWER':'sum',

                                       'AMBIENT_TEMPERATURE':'mean',

                                       'MODULE_TEMPERATURE':'mean',

                                       'IRRADIATION':'mean',

                                        'PLANT_ID':'mean'})



groupby_df
# change groupby object back to DataFrame

groupby_df = groupby_df.reset_index()



# rearrange the column

groupby_df = groupby_df[['DATE', 'SOURCE_KEY', 'PLANT_ID', 'DAILY_YIELD', 'DC_POWER', 'AC_POWER',

       'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE']]
# change "DATE" column into datetime object

groupby_df['DATE'] = pd.to_datetime(groupby_df['DATE'],format = '%Y-%m-%d')

# set the "DATE" column to be index

groupby_df = groupby_df.set_index(['DATE'])
# change "PLANT_ID" to int object

groupby_df['PLANT_ID'] = groupby_df['PLANT_ID'].astype('int')



# change "SOURCE_KEY" to categorical object

le = LabelEncoder()

groupby_df['SOURCE_KEY'] = le.fit_transform(groupby_df['SOURCE_KEY'])
groupby_df
# define a function to create dataset with past information

def create_prev_day_column(dataset,day_num):

    delta = pd.Timedelta(1,"D")

    original_copy = dataset.copy()

    for day in range(1,day_num+1):

        prev_data = original_copy.copy()

        prev_data = prev_data.shift(day,freq = 'D')

        dataset = pd.merge(dataset,prev_data,how = 'left',on = ["DATE","SOURCE_KEY","PLANT_ID"],suffixes=['','_' + str(day)])

        dataset.replace(np.nan,0,inplace = True)

    

    return dataset



# define a function to create a dictionary for K fold validation

def KFold(dataset,fold_num=10):

    datasets = {}

    n = len(dataset)

    for i in range(fold_num):

        datasets[i] = dataset[i:n:fold_num]

    return datasets
# import additional library

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LinearRegression

from sklearn import metrics
# for 3 days

df = create_prev_day_column(groupby_df,3)



rf_RMSE_score_3day = []

lr_RMSE_score_3day = []

dataset_10_fold = KFold(df)

folds = 10

for fold in range(folds):

    train_data = pd.DataFrame()

    for j in range(folds):

        if j == fold:

            pass

        else:

            train_data = pd.concat([train_data,dataset_10_fold[j]])

    

    test_data = dataset_10_fold[fold]

    

    X_train = train_data.drop(['DAILY_YIELD'],axis = 1)

    Y_train = train_data['DAILY_YIELD']

    

    X_test = test_data.drop(['DAILY_YIELD'],axis = 1)

    Y_test = test_data['DAILY_YIELD']

    

    rf = RandomForestRegressor()

    rf.fit(X_train,Y_train)

    Y_pred = rf.predict(X_test)

    rmse_score = np.sqrt(metrics.mean_squared_error(Y_test,Y_pred))

    rf_RMSE_score_3day.append(rmse_score)

    

    lr = LinearRegression()

    lr.fit(X_train,Y_train)

    Y_pred = lr.predict(X_test)

    rmse_score = np.sqrt(metrics.mean_squared_error(Y_test,Y_pred))

    lr_RMSE_score_3day.append(rmse_score)

    
print('The mean of RMSE from 10 fold cross validation for Random Forest model with 3 days past data is  ')

print(np.mean(rf_RMSE_score_3day))



print('\nThe mean of RMSE from 10 fold cross validation for Logistic Regression model with 3 days past data is  ')

print(np.mean(lr_RMSE_score_3day))
# for 7 days

df = create_prev_day_column(groupby_df,7)



rf_RMSE_score_7day = []

lr_RMSE_score_7day = []

dataset_10_fold = KFold(df)

folds = 10

for fold in range(folds):

    train_data = pd.DataFrame()

    for j in range(folds):

        if j == fold:

            pass

        else:

            train_data = pd.concat([train_data,dataset_10_fold[j]])

    

    test_data = dataset_10_fold[fold]

    

    X_train = train_data.drop(['DAILY_YIELD'],axis = 1)

    Y_train = train_data['DAILY_YIELD']

    

    X_test = test_data.drop(['DAILY_YIELD'],axis = 1)

    Y_test = test_data['DAILY_YIELD']

    

    rf = RandomForestRegressor()

    rf.fit(X_train,Y_train)

    Y_pred = rf.predict(X_test)

    rmse_score = np.sqrt(metrics.mean_squared_error(Y_test,Y_pred))

    rf_RMSE_score_7day.append(rmse_score)

    

    lr = LinearRegression()

    lr.fit(X_train,Y_train)

    Y_pred = lr.predict(X_test)

    rmse_score = np.sqrt(metrics.mean_squared_error(Y_test,Y_pred))

    lr_RMSE_score_7day.append(rmse_score)

    
print('The mean of RMSE from 10 fold cross validation for Random Forest model with 7 days past data is  ')

print(np.mean(rf_RMSE_score_7day))



print('\nThe mean of RMSE from 10 fold cross validation for Logistic Regression model with 7 days past data is  ')

print(np.mean(lr_RMSE_score_7day))