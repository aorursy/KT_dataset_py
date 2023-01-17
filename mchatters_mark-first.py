# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data = pd.read_csv('../input/covid19-global-forecasting-week-5/train.csv')

test_data = pd.read_csv('../input/covid19-global-forecasting-week-5/test.csv')

sample_data = pd.read_csv('../input/covid19-global-forecasting-week-5/submission.csv')

country_data = pd.read_csv('/kaggle/input/country-data-population/country_data_population.csv')



print("Train shape: ", train_data.shape)

print("Test shape: ", test_data.shape)

train_data = pd.merge(train_data,country_data, left_on=['Country_Region'],right_on=['Country'], how='left')

train_data.drop(['Country','Lending Category','UrbanRuralDesignation','GeoRegion','Continent'], axis=1, inplace=True)



test_data = pd.merge(test_data,country_data, left_on=['Country_Region'],right_on=['Country'], how='left')

test_data.drop(['Country','Lending Category','UrbanRuralDesignation','GeoRegion','Continent'], axis=1, inplace=True)



train_data['GeoSubregion'].fillna('', inplace=True)

test_data['GeoSubregion'].fillna('', inplace=True)



train_data['Income Group'].fillna('', inplace=True)

test_data['Income Group'].fillna('', inplace=True)



train_data['Province_State'].fillna('', inplace=True)

test_data['Province_State'].fillna('', inplace=True)



train_data['County'].fillna('', inplace=True)

test_data['County'].fillna('', inplace=True)
pd.set_option('display.max_rows', None)



#turn Target_Value absolute value to a per 1000000 factor

train_data['TargetValue'] = train_data['TargetValue'].abs()

train_data['TargetValue_per_mill'] = (train_data['TargetValue'] /  train_data['Population']) * 1000000

train_data['TargetValue_cumm'] = train_data.groupby(['Country_Region','Province_State','County','Target'])['TargetValue_per_mill'].cumsum()



train_data['log_TargetValue_cumm'] = np.log2(train_data['TargetValue_cumm']+1)



train_data.head(300)
train_data_short = train_data[train_data['Target']=='ConfirmedCases']

test_data_short  = test_data[test_data['Target']=='ConfirmedCases']



train_data_short = train_data_short[['Country_Region','Date','Province_State','County','Target','TargetValue']]

test_data_short  = test_data_short[['Country_Region','Date','Province_State','County','Target']]



all_data = pd.merge(train_data_short, test_data_short ,how='outer', on=['Country_Region','Date','Province_State','County','Target'])



all_data.sort_values(['Country_Region','Province_State','County','Date','Target'], ascending = True, inplace = True)

all_data = all_data.reset_index(drop=True)



all_data.head(300)

#Populate days_since_confirmedcases - not a very pythonic way of doing this 



all_data['days_since_confirmedcases'] = 0.0

all_data['days_since_max_day']        = 0.0

all_data['TargetValue'].fillna(0.0, inplace=True)



count = 0

save_province_state = ""

save_country_region = ""

save_county         = ""

save_confirmedcases = 0.0

first_confirmedcase = 0

max_confirmedcase   = 0

save_max_count      = 0



for index, row in all_data.iterrows():



    if save_province_state != row['Province_State'] or save_country_region != row['Country_Region'] or save_county != row['County']:

        

        save_province_state = row['Province_State'] 

        save_country_region = row['Country_Region']  

        save_county         = row['County']  

        first_confirmedcase = 0 

        save_confirmedcases = 0

        max_confirmedcase   = 0

        

        if save_max_count > 0:

            all_data['days_since_max_day'][save_max_count] = 1

            

        save_max_count           = 0

        count_within_group       = 0



    if row['Target'] == 'ConfirmedCases':



        if (save_confirmedcases == 0.0) and (row['TargetValue'] > 4.0):

            save_confirmedcases = row['TargetValue']

            first_confirmedcase = count 

        

        if first_confirmedcase > 0:

            all_data['days_since_confirmedcases'][count] = count - first_confirmedcase

            

        if row['TargetValue'] > max_confirmedcase:

            max_confirmedcase = row['TargetValue']

            save_max_count    = count 



        count += 1

    

all_data.head(500)
count = 0

save_province_state = ""

save_country_region = ""

save_county         = ""

save_max_count      = 0



for index, row in all_data.iterrows():



    if save_province_state != row['Province_State'] or save_country_region != row['Country_Region'] or save_county != row['County']:

        

        save_province_state = row['Province_State'] 

        save_country_region = row['Country_Region']  

        save_county         = row['County']  

        save_max_count      = 0



    if row['Target'] == 'ConfirmedCases':



        if row['days_since_max_day'] > 0.0:

            save_max_count   = count 

        

        if save_max_count > 0:

            all_data['days_since_max_day'][count] = count - save_max_count



        count += 1

    

all_data.head(500)
all_data.head(500)
train_data = pd.merge(train_data,all_data[['County','Province_State','Country_Region','Date','days_since_confirmedcases','days_since_max_day']], left_on=['County','Province_State','Country_Region','Date'],right_on=['County','Province_State','Country_Region','Date'], how='left')





test_data = pd.merge(test_data,all_data[['County','Province_State','Country_Region','Date','days_since_confirmedcases','days_since_max_day']], left_on=['County','Province_State','Country_Region','Date'],right_on=['County','Province_State','Country_Region','Date'], how='left')

import gc

del all_data,train_data_short,test_data_short

gc.collect()
from datetime import datetime

#convert Date to number

train_data['Date_dt']   = pd.to_datetime(train_data['Date'])

train_data['Date_days'] = train_data['Date_dt'] - datetime(2019, 1, 1)

train_data['Date_days'] = train_data['Date_days'].apply(lambda x: x.days)



test_data['Date_dt']   = pd.to_datetime(test_data['Date'])

test_data['Date_days'] = pd.to_datetime(test_data['Date_dt']) - datetime(2019, 1, 1)

test_data['Date_days'] = test_data['Date_days'].apply(lambda x: x.days)



train_data.head(10)

print (train_data.dtypes)
from sklearn.preprocessing import LabelEncoder



label_encoder1 = LabelEncoder()

label_encoder2 = LabelEncoder()

label_encoder3 = LabelEncoder()

label_encoder4 = LabelEncoder()

label_encoder5 = LabelEncoder()

label_encoder6 = LabelEncoder()



train_data['Province_State_encode'] = label_encoder1.fit_transform(train_data['Province_State'])

test_data['Province_State_encode'] = label_encoder1.transform(test_data['Province_State'])



train_data['Country_Region_encode'] = label_encoder2.fit_transform(train_data['Country_Region'])

test_data['Country_Region_encode'] = label_encoder2.transform(test_data['Country_Region'])



train_data['GeoSubregion_encode'] = label_encoder3.fit_transform(train_data['GeoSubregion'])

test_data['GeoSubregion_encode'] = label_encoder3.transform(test_data['GeoSubregion'])



train_data['Income Group_encode'] = label_encoder4.fit_transform(train_data['Income Group'])

test_data['Income Group_encode'] = label_encoder4.transform(test_data['Income Group'])



train_data['County_encode'] = label_encoder5.fit_transform(train_data['County'])

test_data['County_encode'] = label_encoder5.transform(test_data['County'])



train_data['Target_encode'] = label_encoder6.fit_transform(train_data['Target'])

test_data['Target_encode'] = label_encoder6.transform(test_data['Target'])



import lightgbm as lgb

import matplotlib as mpl

import matplotlib.pyplot as plt
def process_model(in_features,in_categorical_features,in_target_value,in_train_data,in_test_data):





    build_data = in_train_data[in_train_data.Date_dt < datetime(2020, 4, 20)]

    build_data = build_data[build_data.days_since_confirmedcases > 14]

    val_data   = in_train_data[in_train_data.Date_dt >= datetime(2020, 4, 20)]



    y_train = in_train_data[in_target_value].values

    y_build = build_data[in_target_value].values

    y_val   = val_data[in_target_value].values



    test_ids = in_test_data.ForecastId.astype(int).astype(str).values



    x_train = in_train_data[in_features]

    x_build = build_data[in_features]

    x_val   = val_data[in_features]

    x_test  = in_test_data[in_features]



    print("Build shape: ", build_data.shape)

    print("Val shape: ", val_data.shape)

    print("Train shape: ", in_train_data.shape)

    print("Test shape: ", in_test_data.shape)



    dtrain = lgb.Dataset(x_train, label = y_train, categorical_feature = in_categorical_features)

    dbuild = lgb.Dataset(x_build, label = y_build, categorical_feature = in_categorical_features)

    dval = lgb.Dataset(x_val, label = y_val, categorical_feature = in_categorical_features)



    params = {

    "objective": "regression_l2",

    "num_leaves": 300,

    "learning_rate": 0.013,

    "bagging_fraction": 0.91,

    "feature_fraction": 0.81,

    "reg_alpha": 0.13,

    "reg_lambda": 0.13,

    "metric": "mae",

    "seed": 2357

    }



    model_lgb_val = lgb.train(params, train_set = dbuild, valid_sets = [dval], num_boost_round = 2000, early_stopping_rounds = 100, verbose_eval = 50)

    model_lgb     = lgb.train(params, train_set = dtrain, num_boost_round = model_lgb_val.best_iteration)

    

    y_pred = model_lgb.predict(x_test)



    #feature importance

    df_model = pd.DataFrame({"feature": features, "importance": model_lgb.feature_importance()})



    df_model.sort_values("importance", ascending = False, inplace = True)



    print(df_model)

    

    #dataset

    predictions = pd.DataFrame(y_pred)

    in_test_data['Pred_TargetValue']=predictions.iloc[:, 0].values

    in_test_data['Pred_TargetValue']=in_test_data['Pred_TargetValue'].clip(lower=0)

    

    #Create Original Values

    in_test_data['Pred_TargetValue_total'] = np.power((in_test_data['Pred_TargetValue']),2) - 1

    in_test_data['Pred_TargetValue_total'] = in_test_data['Pred_TargetValue_total'] * (in_test_data['Population'] / 1000000)

    in_test_data['Pred_TargetValue_by_day'] = in_test_data.groupby(['Country_Region','Province_State','County','Target'])['Pred_TargetValue_total'].diff().fillna(0)

    

    all_data_with_pred = pd.merge(in_train_data, in_test_data ,how='outer', on=['Country_Region','Date','Province_State','County','Target'])



    all_data_with_pred.sort_values(['Country_Region','Province_State','County','Date','Target'], ascending = True, inplace = True)

    all_data_with_pred = all_data_with_pred.reset_index(drop=True)

    

    return all_data_with_pred



def display_comparison(in_target_value,in_pred_target_value,in_region,in_target,in_df):

    plot_df = in_df.query("Country_Region=='"+in_region+"' and Target=='"+in_target+"'")

    

    plt.plot(plot_df[in_target_value].values)

    plt.plot(plot_df[in_pred_target_value].values)

    plt.title("Comparison between the actual data and our predictions for the number of cases")

    plt.ylabel('Number of cases')

    plt.xlabel('Date')

    plt.xticks(range(len(plot_df.Date.values)),plot_df.Date.values,rotation='vertical')

    plt.legend(['Groundtruth', 'Prediction'], loc='best')

    plt.show()

categorical_features = ['County_encode','Province_State_encode','Country_Region_encode','GeoSubregion_encode','Income Group_encode','Target_encode']

features = ['County_encode','Province_State_encode','Country_Region_encode','GeoSubregion_encode','Income Group_encode','Target_encode','days_since_confirmedcases','days_since_max_day','Population','Population Density']



target_value_to_predict = 'log_TargetValue_cumm'
#eval_train_data = train_data.query("Province_State=='' and Target=='ConfirmedCases'")

#eval_test_data  = test_data.query("Province_State=='' and Target=='ConfirmedCases'")



#eval_all_data_with_pred = process_model(features,categorical_features,target_value_to_predict,eval_train_data,eval_test_data)

#eval_all_data_with_pred.to_csv('temp_out.csv', index = False)



#display_comparison(target_value_to_predict,'Pred_TargetValue','Germany','ConfirmedCases',eval_all_data_with_pred)

#display_comparison('TargetValue','Pred_TargetValue_by_day','Germany','ConfirmedCases',eval_all_data_with_pred)



#display_comparison(target_value_to_predict,'Pred_TargetValue','United Kingdom','ConfirmedCases',eval_all_data_with_pred)

#display_comparison('TargetValue','Pred_TargetValue_by_day','United Kingdom','ConfirmedCases',eval_all_data_with_pred)



#display_comparison(target_value_to_predict,'Pred_TargetValue','New Zealand','ConfirmedCases',eval_all_data_with_pred)

#display_comparison('TargetValue','Pred_TargetValue_by_day','New Zealand','ConfirmedCases',eval_all_data_with_pred)





#display_comparison(target_value_to_predict,'Pred_TargetValue','Russia','ConfirmedCases',eval_all_data_with_pred)

#display_comparison('TargetValue','Pred_TargetValue_by_day','Russia','ConfirmedCases',eval_all_data_with_pred)





target_value_to_predict = 'TargetValue'
eval_train_data = train_data.query("Province_State=='' and Target=='ConfirmedCases'")

eval_test_data  = test_data.query("Province_State=='' and Target=='ConfirmedCases'")



eval_all_data_with_pred1 = process_model(features,categorical_features,target_value_to_predict,eval_train_data,eval_test_data)

eval_all_data_with_pred1.to_csv('temp_out1.csv', index = False)



display_comparison(target_value_to_predict,'Pred_TargetValue','Germany','ConfirmedCases',eval_all_data_with_pred1)



display_comparison(target_value_to_predict,'Pred_TargetValue','United Kingdom','ConfirmedCases',eval_all_data_with_pred1)



display_comparison(target_value_to_predict,'Pred_TargetValue','New Zealand','ConfirmedCases',eval_all_data_with_pred1)





display_comparison(target_value_to_predict,'Pred_TargetValue','Russia','ConfirmedCases',eval_all_data_with_pred1)





eval_train_data = train_data.query("Province_State!='' and Target=='ConfirmedCases'")

eval_test_data  = test_data.query("Province_State!='' and Target=='ConfirmedCases'")



eval_all_data_with_pred2 = process_model(features,categorical_features,target_value_to_predict,eval_train_data,eval_test_data)

eval_all_data_with_pred2.to_csv('temp_out2.csv', index = False)



display_comparison(target_value_to_predict,'Pred_TargetValue','Australia','ConfirmedCases',eval_all_data_with_pred2)



display_comparison(target_value_to_predict,'Pred_TargetValue','China','ConfirmedCases',eval_all_data_with_pred2)



eval_train_data = train_data.query("Province_State=='' and Target=='Fatalities'")

eval_test_data  = test_data.query("Province_State=='' and Target=='Fatalities'")



eval_all_data_with_pred3 = process_model(features,categorical_features,target_value_to_predict,eval_train_data,eval_test_data)

eval_all_data_with_pred3.to_csv('temp_out3.csv', index = False)



display_comparison(target_value_to_predict,'Pred_TargetValue','Germany','Fatalities',eval_all_data_with_pred3)



display_comparison(target_value_to_predict,'Pred_TargetValue','United Kingdom','Fatalities',eval_all_data_with_pred3)



display_comparison(target_value_to_predict,'Pred_TargetValue','New Zealand','Fatalities',eval_all_data_with_pred3)





display_comparison(target_value_to_predict,'Pred_TargetValue','Russia','Fatalities',eval_all_data_with_pred3)





eval_train_data = train_data.query("Province_State!='' and Target=='Fatalities'")

eval_test_data  = test_data.query("Province_State!='' and Target=='Fatalities'")



eval_all_data_with_pred4 = process_model(features,categorical_features,target_value_to_predict,eval_train_data,eval_test_data)

eval_all_data_with_pred4.to_csv('temp_out4.csv', index = False)



display_comparison(target_value_to_predict,'Pred_TargetValue','Australia','Fatalities',eval_all_data_with_pred4)



display_comparison(target_value_to_predict,'Pred_TargetValue','China','Fatalities',eval_all_data_with_pred4)





eval_all_data_with_pred = pd.concat([eval_all_data_with_pred1,eval_all_data_with_pred2,eval_all_data_with_pred3,eval_all_data_with_pred4], axis=0)



prediction_data = eval_all_data_with_pred.loc[eval_all_data_with_pred['ForecastId'].notnull(),['Country_Region','Province_State','County','Date','Target','ForecastId', 'Pred_TargetValue' ]]

prediction_data.to_csv('prediction_data.csv', index = False)

## submission

#df_pred_q05 = pd.DataFrame({"ForecastId_Quantile": test_ids + "_0.05", "TargetValue": 0.85 * y_pred})



df_pred_q05 = prediction_data[['ForecastId','Pred_TargetValue']]

df_pred_q05['ForecastId'] = df_pred_q05['ForecastId'].astype(int).astype(str) + "_0.05"

df_pred_q05['Pred_TargetValue'] = df_pred_q05['Pred_TargetValue'].values * 0.85



df_pred_q50 = prediction_data[['ForecastId','Pred_TargetValue']]

df_pred_q50['ForecastId'] = df_pred_q50['ForecastId'].astype(int).astype(str) + "_0.5"





df_pred_q95 = prediction_data[['ForecastId','Pred_TargetValue']]

df_pred_q95['ForecastId'] = df_pred_q95['ForecastId'].astype(int).astype(str) + "_0.95"

df_pred_q95['Pred_TargetValue'] = df_pred_q95['Pred_TargetValue'].values * 1.15

df_pred_q95['Pred_TargetValue'] = df_pred_q95['Pred_TargetValue'].clip(lower=1)



df_submit = pd.concat([df_pred_q05, df_pred_q50, df_pred_q95])



df_submit.columns = ['ForecastId_Quantile','TargetValue']

print(df_submit.shape)

df_submit.to_csv('submission.csv', index = False)