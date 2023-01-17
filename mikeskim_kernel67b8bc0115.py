import numpy as np

import pandas as pd

from datetime import timedelta, datetime

from collections import defaultdict

from xgboost import XGBRegressor



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



PREFIX = "/kaggle/input/covid19-global-forecasting-week-3/"

# Any results you write to the current directory are saved as output.


train_df = pd.read_csv(PREFIX+"train.csv", parse_dates=["Date"])

test_df = pd.read_csv(PREFIX+"test.csv", parse_dates=["Date"])

sub_df = pd.read_csv(PREFIX+"submission.csv")





# In[3]:





LAG_DAYS = 28



N_TREES = 1000



MIN_DATE_TRAIN = train_df["Date"].min()

MID_DATE_TRAIN = train_df["Date"].min()  + timedelta(days=LAG_DAYS)

MAX_DATE_TRAIN = train_df["Date"].max()

START_DATE_TRAIN = MID_DATE_TRAIN + timedelta(days=1)

SPLIT_DATE_TRAIN = START_DATE_TRAIN + timedelta(days=35)

END_DATE_TRAIN = SPLIT_DATE_TRAIN + timedelta(days=8)



MIN_DATE_TEST = test_df["Date"].min()

MAX_DATE_TEST = test_df["Date"].max()

print(MIN_DATE_TRAIN)

print(MID_DATE_TRAIN)

print(END_DATE_TRAIN)

print(MAX_DATE_TRAIN)

print(MIN_DATE_TEST)

print(MAX_DATE_TEST)

print('done')





# In[4]:





both_df = pd.concat([train_df,test_df[test_df["Date"]>MAX_DATE_TRAIN]],axis=0)

print(train_df.shape)

print(test_df.shape)

print(both_df.shape)

both_df.head(3)





# In[5]:





oof_df = train_df[train_df["Date"]<MID_DATE_TRAIN]

def create_oof_dict(oof_df):

    first_death_list = []

    first_case_list = []

    country_list = oof_df["Country_Region"].unique().tolist()

    for country in country_list:

        df = oof_df[oof_df['Country_Region']==country]

        df1 = df[df["ConfirmedCases"]>0]

        if df1.shape[0] > 0:

            first_case_list.append(df1["Date"].min())

        else:

            first_case_list.append(MAX_DATE_TEST)

        df2 = df[df["Fatalities"]>0]

        if df2.shape[0] > 0:

            first_death_list.append(df2["Date"].min())

        else:

            first_death_list.append(MAX_DATE_TEST)

        



    first_df = pd.DataFrame()

    first_df['country'] = country_list

    first_df['first_death_date'] = first_death_list

    first_df['first_case_date'] = first_case_list



    first_df['first_death_date'] = (first_df['first_death_date']-MIN_DATE_TRAIN).dt.days

    first_df['first_case_date'] = (first_df['first_case_date']-MIN_DATE_TRAIN).dt.days



    death_map = dict(zip(first_df['country'].tolist(),first_df['first_death_date'].tolist()))

    case_map = dict(zip(first_df['country'].tolist(),first_df['first_case_date'].tolist()))

    return case_map, death_map





case_map, death_map = create_oof_dict(oof_df)

print("done")





# In[6]:









def replace_state_nan(df):

    df = df.copy()

    bool_index = df['Province_State'].isnull()

    df.loc[bool_index, 'Province_State'] = df.loc[bool_index,'Country_Region'] + "_NaN"

    return df



def create_days(df):

    df = df.copy()

    df['Days'] = (df["Date"]-MIN_DATE_TRAIN).dt.days.astype(int)

    

    df['weekday'] = df["Date"].dt.dayofweek

    

    return df



def create_CFR(df):

    df = df.copy()

    df["CFR"] = df["Fatalities"]/df["ConfirmedCases"]

    df["CFR"].fillna(0, inplace=True)

    return df



korea_dummy_dict = defaultdict(int)

korea_dummy_dict['Korea, South'] =  1



hubei_dummy_dict = defaultdict(int)

hubei_dummy_dict['Hubei'] =  1



iran_dummy_dict = defaultdict(int)

iran_dummy_dict['Iran'] =  1



us_dummy_dict = defaultdict(int)

us_dummy_dict['US'] =  1



italy_dummy_dict = defaultdict(int)

italy_dummy_dict['Italy'] =  1





COUNTRY_VC = train_df["Country_Region"].value_counts()

def hash_country(df):

    df = df.copy()

    df['country_hash'] = pd.util.hash_array(df['Country_Region'])

    df['country_hash'] = df['country_hash'].rank(pct=True)

    

    df['country_count'] = df['Country_Region'].map(COUNTRY_VC)

    

    df['state_hash'] = pd.util.hash_array(df['Province_State'])

    df['state_hash'] = df['Province_State'].rank(pct=True)



    df['first_death'] = df["Country_Region"].map(death_map)

    df['first_case'] = df["Country_Region"].map(case_map)

    

    df['dummy_korea'] = df["Country_Region"].map(korea_dummy_dict)

    df['dummy_hubei'] = df["Province_State"].map(hubei_dummy_dict)

    df['dummy_iran'] = df["Country_Region"].map(iran_dummy_dict)

    df['dummy_italy'] = df["Country_Region"].map(italy_dummy_dict)

    

    return df







def create_features(df):

    df = replace_state_nan(df)

    df = hash_country(df)

    df = create_days(df)

    #df = create_CFR(df)

    return df







# In[7]:





both_df = create_features(both_df)

both_df['state_date_key'] = both_df["Province_State"].astype(str) + "_" + both_df["Date"].astype(str)



case_dict = dict(zip(both_df['state_date_key'].tolist(), both_df['ConfirmedCases'].tolist()))

fatal_dict = dict(zip(both_df['state_date_key'].tolist(), both_df['Fatalities'].tolist()))



both_df.head(3)





# In[8]:





target_to_features_dict = {}

STATIC_FEATURES_LIST = ["country_hash","country_count","state_hash",

                        "first_death",

                        "first_case",

                        "Days","weekday",

                        "dummy_korea","dummy_hubei","dummy_iran"]

province_list = both_df["Province_State"].unique().tolist()

date_array =  both_df[(both_df["Date"]>START_DATE_TRAIN) & (both_df["Date"]<=END_DATE_TRAIN)]["Date"].unique()



both_df["CFR"] = both_df["Fatalities"]/both_df["ConfirmedCases"]

both_df["CFR"].fillna(0, inplace=True)





def update_target_dict(both_df, current_date, current_province_state):

        features_list = []

        target_df = both_df[(both_df["Province_State"]==current_province_state) & (both_df["Date"] == current_date)]

        cases = target_df["ConfirmedCases"].values[0]

        fatals = target_df["Fatalities"].values[0]

        current_key = target_df["state_date_key"].values[0]

        

        prior_df = both_df[(both_df["Province_State"]==current_province_state) & 

                           (both_df["Date"] < current_date)].copy()

        prior_df.sort_values("Date",inplace=True)

        prior_df = prior_df.tail(LAG_DAYS).copy()

        prior_df.reset_index(drop=True, inplace=True)

        



        features_list += [prior_df["Fatalities"].max()]

        features_list += [prior_df["ConfirmedCases"].max()]

        features_list += [prior_df["CFR"].max()]        

        features_list += [(prior_df["Fatalities"]*prior_df["ConfirmedCases"]).max()] # good

        features_list += [prior_df["Fatalities"].diff().abs().max()]

        features_list += [prior_df["ConfirmedCases"].diff().abs().max()]

        features_list += [prior_df["CFR"].diff().abs().max()]



        features_list += target_df[STATIC_FEATURES_LIST].values.tolist()[0]



        for f in ["ConfirmedCases","Fatalities","CFR"]: 

            features_list += prior_df[f].tolist()

    

        features_list += (-prior_df["ConfirmedCases"]+prior_df["Fatalities"]).tolist()



        target_to_features_dict[(current_province_state, current_date, current_key)] = [cases,fatals,features_list]







for current_province_state in province_list:

    for current_date in date_array:

        update_target_dict(both_df, current_date, current_province_state)



print("done")





# In[9]:





def list_to_matrix(x):

    return np.array([np.array(xi) for xi in x])



def create_train_X_y(target_to_features_dict):

    X_train = []

    y_key = []

    y_train_case = []

    y_train_fatal = []

    for k,v in target_to_features_dict.items():

        X_train.append(v[2])

        y_train_case.append(v[0])

        y_train_fatal.append(v[1])

        y_key.append(k[2])

    X_train = list_to_matrix(X_train)

    return X_train, y_train_case, y_train_fatal, y_key



X_train, y_train_case, y_train_fatal, y_key = create_train_X_y(target_to_features_dict)

print(X_train.shape)

print("done")





# In[10]:





def create_test_X_y(target_to_features_dict, date_array_0):

    X_test = []

    y_key = []

    y_test_case = []

    y_test_fatal = []

    for k,v in target_to_features_dict.items():

        if k[1] == date_array_0: #date_array[0]:

            X_test.append(v[2])

            y_test_case.append(v[0])

            y_test_fatal.append(v[1])

            y_key.append(k[2])

    X_test = list_to_matrix(X_test)

    return X_test, y_test_case, y_test_fatal, y_key





# In[11]:





np.random.seed(1)



def rmsle(y_true, y_pred):

    return 'RMSLE', np.sqrt(np.mean(np.power(np.log1p(y_pred) - np.log1p(y_true), 2))), False



num_features = X_train.shape[1]

mc = ["0"]*(num_features-4)

MC_STR = "(1,1,1,1," + ",".join(mc)  + ")" 



def xgb_reg(MC_STR,N_TREES):

    return XGBRegressor(n_estimators=N_TREES, 

                   booster="gbtree", #"dart",

#                   monotone_constraints = MC_STR, 

                   tree_method = "exact",

                   max_depth=6,

                   subsample=0.99,

                   colsample_bytree=0.99,

                   min_child_weight = 1,

#                   one_drop = 1,

#                   rate_drop = 0.1, 

                   objective = "reg:squarederror",

                   reg_alpha=0.1)





def dual_xgb_fit(X_train, y_train_case, y_train_fatal):

    gbm_case = xgb_reg(MC_STR,N_TREES)

    gbm_case.fit(X_train, np.log1p(y_train_case))



    gbm_fatal = xgb_reg(MC_STR,N_TREES)

    gbm_fatal.fit(X_train, np.log1p(y_train_fatal))

    return gbm_case, gbm_fatal

    



gbm_case, gbm_fatal = dual_xgb_fit(X_train, y_train_case, y_train_fatal)





print("done")





# In[12]:





#date_array =  both_df[(both_df["Date"]>START_DATE_TRAIN) & (both_df["Date"]<END_DATE_TRAIN)]["Date"].unique()

print(both_df["state_date_key"].nunique())

print(both_df.shape)





# In[13]:





print(both_df["Date"][both_df["Fatalities"].isnull()].min())





# In[14]:





date_array =  [both_df["Date"][both_df["Fatalities"].isnull()].min()]

print(date_array)

both_df["CFR"] = both_df["Fatalities"]/both_df["ConfirmedCases"]

both_df["CFR"].fillna(0, inplace=True)



for current_province_state in province_list:

    for current_date in date_array:

        update_target_dict(both_df, current_date, current_province_state)



print("done")





# In[15]:





X_test, y_test_case, y_test_fatal, y_key = create_test_X_y(target_to_features_dict, date_array[0])

print(X_test.shape)





# In[16]:





y_pred_case = np.expm1(gbm_case.predict(X_test))

y_pred_fatal = np.expm1(gbm_fatal.predict(X_test))

print("done")





# In[17]:





for j in range(len(y_key)):

    key_val = y_key[j]

    case_dict[key_val] = y_pred_case[j]  

    fatal_dict[key_val] = y_pred_fatal[j]

    

print("done")





# In[18]:





both_df["ConfirmedCases"] = both_df["state_date_key"].map(case_dict)

both_df["Fatalities"] = both_df["state_date_key"].map(fatal_dict)



both_df["CFR"] = both_df["Fatalities"]/both_df["ConfirmedCases"]

both_df["CFR"].fillna(0, inplace=True)



print("done")





# In[19]:





# repeat loop



#date_array =  [both_df["Date"][both_df["Fatalities"].isnull()].min()]

# keep prior date, now filled with prediction

print(date_array)

both_df["CFR"] = both_df["Fatalities"]/both_df["ConfirmedCases"]

both_df["CFR"].fillna(0, inplace=True)



for current_province_state in province_list:

    for current_date in date_array:

        update_target_dict(both_df, current_date, current_province_state)



print("done")





# In[20]:





X_train, y_train_case, y_train_fatal, y_key = create_train_X_y(target_to_features_dict)

print(X_train.shape)

print("done")





# In[21]:







gbm_case, gbm_fatal = dual_xgb_fit(X_train, y_train_case, y_train_fatal)

print("done")





# In[22]:





date_array =  [both_df["Date"][both_df["Fatalities"].isnull()].min()]

print(date_array)

both_df["CFR"] = both_df["Fatalities"]/both_df["ConfirmedCases"]

both_df["CFR"].fillna(0, inplace=True)



for current_province_state in province_list:

    for current_date in date_array:

        update_target_dict(both_df, current_date, current_province_state)



print("done")





# In[23]:





X_test, y_test_case, y_test_fatal, y_key = create_test_X_y(target_to_features_dict, date_array[0])

print(X_test.shape)





# In[24]:





y_pred_case = np.expm1(gbm_case.predict(X_test))

y_pred_fatal = np.expm1(gbm_fatal.predict(X_test))

print("done")





# In[25]:





for j in range(len(y_key)):

    key_val = y_key[j]

    case_dict[key_val] = y_pred_case[j]  

    fatal_dict[key_val] = y_pred_fatal[j]

    

print(len(case_dict.keys()))

print("done")





# In[26]:





both_df["ConfirmedCases"] = both_df["state_date_key"].map(case_dict)

both_df["Fatalities"] = both_df["state_date_key"].map(fatal_dict)



#both_df["CFR"] = both_df["Fatalities"]/both_df["ConfirmedCases"]

#both_df["CFR"].fillna(0, inplace=True)



print("done")





# In[ ]:





null_count = both_df["Fatalities"].isnull().sum()

print(null_count)



while null_count > 0:

    print(null_count)

    

    both_df["CFR"] = both_df["Fatalities"]/both_df["ConfirmedCases"]

    both_df["CFR"].fillna(0, inplace=True)

    

    for current_province_state in province_list:

        for current_date in date_array:

            update_target_dict(both_df, current_date, current_province_state)





    X_train, y_train_case, y_train_fatal, y_key = create_train_X_y(target_to_features_dict)

    

    gbm_case, gbm_fatal = dual_xgb_fit(X_train, y_train_case, y_train_fatal)







    date_array =  [both_df["Date"][both_df["Fatalities"].isnull()].min()]

    print(date_array)



    for current_province_state in province_list:

        for current_date in date_array:

            update_target_dict(both_df, current_date, current_province_state)



    X_test, y_test_case, y_test_fatal, y_key = create_test_X_y(target_to_features_dict, date_array[0])

    print(X_test.shape)



    y_pred_case = np.expm1(gbm_case.predict(X_test))

    y_pred_fatal = np.expm1(gbm_fatal.predict(X_test))



    for j in range(len(y_key)):

        key_val = y_key[j]

        case_dict[key_val] = y_pred_case[j]  

        fatal_dict[key_val] = y_pred_fatal[j]



    print(len(case_dict.keys()))





    both_df["ConfirmedCases"] = both_df["state_date_key"].map(case_dict)

    both_df["Fatalities"] = both_df["state_date_key"].map(fatal_dict)

    null_count = both_df["Fatalities"].isnull().sum()



print("done")





# In[ ]:





both_df.tail(10)





# In[ ]:





test_df = pd.read_csv(PREFIX+"test.csv", parse_dates=["Date"])

test_df = replace_state_nan(test_df)

test_df.head(3)

print(test_df.shape)

merge_df = both_df[["Province_State","Date","ConfirmedCases","Fatalities"]].copy()

merge_df.sort_values(["Province_State","Date"], inplace=True)

print(merge_df.head(5))

merge_df["ConfirmedCases"] = merge_df.groupby('Province_State')["ConfirmedCases"].cummax()

merge_df["Fatalities"] = merge_df.groupby('Province_State')["Fatalities"].cummax()

test_df = test_df.merge(merge_df, how="inner",

                        on=["Province_State","Date"])

print(test_df.shape)





# In[ ]:





test_df[["ForecastId","ConfirmedCases","Fatalities"]].to_csv("submission.csv", index=False)

print("done")





# In[ ]:





# END HERE

#both_df["Province_State"].unique()





# In[ ]: