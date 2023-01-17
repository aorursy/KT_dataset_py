import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

color = sns.color_palette()



import random

random.seed(1556)



from sklearn import model_selection, preprocessing

import xgboost as xgb







pd.options.mode.chained_assignment = None  # default='warn'

pd.set_option('display.max_columns', 500)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



macro_cols = ["balance_trade", "balance_trade_growth", "eurrub", "average_provision_of_build_contract",

"micex_rgbi_tr", "micex_cbi_tr", "deposits_rate", "mortgage_value", "mortgage_rate",

"income_per_cap", "rent_price_4+room_bus", "museum_visitis_per_100_cap", "apartment_build"]



train_df = pd.read_csv("../input/train.csv", parse_dates=['timestamp'])

test_df = pd.read_csv("../input/test.csv", parse_dates=['timestamp'])

macro_df = pd.read_csv("../input/macro.csv", parse_dates=['timestamp'], usecols=['timestamp'] + macro_cols)

train_df = pd.merge(train_df, macro_df, how='left', on='timestamp')

test_df = pd.merge(test_df, macro_df, how='left', on='timestamp')

print(train_df.shape, test_df.shape)



# truncate the extreme values in price_doc #

ulimit = np.percentile(train_df.price_doc.values, 99)

llimit = np.percentile(train_df.price_doc.values, 1)

train_df['price_doc'].ix[train_df['price_doc']>ulimit] = ulimit

train_df['price_doc'].ix[train_df['price_doc']<llimit] = llimit





for f in train_df.columns:

    if train_df[f].dtype=='object':

        #print(f)

        lbl = preprocessing.LabelEncoder()

        lbl.fit(list(train_df[f].values.astype('str')) + list(test_df[f].values.astype('str')))

        train_df[f] = lbl.transform(list(train_df[f].values.astype('str')))

        test_df[f] = lbl.transform(list(test_df[f].values.astype('str')))

        

#print("label encoder...")



# year and month #

train_df["yearmonth"] = train_df["timestamp"].dt.year*100 + train_df["timestamp"].dt.month

test_df["yearmonth"] = test_df["timestamp"].dt.year*100 + test_df["timestamp"].dt.month



# year and week #

train_df["yearweek"] = train_df["timestamp"].dt.year*100 + train_df["timestamp"].dt.weekofyear

test_df["yearweek"] = test_df["timestamp"].dt.year*100 + test_df["timestamp"].dt.weekofyear



# year #

train_df["year"] = train_df["timestamp"].dt.year

test_df["year"] = test_df["timestamp"].dt.year



# month of year #

train_df["month_of_year"] = train_df["timestamp"].dt.month

test_df["month_of_year"] = test_df["timestamp"].dt.month



# week of year #

train_df["week_of_year"] = train_df["timestamp"].dt.weekofyear

test_df["week_of_year"] = test_df["timestamp"].dt.weekofyear



# day of week #

train_df["day_of_week"] = train_df["timestamp"].dt.weekday

test_df["day_of_week"] = test_df["timestamp"].dt.weekday



train_df["month_year"]= train_df["year"] * 100 + train_df["day_of_week"]

test_df["month_year"]= test_df["year"] * 100 + test_df["day_of_week"]



# ratio of living area to full area #

train_df["ratio_life_sq_full_sq"] = train_df["life_sq"] / np.maximum(train_df["full_sq"].astype("float"),1)

test_df["ratio_life_sq_full_sq"] = test_df["life_sq"] / np.maximum(test_df["full_sq"].astype("float"),1)

train_df["ratio_life_sq_full_sq"].ix[train_df["ratio_life_sq_full_sq"]<0] = 0

train_df["ratio_life_sq_full_sq"].ix[train_df["ratio_life_sq_full_sq"]>1] = 1

test_df["ratio_life_sq_full_sq"].ix[test_df["ratio_life_sq_full_sq"]<0] = 0

test_df["ratio_life_sq_full_sq"].ix[test_df["ratio_life_sq_full_sq"]>1] = 1



# ratio of kitchen area to living area #

train_df["ratio_kitch_sq_life_sq"] = train_df["kitch_sq"] / np.maximum(train_df["life_sq"].astype("float"),1)

test_df["ratio_kitch_sq_life_sq"] = test_df["kitch_sq"] / np.maximum(test_df["life_sq"].astype("float"),1)

train_df["ratio_kitch_sq_life_sq"].ix[train_df["ratio_kitch_sq_life_sq"]<0] = 0

train_df["ratio_kitch_sq_life_sq"].ix[train_df["ratio_kitch_sq_life_sq"]>1] = 1

test_df["ratio_kitch_sq_life_sq"].ix[test_df["ratio_kitch_sq_life_sq"]<0] = 0

test_df["ratio_kitch_sq_life_sq"].ix[test_df["ratio_kitch_sq_life_sq"]>1] = 1



# ratio of kitchen area to full area #

train_df["ratio_kitch_sq_full_sq"] = train_df["kitch_sq"] / np.maximum(train_df["full_sq"].astype("float"),1)

test_df["ratio_kitch_sq_full_sq"] = test_df["kitch_sq"] / np.maximum(test_df["full_sq"].astype("float"),1)

train_df["ratio_kitch_sq_full_sq"].ix[train_df["ratio_kitch_sq_full_sq"]<0] = 0

train_df["ratio_kitch_sq_full_sq"].ix[train_df["ratio_kitch_sq_full_sq"]>1] = 1

test_df["ratio_kitch_sq_full_sq"].ix[test_df["ratio_kitch_sq_full_sq"]<0] = 0

test_df["ratio_kitch_sq_full_sq"].ix[test_df["ratio_kitch_sq_full_sq"]>1] = 1



# floor of the house to the total number of floors in the house #

train_df["ratio_floor_max_floor"] = train_df["floor"] / train_df["max_floor"].astype("float")

test_df["ratio_floor_max_floor"] = test_df["floor"] / test_df["max_floor"].astype("float")



# num of floor from top #

train_df["floor_from_top"] = train_df["max_floor"] - train_df["floor"]

test_df["floor_from_top"] = test_df["max_floor"] - test_df["floor"]



train_df["extra_sq"] = train_df["full_sq"] - train_df["life_sq"]

test_df["extra_sq"] = test_df["full_sq"] - test_df["life_sq"]



train_df["age_of_building"] = train_df["build_year"] - train_df["year"]

test_df["age_of_building"] = test_df["build_year"] - test_df["year"]



train_df["num_room_floor"] = train_df["num_room"] / train_df["floor"]

test_df["num_room_floor"] = test_df["num_room"] / test_df["floor"]



train_df["num_room_full_sq"] = train_df["full_sq"] / train_df["num_room"]

test_df["num_room_full_sq"] = test_df["full_sq"] / test_df["num_room"]



def add_count(df, group_col):

    grouped_df = df.groupby(group_col)["id"].aggregate("count").reset_index()

    grouped_df.columns = [group_col, "count_"+group_col]

    df = pd.merge(df, grouped_df, on=group_col, how="left")

    return df



train_df = add_count(train_df, "yearmonth")

test_df = add_count(test_df, "yearmonth")



train_df = add_count(train_df, "yearweek")

test_df = add_count(test_df, "yearweek")



train_df["apartment"]= train_df["sub_area"]*1000 + train_df["metro_km_avto"].astype("float")

test_df["apartment"]= test_df["sub_area"]*1000 + test_df["metro_km_avto"].astype("float")



train_df["apartment_monthyear"]= train_df["apartment"]* 0.00001 + train_df["month_year"].astype("float")

test_df["apartment_monthyear"]= test_df["apartment"]* 0.00001 + test_df["month_year"].astype("float")



train_df["ratio_preschool"] = train_df["children_preschool"] / train_df["preschool_quota"].astype("float")

test_df["ratio_preschool"] = test_df["children_preschool"] / test_df["preschool_quota"].astype("float")



train_df["ratio_school"] = train_df["children_school"] / train_df["school_quota"].astype("float")

test_df["ratio_school"] = test_df["children_school"] / test_df["school_quota"].astype("float")



train_df["population_density"]= train_df["raion_popul"] / train_df["area_m"].astype("float")

test_df["population_density"]= test_df["raion_popul"] / test_df["area_m"].astype("float")



train_df["young_pop_ratio"]= train_df["young_all"] / train_df["full_all"].astype("float")

test_df["young_pop_ratio"]= test_df["young_all"] / test_df["full_all"].astype("float")



train_df["work_pop_ratio"]= train_df["work_all"] / train_df["full_all"].astype("float")

test_df["work_pop_ratio"]= test_df["work_all"] / test_df["full_all"].astype("float")



train_df["kindergarten_to_greenzone"]= (train_df["kindergarten_km"] - train_df["green_zone_km"]).astype("float") + 1

test_df["kindergarten_to_greenzone"]= (test_df["kindergarten_km"] - test_df["green_zone_km"]).astype("float") + 1



train_df["male_subs"]= train_df["male_f"] / train_df["full_all"].astype("float")

test_df["male_subs"]= test_df["male_f"] / test_df["full_all"].astype("float")



train_df["female_subs"]= train_df["female_f"] / train_df["full_all"].astype("float")

test_df["female_subs"]= test_df["female_f"] / test_df["full_all"].astype("float")



train_df["higher_educ"]= train_df["university_top_20_raion"] / train_df["additional_education_raion"].astype("float")

test_df["higher_educ"]= test_df["university_top_20_raion"] / test_df["additional_education_raion"].astype("float")



train_df["higher_educ_"]= train_df["university_km"] / train_df["additional_education_km"].astype("float")

test_df["higher_educ_"]= test_df["university_km"] / test_df["additional_education_km"].astype("float")



train_df["rail_to_kindergaten"]= (train_df["railroad_km"] - train_df["kindergarten_km"].astype("float")) +1

test_df["rail_to_kindergaten"]= (test_df["railroad_km"] - test_df["kindergarten_km"].astype("float")) +1



train_df["rail_to_kindergaten_ration"]= (train_df["railroad_km"] / train_df["kindergarten_km"].astype("float")) +1

test_df["rail_to_kindergaten_ration"]= (test_df["railroad_km"] / test_df["kindergarten_km"].astype("float")) +1



train_df["rail_to_uni"]= (train_df["railroad_km"] - train_df["university_km"].astype("float")) +1

test_df["rail_to_uni"]= (test_df["railroad_km"] - test_df["university_km"].astype("float")) +1



train_df["rail_to_greenzone"]= (train_df["railroad_km"] - train_df["green_zone_km"]).astype("float") + 1

test_df["rail_to_greenzone"]= (test_df["railroad_km"] - test_df["green_zone_km"]).astype("float") + 1



train_df["rail_to_greenzone_ration"]= (train_df["railroad_km"] / train_df["green_zone_km"]).astype("float") + 1

test_df["rail_to_greenzone_ration"]= (test_df["railroad_km"] / test_df["green_zone_km"]).astype("float") + 1



train_df["rail_to_park"]= (train_df["railroad_km"] - train_df["park_km"]).astype("float") + 1

test_df["rail_to_park"]= (test_df["railroad_km"] - test_df["park_km"]).astype("float") + 1



train_df["rail_to_park"]= (train_df["railroad_km"] - train_df["park_km"]).astype("float") + 1

test_df["rail_to_park"]= (test_df["railroad_km"] - test_df["park_km"]).astype("float") + 1



train_df["rail_to_ice"]= (train_df["railroad_km"] - train_df["ice_rink_km"]).astype("float") + 1

test_df["rail_to_ice"]= (test_df["railroad_km"] - test_df["ice_rink_km"]).astype("float") + 1



train_df["rail_to_stadium"]= (train_df["railroad_km"] - train_df["stadium_km"]).astype("float") + 1

test_df["rail_to_stadium"]= (test_df["railroad_km"] - test_df["stadium_km"]).astype("float") + 1



train_df["rail_to_church"]= (train_df["railroad_km"] - train_df["big_church_km"]).astype("float") + 1

test_df["rail_to_church"]= (test_df["railroad_km"] - test_df["big_church_km"]).astype("float") + 1



train_df["rail_to_shop"]= (train_df["railroad_km"] - train_df["shopping_centers_km"]).astype("float") + 1

test_df["rail_to_shop"]= (test_df["railroad_km"] - test_df["shopping_centers_km"]).astype("float") + 1



train_df["rail_to_museum"]= (train_df["railroad_km"] - train_df["museum_km"]).astype("float") + 1

test_df["rail_to_museum"]= (test_df["railroad_km"] - test_df["museum_km"]).astype("float") + 1



train_df["rail_imp"]= (train_df["railroad_km"] - train_df["railroad_station_walk_km"]).astype("float") + 1

test_df["rail_imp"]= (test_df["railroad_km"] - test_df["railroad_station_walk_km"]).astype("float") + 1



train_df["rail_mvt"]= (train_df["railroad_station_walk_km"] / train_df["railroad_station_walk_min"]).astype("float") 

test_df["rail_mvt"]= (test_df["railroad_station_walk_km"] / test_df["railroad_station_walk_min"]).astype("float")



train_df["pre_to_public"]= (train_df["kindergarten_km"] - train_df["public_transport_station_km"]).astype("float") + 1

test_df["pre_to_public"]= (test_df["kindergarten_km"] - test_df["public_transport_station_km"]).astype("float") + 1



train_df["pre_to_public_ratio"]= (train_df["kindergarten_km"] / train_df["public_transport_station_km"]).astype("float") + 1

test_df["pre_to_public_ratio"]= (test_df["kindergarten_km"] / test_df["public_transport_station_km"]).astype("float") + 1



train_df["kinder_radiation_ratio"]= (train_df["kindergarten_km"] / train_df["radiation_km"]).astype("float") 

test_df["kinder_radiation_ratio"]= (test_df["kindergarten_km"] / test_df["radiation_km"]).astype("float") 



train_df["pre_to_public"]= (train_df["kindergarten_km"] - train_df["public_healthcare_km"]).astype("float") + 1

test_df["pre_to_public"]= (test_df["kindergarten_km"] - test_df["public_healthcare_km"]).astype("float") + 1



train_df["kinder_indus_ratio"]= (train_df["kindergarten_km"] / train_df["industrial_km"]).astype("float") + 1

test_df["kinder_indus_ratio"]= (test_df["kindergarten_km"] / test_df["industrial_km"]).astype("float") + 1



train_df["kinder_indus_"]= (train_df["kindergarten_km"] - train_df["industrial_km"]).astype("float") + 1

test_df["kinder_indus_"]= (test_df["kindergarten_km"] - test_df["industrial_km"]).astype("float") + 1



train_df["rail_to_work"]= (train_df["railroad_km"] - train_df["workplaces_km"]).astype("float") + 1

test_df["rail_to_work"]= (test_df["railroad_km"] - test_df["workplaces_km"]).astype("float") + 1



train_df["rail_to_public"]= (train_df["kindergarten_km"] - train_df["public_transport_station_km"]).astype("float") + 1

test_df["rail_to_public"]= (test_df["kindergarten_km"] - test_df["public_transport_station_km"]).astype("float") + 1



train_df["green_zone_reactor_km"]= (train_df["nuclear_reactor_km"] - train_df["green_zone_km"]).astype("float") + 1

test_df["green_zone_reactor_km"]= (test_df["nuclear_reactor_km"] - test_df["green_zone_km"]).astype("float") + 1



train_df["state_age"]= (train_df["state"]*100 + train_df["age_of_building"]).astype("float") 

test_df["state_age"]= (test_df["state"]*100 + test_df["age_of_building"]).astype("float")



train_df["swim_pool_rail"]= (train_df["swim_pool_km"]- train_df["railroad_km"]).astype("float") 

test_df["swim_pool_rail"]= (test_df["swim_pool_km"]- test_df["railroad_km"]).astype("float") 



train_df["swim_pool_public"]= (train_df["swim_pool_km"]- train_df["public_transport_station_km"]).astype("float") 

test_df["swim_pool_public"]= (test_df["swim_pool_km"]- test_df["public_transport_station_km"]).astype("float") 



train_df["swim_pool_shop"]= (train_df["swim_pool_km"]/ train_df["shopping_centers_km"]).astype("float") +1

test_df["swim_pool_shop"]= (test_df["swim_pool_km"]/ test_df["shopping_centers_km"]).astype("float") +1



train_df["morning_rush"]= (train_df["workplaces_km"]- train_df["school_km"]).astype("float") 

test_df["morning_rush"]= (test_df["workplaces_km"]- test_df["school_km"]).astype("float") 



train_df["eurrub_mort"]= (train_df["eurrub"] * train_df["mortgage_rate"]).astype("float") 

test_df["eurrub_mort"]= (test_df["eurrub"] * test_df["mortgage_rate"]).astype("float")



train_df["rad_greenzone"]= (train_df["radiation_km"] - train_df["green_zone_km"]).astype("float")

test_df["rad_greenzone"]= (test_df["radiation_km"] - test_df["green_zone_km"]).astype("float")



train_df["rad_greenzone_ratio"]= (train_df["radiation_km"] / train_df["green_zone_km"]).astype("float")

test_df["rad_greenzone_ratio"]= (test_df["radiation_km"] / test_df["green_zone_km"]).astype("float")



train_df["rad_water_ratio"]= (train_df["radiation_km"] / train_df["water_km"]).astype("float")

test_df["rad_water_ratio"]= (test_df["radiation_km"] / test_df["water_km"]).astype("float")



train_df["water_green_ratio"]= (train_df["water_km"] / train_df["green_zone_km"]).astype("float")

test_df["water_green_ratio"]= (test_df["water_km"] / test_df["green_zone_km"]).astype("float")







y_train = train_df["price_doc"]

x_train = train_df.drop(["id", "timestamp", "price_doc"], axis=1)

x_test = test_df.drop(["id", "timestamp"], axis=1)



test_id=test_df.id



xgb_params = {

    'eta': 0.05,

    'max_depth': 5,

    'subsample': 0.7,

    'colsample_bytree': 0.7,

    'objective': 'reg:linear',

    'eval_metric': 'rmse',

    'silent': 1

}



dtrain = xgb.DMatrix(x_train, y_train)

dtest = xgb.DMatrix(x_test)



'''

cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=1000, early_stopping_rounds=20,

    verbose_eval=50, show_stdv=False)

'''

    

print(" ")

print("training...")

    

#num_boost_rounds = len(cv_output)

#num_boost_rounds= 489 try4 best sub, 410 new

num_boost_rounds = 410



print("num_boost_rounds:", num_boost_rounds)

model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round= num_boost_rounds)



y_predict = model.predict(dtest)

output = pd.DataFrame({'id': test_id, 'price_doc': y_predict})



#output.to_csv('xgbSub.csv', index=False)
fig, ax = plt.subplots(1, 1, figsize=(8, 16))



xgb.plot_importance(model, max_num_features=50, height=0.5, ax=ax)
fig, ax = plt.subplots(1, 1, figsize=(8, 16))



xgb.plot_importance(model, max_num_features=70, height=0.5, ax=ax)
