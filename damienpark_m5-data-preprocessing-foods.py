import pandas as pd

import numpy as np

import tqdm

import gc



import matplotlib.pyplot as plt

import seaborn as sns



# from statsmodels.tsa.statespace import sarimax

# import statsmodels as sm



# from sklearn.preprocessing import MinMaxScaler

# from sklearn.model_selection import train_test_split

# from sklearn.metrics import mean_squared_error

# import keras



pd.set_option("display.max_columns", 100)

pd.set_option("display.max_rows", 500)
train = pd.read_csv("../input/m5-forecasting-accuracy/sales_train_validation.csv")

calendar = pd.read_csv("../input/m5-forecasting-accuracy/calendar.csv")

sell = pd.read_csv("../input/m5-forecasting-accuracy/sell_prices.csv")

# sample = pd.read_csv("../input/m5-forecasting-accuracy/sample_submission.csv")



# data type

train.iloc[:, 6:] = train.iloc[:, 6:].astype("int16")

calendar.iloc[:, 11:] = calendar.iloc[:, 11:].astype("int8")

sell.sell_price = sell.sell_price.astype("float16")
train.id.replace(regex="_validation", value="", inplace=True)

sell["id"] = sell.item_id+"_"+sell.store_id

sell.drop(columns=["store_id", "item_id"], inplace=True)
# train dataset

# setting columns which we use

train_col = ['id', 'date', 

             'FOODS_1', 'FOODS_2', 'FOODS_3', 

             'HOBBIES_1', "HOBBIES_2", 

             'HOUSEHOLD_1', 'HOUSEHOLD_2', 

             'FOODS', 'HOBBIES', 

             'CA_1', 'CA_2', 'CA_3', 'CA_4', 

             'TX_1', 'TX_2', 'TX_3', 

             'WI_1', 'WI_2', 'WI_3', 

             'CA', 'TX', 

             "snap_CA", "snap_TX", "snap_WI", 

             'Cultural', 'National', 'Religious', 'Sporting', 

             "release", 

             'sell_price', 'sales']



# event one-hot encoding

train_event = calendar.loc[:, ["d", "event_name_1", "event_type_1", "event_name_2", "event_type_2"]]

train_event = train_event.melt(id_vars="d")

train_event = train_event[~train_event.value.isnull()].reset_index(drop=True)

train_event.variable.replace(to_replace={"event_name_1":"event_name", "event_name_2":"event_name", 

                                   "event_type_1":"event_type", "event_type_2":"event_type"}, 

                             inplace=True)



train_event_type = train_event.query("variable=='event_type'").loc[:, ["d"]].merge(pd.get_dummies(train_event.query("variable=='event_type'").value), 

                                                                                   left_index=True, right_index=True, how="left")

train_event_name = train_event.query("variable=='event_name'").loc[:, ["d"]].merge(pd.get_dummies(train_event.query("variable=='event_name'").value), 

                                                                                   left_index=True, right_index=True, how="left")

train_event_type = train_event_type.groupby("d").sum()

train_event_name = train_event_name.groupby("d").sum()

train_event_type.reset_index(inplace=True)

train_event_name.reset_index(inplace=True)



# train one-hot encoding

train_dept_id = pd.get_dummies(train.dept_id)

train_cat_id = pd.get_dummies(train.cat_id)

train_store_id = pd.get_dummies(train.store_id)

train_state_id = pd.get_dummies(train.state_id)



# merge all data

df_dummies = pd.concat([train[["id"]], train_dept_id, train_cat_id, train_store_id, train_state_id], axis=1)

train_df = df_dummies.merge(train.iloc[:, 6:], left_index=True, right_index=True, how="left")



del train, train_dept_id, train_cat_id, train_store_id, train_state_id

gc.collect()



train_df = train_df.melt(id_vars=list(df_dummies.columns), 

                         var_name="d", value_name="sales")



train_df = train_df.merge(calendar.loc[:, ["date", "d", "wm_yr_wk", "snap_CA", "snap_TX", "snap_WI"]], 

                          left_on="d", right_on="d", how="left")



del calendar

gc.collect()



train_df = train_df.merge(train_event_type, on="d", how="left")

train_df = train_df.merge(sell.loc[:, ["id", "wm_yr_wk", "sell_price"]], 

                          on=["id", "wm_yr_wk"], how="left")

# df = df.loc[:, col]

train_df.fillna(0, inplace=True)

train_df.loc[:, ['snap_CA', 'snap_TX', 'snap_WI', 

                 'Cultural', 'National', 'Religious', 'Sporting']] = train_df.loc[:, ['snap_CA', 'snap_TX', 'snap_WI', 

                                                                                      'Cultural', 'National', 'Religious', 'Sporting']].astype("uint8")

train_df.date = pd.to_datetime(train_df.date)

# train_df.set_index(["id", "date"], inplace=True)



del sell

gc.collect()



train_df["release"] = 1

train_df.loc[train_df.sell_price==0, "release"] = 0

train_df.release = train_df.release.astype("uint8")



# reordering columns

train_df = train_df.loc[:, train_col]
train_df.tail()
train = pd.read_csv("../input/m5-forecasting-accuracy/sales_train_validation.csv")

calendar = pd.read_csv("../input/m5-forecasting-accuracy/calendar.csv")

sell = pd.read_csv("../input/m5-forecasting-accuracy/sell_prices.csv")

# sample = pd.read_csv("../input/m5-forecasting-accuracy/sample_submission.csv")



# data type

train.iloc[:, 6:] = train.iloc[:, 6:].astype("int16")

calendar.iloc[:, 11:] = calendar.iloc[:, 11:].astype("int8")

sell.sell_price = sell.sell_price.astype("float16")
train.id.replace(regex="_validation", value="", inplace=True)

sell["id"] = sell.item_id+"_"+sell.store_id

sell.drop(columns=["store_id", "item_id"], inplace=True)
# test dataset

# setting columns which we use

test_col = ['id', 'date', 

            'FOODS_1', 'FOODS_2', 'FOODS_3', 

            'HOBBIES_1', "HOBBIES_2", 

            'HOUSEHOLD_1', 'HOUSEHOLD_2', 

            'FOODS', 'HOBBIES', 

            'CA_1', 'CA_2', 'CA_3', 'CA_4', 

            'TX_1', 'TX_2', 'TX_3', 

            'WI_1', 'WI_2', 'WI_3', 

            'CA', 'TX', 

            "snap_CA", "snap_TX", "snap_WI", 

            'Cultural', 'National', 'Religious', 'Sporting', 

            "release", 

            'sell_price', 'sales']



# event one-hot encoding

test_event = calendar.query("date>'2016-04-24'").loc[:, ["d", "event_name_1", "event_type_1", "event_name_2", "event_type_2"]]

test_event = test_event.melt(id_vars="d")

test_event = test_event[~test_event.value.isnull()].reset_index(drop=True)

test_event.variable.replace(to_replace={"event_name_1":"event_name", "event_name_2":"event_name", 

                                        "event_type_1":"event_type", "event_type_2":"event_type"}, 

                            inplace=True)



test_event_type = test_event.query("variable=='event_type'").loc[:, ["d"]].merge(pd.get_dummies(test_event.query("variable=='event_type'").value), 

                                                                       left_index=True, right_index=True, how="left")

test_event_name = test_event.query("variable=='event_name'").loc[:, ["d"]].merge(pd.get_dummies(test_event.query("variable=='event_name'").value), 

                                                                       left_index=True, right_index=True, how="left")

test_event_type = test_event_type.groupby("d").sum()

test_event_name = test_event_name.groupby("d").sum()

test_event_type.reset_index(inplace=True)

test_event_name.reset_index(inplace=True)

test_event_type = calendar.query("date>'2016-04-24'").loc[:, ["d", "date"]].merge(test_event_type, how="left")

test_event_name = calendar.query("date>'2016-04-24'").loc[:, ["d", "date"]].merge(test_event_name, how="left")

test_event_type.date = pd.to_datetime(test_event_type.date)

test_event_name.date = pd.to_datetime(test_event_name.date)

test_event_type.fillna(0, inplace=True)

test_event_name.fillna(0, inplace=True)

test_event_type.iloc[:, 2:] = test_event_type.iloc[:, 2:].astype("uint8")

test_event_name.iloc[:, 2:] = test_event_name.iloc[:, 2:].astype("uint8")





sample = pd.read_csv("../input/m5-forecasting-accuracy/sample_submission.csv")



validation = sample[sample.id.str.contains("validation")].add_prefix("val_")

evaluation = sample[sample.id.str.contains("evaluation")].add_prefix("evl_")

validation.rename(columns={"val_id":"id"}, inplace=True)

evaluation.rename(columns={"evl_id":"id"}, inplace=True)

validation.id.replace(regex="_validation", value="", inplace=True)

evaluation.id.replace(regex="_evaluation", value="", inplace=True)



validation = validation.melt(id_vars="id", var_name="F", value_name="sales")

evaluation = evaluation.melt(id_vars="id", var_name="F", value_name="sales")



val = ['val_F1', 'val_F2', 'val_F3', 'val_F4', 'val_F5', 'val_F6', 

       'val_F7', 'val_F8', 'val_F9', 'val_F10', 'val_F11', 'val_F12', 

       'val_F13', 'val_F14', 'val_F15', 'val_F16', 'val_F17', 'val_F18', 

       'val_F19', 'val_F20', 'val_F21', 'val_F22', 'val_F23', 'val_F24', 

       'val_F25', 'val_F26', 'val_F27', 'val_F28']

evl = ['evl_F1', 'evl_F2', 'evl_F3', 'evl_F4', 'evl_F5', 'evl_F6', 

       'evl_F7', 'evl_F8', 'evl_F9', 'evl_F10', 'evl_F11', 'evl_F12', 

       'evl_F13', 'evl_F14', 'evl_F15', 'evl_F16', 'evl_F17', 'evl_F18', 

       'evl_F19', 'evl_F20', 'evl_F21', 'evl_F22', 'evl_F23', 'evl_F24', 

       'evl_F25', 'evl_F26', 'evl_F27', 'evl_F28']

val = pd.DataFrame(data={"F":val, "date":pd.date_range("2016-04-25", "2016-05-22")})

evl = pd.DataFrame(data={"F":evl, "date":pd.date_range("2016-05-23", "2016-06-19")})



validation = validation.merge(val).drop(columns=["F"])

evaluation = evaluation.merge(evl).drop(columns=["F"])



for i in ['FOODS_1', 'FOODS_2', 'FOODS_3', 'HOBBIES_1', 'HOBBIES_2',

          'HOUSEHOLD_1', 'HOUSEHOLD_2', 'FOODS', 'HOBBIES', 'HOUSEHOLD', 

          'CA_1', 'CA_2', 'CA_3', 'CA_4', 

          'TX_1', 'TX_2', 'TX_3', 

          'WI_1', 'WI_2', 'WI_3',

          'CA', 'TX', 'WI']:

    validation[i] = 0

    validation.loc[validation.id.str.contains(i), i] = 1



for i in ['FOODS_1', 'FOODS_2', 'FOODS_3', 'HOBBIES_1', 'HOBBIES_2',

          'HOUSEHOLD_1', 'HOUSEHOLD_2', 'FOODS', 'HOBBIES', 'HOUSEHOLD', 

          'CA_1', 'CA_2', 'CA_3', 'CA_4', 

          'TX_1', 'TX_2', 'TX_3', 

          'WI_1', 'WI_2', 'WI_3',

          'CA', 'TX', 'WI']:

    evaluation[i] = 0

    evaluation.loc[evaluation.id.str.contains(i), i] = 1



calendar.date = pd.to_datetime(calendar.date)

validation = validation.merge(calendar.loc[:, ["date", "wm_yr_wk", "snap_CA", "snap_TX", "snap_WI"]], 

                              left_on="date", right_on="date", how="left")

evaluation = evaluation.merge(calendar.loc[:, ["date", "wm_yr_wk", "snap_CA", "snap_TX", "snap_WI"]], 

                              left_on="date", right_on="date", how="left")

validation = validation.merge(sell.loc[:, ["id", "wm_yr_wk", "sell_price"]], 

                              on=["id", "wm_yr_wk"], how="left")

evaluation = evaluation.merge(sell.loc[:, ["id", "wm_yr_wk", "sell_price"]], 

                              on=["id", "wm_yr_wk"], how="left")



validation["release"] = 1

validation.loc[validation.sell_price==0, "release"] = 0

validation.release = validation.release.astype("uint8")

evaluation["release"] = 1

evaluation.loc[evaluation.sell_price==0, "release"] = 0

evaluation.release = evaluation.release.astype("uint8")



validation = validation.merge(test_event_type, on="date", how="left")

evaluation = evaluation.merge(test_event_type, on="date", how="left")



# reordering columns

validation = validation.loc[:, test_col]

evaluation = evaluation.loc[:, test_col]
%%time

train_df.query("FOODS_1==1").reset_index(drop=True).to_csv("../working/df_FOODS_1.csv", index=False)

train_df.query("FOODS_2==1").reset_index(drop=True).to_csv("../working/df_FOODS_2.csv", index=False)

train_df.query("FOODS_3==1").reset_index(drop=True).to_csv("../working/df_FOODS_3.csv", index=False)



# train_df.query("HOBBIES_1==1").reset_index(drop=True).to_csv("../working/df_HOBBIES_1.csv", index=False)

# train_df.query("HOBBIES_2==1").reset_index(drop=True).to_csv("../working/df_HOBBIES_2.csv", index=False)



# train_df.query("HOUSEHOLD_1==1").reset_index(drop=True).to_csv("../working/df_HOUSEHOLD_1.csv", index=False)

# train_df.query("HOUSEHOLD_2==1").reset_index(drop=True).to_csv("../working/df_HOUSEHOLD_2.csv", index=False)
%%time

validation.reset_index(drop=True).to_csv("../working/validation.csv", index=False)

evaluation.reset_index(drop=True).to_csv("../working/evaluation.csv", index=False)
df = train_df

df["item_id"] = df.id.apply(func=lambda x:x[:-5])
# _ = df.query("FOODS_1==1 and CA==1 and release==1").groupby(["item_id", "snap_CA"]).sales.mean()

# _ = _.reset_index()
_ = df.query("FOODS_1==1 and CA==1 and release==1").loc[:, ["item_id", "snap_CA", "sales"]]

_ = _.reset_index(drop=True)

item_list = np.sort(pd.unique(_.item_id))



start = 0

count = len(item_list)//44+1



plt.figure(figsize=(30, 30))

plt.suptitle("FOODS_1 and CA")

for idx, val in enumerate(range(44, count*44+1, 44)):

    plt.subplot(count, 1, idx+1)

    temp = item_list[start:val]

    start = val

    sns.boxplot(data=_.query("item_id in @temp").sort_values("item_id"), x="item_id", y="sales", hue="snap_CA")

    plt.xticks(rotation=45)

plt.show()
_ = df.query("FOODS_1==1 and TX==1 and release==1").loc[:, ["item_id", "snap_TX", "sales"]]

_ = _.reset_index(drop=True)

item_list = np.sort(pd.unique(_.item_id))



start = 0

count = len(item_list)//44+1



plt.figure(figsize=(30, 30))

plt.suptitle("FOODS_1 and TX")

for idx, val in enumerate(range(44, count*44+1, 44)):

    plt.subplot(count, 1, idx+1)

    temp = item_list[start:val]

    start = val

    sns.boxplot(data=_.query("item_id in @temp").sort_values("item_id"), x="item_id", y="sales", hue="snap_TX")

    plt.xticks(rotation=45)

plt.show()
_ = df.query("FOODS_1==1 and CA==0 and TX==0 and release==1").loc[:, ["item_id", "snap_WI", "sales"]]

_ = _.reset_index(drop=True)

item_list = np.sort(pd.unique(_.item_id))



start = 0

count = len(item_list)//44+1



plt.figure(figsize=(30, 30))

plt.suptitle("FOODS_1 and WI")

for idx, val in enumerate(range(44, count*44+1, 44)):

    plt.subplot(count, 1, idx+1)

    temp = item_list[start:val]

    start = val

    sns.boxplot(data=_.query("item_id in @temp").sort_values("item_id"), x="item_id", y="sales", hue="snap_WI")

    plt.xticks(rotation=45)



plt.show()
_ = df.query("FOODS_2==1 and CA==1 and release==1").loc[:, ["item_id", "snap_CA", "sales"]]

_ = _.reset_index(drop=True)

item_list = np.sort(pd.unique(_.item_id))



start = 0

count = len(item_list)//50+1



plt.figure(figsize=(50, 30))

plt.suptitle("FOODS_2 and CA")

for idx, val in enumerate(range(50, count*50+1, 50)):

    plt.subplot(count, 1, idx+1)

    temp = item_list[start:val]

    start = val

    sns.boxplot(data=_.query("item_id in @temp").sort_values("item_id"), x="item_id", y="sales", hue="snap_CA")

    plt.xticks(rotation=45)

plt.show()
_ = df.query("FOODS_2==1 and TX==1 and release==1").loc[:, ["item_id", "snap_TX", "sales"]]

_ = _.reset_index(drop=True)

item_list = np.sort(pd.unique(_.item_id))



start = 0

count = len(item_list)//50+1



plt.figure(figsize=(50, 30))

plt.suptitle("FOODS_2 and TX")

for idx, val in enumerate(range(50, count*50+1, 50)):

    plt.subplot(count, 1, idx+1)

    temp = item_list[start:val]

    start = val

    sns.boxplot(data=_.query("item_id in @temp").sort_values("item_id"), x="item_id", y="sales", hue="snap_TX")

    plt.xticks(rotation=45)

plt.show()
_ = df.query("FOODS_2==1 and CA==0 and TX==0 and release==1").loc[:, ["item_id", "snap_WI", "sales"]]

_ = _.reset_index(drop=True)

item_list = np.sort(pd.unique(_.item_id))



start = 0

count = len(item_list)//50+1



plt.figure(figsize=(50, 30))

plt.suptitle("FOODS_2 and WI")

for idx, val in enumerate(range(50, count*50+1, 50)):

    plt.subplot(count, 1, idx+1)

    temp = item_list[start:val]

    start = val

    sns.boxplot(data=_.query("item_id in @temp").sort_values("item_id"), x="item_id", y="sales", hue="snap_WI")

    plt.xticks(rotation=45)



plt.show()
_ = df.query("FOODS_3==1 and CA==1 and release==1").loc[:, ["item_id", "snap_CA", "sales"]]

_ = _.reset_index(drop=True)

item_list = np.sort(pd.unique(_.item_id))



start = 0

count = len(item_list)//50+1



plt.figure(figsize=(50, 30))

plt.suptitle("FOODS_3 and CA")

for idx, val in enumerate(range(50, count*50+1, 50)):

    plt.subplot(count, 1, idx+1)

    temp = item_list[start:val]

    start = val

    sns.boxplot(data=_.query("item_id in @temp").sort_values("item_id"), x="item_id", y="sales", hue="snap_CA")

    plt.xticks(rotation=45)

plt.show()
_ = df.query("FOODS_3==1 and TX==1 and release==1").loc[:, ["item_id", "snap_TX", "sales"]]

_ = _.reset_index(drop=True)

item_list = np.sort(pd.unique(_.item_id))



start = 0

count = len(item_list)//50+1



plt.figure(figsize=(50, 30))

plt.suptitle("FOODS_3 and TX")

for idx, val in enumerate(range(50, count*50+1, 50)):

    plt.subplot(count, 1, idx+1)

    temp = item_list[start:val]

    start = val

    sns.boxplot(data=_.query("item_id in @temp").sort_values("item_id"), x="item_id", y="sales", hue="snap_TX")

    plt.xticks(rotation=45)

plt.show()
_ = df.query("FOODS_3==1 and CA==0 and TX==0 and release==1").loc[:, ["item_id", "snap_WI", "sales"]]

_ = _.reset_index(drop=True)

item_list = np.sort(pd.unique(_.item_id))



start = 0

count = len(item_list)//50+1



plt.figure(figsize=(50, 30))

plt.suptitle("FOODS_3 and WI")

for idx, val in enumerate(range(50, count*50+1, 50)):

    plt.subplot(count, 1, idx+1)

    temp = item_list[start:val]

    start = val

    sns.boxplot(data=_.query("item_id in @temp").sort_values("item_id"), x="item_id", y="sales", hue="snap_WI")

    plt.xticks(rotation=45)



plt.show()