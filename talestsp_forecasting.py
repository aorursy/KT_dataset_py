import pandas as pd

import numpy as np

from sklearn.linear_model import LinearRegression



import my_dao

import time_utils

import pretties

import stats

import process

import evaluation

import plotter

import download



from bokeh.plotting import show, output_notebook
pretties.max_data_frame_columns()

pretties.decimal_notation()

output_notebook()
train = my_dao.load_dataset("train")

train = train.groupby("store_dept").apply(process.train_sales_semantic_enrichment)



test = my_dao.load_dataset("test")



feat = my_dao.load_features()

feat = process.features_semantic_enrichment(feat)



stores = my_dao.load_stores()
train = train.merge(feat, how="left", left_on=["Store", "Date"], right_on=["Store", "Date"], suffixes=["", "_y"])

del train["IsHoliday_y"]

del train["timestamp_y"]

train = train.merge(stores, how="left", left_on=["Store"], right_on=["Store"])
test = test.merge(feat, how="left", left_on=["Store", "Date"], right_on=["Store", "Date"], suffixes=["", "_y"])

del test["IsHoliday_y"]

del test["timestamp_y"]

test = test.merge(stores, how="left", left_on=["Store"], right_on=["Store"])
cols = ['Date', 'Store', 'Dept', 'Weekly_Sales', 'pre_holiday', 'IsHoliday', 'pos_holiday', 'Fuel_Price', 

        'CPI', 'Unemployment', 'celsius', 'datetime', 'Type', 'sales_diff', 'sales_diff_p',

        'Size', 'Temperature', 'timestamp', 'store_dept', "day_n", "week_n", "month_n", "wm_date", "up_diff", "celsius_diff", "year"]



train = train[cols]

print("Shape: {}".format(train.shape))

train.sample(6)
print("Train\n")

print("Initial date: {}".format(train["Date"].iloc[0]))

print("Final date  : {}".format(train["Date"].iloc[-1]))

print("Time interval (months): {}".format(time_utils.time_interval_months(train["Date"])))

print("Time interval (years) : {}".format(time_utils.time_interval_months(train["Date"]) / 12))
print("Test\n")

print("Initial date: {}".format(test["Date"].iloc[0]))

print("Final date  : {}".format(test["Date"].iloc[-1]))

print("Time interval (months): {}".format(time_utils.time_interval_months(test["Date"])))

print("Time interval (years) : {}".format(time_utils.time_interval_months(test["Date"]) / 12))
timestamp_threshold = time_utils.str_datetime_to_timestamp("2012-02-01", "%Y-%m-%d") #24 months from the first entry



use_train = train[train["timestamp"] <= timestamp_threshold]

use_valid = train[train["timestamp"] > timestamp_threshold]
print("Fitting dataset time interval\n")

print(use_train["Date"].head(1).append(use_train["Date"].tail(1)))

print()

print("Time interval (months): {}".format(time_utils.time_interval_months(use_train["Date"])))

print("Time interval (years) : {}".format(time_utils.time_interval_months(use_train["Date"]) / 12))
print("Validation dataset time interval\n")

print(use_valid["Date"].head(1).append(use_valid["Date"].tail(1)))

print()

print("Time interval (months): {}".format(time_utils.time_interval_months(use_valid["Date"])))

print("Time interval (years) : {}".format(time_utils.time_interval_months(use_valid["Date"]) / 12))
try:

    wm_data_train = my_dao.load_week_month_data("wm_data_train")

except FileNotFoundError:

    wm_data_train = process.wm_data(use_train)

    my_dao.save_week_month_data(wm_data_train, "wm_data_train")
wm_data_train = process.format_wm_data_colnames(wm_data_train, "train")

wm_data_train.sample(4)
try:

    wm_data_valid = my_dao.load_week_month_data("wm_data_valid")

except FileNotFoundError:

    wm_data_valid = process.wm_data(use_valid)

    my_dao.save_week_month_data(wm_data_valid, "wm_data_valid")
wm_data_valid = process.format_wm_data_colnames(wm_data_valid, "valid")

wm_data_valid.sample(4)
xy = pd.merge(wm_data_train, wm_data_valid, 

              left_on=["wm_date", "store_dept"], right_on=["wm_date", "store_dept"], 

              how="inner", suffixes=["_train", "_valid"])



xy["Store"] = xy["year1_sales_train"]

xy["Dept"] = xy["year1_size_train"]

xy["Date"] = xy["Date_train"]



print("Total groups: ", len(xy.drop_duplicates(["wm_date", "store_dept"])))

display(xy.head(4))

print("Hey, look the table above and check the first two columns (store_dept and wm_date)")

print("They mean Store 29, Dept 5, Month_n 7 and 4th week of the month.")

print("The following columns are the fields values for each year.")
%matplotlib inline

xy.plot.scatter("year0_sales_train", "year1_sales_train", 

                title="scatter plot - sales year0 vs sales year1", 

               ylim=(0, 200000), xlim=(0, 200000), alpha=0.15, figsize=(6,6))
%matplotlib inline

xy.plot.scatter("year1_sales_train", "year0_sales_valid", color="magenta",

                title="scatter plot - sales year1 vs sales year2", 

               ylim=(0, 200000), xlim=(0, 200000), alpha=0.3, figsize=(6,6))
print("NAs count on first year")

stats.freq(xy["year0_sales_train"].isna())
print("NAs count on second year")

stats.freq(xy["year1_sales_train"].isna())
print("NAs count on third year")

stats.freq(xy["year0_sales_valid"].isna())
not_na_xy = xy[(xy["year0_sales_train"].notna()) & (xy["year1_sales_train"].notna()) & (xy["year0_sales_valid"].notna())]
key_colnames = ["Store", "Dept", "Date"] #column names need to build submission file
fitting_cols = ["year0_sales_train", "year0_size_train"] #first year data

x = not_na_xy[fitting_cols] 

x.head(4)
y = not_na_xy[["year1_sales_train"]] #first year target

y.head(4)
reg = LinearRegression().fit(x, y)

print(reg.score(x, y))

print("Function:")



print("y = {} * {} + {} * {} + {}".format(round(reg.coef_[0][0], 3), x.columns[0], round(reg.coef_[0][1], 3), x.columns[1], round(reg.intercept_[0]), 3))
x_valid = not_na_xy#[["year1_sales_train", "year1_size_train"] + key_colnames + ["year1_isholiday_train"]]

x_valid.head(4)
y_pred = reg.predict(x_valid[fitting_cols])
%matplotlib inline

pd.DataFrame(y_pred)[0].plot.hist(title="Validation Predicted")

y_valid = not_na_xy[["year0_sales_valid"]]

y_valid.plot.hist(title="Validation Real", color="magenta")
x_valid[fitting_cols]
x_valid = x_valid.rename({"year0_isholiday_valid": "IsHoliday", "year0_sales_valid": "Weekly_Sales"}, axis=1)



subm = evaluation.build_submission_df(test_df=x_valid[fitting_cols + key_colnames], 

                                      target_predicted=y_pred)



print("Validation prediction evaluation:\n")

print(evaluation.evaluate(subm, x_valid))
test_cols = [fitting_col.replace("0", "1") for fitting_col in fitting_cols]
test.sample(4)
test["Date"].head(1).append(test["Date"].tail(1))
try:

    wm_data_test = my_dao.load_week_month_data("wm_data_test")

except FileNotFoundError:

    wm_data_test = process.wm_data(test)

    my_dao.save_week_month_data(wm_data_test, "wm_data_test")
wm_data_test = process.format_wm_data_colnames(wm_data_test, "test")
try:

    wm_data_train_valid = my_dao.load_week_month_data("wm_data_train_valid")

except FileNotFoundError:

    wm_data_train_valid = process.wm_data(train)

    my_dao.save_week_month_data(wm_data_train_valid, "wm_data_train_valid")
wm_data_train_valid = process.format_wm_data_colnames(wm_data_train_valid, "train")
xy_test = pd.merge(wm_data_train_valid, wm_data_test, 

                   left_on=["wm_date", "store_dept"], right_on=["wm_date", "store_dept"], 

                   how="right", suffixes=["_train", "_test"])



xy_test.sample(5)
print("NAs frequency over predictive columns\n")

for test_col in test_cols:

    print(test_col)

    print(stats.freq(xy_test[test_col].isna()))

    print()
not_na_xy_test = xy_test[xy_test[test_cols].notna().all(1)]

na_xy_test = xy_test[~xy_test.index.isin(not_na_xy_test.index)]
stats.freq(na_xy_test["store_dept"].isin(wm_data_train_valid["store_dept"]))
na_xy_test_filled = na_xy_test.apply(lambda row : process.dummy_fill_store_dept_median(row, wm_data_train_valid, test_cols), axis=1)

xy_test = not_na_xy_test.append(na_xy_test_filled)



print("NAs frequency over predictive columns\n")

for test_col in test_cols:

    print(test_col)

    print(stats.freq(xy_test[test_col].isna()))

    print()
not_na_xy_test = xy_test[xy_test[test_cols].notna().all(1)]

na_xy_test = xy_test[~xy_test.index.isin(not_na_xy_test.index)]



na_xy_test_filled = na_xy_test.apply(lambda row : process.dummy_fill_store_median(row, wm_data_train_valid, test_cols), axis=1)

xy_test = not_na_xy_test.append(na_xy_test_filled)



print("NAs frequency over predictive columns\n")

for test_col in test_cols:

    print(test_col)

    print(stats.freq(xy_test[test_col].isna()))

    print()
x_test = not_na_xy_test.append(na_xy_test_filled)#[["year1_sales_train", "year1_size_train"] + key_colnames + ["IsHoliday_test"]]

x_test.head(4)
y_test_pred = reg.predict(x_test[test_cols])
subm = evaluation.build_submission_df(test_df=x_test, 

                                      target_predicted=y_test_pred,

                                      store_colname="Store_test", 

                                      dept_colname="Dept_test", 

                                      date_colname="Date_test")



# subm["Weekly_Sales"] = subm["Weekly_Sales"].apply(lambda ws : round(ws, 4))
subm.reset_index().to_csv("submission.csv", index=False)
download.create_download_link(subm, "Submission Download", "submission.csv")