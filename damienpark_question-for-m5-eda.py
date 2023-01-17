import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

pd.set_option("display.max_columns", 100)
pd.set_option("display.max_rows", 500)
train = pd.read_csv("../input/m5-forecasting-accuracy/sales_train_validation.csv")
calendar = pd.read_csv("../input/m5-forecasting-accuracy/calendar.csv")
sell = pd.read_csv("../input/m5-forecasting-accuracy/sell_prices.csv")
sample = pd.read_csv("../input/m5-forecasting-accuracy/sample_submission.csv")
display(train.tail())
display(calendar.tail())
display(sell.tail())
display(sample.tail())
temp = sell.query("item_id in ('HOBBIES_1_001', 'HOBBIES_2_001') and store_id=='CA_1'")
sns.lineplot(x="wm_yr_wk", y="sell_price", hue="item_id", data=temp)
plt.show()

display(train.groupby("dept_id").item_id.size())
train.groupby(["store_id", "cat_id", "dept_id"])["id", "item_id"].nunique()
temp = train.melt(id_vars=["id"], 
                  var_name="d", value_name="sales")
temp = temp.merge(calendar[["d", "date"]], on="d")

_ = temp.groupby("id").sum()
sum(_.sales==0)
sell.groupby(["item_id", "store_id"]).nunique()
temp = train.groupby(["cat_id"]).mean().T
temp.columns.name = None
temp = temp.reset_index().merge(calendar.loc[:, ["d", "date"]], 
                                left_on="index", right_on="d")
temp.date = pd.to_datetime(temp.date)

plt.figure(figsize=(30, 10))
for i in ["FOODS", "HOBBIES", "HOUSEHOLD"]:
    plt.plot(temp["date"], temp[i], label=i, alpha=.4)
    plt.gca().xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y-%m-%d"))
    plt.gca().xaxis.set_minor_formatter(matplotlib.dates.DateFormatter("%Y-%m-%d"))
plt.legend()
plt.xticks(rotation=45)
plt.margins(x=0.01)
plt.title("mean sales for each cat_id")
plt.show()
plt.figure(figsize=(30, 20))
plt.subplot(2, 1, 1)
plt.plot(temp["date"][1:], temp["FOODS"].diff()[1:], label="FOODS", alpha=.4)
plt.gca().xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y-%m-%d"))
plt.legend()
plt.xticks(rotation=45)
plt.margins(x=0.01)

plt.subplot(2, 1, 2)
for i in ["HOBBIES", "HOUSEHOLD"]:
    plt.plot(temp["date"][1:], temp[i].diff()[1:], label=i, alpha=.4)
    plt.gca().xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y-%m-%d"))
plt.legend()
plt.xticks(rotation=45)
plt.margins(x=0.01)
plt.show()
plt.figure(figsize=(30, 20))
plt.subplot(2, 1, 1)
plt.plot(temp["date"][1:], temp["FOODS"].diff()[1:], label="FOODS", alpha=.4)
plt.gca().xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y-%m-%d"))
plt.legend()
plt.xticks(rotation=45)
plt.margins(x=0.01)

plt.subplot(2, 1, 2)
for i in ["HOBBIES", "HOUSEHOLD"]:
    plt.plot(temp["date"][1:], temp[i].diff()[1:], label=i, alpha=.4)
    plt.gca().xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y-%m-%d"))
plt.legend()
plt.xticks(rotation=45)
plt.margins(x=0.01)
plt.show()
temp = train.groupby(["cat_id"]).mean().T
temp.columns.name = None
temp = temp.reset_index().melt(id_vars="index", value_vars=["FOODS", "HOBBIES", "HOUSEHOLD"])

plt.figure(figsize=(20, 10))
sns.boxplot(data=temp, x="variable", y="value")
plt.show()
temp = train.groupby(["dept_id"]).mean().T
temp.columns.name = None
temp = temp.reset_index().merge(calendar.loc[:, ["d", "date"]], 
                                left_on="index", right_on="d")
temp.date = pd.to_datetime(temp.date)

plt.figure(figsize=(30, 30))
plt.subplot(3, 1, 1)
for i in ["FOODS_1", "FOODS_2", "FOODS_3"]:
    plt.plot(temp["date"], temp[i], label=i, alpha=.4)
    plt.gca().xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y-%m-%d"))
plt.legend()
plt.xticks(rotation=45)
plt.margins(x=0.01)

plt.subplot(3, 1, 2)
for i in ["HOBBIES_1", "HOBBIES_2"]:
    plt.plot(temp["date"], temp[i], label=i, alpha=.4)
    plt.gca().xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y-%m-%d"))
plt.legend()
plt.xticks(rotation=45)
plt.margins(x=0.01)

plt.subplot(3, 1, 3)
for i in ["HOUSEHOLD_1", "HOUSEHOLD_2"]:
    plt.plot(temp["date"], temp[i], label=i, alpha=.4)
    plt.gca().xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y-%m-%d"))
plt.legend()
plt.xticks(rotation=45)
plt.margins(x=0.01)

plt.title("mean sales for each dept_id")
plt.show()
temp = train.groupby(["dept_id"]).mean().T
temp.columns.name = None
temp = temp.reset_index().melt(id_vars="index", value_vars=list(temp.columns))

plt.figure(figsize=(20, 10))
sns.boxplot(data=temp, x="variable", y="value")
plt.show()
temp = train.groupby(["state_id"]).mean().T
temp.columns.name = None
temp = temp.reset_index().merge(calendar.loc[:, ["d", "date"]], 
                                left_on="index", right_on="d")
temp.date = pd.to_datetime(temp.date)

plt.figure(figsize=(30, 10))
for i in ["CA", "TX", "WI"]:
    plt.plot(temp["date"], temp[i], label=i, alpha=.4)
    plt.gca().xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y-%m-%d"))
plt.legend()
plt.xticks(rotation=45)
plt.margins(x=0.01)
plt.title("mean sales for each state_id")
plt.show()
temp = train.groupby(["state_id"]).mean().T
temp.columns.name = None
temp = temp.reset_index().melt(id_vars="index", value_vars=list(temp.columns))

plt.figure(figsize=(20, 10))
sns.boxplot(data=temp, x="variable", y="value")
plt.show()
temp = train.groupby(["store_id"]).mean().T
temp.columns.name = None
temp = temp.reset_index().merge(calendar.loc[:, ["d", "date"]], 
                                left_on="index", right_on="d")
temp.date = pd.to_datetime(temp.date)

plt.figure(figsize=(30, 30))
plt.subplot(3, 1, 1)
for i in ["CA_1", "CA_2", "CA_3", "CA_4"]:
    plt.plot(temp["date"], temp[i], label=i, alpha=.4)
    plt.gca().xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y-%m-%d"))
plt.legend()
plt.xticks(rotation=45)
plt.margins(x=0.01)

plt.subplot(3, 1, 2)
for i in ["TX_1", "TX_2", "TX_3"]:
    plt.plot(temp["date"], temp[i], label=i, alpha=.4)
    plt.gca().xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y-%m-%d"))
plt.legend()
plt.xticks(rotation=45)
plt.margins(x=0.01)

plt.subplot(3, 1, 3)
for i in ["WI_1", "WI_2", "WI_3"]:
    plt.plot(temp["date"], temp[i], label=i, alpha=.4)
    plt.gca().xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y-%m-%d"))
plt.legend()
plt.xticks(rotation=45)
plt.margins(x=0.01)

plt.title("mean sales for each store_id")
plt.show()
temp = train.groupby(["store_id"]).mean().T
temp.columns.name = None
temp = temp.reset_index().melt(id_vars="index", value_vars=list(temp.columns))

plt.figure(figsize=(20, 10))
sns.boxplot(data=temp, x="variable", y="value")
plt.show()