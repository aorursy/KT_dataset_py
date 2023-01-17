import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')
import os

print(os.listdir("../input"))

raw_data = pd.read_csv("../input/beer-consumption-sao-paulo/Consumo_cerveja.csv")

raw_data.head()
raw_data.columns = ["Date", "Avg Temp", "Min Temp", "Max Temp", "Rainfall", "Weekend", "Beer Consumption"]

raw_data.head()
raw_data.shape
raw_data.info()
raw_data.isnull().sum()
pd.set_option("display.max_rows", 15)
raw_data.iloc[364:]
after_drop_data = raw_data.drop(range(365, 941))

after_drop_data.shape
after_drop_data.tail()
temp_data = after_drop_data[["Min Temp", "Max Temp", "Avg Temp", "Rainfall"]]

temp_data.head()
import re

for index, row in temp_data.iterrows():

    for j in ["Min Temp", "Max Temp", "Avg Temp", "Rainfall"]:

        row[j] = re.sub(r',', '.', row[j])

temp_data.head()
temp_data = temp_data.astype("float64")
tidy_data = after_drop_data

for j in ["Min Temp", "Max Temp", "Avg Temp", "Rainfall"]:

    tidy_data[j] = temp_data[j]

tidy_data.head()
tidy_data.info()
date_data = pd.DataFrame(tidy_data["Date"])

date_data.head()
months = np.array([])

days = np.array([])

for index, row in tidy_data.iterrows():

    months = np.append(months, re.search(r'-(.+?)-', row["Date"]).group(1))

    days = np.append(days, re.search(r'-..-(.+?)$', row["Date"]).group(1))

date_data["Month"] = months

date_data["Day"] = days

date_data[["Month", "Day"]] = date_data[["Month", "Day"]].astype("float64")

date_data.head()
final_data = pd.merge(tidy_data, date_data, on="Date", how="inner")

final_data = final_data.drop(["Date"], axis=1)

#Re-order to have a y on right side 

final_data = final_data[["Avg Temp", "Min Temp", "Max Temp", "Rainfall", "Weekend", "Month", "Day", "Beer Consumption"]]

final_data.head()
from sklearn.model_selection import train_test_split

train, test = train_test_split(final_data, test_size = 0.2, random_state = 25) 

print(train.shape)

print(test.shape)
df = train

df.describe()
df.hist(figsize = (15, 18))
df.boxplot(figsize=(8, 8), column = ["Min Temp", "Avg Temp", "Max Temp", "Beer Consumption"])
corr = df.corr()

sns.heatmap(corr)
fig, ax = plt.subplots(1, 3, figsize=(14, 4))

ax[0].scatter(df["Beer Consumption"], df["Min Temp"])

ax[1].scatter(df["Beer Consumption"], df["Avg Temp"])

ax[2].scatter(df["Beer Consumption"], df["Max Temp"])
by_month = df[["Month", "Beer Consumption"]]

by_month = by_month.groupby("Month").mean().reset_index()

by_month["Name"] = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

by_month.sort_values(by=["Beer Consumption"], inplace=True, kind="heapsort")

by_month.reset_index(inplace=True)

by_month.drop(["index"], axis=1, inplace=True)

plt.figure(figsize=(12, 4))

plt.bar(by_month["Name"], by_month["Beer Consumption"])
order_Month = by_month["Month"].tolist()

order_Month = {v : k+1 for k, v in enumerate(order_Month)}

order_Month
def editMonth(df):

    for index, row in df.iterrows():

        #print(row["Month"], order_Month[row["Month"]])

        row["Month"] = order_Month[row["Month"]]

    return df

df = editMonth(df)

df.head()
plt.scatter(df["Month"], df["Beer Consumption"])
print(final_data[["Month", "Beer Consumption"]].corr())

print(df[["Month", "Beer Consumption"]].corr())
by_day = df[["Day", "Beer Consumption"]]

by_day = by_day.groupby("Day").mean().sort_values(by="Beer Consumption").reset_index()

plt.figure(figsize=(16, 4))

plt.bar(range(1, 32), by_day["Beer Consumption"], tick_label=by_day["Day"])
order_Day = by_day["Day"].tolist()

order_Day = {v : k+1 for k, v in enumerate(order_Day)}

print(order_Day)

def editDay(df):

    for index, row in df.iterrows():

        #print(row["Day"], order_Day[row["Day"]])

        row["Day"] = order_Day[row["Day"]]

    return df

df = editDay(df)

df.head()
plt.scatter(df["Day"], df["Beer Consumption"])
print(final_data[["Day", "Beer Consumption"]].corr())

print(df[["Day", "Beer Consumption"]].corr())
co = df.corr()

sns.heatmap(co)
rest_data = df[["Rainfall", "Weekend", "Beer Consumption"]]

fig, ax = plt.subplots(1, 2)

ax[0].scatter(rest_data["Weekend"], rest_data["Beer Consumption"], c=rest_data["Weekend"])

ax[1].scatter(rest_data["Rainfall"], rest_data["Beer Consumption"])
rest_data.corr()
plt.hist(x = [rest_data["Beer Consumption"].where(rest_data["Weekend"]==1).dropna(), rest_data["Beer Consumption"].where(rest_data["Weekend"]==0).dropna()], color=["blue", "red"], histtype="step")
plt.hist(rest_data["Rainfall"], log=True)
def logRainfall(df):

    df["Rainfall"] = df["Rainfall"].apply(np.log)

    #df["Rainfall"].replace(to_replace = (-np.inf), value=0)

    df.loc[df["Rainfall"] == -np.inf, "Rainfall"] = 0

    return df

df = logRainfall(df)

df["Rainfall"]
sns.heatmap(df.corr())

print(df.corr()["Beer Consumption"])
test = editMonth(test)

test = editDay(test)

test = logRainfall(test)
test
x_train, y_train = df.drop(["Beer Consumption"], axis=1), df["Beer Consumption"]

x_test, y_test = test.drop(["Beer Consumption"], axis=1), test["Beer Consumption"]

print(x_train.shape, y_train.shape)

print(x_test.shape, y_test.shape)
from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(x_train, y_train)

lr.coef_
predict = lr.predict(x_test)

print(predict[:5])

print(y_test[:5])
lr.score(x_test, y_test)#r2 score
from sklearn.metrics import mean_squared_error

ans = np.sqrt(mean_squared_error(y_test, predict))
ans