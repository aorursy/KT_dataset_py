import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import scipy as sp
data = pd.read_csv("../input/BreadBasket_DMS.csv")
data.info()
data.head(5)
data["DateTime"] = pd.to_datetime((data.Date + " " + data.Time))
data.drop(columns=["Date", "Time"], inplace=True)
data["DateTime"].dt.hour.value_counts().sort_index()
data = (data.loc[(data["DateTime"].dt.hour.between(8, 17, inclusive=True))]
        .reset_index(drop=True))

data["Weekday"] = data["DateTime"].dt.day_name()
data["Hour"] = data["DateTime"].dt.hour
np.sort(data.Item.unique())
data.loc[data["Item"]=="NONE"].shape[0]
data.drop(index=data[data["Item"] == "NONE"].index, inplace=True)
data.reset_index(drop=True, inplace=True)

data.head(5)
totsales = data.groupby(data.DateTime.dt.date).Item.count().to_frame()
totsales.index = pd.to_datetime(totsales.index)

totsales.rename(columns={"Item":"ItemsSold"}, inplace=True)
totsales["UniqueTrans"] = data.groupby(data.DateTime.dt.date).Transaction.nunique()
totsales["ItemsperTrans"] = totsales.ItemsSold / totsales.UniqueTrans
totsales["Weekday"] = totsales.index.day_name()

totsales.head(5)
def subplot_properties(titles):
    fig = plt.gcf() # get current figure
    for i, axis in enumerate(fig.axes): # iterate over the axes (subplots) stored in the figure
        axis.yaxis.grid(True)
        axis.set_xlabel("")
        axis.set_ylabel("")
        sns.despine(ax=axis, left=True, bottom=True)
        axis.set_title(titles[i], fontweight="bold")
        plt.sca(axis)
        plt.yticks(fontweight="light")
    sns.despine(ax=axis, left=True, bottom=False, offset=15)
    plt.xticks(fontweight="light")
    fig.subplots_adjust(hspace=0.25)

%config InlineBackend.figure_format = 'svg'
sns.set(font="sans-serif", palette="deep", style="white", context="notebook")

fig, ax = plt.subplots(3,1,figsize=(10,8), sharex=True)

ax[0].plot(totsales.index, totsales.ItemsSold)
ax[1].plot(totsales.index, totsales.UniqueTrans)
ax[2].plot(totsales.index, totsales.ItemsperTrans)

plottitles = ["Total Items Sold", "Total Transactions", "Items Sold per Transaction"]
subplot_properties(plottitles)

plt.show()
totsales.iloc[:,:3].corr()
sp.stats.variation(a=totsales.iloc[:,:3])
totsales.groupby("Weekday").ItemsSold.mean().to_frame().sort_values(by="ItemsSold", ascending=False)
weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

fig, ax = plt.subplots(3, 1, figsize=(10,8), sharex=True)

sns.boxplot(x="Weekday", y="ItemsSold", data=totsales, order=weekdays, ax=ax[0])
sns.boxplot(x="Weekday", y="UniqueTrans", data=totsales, order=weekdays, ax=ax[1])
sns.boxplot(x="Weekday", y="ItemsperTrans", data=totsales, order=weekdays, ax=ax[2])

plottitles2 = [i + " by Weekday" for i in plottitles]
subplot_properties(plottitles2)

plt.show()