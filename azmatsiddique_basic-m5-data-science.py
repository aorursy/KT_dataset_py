import pandas as pd 

import datetime

cal= pd.read_csv(r"/kaggle/input/m5-forecasting-accuracy/calendar.csv")

sell= pd.read_csv(r"/kaggle/input/m5-forecasting-accuracy/sell_prices.csv")

train= pd.read_csv(r"/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv")

ss= pd.read_csv(r"/kaggle/input/m5-forecasting-accuracy/sample_submission.csv")
cal.shape
cal.isnull().sum()
sell.head()
# includes all the items of 3 states= 30490

# shows sale history of every item



train.head()
train.shape
ss.head()
sell.head()
train_state= train.drop(labels= ['id', 'item_id', 'dept_id', 'cat_id', 'store_id'],axis=1)

train_state= train_state.groupby("state_id", as_index=False).sum()

train_state= train_state.T

train_state= train_state.rename(columns=train_state.iloc[0]).drop(train_state.index[0])

train_state= train_state.reset_index()

train_state= train_state.rename(columns={"index":"d"})

train_state= pd.merge(train_state, cal, how="inner", on="d")
import plotly.graph_objects as go

fig = go.Figure()



fig.add_trace(go.Scatter(x=train_state["date"], y=train_state["CA"], name="CA",line_color='deepskyblue',opacity=0.8))

fig.add_trace(go.Scatter(x=train_state["date"], y=train_state["TX"], name="TX",line_color='magenta', opacity=0.8))

fig.add_trace(go.Scatter(x=train_state["date"], y=train_state["WI"], name="WI",line_color='greenyellow',opacity=0.8))

               

fig.update_layout(title_text='Total Monthly Sale per State', xaxis_rangeslider_visible=True)

fig.show()
train_m= train_state.copy()

train_m["year"]= pd.DatetimeIndex(train_state["date"]).year

train_m= train_m[["CA","TX","WI","year","month","date"]]

train_m= train_m.groupby(["year","month"], as_index=False).sum()

train_m= train_m.drop(labels=["date"], axis=1)



train_d=train_state.copy()

train_d["year"]= pd.DatetimeIndex(train_state["date"]).year
train_d
train_m
train1=train.copy()

train1["Total Sale"]= train1.sum(axis=1)
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



plt.figure(figsize=(12,7))

plt.style.use('seaborn-darkgrid')



sns.boxenplot(x="store_id", y="Total Sale",color="r",scale="linear", data=train1)

plt.title("Total Sale per Store", fontsize=18)
train2= train.drop(labels= ["id", "dept_id", "item_id"], axis=1)

train2= train2.groupby(["state_id", "store_id", "cat_id"], as_index=False).sum()

train2["Total Sale"]=train2.sum(axis=1)

train2= train2[["state_id", "store_id", "cat_id", "Total Sale"]]
import plotly.express as px



fig = px.bar(train2, x="store_id", y="Total Sale",color="cat_id",  barmode="group",  facet_row="state_id", 

             category_orders={"state_id": ["CA", "TX", "WI"]}, title="Total Sale per Category per Store",height=700)

fig.show()
train_cat= train.drop(labels= ['id', 'item_id', 'dept_id', 'store_id', 'state_id'],axis=1)

train_cat= train_cat.groupby("cat_id", as_index=False).sum()

train_cat= train_cat.T

train_cat= train_cat.rename(columns=train_cat.iloc[0]).drop(train_cat.index[0])

train_cat= train_cat.reset_index()

train_cat= train_cat.rename(columns={"index":"d"})

train_cat= pd.merge(train_cat, cal, how="inner", on="d")



train_m= train_cat[["FOODS","HOBBIES","HOUSEHOLD","month"]].groupby("month", as_index=False).sum()

train_d= train_cat[["FOODS","HOBBIES","HOUSEHOLD","weekday"]].groupby("weekday", as_index=False).sum()
import matplotlib.pylab as pl

import matplotlib.gridspec as gridspec

gs=gridspec.GridSpec(1,2)

plt.figure(figsize=(15,5))



ax=pl.subplot(gs[0,0])

plt.bar(train_m["month"], train_m["FOODS"], color="salmon")

plt.bar(train_m["month"], train_m["HOUSEHOLD"], color="orange")

plt.bar(train_m["month"], train_m["HOBBIES"], color="cyan")

plt.ylabel("Overall Sale", fontsize=22)

plt.legend(["FOODS", "HOUSEHOLD", "HOBBIES"])



ax=pl.subplot(gs[0,1])

plt.plot(train_d["weekday"], train_d["FOODS"], color="salmon", marker="o", markersize=8, linewidth=3.5)

plt.plot(train_d["weekday"], train_d["HOUSEHOLD"], color="orange", marker="o", markersize=8, linewidth=3.5)

plt.plot(train_d["weekday"], train_d["HOBBIES"], color="cyan", marker="o", markersize=8, linewidth=3.5)

plt.legend(["FOODS", "HOUSEHOLD", "HOBBIES"])



plt.tight_layout()
import matplotlib.pylab as pl

import matplotlib.gridspec as gridspec



CA= train_state[train_state["snap_CA"]==0]

CA_snap= train_state[train_state["snap_CA"]==1]



TX= train_state[train_state["snap_TX"]==0]

TX_snap= train_state[train_state["snap_TX"]==1]



WI= train_state[train_state["snap_WI"]==0]

WI_snap= train_state[train_state["snap_WI"]==1]



gs=gridspec.GridSpec(3,3)

plt.figure(figsize=(12,8))



ax=pl.subplot(gs[0,0])

sns.set(style="darkgrid")

ax = sns.countplot(x="snap_CA", data=train_state)

plt.ylabel("No. of Days", fontsize=10)

plt.title("CA",fontsize=15)



ax=pl.subplot(gs[0,1])

sns.set(style="darkgrid")

ax = sns.countplot(x="snap_TX", data=train_state)

# plt.ylabel("No. of Days", fontsize=10)

plt.title("TX",fontsize=15)



ax=pl.subplot(gs[0,2])

sns.set(style="darkgrid")

ax = sns.countplot(x="snap_WI", data=train_state)

# plt.ylabel("No. of Days", fontsize=10)

plt.title("WI",fontsize=15)



ax=pl.subplot(gs[1:,0])

plt.pie([CA["CA"].sum(), CA_snap["CA"].sum()], autopct="%1.1f%%", colors=["blue", "orange"],

        startangle=90, explode=[0, 0.1], shadow=True)

plt.legend(["sale without SNAP purchase", "sale with SNAP purchase"], loc="best", fontsize=9)



ax=pl.subplot(gs[1:,1])

plt.pie([TX["TX"].sum(), TX_snap["TX"].sum()], autopct="%1.1f%%", colors=["blue", "pink"],

        startangle=90, explode=[0, 0.1], shadow=True)

plt.legend(["sale without SNAP purchase", "sale with SNAP purchase"], loc="lower right", fontsize=9)



ax=pl.subplot(gs[1:,2])

plt.pie([WI["WI"].sum(), WI_snap["WI"].sum()], autopct="%1.1f%%", colors=["blue", "yellowgreen"],

        startangle=90, explode=[0, 0.1], shadow=True)

plt.legend(["sale without SNAP purchase", "sale with SNAP purchase"], loc="lower right", fontsize=9)  



plt.tight_layout()
event1= cal[cal["event_name_1"].notnull()]

event1.loc[85, "event_name_1"]="OrthodoxEaster + Easter"

event1.loc[827, "event_name_1"]="OrthodoxEaster + Cinco De Mayo"

event1.loc[1177, "event_name_1"]="Easter + OrthodoxEaster"

event1.loc[1233, "event_name_1"]="NBAFinalsEnd + Father's day"

event1.loc[1968, "event_name_1"]="NBAFinalsEnd + Father's day"



event1= pd.merge(train_state[["d","CA","TX","WI"]], event1[["d","event_name_1"]], on="d", how="inner").drop(labels=["d"], axis=1)

event1["Total Sale"]= event1["CA"] + event1["TX"] + event1["WI"]

event1= event1.groupby("event_name_1", as_index=False).sum() 



plt.figure(figsize=(15,7))

plt.style.use("bmh")



plt.plot(event1["event_name_1"],event1["Total Sale"], color="red", marker="o", markerfacecolor="black", linewidth=1)

plt.xlabel("Events", fontsize=15)

plt.ylabel("Total Sale", fontsize=15)

plt.title("Overall Sale on Holidays", fontsize=20)

plt.xticks(rotation="vertical")
import seaborn as sns; sns.set()

import matplotlib.pyplot as plt

plt.figure(figsize=(15,7))

ax = sns.lineplot(event1["event_name_1"],event1["Total Sale"])