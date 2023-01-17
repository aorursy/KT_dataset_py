import numpy as np

import pandas as pd

pd.set_option("display.max_columns",100)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

params={"axes.titlesize":16,

        "axes.titleweight":"bold",

        "axes.titlelocation":"center"}

plt.rcParams.update(params)

import warnings

warnings.filterwarnings("ignore")
df=pd.read_csv("../input/hotel-booking-demand/hotel_bookings.csv")

df.head(3)
print("There were {num} bookings!".format(num=df.shape[0]))
df.info()
df.isnull().sum().sort_values(ascending=False)
null_replacements={"company":0,"agent":0,"country":"Unknown","children":0}

df.fillna(null_replacements,inplace=True)
df.isnull().sum()
df.describe().T
df.drop(df.loc[(df["adults"]==0)&(df["children"]==0)&(df["babies"]==0)].index,inplace=True)
print("Now instead of the inital 119390, we have {num} bookings.".format(num=df.shape[0]))
df["adr"].loc[df["adr"]<0]=0
df["adr"].min()
df["adr"].loc[df["hotel"]=="City Hotel"].sort_values(ascending=False).head(5)
df["adr"].loc[df["adr"]==5400]=540
df["adr"].loc[df["hotel"]=="City Hotel"].sort_values(ascending=False).head(5)
for col in df.select_dtypes(include=["object"]).drop(["country","reservation_status_date"],axis=1):

    print(col)

    print(df.select_dtypes(include=["object"]).drop(["country","reservation_status_date"],axis=1)[col].value_counts())

    print("")
df["meal"].replace("Undefined","SC",inplace=True)
df["meal"].value_counts()
hotel=pd.DataFrame(df["hotel"].value_counts().to_list(),index=df["hotel"].value_counts().index.to_list())



hotel[0].plot(kind="pie",figsize=(8,6),autopct="%1.1f%%",startangle=90,explode=[0,0.05],colors=["skyblue","lightsalmon"],textprops={"fontsize":14})

plt.ylabel("")

plt.title("Which hotels are guests booking?",y=0.93)
ax=sns.countplot(x=df["hotel"],hue=df["is_canceled"],palette={0:"lightgreen",1:"palevioletred"},saturation=0.7)

plt.title("Number of bookings that were canceled at each hotel",y=1.03)

plt.xlabel("")

plt.ylabel("Count")

plt.legend(("Not Canceled","Canceled"))
ax=sns.countplot(df["deposit_type"].loc[df["is_canceled"]==1],palette={"No Deposit":"paleturquoise","Non Refund":"lavender","Refundable":"black"})

plt.xlabel("")

plt.ylabel("Count")

plt.title("Number of deposit types from cancelled bookings")



i=0

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,height+0.1,

        df["deposit_type"].loc[df["is_canceled"]==1].value_counts()[i],ha="center")

    i += 1
df["is_canceled"].value_counts()
sns.countplot(df["is_repeated_guest"],hue=df["is_canceled"],palette={0:"lightgreen",1:"palevioletred"},saturation=0.7)

plt.title("Number of first time and repeated guests who canceled",y=1.03)

plt.xlabel("")

plt.xticks((0,1),("First Time Guest","Repeated Guest"))

plt.ylabel("Count")

plt.legend(("Not Canceled","Canceled"))
fig=plt.figure()

ax1=fig.add_axes([0,0,1,1])

ax2=fig.add_axes([0.3,0.3,0.6,0.6])

sns.countplot(df["previous_cancellations"].loc[(df["is_canceled"]==1)],palette="YlOrRd",ax=ax1)

sns.countplot(df["previous_cancellations"].loc[(df["is_canceled"]==1)&(df["previous_cancellations"]!=0)&(df["previous_cancellations"]!=1)],palette="YlOrRd",ax=ax2)

ax1.set_title("Number of cancellations prior to booking the current cancelled booking",fontweight="bold",fontsize=16,y=1.01)

ax1.set_xlabel("Number of cancellations")

ax2.set_xlabel("")

ax1.set_ylabel("Count")

ax2.set_ylabel("")
df["previous_cancellations"].loc[(df["is_canceled"]==1)].value_counts()
waitinglist=pd.DataFrame({"Days":df["days_in_waiting_list"].loc[(df["is_canceled"]==1)].value_counts().head(10).index,"Count":df["days_in_waiting_list"].loc[(df["is_canceled"]==1)].value_counts().head(10).values})

waitinglistorder=[0,39,31,44,35,46,69,45,41,62]



fig=plt.figure()

ax1=fig.add_axes([0,0,1,1])

ax2=fig.add_axes([0.3,0.3,0.6,0.6])

sns.barplot(x=waitinglist["Days"],y=waitinglist["Count"],order=waitinglistorder,palette="Purples",ax=ax1)

sns.barplot(x=waitinglist["Days"].drop([0]),y=waitinglist["Count"].drop([0]),order=waitinglistorder,palette="Purples",ax=ax2)

ax1.set_title("Number of days the cancelled booking was in the waiting list before it was confirmed",fontweight="bold",fontsize=16)

ax1.set_xlabel("Number of days")

ax2.set_xlabel("")

ax1.set_ylabel("Count")

ax2.set_ylabel("")

ax2.set_xlim([0.5,9.5])
import folium

import json



country_geo=json.load(open("../input/python-folio-country-boundaries/world-countries.json"))

country=pd.DataFrame({"Country":df["country"].value_counts().index.to_list(),"Number":df["country"].value_counts().to_list()})



m=folium.Map(width=600,height=400,location=[39.3999,8.2245],zoom_start=2)

m.choropleth(geo_data=country_geo,data=country,columns=["Country","Number"],key_on="feature.id",fill_color="PuRd",fill_opacity=0.7,line_opacity=2,legend_name="Number of Guests")

m
plt.figure(figsize=(12,6))

sns.barplot(x=country["Country"].head(20),y=country["Number"].head(20),palette="Pastel2")

plt.xlabel("")

plt.ylabel("Number of Guests")

plt.title("Top 20 guests country of origin")
fig,axes=plt.subplots(1,3,sharey=True,figsize=(12,6))

sns.countplot(df["adults"],hue=df["hotel"],palette=["skyblue","lightsalmon"],ax=axes[0])

sns.countplot(df["children"],hue=df["hotel"],palette=["skyblue","lightsalmon"],ax=axes[1])

sns.countplot(df["babies"],hue=df["hotel"],palette=["skyblue","lightsalmon"],ax=axes[2])

axes[0].set_xlabel("Number of Adults")

axes[1].set_xlabel("Number of Children")

axes[2].set_xlabel("Number of Babies")

axes[0].set_ylabel("Count")

axes[1].set_ylabel("")

axes[2].set_ylabel("")

axes[0].get_legend().set_visible(False)

axes[1].legend(ncol=2,bbox_to_anchor=(0.95,1.09))

axes[2].get_legend().set_visible(False)

plt.suptitle("Number of adults, children and babies",fontweight="bold",fontsize=16,y=0.98)
fig,axes=plt.subplots(1,2,sharey=True,figsize=(10,4))

sns.countplot(df["children"].loc[df["adults"]==0],palette="Oranges",ax=axes[0])

sns.countplot(df["babies"].loc[df["adults"]==0],palette="Blues",ax=axes[1])

axes[0].set_xlabel("Number of Children")

axes[1].set_xlabel("Number of Babies")

axes[0].set_ylabel("Count")

axes[1].set_ylabel("")

plt.suptitle("Number of children and babies on holiday without an adult",fontweight="bold",fontsize=16,y=0.97)
df["children"].loc[(df["adults"]==0)&(df["babies"]==1)]
sns.countplot(df["customer_type"],order=df["customer_type"].value_counts().index,palette="RdPu")

plt.title("Types of guests")

plt.xlabel("")

plt.ylabel("Count")
df["customer_type"].value_counts()
fig,axes=plt.subplots(1,2,sharey=True,figsize=(14,4))

sns.countplot(df["market_segment"],order=df["market_segment"].value_counts().index,palette="twilight_shifted_r",ax=axes[0])

sns.countplot(df["distribution_channel"],order=df["distribution_channel"].value_counts().index,palette="Greens",ax=axes[1])

axes[0].set_xlabel("Market segment designation")

axes[1].set_xlabel("Booking distribution channel")

axes[0].set_xticklabels(df["market_segment"].value_counts().index,rotation=40)

axes[1].set_xticklabels(df["distribution_channel"].value_counts().index,rotation=40)

axes[0].set_ylabel("Count")

axes[1].set_ylabel("")

plt.suptitle("Market segments and distribution channels",fontweight="bold",fontsize=16,y=0.97)
df["agent"].value_counts().head(3)
fig,axes=plt.subplots(1,2,sharey=True,figsize=(10,4))

sns.countplot(df["total_of_special_requests"],palette="YlOrBr",ax=axes[0])

sns.countplot(df["required_car_parking_spaces"],palette="YlGn",ax=axes[1])

axes[0].set_xlabel("Number of special requested")

axes[1].set_xlabel("Number of car parking spaces requested")

axes[0].set_ylabel("Count")

axes[1].set_ylabel("")

plt.suptitle("Number of requests",fontweight="bold",fontsize=16,y=0.97)
fig,ax=plt.subplots(figsize=(8,4),subplot_kw=dict(aspect="equal"))



labels=["Bed and Breakfast","Half Board","No Meal Package","Full Board"]

values=df["meal"].value_counts().to_list()



wedges,texts=ax.pie(values,wedgeprops=dict(width=0.4),startangle=215,colors=["mediumaquamarine","moccasin","paleturquoise","lightpink"])



kw=dict(arrowprops=dict(arrowstyle="-"),bbox=dict(boxstyle="square,pad=0.3",fc="w",ec="k",lw=0.72),zorder=0,va="center")



for i,p in enumerate(wedges):

    ang=(p.theta2-p.theta1)/2.+p.theta1

    y=np.sin(np.deg2rad(ang))

    x=np.cos(np.deg2rad(ang))

    horizontalalignment={-1:"right",1:"left"}[int(np.sign(x))]

    connectionstyle="angle,angleA=0,angleB={}".format(ang)

    kw["arrowprops"].update({"connectionstyle":connectionstyle})

    ax.annotate(labels[i],xy=(x,y),xytext=(1.35*np.sign(x),1.4*y),horizontalalignment=horizontalalignment,**kw)



ax.set_title("Types of meals booked")



plt.show()
df["meal"].value_counts()
fig=plt.figure(figsize=(12,6))

ax1=plt.subplot2grid((4,1),(0,0))

ax2=plt.subplot2grid((4,1),(1,0),rowspan=3)

sns.boxplot(df["lead_time"],whis=30,color="plum",ax=ax1)

sns.distplot(a=df["lead_time"],kde=False,color="plum",hist_kws=dict(edgecolor="black",linewidth=2),ax=ax2)

plt.suptitle("Number of days between the booking date and the arrival date",fontweight="bold",fontsize=16,y=0.93)

ax1.set_xlabel("")

ax1.set_xticklabels([])

ax2.set_xlabel("Number of days")

ax2.set_ylabel("Count")
df["lead_time"].describe()
sns.countplot(df["arrival_date_year"],hue=df["hotel"],palette=["lightsalmon","skyblue"])

plt.title("Number of arrivals for 2015, 2016 and 2017")

plt.xlabel("")

plt.ylabel("Count")

plt.legend()
months_map={"January":1,"February":2,"March":3,"April":4,"May":5,"June":6,"July":7,"August":8,"September":9,"October":10,"November":11,"December":12}

df["arrival_date_month"]=df["arrival_date_month"].map(months_map)



plt.figure(figsize=(12,6))

sns.countplot(df["arrival_date_month"],hue=df["hotel"],palette=["lightsalmon","skyblue"])

sns.lineplot(x=df["arrival_date_month"].value_counts().index-1,y=df["arrival_date_month"].value_counts().values,color="mediumpurple",marker="o",label="Total")

plt.title("Number of arrivals in each month")

plt.xlabel("Month")

plt.ylabel("Count")
plt.figure(figsize=(12,6))

sns.countplot(df["arrival_date_day_of_month"],hue=df["hotel"],palette=["lightsalmon","skyblue"])

sns.lineplot(x=df["arrival_date_day_of_month"].value_counts().index-1,y=df["arrival_date_day_of_month"].value_counts().values,color="mediumpurple",marker="o",label="Total")

plt.title("Number of arrivals on each day of the month")

plt.xlabel("Day of the month")

plt.ylabel("Count")
fig,axes=plt.subplots(1,2,sharey=True,figsize=(16,6))

sns.countplot(df["stays_in_weekend_nights"],hue=df["hotel"],palette=["lightsalmon","skyblue"],ax=axes[0])

sns.countplot(df["stays_in_week_nights"],hue=df["hotel"],palette=["lightsalmon","skyblue"],ax=axes[1])

axes[0].set_xlabel("Number of weekend nights (Saturday or Sunday)")

axes[1].set_xlabel("Number of weekday nights (Monday to Friday)")

axes[0].set_ylabel("Count")

axes[1].set_ylabel("")

axes[0].get_legend().set_visible(False)

axes[1].legend(ncol=2,bbox_to_anchor=(0.15,1.09))

plt.suptitle("Number of weekend/weekday nights booked",fontweight="bold",fontsize=16,y=0.98)
fig,axes=plt.subplots(figsize=(10,4))

sns.lineplot(x=df["arrival_date_month"],y=df["adr"],hue=df["hotel"],palette=["lightsalmon","skyblue"],ci=95)

plt.title("Average daily rate")

plt.ylabel("Average daily rate (€)")

plt.xlabel("Month")

plt.xticks(range(1,13))

handles,labels=axes.get_legend_handles_labels()

axes.legend(handles=handles[1:],labels=labels[1:])
df.groupby(["hotel"]).adr.agg(["mean","std","min","median","max"])
df.groupby(["hotel","arrival_date_month"]).adr.agg(["mean","std","min","median","max"])