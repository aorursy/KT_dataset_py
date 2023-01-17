import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import plotly.graph_objects as go
data = pd.read_csv("../input/gtd/globalterrorismdb_0718dist.csv", encoding = "ISO-8859-1")
data.columns
data.rename(columns={'iyear':'Year','imonth':'Month','iday':'Day','country_txt':'Country','region_txt':'Region','attacktype1_txt':'AttackType','target1':'Target','nkill':'Killed','nwound':'Wounded','summary':'Summary','gname':'Group','targtype1_txt':'Target_type','weaptype1_txt':'Weapon_type','motive':'Motive'},inplace=True)
data = data[['Year','Month','Day','Country','Region','city','latitude','longitude','AttackType','Killed','Wounded','Target','Summary','Group','Target_type','Weapon_type','Motive']]
data.head(30)
data.isnull().sum()
data.info()
data.describe()
data.corr
data.columns
data.AttackType.value_counts()
data.AttackType.value_counts().plot( kind= "pie", figsize=(12,12),shadow=True, startangle=20,autopct='%1.1f%%',labels=None)

plt.legend(title="Attack Type",

          loc="center",

          bbox_to_anchor=(0.1, 0, 2, 2), labels = data.AttackType.value_counts().index)
data.Country.value_counts()
data.Country.value_counts()[:20].plot( kind = "barh",color = "g", title = " Top 20 Country")
data.Region.value_counts()
data.Killed.value_counts()[:5]
list1 = []

for i in range(1, 6):

    list1.append("{} killed".format(i))

list1.append("5+ killed")
list2 = []

for i in range(5):

    data_i = data.Killed.value_counts()[i]

    list2.append(data_i)

list2.append(data.Killed.value_counts().sum())

list2
fig = go.Figure(data=[go.Pie(labels=list1, values=list2, hole = 0.3)])

fig.update_layout(annotations=[dict(text='Killed', x=0.5, y=0.5, font_size=30, showarrow=False)])

fig.update_layout(title_text='Ratio of The Number of Killed')

fig.show()
data.Target.value_counts()[:30]
data.Group.value_counts()
data.Group.value_counts()[:20].to_frame()

ax = data.Group.value_counts().drop("Unknown")[:20].plot(kind ="barh",figsize=(12, 8),color='#86bf03',width=0.75, title = "Groups")

ax.set_xlabel("Number Of Death", labelpad=5, weight='bold', size=10)
Group = data.Group.value_counts()
ExceptUnknown = Group.drop ("Unknown").sum()

Unknown = data.Group.value_counts().Unknown
values =[ExceptUnknown, Unknown]

values
fig = go.Figure(data=[go.Pie(labels=["ExceptUnknown","Unknown"], values=[ExceptUnknown, Unknown], hole = 0.4,)])

fig.update_layout(annotations=[dict(text = "$Ratio$", x=0.5, y=0.5, font_size=30, showarrow=False)],title ="Groups vs Unknown")

fig.show()
data.city.value_counts()[:10]
Badhdad = data[data.city=="Baghdad"]
list4 = []

for i in range (1,12):

    city_i = data.city.value_counts().index[i]

    list4.append(city_i)
list4
data.city.value_counts().index[10]
fig = plt.figure(figsize=(15,10))

fig.suptitle('Number of Death of Top Ten Cities')



for i in range(1,11):

    fig.subplots_adjust(hspace=0.7, wspace = 0.2)

    df0= data[data.city == data.city.value_counts().index[i]]

    fig.add_subplot(5,2,i)

    df0.Year.plot(kind = "hist",title =data.city.value_counts().index[i], legend=False)

    plt.xlabel("Years")

    plt.ylabel("Number of Death")



plt.show()
fig = plt.figure(figsize=(15,10))

fig.suptitle('Number of Death of Top Nine Cities')



for i in range(1,10):

    fig.subplots_adjust(hspace=0.7, wspace = 0.2)

    df_city= data[data.city == data.city.value_counts().index[i]]

    fig.add_subplot(3,3,i)

    df_city.Year.plot(kind="hist",title =data.city.value_counts().index[i], legend=False)

    plt.title("{}".format(data.city.value_counts().index[i]))

    plt.ylabel('Number of Death')

    plt.xlabel('Years')



plt.show()

data.Year.value_counts()
data.Year.plot(kind = "hist", color="r",grid = True, bins = 100,figsize = (14, 5))

plt.title("Number of Terrorist Attack Each Year")

plt.ylabel("Number of Attack")

plt.xlabel("Year")
data.Year.plot(kind = "hist", color="r",grid = True, bins = 100,figsize = (14, 5), cumulative = True)

plt.title("Cummilative Terrorist Attack Number")

plt.ylabel("Number of Attack")

plt.xlabel("Year")
data["Wounded"].dropna(inplace= True)
assert data["Wounded"].notnull().all()
data.Wounded.astype("int64")
first_filter = data.Wounded > 50
second_filter = data.Killed > 500
data[first_filter & second_filter]
data2 = data.copy()
data2
data2.dropna()
data_iraq = data[data2.Country == "Iraq"]
data_iraq.dropna()
data4 = data_iraq.dropna()
data4
melted = pd.melt(frame = data4, id_vars ="city", value_vars=["Killed"])

melted

plt.subplot(1,2,1)

data_iraq.dropna().boxplot(column = "latitude")

plt.subplot(1,2,2)

data_iraq.dropna().boxplot(column = "longitude")

plt.show()
plt.subplot(1,2,1)

green_diamond = dict(markerfacecolor='g', marker='d')

red_square = dict(markerfacecolor='r', marker='s')

data2.dropna().boxplot(column = "latitude",flierprops=green_diamond)

plt.subplot(1,2,2)

data2.dropna().boxplot(column = "longitude", flierprops=red_square)

plt.show()
data2.dropna().set_index(["Country", "city","Year", "Month"])
data2.dropna().groupby("Country").Killed.mean().nlargest(20)

data2.dropna().groupby("Country").Killed.sum().nlargest(20)
data.shape
data3 = data2.dropna().copy()
data3

data3["YMD"] = data3["Year"].map(str)+"-"+data3["Month"].map(str) +"-"+ data3["Day"].map(str)
data5 = data3.copy().drop(["Day","Month","Year"], axis = 1)
datatime_object = pd.to_datetime(data5["YMD"],errors='coerce')
data5["date"] = datatime_object
data5 = data5.set_index("date")
data6 = data5.resample("A").mean().dropna()
data6.Killed.plot(kind="line", figsize = (15,10))

plt.ylabel("Mean Value of Number of Death by Year")
data5.resample("M").mean().dropna()
data8 = data5.resample("M").sum()
data8.Killed.plot(kind="line", figsize = (15,10))

plt.ylabel("Summation of Number of Death")
data7 = data5.resample("M").mean().dropna()
data7.Killed.plot(kind="line", figsize = (15,10))

plt.ylabel("Mean Value of Number of Death by Mounth")
dataFoTT = data2.dropna().Target_type.value_counts()
dataSoTT = data2.dropna().groupby("Target_type").Killed.sum()
Ratio = dataSoTT/dataFoTT

Ratio
Ratio.plot(kind="barh",color="r", figsize = (15,10), title= "Risk of Death in Target Type ")
