import pandas as pd

import matplotlib.pyplot as plt

from matplotlib import style

%matplotlib inline





style.use("ggplot")

import plotly

import plotly.express as px

import plotly.graph_objects as go

import plotly.offline as pyo

from plotly.offline import init_notebook_mode,plot,iplot

import seaborn as sns



plt.rcParams["figure.figsize"]=17,8

import cufflinks as cf



import folium

pyo.init_notebook_mode(connected = True)

cf.go_offline()
df =pd.read_excel("../input/covid19/Covid cases in India.xlsx")

df.head()
df.drop(["S. No."] ,axis =1 ,inplace=True)

df
df["Total cases"] = df["Total Confirmed cases (Indian National)"] + df["Total Confirmed cases ( Foreign National )"]
df
df["Total cases"].sum()
df["Active cases"] = df["Total cases"]-(df["Death"]+df["Cured"])

df
df["Active cases"].sum()
df.style.background_gradient(cmap ="Reds")
total_active_cases = df.groupby(by="Name of State / UT")['Active cases'].sum().sort_values(ascending =False).to_frame()

total_active_cases.style.background_gradient(cmap="Reds")
df.plot(kind="bar" ,x="Name of State / UT" , y="Total cases",color="red")

df.iplot(kind="bar",x="Name of State / UT" , y="Total cases")
plt.bar(df["Name of State / UT"] , df["Total cases"],)



px.bar(df,x="Name of State / UT" , y="Total cases")
df.plot(kind="scatter" ,x="Name of State / UT" , y="Total cases")



df.iplot(kind="scatter",x="Name of State / UT" , y="Total cases",mode="markers+lines")

px.scatter(df,x="Name of State / UT" , y="Total cases",)
plt.scatter(df["Name of State / UT"] , df["Total cases"])
# px.scatter(df,x="Name of State / UT" , y="Total cases")



#object oriented plotly

fig = go.Figure()

fig.add_trace(go.Bar(x=df['Name of State / UT'] ,y=df["Total cases"]))

fig





# fig.update_layout(title ="Total cases in india",xaxis = dict(tital ="State") , yaxis = dict(tital="total cases"))
x=px.bar(df , df["Name of State / UT"] , y=df["Total cases"])

x
indian_cord = pd.read_excel('../input/covid19/Indian Coordinates.xlsx')

indian_cord.head()
df_full = pd.merge(indian_cord ,df,on="Name of State / UT")

# df_full.head()
map = folium.Map(location=[20,70],zoom_start=4,tiles="stamenterrain",width="80%" , height="55%",)



for lat,long,value,name in zip(df_full['Latitude'],df_full['Longitude'],df_full['Total cases'],df_full['Name of State / UT']):

    folium.CircleMarker([lat,long],radius=value*0.5 , popup=('<strong>State</strong>: ' + str(name).capitalize() + '<br>''<strong>Total Cases</strong>: ' + str(value) + '<br>') ,color="red",fill_color="red" , fill_opacity =0.2).add_to(map)



map
a=zip(df_full['Latitude'],df_full['Longitude'],df_full['Total cases'],df_full['Name of State / UT'])

print(tuple(a))
f, ax = plt.subplots(figsize=(12, 8))

data = df_full[['Name of State / UT','Total cases','Cured','Death']]

data.sort_values('Total cases',ascending=False,inplace=True)

sns.set_color_codes("pastel")

sns.barplot(x="Total cases", y="Name of State / UT", data=data,label="Total", color="red")

sns.set_color_codes("muted")

sns.barplot(x="Cured", y="Name of State / UT", data=data, label="Cured", color="green")

ax.legend(ncol=2, loc="lower right", frameon=True)

ax.set(xlim=(0, 35), ylabel="",xlabel="Cases")

sns.despine(left=True, bottom=True)
dbd_india = pd.read_excel('../input/covid19/per_day_cases.xlsx' ,parse_dates=True , sheet_name="India")

# dbd_india

dbd_italy = pd.read_excel('../input/covid19/per_day_cases.xlsx' ,parse_dates=True , sheet_name="Italy")

dbd_Korea = pd.read_excel('../input/covid19/per_day_cases.xlsx' ,parse_dates=True , sheet_name="Korea")

dbd_Wuhan = pd.read_excel('../input/covid19/per_day_cases.xlsx' ,parse_dates=True , sheet_name="Wuhan")

dbd_india.head()
fig = plt.figure(figsize=(10,5))

axes = fig.add_axes([0,0,1,1])

axes.bar(dbd_india["Date"] , dbd_india["Total Cases"])

axes.set_title("Total number of confirmed cases in India")

plt.show()







fig = px.bar(dbd_india,dbd_india["Date"] , dbd_india["Total Cases"],color=dbd_india["Total Cases"],title="Total number of confirmed cases in India")

fig