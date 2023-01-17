import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
#Active passenger airport
df_passenger=pd.read_csv("../input/russian-passenger-air-service-20072020/russian_passenger_air_service_2.csv")
df_passenger=df_passenger[df_passenger["Whole year"]!=0]
df_passenger
#Active cargo airport
df_cargo=pd.read_csv("../input/russian-passenger-air-service-20072020/russian_air_service_CARGO_AND_PARCELS.csv")
df_cargo=df_cargo[df_cargo["Whole year"]!=0]
df_cargo
a_pass_air=set(df_passenger["Airport name"].unique())
a_corgo_air=set(df_cargo["Airport name"].unique())
#cargo and passenger airports
cap=list(a_pass_air&a_corgo_air)
#only corgo airports
onc=list(a_corgo_air-a_pass_air)
#only passenger airports
onp=list(a_pass_air-a_corgo_air)
#all airports
aap=list(a_pass_air|a_corgo_air)
new_columns=['January', 'February', 'March', 'April', 'May','June', 'July', 'August', 'September', 'October', 'November','December', 'Whole year']
new_dataframe=pd.DataFrame(columns=new_columns)

for i in aap:
    if i in onp:
        a=df_passenger[df_passenger["Airport name"]==i][new_columns]
        a["Airport name"]=i
        a["Type"]="Only passenger"
        a["Year"]=df_passenger[df_passenger["Airport name"]==i]["Year"]
        a["Airport coordinates"]=df_passenger[df_passenger["Airport name"]==i]["Airport coordinates"]
        new_dataframe=new_dataframe.append(a)
    elif i in onc:
        a=df_cargo[df_cargo["Airport name"]==i][new_columns]
        a["Airport name"]=i
        a["Type"]="Only passenger"
        a["Year"]=df_cargo[df_cargo["Airport name"]==i]["Year"]
        a["Airport coordinates"]=df_cargo[df_cargo["Airport name"]==i]["Airport coordinates"]
        new_dataframe=new_dataframe.append(a)
    else:
        a=df_cargo[df_cargo["Airport name"]==i][new_columns]+df_passenger[df_passenger["Airport name"]==i][new_columns]
        a["Airport name"]=i
        a["Type"]="Passenger and Cargo"
        a["Year"]=df_passenger[df_passenger["Airport name"]==i]["Year"]
        a["Airport coordinates"]=df_passenger[df_passenger["Airport name"]==i]["Airport coordinates"]
        new_dataframe=new_dataframe.append(a)
new_dataframe=new_dataframe.sort_values(by=["Airport name","Year"])
year=new_dataframe["Year"]
airport=new_dataframe["Airport name"]
new_index=list(zip(airport,year))
new_index=pd.MultiIndex.from_tuples(new_index)
new_dataframe=new_dataframe.set_index(new_index).drop(["Year","Airport name"],axis=1)
new_dataframe
new_dataframe.loc["Abakan"]
new_dataframe.loc["Tambov"]
fig,ax=plt.subplots(figsize=[25,7])
x_bar=['January', 'February', 'March', 'April', 'May', 'June', 'July','August', 'September', 'October', 'November', 'December']
x_loc=np.arange(len(x_bar))
y1=list(new_dataframe.loc["Abakan"].loc[2019][x_bar])
y2=list(new_dataframe.loc["Tambov"].loc[2019][x_bar])

a=0.4

ax.bar(x_loc-a/2,y1,width=a)
ax.bar(x_loc+a/2,y2,width=a)
ax.legend(["Abakan","Tambov"])

ax.set_title("Abakan and Tambov total values")
ax.set_xlabel("Months")
ax.set_ylabel("Values")


plt.xticks(x_loc,x_bar);
