import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import plotly.graph_objects as go

import warnings
warnings.filterwarnings("ignore")

import datetime

#Define plot style
plt.style.use("fivethirtyeight")

import gc
df = pd.read_csv(r"../input/ashrae-global-thermal-comfort-database-ii/ashrae_db2.01.csv")
df_city = pd.read_csv(r"../input/cities-of-the-world/cities15000.csv",encoding = "ISO-8859-1") # Location data
df.head()
df.info()
to_drop = ["Air temperature (F)",#Duplicate data, we have it in celsius.
           "Ta_h (F)","Ta_m (F)",#Duplicate data.
           "Ta_l (F)","Operative temperature (F)", #Duplicate data.
           "Radiant temperature (F)", #Duplicate data.
           "Globe temperature (F)", #Duplicate data
           "Tg_h (F)", #Duplicate data
           "Tg_m (F)", #Duplicate data
           "Tg_l (F)", #Duplicate data
           "Publication (Citation)", #Unnecessary for the analysis
           "Data contributor",#Unnecessary for the analysis
           "Database", #Unnecessary for the analysis
           "Air velocity (fpm)", #Duplicate data
           "Velocity_h (fpm)", #Duplicate data
           "Velocity_m (fpm)", #Duplicate data
           "Velocity_l (fpm)", #Duplicate data
           "Outdoor monthly air temperature (F)",#Duplicate data
           "Blind (curtain)", #Unnecessary for the analysis
           'Fan', 'Window', #Unnecessary for the analysis
           'Door','Heater', #Unnecessary for the analysis
           'activity_10', #Unnecessary for the analysis
           'activity_20', #Unnecessary for the analysis
           'activity_30', #Unnecessary for the analysis
           'activity_60' #Unnecessary for the analysis
          ]

df.drop(to_drop, axis=1, inplace=True)
geo_df = df.groupby("City")["City"].agg("size")
geo_df = geo_df.reset_index(name="Count")

df_city = df_city[["asciiname", "latitude", "longitude"]]
df_city.rename(columns = {"asciiname" : "City","latitude" : "Lat", "longitude" : "Lng"},inplace=True)
df_city.drop_duplicates(subset="City",inplace=True)

#Midland is in the UK
df_city.loc[(df_city.City == "Midland"),"Lat"]= 52.489471
df_city.loc[(df_city.City == "Midland"),"Lng"]= -1.898575

geo_df = pd.merge(geo_df,df_city,how="left", on="City")

geo_df.sort_values(by="Count",ascending=False, inplace=True)
fig = go.Figure(go.Scattergeo(lon=geo_df["Lng"],
                              lat=geo_df["Lat"],
                              text=geo_df["City"] + "<br>Count: " + geo_df["Count"].astype(str),
                              marker = dict(
                                  size = geo_df["Count"]/1000,
                                  line_width = 0,sizemin=5)
                             )
               )


fig.update_layout(title_text = "Geographical distribution fo the buildings")

fig.update_geos(projection_type="natural earth")

fig.show()
#Data
geo_df.sort_values(by="Count",ascending=True, inplace=True)

#Plot
ax, fig = plt.subplots(figsize=(10,5))

plt.barh(geo_df["City"][-10:],geo_df["Count"][-10:])

plt.ylabel("Cities", fontsize=18, alpha=.75)
plt.xlabel("Number", fontsize=18, alpha=.75)

plt.yticks(alpha=0.75,weight="bold")
plt.xticks(alpha=0.75)

plt.title("Most enteries per city",alpha=0.75,weight="bold",fontsize=20, loc="left")
#Plot
ax, fig = plt.subplots(figsize=(10,5))

plt.barh(df["Country"].value_counts().index,df["Country"].value_counts())

plt.ylabel("Countries", fontsize=18, alpha=.75)
plt.xlabel("Number", fontsize=18, alpha=.75)

plt.yticks(alpha=0.75, fontsize=10)
plt.xticks(alpha=0.75)

plt.title("Most enteries per country",alpha=0.75,weight="bold",fontsize=20, loc="left")

del geo_df
cooling_geo = df.groupby(["City","Cooling startegy_building level"])["City"].agg("size")
cooling_geo = cooling_geo.reset_index(name="Count")
cooling_geo = pd.merge(cooling_geo,df_city,how="left", on="City")

gc.collect()
fig = go.Figure()

for i in cooling_geo["Cooling startegy_building level"].unique():

    df_part = cooling_geo[cooling_geo["Cooling startegy_building level"] == i]
    fig.add_trace(go.Scattergeo(
        lon = df_part["Lng"],
        lat = df_part["Lat"],
        text= df_part["City"] + "<br>Count: " + df_part["Count"].astype(str),
        name = i,
        marker = dict(
            size = df_part["Count"]/25,
            line_color='rgb(40,40,40)',
            line_width=1.5,
            sizemode = 'area'
        )
    ))

fig.update_layout(dict(
        title = "Geographical cooling strategies (click on the legend to filter data)",
        height=450,
        geo = dict(
            showland = True,
            landcolor = "rgb(250, 250, 250)",
            subunitcolor = "rgb(217, 217, 217)",
            countrycolor = "rgb(217, 217, 217)",
            countrywidth = 0.5,
            subunitwidth = 0.5
        )))
fig.update_geos(projection_type="natural earth")

fig.show()
table = pd.pivot_table(df[["Climate","Cooling startegy_building level"]],
                       index=["Climate"],columns=["Cooling startegy_building level"],
                       aggfunc=len,
                       fill_value=0)
def conv_to_per(df):

    """
    Converts columns to %
    """
    to_drop = []
    df["Sum"] = np.sum(table,axis=1)
    for i in df:
        
        df[i + " percent"] = np.round(df[i] / df["Sum"] * 100,2)
        to_drop.append(i)

    df = df.drop(to_drop,axis=1)
    df = df.drop("Sum percent",axis=1)

    return df
#Data
table = conv_to_per(table)

#Plot
table.plot.barh(stacked=True,figsize=(15,10))

plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, ncol=5)

plt.title("Percentage of cooling strategies per climate",
          alpha=0.75,weight="bold",fontsize=20, loc="left")
fig, axs = plt.subplots(figsize=(15,40))

lab = table.columns

for i in range(19):
    
    data = table.iloc[i,:]
    data_adjusted = np.concatenate((data,[data[0]]))
    label_palce = np.linspace(start=0, stop=2*np.pi, num=len(data_adjusted))

    plt.subplot(10,2,1+i,polar=True)
    plt.title(table.index[i], fontsize=18)
    plt.tight_layout()
    plt.subplots_adjust(top=1.5)
    plt.plot(label_palce, data_adjusted)
    lines, labels = plt.thetagrids(np.degrees(label_palce), labels=lab)
    
ax, fig = plt.subplots(figsize=(10,7))

plt.bar(df["Thermal preference"].value_counts().index,df["Thermal preference"].value_counts())

plt.ylabel("Number", fontsize=18, alpha=.75)
plt.xlabel("Subject temperature satisfaction", fontsize=18, alpha=.75)

plt.yticks(alpha=0.75, fontsize=10)
plt.xticks(alpha=0.75)

plt.title("Subjective temperature review",alpha=0.75,weight="bold",fontsize=20, loc="left")

gc.collect()
ax, fig = plt.subplots(figsize=(10,5))

sns.distplot(df["Thermal sensation"].dropna())

plt.ylabel("",alpha=.75)
plt.xlabel("Subject thermal sensation", fontsize=18, alpha=.75)

plt.yticks(alpha=0.75, fontsize=10)
plt.xticks(alpha=0.75)

plt.title("Distribution of thermal sensation",alpha=0.75,weight="bold",fontsize=20, loc="left")

gc.collect()
fig, ax = plt.subplots(figsize=(12,7))

#Data
df_numeric = df[["Age","Sex","Clo","Met","Subject«s height (cm)","Subject«s weight (kg)","Thermal sensation"]]
df_numeric = df_numeric.corr()

#Heatmap
ax = sns.heatmap(df_numeric, annot=True,annot_kws={"size": 14},linewidths=.5,center=0,cbar=False)

#Heatmap bug fix
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)

del df_numeric

gc.collect()
termal_pref = df.groupby(["City","Thermal preference"])["City"].agg("size")
termal_pref = termal_pref.reset_index(name="Count")


termal_pref = pd.merge(termal_pref,df_city,how="left", on="City")
fig = go.Figure()

for i in termal_pref["Thermal preference"].unique():

    df_part = termal_pref[termal_pref["Thermal preference"] == i]
    fig.add_trace(go.Scattergeo(
        lon = df_part["Lng"],
        lat = df_part["Lat"],
        text= df_part["City"] + "<br>Count: " + df_part["Count"].astype(str),
        name = i,
        marker = dict(
            size = df_part["Count"]/25,
            line_color='rgb(40,40,40)',
            line_width=1.5,
            sizemode = 'area'
        )
    ))

fig.update_layout(dict(
        title = "Geographical thermal preference (click on the legend to filter data)",
        height=450,
        geo = dict(
            showland = True,
            landcolor = "rgb(250, 250, 250)",
            subunitcolor = "rgb(217, 217, 217)",
            countrycolor = "rgb(217, 217, 217)",
            countrywidth = 0.5,
            subunitwidth = 0.5
        )))
fig.update_geos(projection_type="natural earth")

fig.show()
#Data
warmer_df = df[df["Thermal preference"] == "warmer"]

#Plot
ax, fig = plt.subplots(figsize=(10,5))

plt.barh(warmer_df["Climate"].value_counts().index,warmer_df["Climate"].value_counts())

plt.ylabel("Climate",alpha=.75)
plt.xlabel("Count", fontsize=18, alpha=.75)

plt.yticks(alpha=0.75, fontsize=10)
plt.xticks(alpha=0.75)

plt.title("Count of warmer requests per climate", alpha=.75, fontsize=20, weight="bold", loc="left")

gc.collect()
warmer_season = warmer_df[warmer_df["Climate"].isin(warmer_df["Climate"].value_counts().index[:3])]

warmer_season = warmer_season[["Season",
                               "Climate",
                               "Operative temperature (C)",
                               "Outdoor monthly air temperature (C)"]] 
i = 0
axs, fig = plt.subplots(figsize=(15,5),sharex=True)

for climate in warmer_season["Climate"].unique():
    
    df = warmer_season.query("Climate == @climate")
    
    #25% as sample since my CPU is going to burn up
    sample_size = df.sample(frac=0.25)
    
    try:
        plt.subplot(1,3,1+i)
        i+=1
        sns.swarmplot(x="Season",
                      y="Operative temperature (C)",
                      color="#008FD5",
                      alpha=0.75,
                      data=sample_size,
                      label="Op"
                     )
        sns.swarmplot(x="Season",
                      y="Outdoor monthly air temperature (C)",
                      color="#FF2700",
                      alpha=0.75,
                      data=sample_size,
                      label="Outside"
                     )
        
        plt.title(climate, fontsize=18, alpha=0.75)
        plt.ylabel("Temperature", fontsize=18)
        
    except:
        pass
    
plt.text(x=-11,y=30, s="Operative vs Outside temperature", fontsize=25, weight="bold", alpha=0.75)
ax, fig = plt.subplots(figsize=(10,5))

sns.distplot(warmer_df["Operative temperature (C)"].dropna(), label="Operative temperature")
sns.distplot(warmer_df["Outdoor monthly air temperature (C)"].dropna(), label="Outdoor temperature")

plt.xlabel("Temperature", fontsize=15, alpha=0.75, weight="bold")

plt.title("Temperature distribution: Operative vs Outdoor", fontsize=20, alpha=0.75, weight="bold", loc="left")
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
          fancybox=True, ncol=5)
#Lets see how geographic factors affect the pref

cooler_df = df[df["Thermal preference"] == "cooler"]

plt.barh(cooler_df["Climate"].value_counts().index,cooler_df["Climate"].value_counts())
plt.barh(cooler_df["Cooling startegy_building level"].value_counts().index,cooler_df["Cooling startegy_building level"].value_counts())
sns.distplot(cooler_df["Operative temperature (C)"].dropna())
op_temp = df.groupby("City")["Operative temperature (C)"].agg("mean")
op_temp = op_temp.reset_index(name="Mean")

op_temp = pd.merge(op_temp,df_city,how="left", on="City")
fig = go.Figure()

df_part = op_temp.dropna()
df_part = df_part[df_part["Mean"] <= 22]
fig.add_trace(go.Scattergeo(
    lon = df_part["Lng"],
    lat = df_part["Lat"],
    text= df_part["City"] + "<br>Mean Operative Temperature: " + round(df_part["Mean"],2).astype(str),
    name = "<= 22",
    marker = dict(
        size = round(df_part["Mean"]/2,2),
        line_color='rgb(40,40,40)',
        line_width=1.5
    )
))

df_part = op_temp.dropna()
df_part = df_part[df_part["Mean"] > 22]
fig.add_trace(go.Scattergeo(
    lon = df_part["Lng"],
    lat = df_part["Lat"],
    text= df_part["City"] + "<br>Mean Operative Temperature: " + round(df_part["Mean"],2).astype(str),
    name = "> 22",
    marker = dict(
        size = round(df_part["Mean"]/2,2),
        line_color='rgb(40,40,40)',
        line_width=1.5
    )
))

fig.update_layout(dict(
        title = "Geographical operative temperature",
        height=450,
        geo = dict(
            showland = True,
            landcolor = "rgb(250, 250, 250)",
            subunitcolor = "rgb(217, 217, 217)",
            countrycolor = "rgb(217, 217, 217)",
            countrywidth = 0.5,
            subunitwidth = 0.5
        )))
fig.update_geos(projection_type="natural earth")

fig.show()
