import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sb



sb.set(style="whitegrid" , rc={'figure.figsize':(10,5)} , palette = "Blues")

pd.options.display.max_rows = 500 

csv_path = "/kaggle/input/pollution-india-2010/pollution_india_2010 (1).csv"

df = pd.read_csv(csv_path)

df.head()
df.info()
non_null_df = df.replace("Null" , np.nan).dropna(axis = 0)
non_null_df.info()
non_null_df[["NO2","PM10","SO2"]] = non_null_df[["NO2","PM10","SO2"]].astype("int")
non_null_df.info()
# district wise NO2 emission analysis



district_no2 = non_null_df[["City" , "NO2"]]

top10_district_no2 = district_no2.sort_values(by = "NO2" , axis = 0 , ascending= False)[:10]
sb.barplot(x="City", y="NO2", data=top10_district_no2)

plt.title(" Top 10 districts in India with highest NO2 emission")

plt.xticks(rotation=45)
state_no2_pm10_so2= pd.DataFrame(non_null_df.groupby("State")["NO2","PM10","SO2"].mean().reset_index())

state_no2_pm10_so2.head()

sb.clustermap(state_no2_pm10_so2[["NO2","PM10","SO2"]], method="single")
sb.heatmap(state_no2_pm10_so2.corr())

plt.title("NO2, SO2, PM10 Correlation heatmap")
f, ax = plt.subplots(figsize=(6, 15))



sb.set_color_codes("pastel")

sb.barplot(x = "PM10" , y= "State" , data = state_no2_pm10_so2 , label = "PM10 Content" , color = "b" )

sb.set_color_codes("muted")

sb.barplot(x = "SO2" , y= "State" , data = state_no2_pm10_so2 , label = "SO2 Content" , color = "b")





ax.legend(ncol=2, loc="best", frameon=True)

ax.set(xlim=(0, 200), ylabel="",

       xlabel="Statewise Air Pollution Content Level")

sb.despine(left=True, bottom=True)
state_no2_pm10_so2.set_index('State').plot(kind='barh', stacked=True , cmap = "Set1" )

plt.title("Statewise NO2, SO2, PM10 Level")
y = non_null_df["NO2"]

x = non_null_df["SO2"]

my_color=np.where(y>=20, 'orange', 'skyblue')

 

THRESHOLD = 20 

import seaborn as sns

plt.vlines(x=x, ymin=THRESHOLD, ymax=y, color=my_color, alpha=0.4)

plt.scatter(x, y, color=my_color, s=5, alpha=1)

 

# Add title and axis names

plt.title("NO2 vs SO2 {THRESHOLD} threshold distribution", loc='left')

plt.xlabel('SO2')

plt.ylabel('NO2')

THRESHOLD = 35



y = non_null_df["NO2"]

x = non_null_df["PM10"]

my_color=np.where(y>=THRESHOLD, 'red', 'skyblue')

 



import seaborn as sns

plt.vlines(x=x, ymin=THRESHOLD, ymax=y, color=my_color, alpha=0.4)

plt.scatter(x, y, color=my_color, s=5, alpha=1)

 



plt.title(f"NO2 vs PM10 with {THRESHOLD} threshold distribution", loc='left')

plt.xlabel('PM10')

plt.ylabel('NO2')

sb.lineplot(data=state_no2_pm10_so2[["NO2" , "SO2", "PM10"]], palette="tab10", linewidth=2.5)
sb.lmplot(x= "NO2", y="PM10", hue="State",

               truncate=True, height=5, data=state_no2_pm10_so2)

plt.title("State wise PM10 vs NO2 Distribution")
sb.lmplot(x= "SO2", y="PM10", hue="State",

               truncate=True, height=5, data=state_no2_pm10_so2)

plt.title("State wise PM10 vs SO2 Distribution")
from mpl_toolkits import mplot3d
fig = plt.figure(figsize = (8 , 5))

ax = plt.axes(projection="3d")



NO2 = non_null_df["NO2"] 

SO2 = non_null_df["SO2"]

PM10 = non_null_df["PM10"]

ax.scatter3D(NO2, SO2, PM10, c=PM10, cmap='hsv')

ax.set_xlabel('NO2')

ax.set_ylabel('SO2')

ax.set_zlabel('PM10')

plt.show()

non_null_df["State"].unique()
def statewise_data(state_name): 

    try:

        return non_null_df[non_null_df["State"] == state_name]

    except Exception as e:

        return e
statewise_data("Karnataka").sort_values(by = "PM10").set_index("City")[["PM10" , "SO2" , "NO2"]].plot(kind='barh', stacked=True )

plt.title(" Karnataka State Air Pollutants")
statewise_data("Delhi").sort_values(by = "PM10").set_index("City")[["PM10" , "SO2" , "NO2"]].plot(kind='barh', stacked=True )

plt.title(" Delhi State Air Pollutants")
statewise_data("Goa").sort_values(by = "PM10").set_index("City")[["PM10" , "SO2" , "NO2"]].plot(kind='barh', stacked=True )

plt.title(" Goa State Air Pollutants")
statewise_data("West Bengal").sort_values(by = "PM10").set_index("City")[["PM10" , "SO2" , "NO2"]].plot(kind='barh', stacked=True )

plt.title(" West Bengal State Air Pollutants")
non_null_df.sort_values(by = "NO2")[["City" , "NO2"]].set_index("City")[:10].plot(kind='barh')
non_null_df.sort_values(by = "SO2")[["City" , "SO2"]].set_index("City")[:10].plot(kind='barh')
non_null_df.sort_values(by = "PM10")[["City" , "PM10"]].set_index("City")[:10].plot(kind='barh')