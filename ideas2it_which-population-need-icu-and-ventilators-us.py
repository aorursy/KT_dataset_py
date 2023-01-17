# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
import glob  
import matplotlib.pyplot as plt
import plotly as py
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from scipy.stats import spearmanr
df_us_states = pd.read_csv("../input/uncover/UNCOVER/covid_tracking_project/covid-sources-for-us-states.csv")
df_us_cases_states = pd.read_csv("../input/uncover/UNCOVER/covid_tracking_project/covid-statistics-by-us-states-totals.csv")
df_us_cases_states = df_us_cases_states[["state","positive"]]
df_us_cases_states = df_us_cases_states.merge(df_us_states[["state","name"]],on="state",how="left")
hosp_cap = pd.read_csv("../input/uncover/UNCOVER/harvard_global_health_institute/hospital-capacity-by-state-40-population-contracted.csv")
hosp_cap = hosp_cap[0:-1]
hosp_cap = hosp_cap.merge(df_us_cases_states,on="state",how="left")[0:-1]
#hosp_cap[["state","name","total_hospital_beds","total_icu_beds","hospital_bed_occupancy_rate","icu_bed_occupancy_rate","positive","population_65","available_icu_beds"]]
hosp_cap["ICU_avaliable_per_10k_65pop"] = hosp_cap["available_icu_beds"]*10000/hosp_cap["population_65"]
hosp_cap["ICU_avaliable_per_10k_65pop"] = hosp_cap["ICU_avaliable_per_10k_65pop"]
hosp_cap = hosp_cap.sort_values("ICU_avaliable_per_10k_65pop",ascending=True)
hosp_cap_positive = hosp_cap.sort_values("positive",ascending=False)
plt.figure(figsize=(15,5))
plt.bar(hosp_cap_positive.name,hosp_cap_positive.positive)
plt.xticks(rotation="vertical")
plt.title("positive cases per state")
for i in range(1,16):
    print("First ",i," states")
    print("Percentage of cases",round(hosp_cap_positive["positive"][0:i].sum()*100/hosp_cap_positive.positive.sum()),"%")
    print("____________________________")
hotspot_df = hosp_cap_positive[["state","name","positive"]][0:15]
#Identifying Hotspot States
print("Hotspot States")
hotspot_states = pd.DataFrame(hosp_cap_positive["name"][0:15].unique()).rename(columns={0:"name"})
hotspot_states
hotspot_states.name.unique()

hotspot_df = hotspot_df.sort_values("positive",ascending=True).reset_index(drop=True).reset_index()
fig = go.Figure(data=go.Choropleth(
    locationmode="USA-states",
    locations = hotspot_df['state'],
    z = hotspot_df['index'],
    text = hotspot_df['name'],
    customdata=hotspot_df['positive'],
    colorscale = [[0, 'rgb(255,255,0)'], [1, 'rgb(255,0,0)']],
    autocolorscale=False,
    reversescale=False,
    marker_line_color='darkgray',
    marker_line_width=0.5,
    hovertemplate="%{text}<br>Cases : %{customdata}<br>",
    colorbar_title='Intensity'
))

fig.update_layout(title_text = 'Hotspot States in USA', showlegend = True, geo = dict( scope = 'usa', landcolor = 'rgb(217, 217, 217)', ) )
fig.show()
plt.figure(figsize=(15,5))
plt.bar(hosp_cap.name,hosp_cap.ICU_avaliable_per_10k_65pop)
plt.xticks(rotation="vertical")
plt.title("ICU Beds available per 10k Senior Citizen Population")
# States with higher elderly population and low ICU beds to elderly population ratio
low_icu_elder = hosp_cap["name"][0:15].unique()
print("Top states with lowest ICU_avaliable_per_10k_65pop ratio : ")
low_icu_elder
#Finding hotspot states with lower IICU_avaliable_per_10k_65pop
print("Hotspot states with lower ICU beds per elderly population")
hotspot_states[hotspot_states["name"].isin(low_icu_elder)]["name"].unique()
hosp_cap["cases_ICU_ratio"] = hosp_cap["available_icu_beds"]/hosp_cap["positive"]
lowerst_icu_availablity_per_total_cases = hosp_cap.sort_values("cases_ICU_ratio",ascending=True)[0:15].name.unique()
print("States with lowest icu availablity")
lowerst_icu_availablity_per_total_cases
low_icu_df = hosp_cap.sort_values("cases_ICU_ratio",ascending=True)[0:15]
hotspot_df["type"] = "hotspot"
low_icu_df = low_icu_df.merge(hotspot_df[["name","type"]],on="name",how="left").fillna("Other")
print("Hotspot states with lower ICU beds per elderly population")
hotspot_states[hotspot_states["name"].isin(lowerst_icu_availablity_per_total_cases)]["name"].unique()
low_icu_df = low_icu_df.sort_values("cases_ICU_ratio",ascending=False).reset_index(drop=True).reset_index()
fig = go.Figure(data=go.Choropleth(
    locationmode="USA-states",
    locations = low_icu_df['state'],
    z = low_icu_df['index'],
    text = low_icu_df['name'],
    customdata=low_icu_df[['cases_ICU_ratio', 'positive',"type"]],
    colorscale = [[0, 'rgb(255,255,0)'], [1, 'rgb(255,0,0)']],
    autocolorscale=False,
    reversescale=False,
    marker_line_color='darkgray',
    marker_line_width=0.5,
    hovertemplate="%{text}<br>Cases : %{customdata[1]}<br>ICU Case Ratio : %{customdata[0]}<br>Type: %{customdata[2]}<extra></extra>",
    colorbar_title='Intensity'
))

fig.update_layout(title_text = 'US States with lowest availablity of ICU beds', showlegend = True, geo = dict( scope = 'usa', landcolor = 'rgb(217, 217, 217)', ) )
fig.show()
hosp_cap_lowest_icu_cases = hosp_cap.sort_values("cases_ICU_ratio",ascending=True)
plt.figure(figsize=(15,5))
plt.bar(hosp_cap_lowest_icu_cases.name,hosp_cap_lowest_icu_cases.cases_ICU_ratio)
plt.xticks(rotation="vertical")
plt.title("Lowest ICU per Cases")
hosp_icu_better = hosp_cap[hosp_cap["available_icu_beds"]>hosp_cap["available_icu_beds"].quantile(0.75)]
sns.regplot((hosp_icu_better[hosp_icu_better["positive"]<60000].ICU_avaliable_per_10k_65pop),(hosp_icu_better[hosp_icu_better["positive"]<60000].positive),x_estimator=np.mean)
vent_df = pd.read_csv("../input/uncover/UNCOVER/ihme/2020_03_30/Hospitalization_all_locs.csv")
vent_df["date"] = vent_df.date.astype("datetime64[ns]")
vent_df.date.head()
vent_df = vent_df[vent_df["date"]=="2020-03-30"]
#Taking relavant fields for invasive ventilators 
# Here InvVen_mean : Needed Invasive Ventilators Means by day(we have taken as on 30)
vent_df = vent_df[["location","InvVen_mean"]].rename(columns={"location":"name"})
hosp_cap_vent = hosp_cap.merge(vent_df,on="name",how="left").fillna(0)
highest_ventilator_req = hosp_cap_vent.sort_values("InvVen_mean",ascending=False)[0:15].name.unique()
print("Topfifteen states in higher need of invasive ventilators")
highest_ventilator_req
hosp_cap_vent_1 = hosp_cap_vent[hosp_cap_vent["positive"]<60000]
sns.regplot((hosp_cap_vent_1[hosp_cap_vent_1["name"].isin(highest_ventilator_req)].positive),hosp_cap_vent_1[hosp_cap_vent_1["name"].isin(highest_ventilator_req)].InvVen_mean)
coef, p = spearmanr(np.log(hosp_cap_vent[hosp_cap_vent["name"].isin(highest_ventilator_req)].positive),hosp_cap_vent[hosp_cap_vent["name"].isin(highest_ventilator_req)].InvVen_mean)
print("Spearman R test on the number of cases and the ventilator needs")
print("Coefficient :",coef)
print("Pvalue :",p)
df = hosp_cap_vent[hosp_cap_vent["name"].isin(highest_ventilator_req)]
df = df.sort_values("InvVen_mean",ascending=True).reset_index(drop=True).reset_index()
fig = go.Figure(data=go.Choropleth(
    locationmode="USA-states",
    locations = df['state'],
    z = df['index'],
    text = df['name'],
    customdata=df[['InvVen_mean', 'positive']],
    colorscale = [[0, 'rgb(255,255,0)'], [1, 'rgb(255,0,0)']],
    autocolorscale=False,
    reversescale=False,
    marker_line_color='darkgray',
    marker_line_width=0.5,
    hovertemplate="%{text}<br>Cases : %{customdata[1]}<br>Avg Ventilators Needed : %{customdata[0]}<extra></extra>",
    colorbar_title='Intensity'
))

fig.update_layout(title_text = 'US States In Need of Ventilators', showlegend = True, geo = dict( scope = 'usa', landcolor = 'rgb(217, 217, 217)', ) )
fig.show()
