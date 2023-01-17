# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #visualization
import seaborn as sns #visualization
import plotly.graph_objs as go #visualization
from plotly.offline import init_notebook_mode, iplot, plot
import warnings
init_notebook_mode(connected=True) 
# filter warnings
warnings.filterwarnings('ignore')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("/kaggle/input/corona-virus-report/covid_19_clean_complete.csv")
data.head(10)
data.tail(10)
data.describe()
data.info()
from datetime import date, timedelta, datetime
data["Date"] = pd.to_datetime(data["Date"])
data.info()
data["Country/Region"].unique()
print("{} countries".format(len(list(data["Country/Region"].unique()))))
sns.pairplot(data, markers = "+")
plt.show()
desc = data.describe()
print(desc[:3])
data.corr() 

f,ax = plt.subplots(figsize=(12, 12))
sns.heatmap(data.corr(), annot=True,annot_kws = {"size": 12}, linewidths=0.5, fmt = '.3f', ax=ax)
plt.title("Correlations", fontsize = 20)
plt.show()
date_list1 = list(data["Date"].unique())
confirmed = []
deaths = []
recovered = []
active = []
for i in date_list1:
    x = data[data["Date"] == i]
    confirmed.append(sum(x["Confirmed"]))
    deaths.append(sum(x["Deaths"]))
    recovered.append(sum(x["Recovered"]))
    active.append(sum(x["Active"]))
data_glob = pd.DataFrame(list(zip(date_list1,confirmed,deaths,recovered,active)),columns = ["Date","Confirmed","Deaths","Recovered","Active"])
data_glob.head()
data_glob.tail()
trace1 = go.Scatter(
x = data_glob["Date"],
y = data_glob["Confirmed"],
mode = "lines",
name = "Confirmed",
line = dict(width = 2.5),
marker = dict(color = 'rgba(4,90,141, 0.8)')
)

trace2 = go.Scatter(
x = data_glob["Date"],
y = data_glob["Deaths"],
mode = "lines",
name = "Deaths",
line = dict(width = 2.5),
marker = dict(color = 'rgba(152,0,67, 0.8)')
)

trace3 = go.Scatter(
x = data_glob["Date"],
y = data_glob["Recovered"],
mode = "lines",
name = "Recovered",
line = dict(width = 2.5),    
marker = dict(color = 'rgba(1,108,89, 0.8)')
)

trace4 = go.Scatter(
x = data_glob["Date"],
y = data_glob["Active"],
mode = "lines",
name = "Active",
line = dict(width = 2.5),
marker = dict(color = 'rgba(84,39,143, 0.8)')
)

data_plt = [trace1,trace2,trace3,trace4]
layout = go.Layout(title = "Global Patient Data",template = "plotly_white",xaxis_title="Date",yaxis_title="Value")
fig = go.Figure(data = data_plt,layout = layout)
iplot(fig)
labels = ["Recovered","Deaths","Active"]
values = [np.sum(data_glob["Recovered"]),np.sum(data_glob["Deaths"]),np.sum(data_glob["Active"])]

fig = go.Figure(data = [go.Pie(labels = labels, values = values,pull = [0.05,0.05,0.05],textinfo='label+percent',insidetextorientation='radial')],layout = go.Layout(title = "Global Patient Percentage"))
fig.show()
data_last = data.tail(1)
data_last_day = data[data["Date"] == data_last["Date"].iloc[0]] 
country_list = list(data_last_day["Country/Region"].unique())
confirmed = []
deaths = []
recovered = []
active = []
for i in country_list:
    x = data_last_day[data_last_day["Country/Region"] == i]
    confirmed.append(sum(x["Confirmed"]))
    deaths.append(sum(x["Deaths"]))
    recovered.append(sum(x["Recovered"]))
    active.append(sum(x["Active"]))
data_maps = pd.DataFrame(list(zip(country_list,confirmed,deaths,recovered,active)),columns = ["Country/Region","Confirmed","Deaths","Recovered","Active"])
data_maps.head()
import plotly.express as px
grp = data_maps.groupby(["Country/Region"])["Confirmed"].max()
grp = grp.reset_index()
fig = px.choropleth(grp, locations = "Country/Region", 
                    color = np.log10(grp["Confirmed"]),
                    hover_name= grp["Country/Region"],
                    hover_data = ["Confirmed"],
                    color_continuous_scale=px.colors.sequential.Plasma,locationmode = "country names")
fig.update_geos(fitbounds = "locations",)
fig.update_layout(title_text = "Confirmed Cases on World Map(Log Scale)")
fig.update_coloraxes(colorbar_title = "Confirmed Cases (Log Scale)", colorscale = "Blues")
fig.show()
grp = data_maps.groupby(["Country/Region"])["Deaths"].max()
grp = grp.reset_index()
fig = px.choropleth(grp, locations = "Country/Region", 
                    color = np.log10(grp["Deaths"]),
                    hover_name= grp["Country/Region"],
                    hover_data = ["Deaths"],
                    color_continuous_scale=px.colors.sequential.Plasma,locationmode = "country names")
fig.update_geos(fitbounds = "locations",)
fig.update_layout(title_text = "Death Cases on World Map(Log Scale)")
fig.update_coloraxes(colorbar_title = "Death Cases (Log Scale)", colorscale = "Reds")
fig.show()
grp = data_maps.groupby(["Country/Region"])["Recovered"].max()
grp = grp.reset_index()
fig = px.choropleth(grp, locations = "Country/Region", 
                    color = np.log10(grp["Recovered"]),
                    hover_name= grp["Country/Region"],
                    hover_data = ["Recovered"],
                    color_continuous_scale=px.colors.sequential.Plasma,locationmode = "country names")
fig.update_geos(fitbounds = "locations",)
fig.update_layout(title_text = "Recovered Cases on World Map(Log Scale)")
fig.update_coloraxes(colorbar_title = "Recovered Cases (Log Scale)", colorscale = "Greens")
fig.show()
grp = data_maps.groupby(["Country/Region"])["Active"].max()
grp = grp.reset_index()
fig = px.choropleth(grp, locations = "Country/Region", 
                    color = np.log10(grp["Active"]),
                    hover_name= grp["Country/Region"],
                    hover_data = ["Active"],
                    color_continuous_scale=px.colors.sequential.Plasma,locationmode = "country names")
fig.update_geos(fitbounds = "locations",)
fig.update_layout(title_text = "Active Cases on World Map(Log Scale)")
fig.update_coloraxes(colorbar_title = "Active Cases (Log Scale)", colorscale = "Purples")
fig.show()
data_last = data.tail(1)
data_last_day = data[data["Date"] == data_last["Date"].iloc[0]] 
country_list = list(data_last_day["Country/Region"].unique())
confirmed = []
deaths = []
recovered = []
active = []
for i in country_list:
    x = data_last_day[data_last_day["Country/Region"] == i]
    confirmed.append(sum(x["Confirmed"]))
    deaths.append(sum(x["Deaths"]))
    recovered.append(sum(x["Recovered"]))
    active.append(sum(x["Active"]))
data_glob_country = pd.DataFrame(list(zip(country_list,confirmed,deaths,recovered,active)),columns = ["Country","Confirmed","Deaths","Recovered","Active"])
data_glob_country.head()
data_glob_country.tail()
confirmed_sorted = data_glob_country.sort_values(by = ["Confirmed"])
confirmed_10 = confirmed_sorted.tail(10)

fig = go.Figure(data = [go.Bar(x = confirmed_10["Country"],
                              y = confirmed_10["Confirmed"],
                              text = confirmed_10["Confirmed"],
                              textposition = "outside",
                              marker=dict(color = confirmed_10["Confirmed"],line = dict(color = "rgb(0,0,0)", width = 1.5),colorbar=dict(title="Confirmed"),colorscale="tempo",))],
                              layout = go.Layout(template= "plotly_white",title = "Confirmed Cases Top 10 Countries",xaxis_title="Country",yaxis_title="Confirmed"))
iplot(fig)
confirmed_sorted = data_glob_country.sort_values(by = ["Deaths"])
confirmed_10 = confirmed_sorted.tail(10)

fig = go.Figure(data = [go.Bar(x = confirmed_10["Country"],
                              y = confirmed_10["Deaths"],
                              text = confirmed_10["Deaths"],
                              textposition = "outside",
                              marker=dict(color = confirmed_10["Deaths"],line = dict(color = "rgb(0,0,0)", width = 1.5),colorbar=dict(title="Deaths"),colorscale="amp",))],
                              layout = go.Layout(template= "plotly_white",title = "Death Cases Top 10 Countries",xaxis_title="Country",yaxis_title="Death"))
iplot(fig)
confirmed_sorted = data_glob_country.sort_values(by = ["Recovered"])
confirmed_10 = confirmed_sorted.tail(10)

fig = go.Figure(data = [go.Bar(x = confirmed_10["Country"],
                              y = confirmed_10["Recovered"],
                              text = confirmed_10["Recovered"],
                              textposition = "outside",
                              marker=dict(color = confirmed_10["Recovered"],line = dict(color = "rgb(0,0,0)", width = 1.5),colorbar=dict(title="Recovered"),colorscale="speed",))],
                              layout = go.Layout(template= "plotly_white",title = "Recovered Cases Top 10 Countries",xaxis_title="Country",yaxis_title="Recovered"))
iplot(fig)
confirmed_sorted = data_glob_country.sort_values(by = ["Active"])
confirmed_10 = confirmed_sorted.tail(10)

fig = go.Figure(data = [go.Bar(x = confirmed_10["Country"],
                              y = confirmed_10["Active"],
                              text = confirmed_10["Active"],
                              textposition = "outside",
                              marker=dict(color = confirmed_10["Active"],line = dict(color = "rgb(0,0,0)", width = 1.5),colorbar=dict(title="Active"),colorscale="matter",))],
                              layout = go.Layout(template= "plotly_white",title = "Active Cases Top 10 Countries",xaxis_title="Country",yaxis_title="Active"))
iplot(fig)
conf_death = data_glob_country["Deaths"]*100/data_glob_country["Confirmed"]
conf_death_df = pd.DataFrame(list(zip(data_glob_country["Country"],conf_death)),columns = ["Country","Deaths_percentage"])
conf_death_sorted = conf_death_df.sort_values(by = ["Deaths_percentage"])
conf_death_10 = conf_death_sorted.tail(10)
conf_death_10.head(10)
fig = go.Figure(data = [go.Bar(x = conf_death_10["Country"],
                              y = conf_death_10["Deaths_percentage"],
                              text = np.round(conf_death_10["Deaths_percentage"],2),
                              textposition = "outside",
                              marker=dict(color = conf_death_10["Deaths_percentage"],line = dict(color = "rgb(0,0,0)", width = 1.5),colorbar=dict(title="Death Percentage(%)"),colorscale="PuRd",))],
                              layout = go.Layout(template= "plotly_white",title = "Death Percentage  Top 10 Countries",xaxis_title="Country",yaxis_title="Death Percentage(%)"))
iplot(fig)
conf_recovered = data_glob_country["Recovered"]*100/data_glob_country["Confirmed"]
conf_recovered_df = pd.DataFrame(list(zip(data_glob_country["Country"],conf_recovered)),columns = ["Country","Recovered_percentage"])
conf_recovered_sorted = conf_recovered_df.sort_values(by = ["Recovered_percentage"])
conf_recovered_10 = conf_recovered_sorted.tail(10)
conf_recovered_10.head(10)
fig = go.Figure(data = [go.Bar(x = conf_recovered_10["Country"],
                              y = conf_recovered_10["Recovered_percentage"],
                              text = np.round(conf_recovered_10["Recovered_percentage"],2),
                              textposition = "outside",
                              marker=dict(color = conf_recovered_10["Recovered_percentage"],line = dict(color = "rgb(0,0,0)", width = 1.5),colorbar=dict(title="Recovered Percentage(%)"),colorscale="PuBuGn",))],
                              layout = go.Layout(template= "plotly_white",title = "Recovered Percentage  Top 10 Countries",xaxis_title="Country",yaxis_title="Recovered Percentage(%)"))
iplot(fig)
conf_active = data_glob_country["Active"]*100/data_glob_country["Confirmed"]
conf_active_df = pd.DataFrame(list(zip(data_glob_country["Country"],conf_active)),columns = ["Country","Active_percentage"])
conf_active_sorted = conf_active_df.sort_values(by = ["Active_percentage"])
conf_active_10 = conf_active_sorted.tail(10)
conf_active_10.head(10)
fig = go.Figure(data = [go.Bar(x = conf_active_10["Country"],
                              y = conf_active_10["Active_percentage"],
                              text = np.round(conf_active_10["Active_percentage"],2),
                              textposition = "outside",
                              marker=dict(color = conf_active_10["Active_percentage"],line = dict(color = "rgb(0,0,0)", width = 1.5),colorbar=dict(title="Active Percentage(%)"),colorscale="Purples",))],
                              layout = go.Layout(template= "plotly_white",title = "Active Percentage Top 10 Countries",xaxis_title="Country",yaxis_title="Active Percentage(%)"))
iplot(fig)
from plotly.subplots import make_subplots
fig = make_subplots(rows=2,cols=2,subplot_titles = ("Confirmed","Deaths","Recovered","Active"))

fig.append_trace(go.Box(y = data_glob["Confirmed"],name = "Confirmed",marker_color = 'rgb(4,90,141)'),row = 1, col = 1)
fig.append_trace(go.Box(y = data_glob["Deaths"], name = "Deaths",marker_color = 'rgb(152,0,67)'),row = 1, col = 2)
fig.append_trace(go.Box(y = data_glob["Recovered"], name = "Recovered",marker_color = 'rgb(1,108,89)'),row = 2, col = 1)
fig.append_trace(go.Box(y = data_glob["Active"], name = "Active",marker_color = 'rgb(84,39,143)'),row = 2, col = 2)

fig.update_xaxes(title_text="Confirmed", row=1, col=1)
fig.update_xaxes(title_text="Deaths", row=1, col=2)
fig.update_xaxes(title_text="Recovered", row=2, col=1)
fig.update_xaxes(title_text="Active", row=2, col=2)

fig.update_yaxes(title_text="Value", row=1, col=1)
fig.update_yaxes(title_text="Value", row=1, col=2)
fig.update_yaxes(title_text="Value", row=2, col=1)
fig.update_yaxes(title_text="Value", row=2, col=2)

fig.update_layout(height = 1000,title_text="Boxplots and Outliers of Data",template = "plotly_white")
iplot(fig)
from sklearn.linear_model import LinearRegression

x = np.array(data_glob.loc[:,'Deaths']).reshape(-1,1)
y = np.array(data_glob.loc[:,'Confirmed']).reshape(-1,1)

reg = LinearRegression()
predict_space = np.linspace(min(x), max(x)).reshape(-1,1)
reg.fit(x,y)
predicted = reg.predict(predict_space)

print('R^2 score : ',reg.score(x, y))

plt.figure(figsize = (13,8))
plt.plot(predict_space, predicted, color='black', linewidth=3,label = "LR Prediction")
plt.scatter(x=x,y=y,label = "Data")
plt.legend()
plt.xlabel('Deaths') 
plt.ylabel('Confirmed')
plt.grid(True, alpha = 0.5)
plt.title("Confirmed vs Deaths")
plt.show()

x = np.array(data_glob.loc[:,'Active']).reshape(-1,1)
y = np.array(data_glob.loc[:,'Confirmed']).reshape(-1,1)

reg = LinearRegression()
predict_space = np.linspace(min(x), max(x)).reshape(-1,1)
reg.fit(x,y)
predicted = reg.predict(predict_space)

print('R^2 score : ',reg.score(x, y))

plt.figure(figsize = (13,8))
plt.plot(predict_space, predicted, color='black', linewidth=3,label = "LR Prediction")
plt.scatter(x=x,y=y,label = "Data")
plt.legend()
plt.xlabel('Active') 
plt.ylabel('Confirmed')
plt.grid(True, alpha = 0.5)
plt.title("Confirmed vs Active")
plt.show()

x = np.array(data_glob.loc[:,'Recovered']).reshape(-1,1)
y = np.array(data_glob.loc[:,'Confirmed']).reshape(-1,1)

reg = LinearRegression()
predict_space = np.linspace(min(x), max(x)).reshape(-1,1)
reg.fit(x,y)
predicted = reg.predict(predict_space)

print('R^2 score : ',reg.score(x, y))

plt.figure(figsize = (13,8))
plt.plot(predict_space, predicted, color='black', linewidth=3,label = "LR Prediction")
plt.scatter(x=x,y=y,label = "Data")
plt.legend()
plt.xlabel('Recovered') 
plt.ylabel('Confirmed')
plt.grid(True, alpha = 0.5)
plt.title("Confirmed vs Recovered")
plt.show()

death_percent = ((data_glob["Deaths"]*100)/data_glob["Confirmed"])
#today = ((data_glob["Deaths"].tolist().index(len(data_glob["Deaths"].tolist())-1)*100)/(data_glob["Confirmed"]).tolist().index(len(data_glob["Confirmed"].tolist())-1))
plt.figure(figsize = (13,8))
plt.plot(data_glob["Date"],death_percent,label = "Death Percent", color = "Red")
plt.legend()
plt.xlabel("Date")
plt.ylabel("Percentage(%)")
plt.grid(True, alpha = 0.7)
plt.title("Death Percentage Per Day")
#plt.text(1,1 ,"Highest Death Percent:"+str(today) ,fontsize = 10, color = "black")
plt.show()
recovered_percent = ((data_glob["Recovered"]*100)/data_glob["Confirmed"])

plt.figure(figsize = (13,8))
plt.plot(data_glob["Date"],recovered_percent,label = "Recovered Percent", color = "Green")
plt.legend()
plt.xlabel("Date")
plt.ylabel("Percentage(%)")
plt.grid(True, alpha = 0.7)
plt.title("Recovered Percentage Per Day")
plt.show()
recovered_percent = ((data_glob["Active"]*100)/data_glob["Confirmed"])

plt.figure(figsize = (13,8))
plt.plot(data_glob["Date"],recovered_percent,label = "Active Percent", color = "Purple")
plt.legend()
plt.xlabel("Date")
plt.ylabel("Percentage(%)")
plt.grid(True, alpha = 0.7)
plt.title("Active Percentage Per Day")
plt.show()
data_glob = data_glob.tail(10)
data_glob.tail(10)
trace1 = go.Bar(x = data_glob["Confirmed"],
               y = data_glob["Date"],
               orientation = "h",
               text = data_glob["Confirmed"],
               textposition = "auto",
               name = "Confirmed",
               marker = dict(color = "rgba(4,90,141,0.8)"))
                #,line = dict(color = "rgb(0,0,0)", width = 1.2)))

trace2 = go.Bar(x = data_glob["Deaths"],
               y = data_glob["Date"],
               orientation = "h",
               text = data_glob["Deaths"],
               textposition = "auto",
               name = "Deaths",
               marker = dict(color = "rgba(152,0,67,0.8)"))
                             #,line = dict(color = "rgb(0,0,0)", width = 1.2)))

trace3 = go.Bar(x = data_glob["Recovered"],
               y = data_glob["Date"],
               orientation = "h",
               text = data_glob["Recovered"],
               textposition = "auto",
               name = "Recovered",
               marker = dict(color = "rgba(1,108,89,0.8)"))
                             #,line = dict(color = "rgb(0,0,0)", width = 1.2)))

trace4 = go.Bar(x = data_glob["Active"],
               y = data_glob["Date"],
               orientation = "h",
               text = data_glob["Active"],
               textposition = "auto",
               name = "Active",
               marker = dict(color = "rgba(84,39,143,0.8)"))
                             #,line = dict(color = "rgb(0,0,0)", width = 1.2)))

data_bar = [trace2,trace3,trace4,trace1]
layout = go.Layout(height = 1000, title = "Last 10 Days", template = "plotly_white",xaxis_title="Value",yaxis_title="Date")
fig = go.Figure(data = data_bar, layout = layout)
iplot(fig)
trace = go.Bar(
    x = data_glob["Date"],
    y = data_glob["Confirmed"],
    text = data_glob["Confirmed"],
    textposition = "outside",
    marker=dict(color = data_glob["Confirmed"],colorbar=dict(
            title="Colorbar"
        ),colorscale="Blues",))

layout = go.Layout(title = "Confirmed Last 10 Days",template = "plotly_white",yaxis_title="Confirmed",xaxis_title="Date")
fig = go.Figure(data = trace, layout = layout)

iplot(fig)
trace = go.Bar(
    x = data_glob["Date"],
    y = data_glob["Deaths"],
    text = data_glob["Deaths"],
    textposition = "outside",
    marker=dict(color = data_glob["Deaths"],colorbar=dict(
            title="Colorbar"
        ),colorscale="Reds",))

layout = go.Layout(title = "Deaths Last 10 Days",template = "plotly_white",yaxis_title="Death",xaxis_title="Date")
fig = go.Figure(data = trace, layout = layout)

iplot(fig)
trace = go.Bar(
    x = data_glob["Date"],
    y = data_glob["Recovered"],
    text = data_glob["Recovered"],
    textposition = "outside",
    marker=dict(color = data_glob["Recovered"],colorbar=dict(
            title="Colorbar"
        ),colorscale="YlGn",))

layout = go.Layout(title = "Recovered Last 10 Days",template = "plotly_white",yaxis_title="Recovered",xaxis_title="Date")
fig = go.Figure(data = trace, layout = layout)

iplot(fig)
trace = go.Bar(
    x = data_glob["Date"],
    y = data_glob["Active"],
    text = data_glob["Active"],
    textposition = "outside",
    marker=dict(color = data_glob["Active"],colorbar=dict(
            title="Colorbar"
        ),colorscale="Purples",))

layout = go.Layout(title = "Active Last 10 Days",template = "plotly_white",yaxis_title="Active",xaxis_title="Date")
fig = go.Figure(data = trace, layout = layout)

iplot(fig)
australia = data[data["Country/Region"] == "Australia"]
australia.head()
date_list1 = list(australia["Date"].unique())
confirmed = []
deaths = []
recovered = []
active = []
for i in date_list1:
    x = australia[australia["Date"] == i]
    confirmed.append(sum(x["Confirmed"]))
    deaths.append(sum(x["Deaths"]))
    recovered.append(sum(x["Recovered"]))
    active.append(sum(x["Active"]))
australia = pd.DataFrame(list(zip(date_list1,confirmed,deaths,recovered,active)),columns = ["Date","Confirmed","Deaths","Recovered","Active"])
#data_aus.head()
australia.tail()
trace1 = go.Scatter(
x = australia["Date"],
y = australia["Confirmed"],
mode = "lines",
name = "Confirmed",
line = dict(width = 2.5),
marker = dict(color = 'rgba(4,90,141, 0.8)')
)

trace2 = go.Scatter(
x = australia["Date"],
y = australia["Deaths"],
mode = "lines",
name = "Deaths",
line = dict(width = 2.5),
marker = dict(color = 'rgba(152,0,67, 0.8)')
)

trace3 = go.Scatter(
x = australia["Date"],
y = australia["Recovered"],
mode = "lines",
name = "Recovered",
line = dict(width = 2.5),    
marker = dict(color = 'rgba(1,108,89, 0.8)')
)

trace4 = go.Scatter(
x = australia["Date"],
y = australia["Active"],
mode = "lines",
name = "Active",
line = dict(width = 2.5),
marker = dict(color = 'rgba(84,39,143, 0.8)')
)

data_plt = [trace1,trace2,trace3,trace4]
layout = go.Layout(title = "Australia Patient Data",template = "plotly_white",xaxis_title="Date",yaxis_title="Value")
fig = go.Figure(data = data_plt,layout = layout)
iplot(fig)
labels = ["Recovered","Deaths","Active"]
values = [np.sum(australia["Recovered"]),np.sum(australia["Deaths"]),np.sum(australia["Active"])]

fig = go.Figure(data = [go.Pie(labels = labels, values = values,pull = [0.05,0.05,0.05],textinfo='label+percent',insidetextorientation='radial')],layout = go.Layout(title = "Australia Patient Percentage"))
fig.show()
from plotly.subplots import make_subplots
fig = make_subplots(rows=2,cols=2,subplot_titles = ("Confirmed","Deaths","Recovered","Active"))

fig.append_trace(go.Box(y = australia["Confirmed"],name = "Confirmed",marker_color = 'rgb(4,90,141)'),row = 1, col = 1)
fig.append_trace(go.Box(y = australia["Deaths"], name = "Deaths",marker_color = 'rgb(152,0,67)'),row = 1, col = 2)
fig.append_trace(go.Box(y = australia["Recovered"], name = "Recovered",marker_color = 'rgb(1,108,89)'),row = 2, col = 1)
fig.append_trace(go.Box(y = australia["Active"], name = "Active",marker_color = 'rgb(84,39,143)'),row = 2, col = 2)

fig.update_xaxes(title_text="Confirmed", row=1, col=1)
fig.update_xaxes(title_text="Deaths", row=1, col=2)
fig.update_xaxes(title_text="Recovered", row=2, col=1)
fig.update_xaxes(title_text="Active", row=2, col=2)

fig.update_yaxes(title_text="Value", row=1, col=1)
fig.update_yaxes(title_text="Value", row=1, col=2)
fig.update_yaxes(title_text="Value", row=2, col=1)
fig.update_yaxes(title_text="Value", row=2, col=2)

fig.update_layout(height = 1000,title_text="Boxplots and Outliers",template = "plotly_white")
iplot(fig)

australia = australia.tail(5)
trace1 = go.Bar(x = australia["Confirmed"],
               y = australia["Date"],
               orientation = "h",
               text = australia["Confirmed"],
               textposition = "auto",
               name = "Confirmed",
               marker = dict(color = "rgba(4,90,141,0.8)"))
                #,line = dict(color = "rgb(0,0,0)", width = 1.2)))

trace2 = go.Bar(x = australia["Deaths"],
               y = australia["Date"],
               orientation = "h",
               text = australia["Deaths"],
               textposition = "auto",
               name = "Deaths",
               marker = dict(color = "rgba(152,0,67,0.8)"))
                             #,line = dict(color = "rgb(0,0,0)", width = 1.2)))

trace3 = go.Bar(x = australia["Recovered"],
               y = australia["Date"],
               orientation = "h",
               text = australia["Recovered"],
               textposition = "auto",
               name = "Recovered",
               marker = dict(color = "rgba(1,108,89,0.8)"))
                             #,line = dict(color = "rgb(0,0,0)", width = 1.2)))

trace4 = go.Bar(x = australia["Active"],
               y = australia["Date"],
               orientation = "h",
               text = australia["Active"],
               textposition = "auto",
               name = "Active",
               marker = dict(color = "rgba(84,39,143,0.8)"))
                             #,line = dict(color = "rgb(0,0,0)", width = 1.2)))

data_bar = [trace2,trace4,trace3,trace1]
layout = go.Layout(height = 1000, title = "Last 10 Days in Australia", template = "plotly_white",yaxis_title="Date",xaxis_title="Value")
fig = go.Figure(data = data_bar, layout = layout)
iplot(fig)

china = data[data["Country/Region"] == "China"]
china.head()
date_list1 = list(china["Date"].unique())
confirmed = []
deaths = []
recovered = []
active = []
for i in date_list1:
    x = china[china["Date"] == i]
    confirmed.append(sum(x["Confirmed"]))
    deaths.append(sum(x["Deaths"]))
    recovered.append(sum(x["Recovered"]))
    active.append(sum(x["Active"]))
china = pd.DataFrame(list(zip(date_list1,confirmed,deaths,recovered,active)),columns = ["Date","Confirmed","Deaths","Recovered","Active"])
#data_aus.head()
china.tail()
trace1 = go.Scatter( 
x = china["Date"],
y = china["Confirmed"],
mode = "lines",
name = "Confirmed",
line = dict(width = 2.5),
marker = dict(color = 'rgba(4,90,141, 0.8)')
)

trace2 = go.Scatter(
x = china["Date"],
y = china["Deaths"],
mode = "lines",
name = "Deaths",
line = dict(width = 2.5),
marker = dict(color = 'rgba(152,0,67, 0.8)')
)

trace3 = go.Scatter(
x = china["Date"],
y = china["Recovered"],
mode = "lines",
name = "Recovered",
line = dict(width = 2.5),    
marker = dict(color = 'rgba(1,108,89, 0.8)')
)

trace4 = go.Scatter(
x = china["Date"],
y = china["Active"],
mode = "lines",
name = "Active",
line = dict(width = 2.5),
marker = dict(color = 'rgba(84,39,143, 0.8)')
)

data_plt = [trace1,trace2,trace3,trace4]
layout = go.Layout(title = "China Patient Data",template = "plotly_white",xaxis_title="Date",yaxis_title="Value")
fig = go.Figure(data = data_plt,layout = layout)

iplot(fig)
labels = ["Recovered","Deaths","Active"]
values = [np.sum(china["Recovered"]),np.sum(china["Deaths"]),np.sum(china["Active"])]

fig = go.Figure(data = [go.Pie(labels = labels, values = values,pull = [0.05,0.05,0.05],textinfo='label+percent',insidetextorientation='radial')],layout = go.Layout(title = "China Patient Percentage"))
fig.show()
from plotly.subplots import make_subplots
fig = make_subplots(rows=2,cols=2,subplot_titles = ("Confirmed","Deaths","Recovered","Active"))

fig.append_trace(go.Box(y = china["Confirmed"],name = "Confirmed",marker_color = 'rgb(4,90,141)'),row = 1, col = 1)
fig.append_trace(go.Box(y = china["Deaths"], name = "Deaths",marker_color = 'rgb(152,0,67)'),row = 1, col = 2)
fig.append_trace(go.Box(y = china["Recovered"], name = "Recovered",marker_color = 'rgb(1,108,89)'),row = 2, col = 1)
fig.append_trace(go.Box(y = china["Active"], name = "Active",marker_color = 'rgb(84,39,143)'),row = 2, col = 2)

fig.update_xaxes(title_text="Confirmed", row=1, col=1)
fig.update_xaxes(title_text="Deaths", row=1, col=2)
fig.update_xaxes(title_text="Recovered", row=2, col=1)
fig.update_xaxes(title_text="Active", row=2, col=2)

fig.update_yaxes(title_text="Value", row=1, col=1)
fig.update_yaxes(title_text="Value", row=1, col=2)
fig.update_yaxes(title_text="Value", row=2, col=1)
fig.update_yaxes(title_text="Value", row=2, col=2)

fig.update_layout(height = 1000,title_text="Boxplots and Outliers",template = "plotly_white")
iplot(fig)

china = china.tail(5)
trace1 = go.Bar(x = china["Confirmed"],
               y = china["Date"],
               orientation = "h",
               text = china["Confirmed"],
               textposition = "auto",
               name = "Confirmed",
               marker = dict(color = "rgba(4,90,141,0.8)"))
                #,line = dict(color = "rgb(0,0,0)", width = 1.2)))

trace2 = go.Bar(x = china["Deaths"],
               y = china["Date"],
               orientation = "h",
               text = china["Deaths"],
               textposition = "auto",
               name = "Deaths",
               marker = dict(color = "rgba(152,0,67,0.8)"))
                             #,line = dict(color = "rgb(0,0,0)", width = 1.2)))

trace3 = go.Bar(x = china["Recovered"],
               y = china["Date"],
               orientation = "h",
               text = china["Recovered"],
               textposition = "auto",
               name = "Recovered",
               marker = dict(color = "rgba(1,108,89,0.8)"))
                             #,line = dict(color = "rgb(0,0,0)", width = 1.2)))

trace4 = go.Bar(x = china["Active"],
               y = china["Date"],
               orientation = "h",
               text = china["Active"],
               textposition = "auto",
               name = "Active",
               marker = dict(color = "rgba(84,39,143,0.8)"))
                             #,line = dict(color = "rgb(0,0,0)", width = 1.2)))

data_bar = [trace4,trace2,trace3,trace1]
layout = go.Layout(height = 1000, title = "Last 10 Days in Australia", template = "plotly_white",yaxis_title="Date",xaxis_title="Value")
fig = go.Figure(data = data_bar, layout = layout)
iplot(fig)

france = data[data["Country/Region"] == "France"]
france.head()
date_list1 = list(france["Date"].unique())
confirmed = []
deaths = []
recovered = []
active = []
for i in date_list1:
    x = france[france["Date"] == i]
    confirmed.append(sum(x["Confirmed"]))
    deaths.append(sum(x["Deaths"]))
    recovered.append(sum(x["Recovered"]))
    active.append(sum(x["Active"]))
france = pd.DataFrame(list(zip(date_list1,confirmed,deaths,recovered,active)),columns = ["Date","Confirmed","Deaths","Recovered","Active"])
france.tail()
trace1 = go.Scatter( 
x = france["Date"],
y = france["Confirmed"],
mode = "lines",
name = "Confirmed",
line = dict(width = 2.5),
marker = dict(color = 'rgba(4,90,141, 0.8)')
)

trace2 = go.Scatter(
x = france["Date"],
y = france["Deaths"],
mode = "lines",
name = "Deaths",
line = dict(width = 2.5),
marker = dict(color = 'rgba(152,0,67, 0.8)')
)

trace3 = go.Scatter(
x = france["Date"],
y = france["Recovered"],
mode = "lines",
name = "Recovered",
line = dict(width = 2.5),    
marker = dict(color = 'rgba(1,108,89, 0.8)')
)

trace4 = go.Scatter(
x = france["Date"],
y = france["Active"],
mode = "lines",
name = "Active",
line = dict(width = 2.5),
marker = dict(color = 'rgba(84,39,143, 0.8)')
)

data_plt = [trace1,trace2,trace3,trace4]
layout = go.Layout(title = "France Patient Data",template = "plotly_white",xaxis_title="Date",yaxis_title="Value")
fig = go.Figure(data = data_plt,layout = layout)
iplot(fig)
labels = ["Recovered","Deaths","Active"]
values = [np.sum(france["Recovered"]),np.sum(france["Deaths"]),np.sum(france["Active"])]

fig = go.Figure(data = [go.Pie(labels = labels, values = values,pull = [0.05,0.05,0.05],textinfo='label+percent',insidetextorientation='radial')],layout = go.Layout(title = "France Patient Percentage"))
fig.show()
from plotly.subplots import make_subplots
fig = make_subplots(rows=2,cols=2,subplot_titles = ("Confirmed","Deaths","Recovered","Active"))

fig.append_trace(go.Box(y = france["Confirmed"],name = "Confirmed",marker_color = 'rgb(4,90,141)'),row = 1, col = 1)
fig.append_trace(go.Box(y = france["Deaths"], name = "Deaths",marker_color = 'rgb(152,0,67)'),row = 1, col = 2)
fig.append_trace(go.Box(y = france["Recovered"], name = "Recovered",marker_color = 'rgb(1,108,89)'),row = 2, col = 1)
fig.append_trace(go.Box(y = france["Active"], name = "Active",marker_color = 'rgb(84,39,143)'),row = 2, col = 2)

fig.update_xaxes(title_text="Confirmed", row=1, col=1)
fig.update_xaxes(title_text="Deaths", row=1, col=2)
fig.update_xaxes(title_text="Recovered", row=2, col=1)
fig.update_xaxes(title_text="Active", row=2, col=2)

fig.update_yaxes(title_text="Value", row=1, col=1)
fig.update_yaxes(title_text="Value", row=1, col=2)
fig.update_yaxes(title_text="Value", row=2, col=1)
fig.update_yaxes(title_text="Value", row=2, col=2)

fig.update_layout(height = 1000,title_text="Boxplots and Outliers",template = "plotly_white")
iplot(fig)

france = france.tail(5)
trace1 = go.Bar(x = france["Confirmed"],
               y = france["Date"],
               orientation = "h",
               text = france["Confirmed"],
               textposition = "auto",
               name = "Confirmed",
               marker = dict(color = "rgba(4,90,141,0.8)"))
                #,line = dict(color = "rgb(0,0,0)", width = 1.2)))

trace2 = go.Bar(x = france["Deaths"],
               y = france["Date"],
               orientation = "h",
               text = france["Deaths"],
               textposition = "auto",
               name = "Deaths",
               marker = dict(color = "rgba(152,0,67,0.8)"))
                             #,line = dict(color = "rgb(0,0,0)", width = 1.2)))

trace3 = go.Bar(x = france["Recovered"],
               y = france["Date"],
               orientation = "h",
               text = france["Recovered"],
               textposition = "auto",
               name = "Recovered",
               marker = dict(color = "rgba(1,108,89,0.8)"))
                             #,line = dict(color = "rgb(0,0,0)", width = 1.2)))

trace4 = go.Bar(x = france["Active"],
               y = france["Date"],
               orientation = "h",
               text = france["Active"],
               textposition = "auto",
               name = "Active",
               marker = dict(color = "rgba(84,39,143,0.8)"))
                             #,line = dict(color = "rgb(0,0,0)", width = 1.2)))

data_bar = [trace2,trace3,trace4,trace1]
layout = go.Layout(height = 1000, title = "Last 10 Days in Australia", template = "plotly_white",yaxis_title="Date",xaxis_title="Value")
fig = go.Figure(data = data_bar, layout = layout)
iplot(fig)

germany = data[data["Country/Region"] == "Germany"]
germany.head()
date_list1 = list(germany["Date"].unique())
confirmed = []
deaths = []
recovered = []
active = []
for i in date_list1:
    x = germany[germany["Date"] == i]
    confirmed.append(sum(x["Confirmed"]))
    deaths.append(sum(x["Deaths"]))
    recovered.append(sum(x["Recovered"]))
    active.append(sum(x["Active"]))
germany = pd.DataFrame(list(zip(date_list1,confirmed,deaths,recovered,active)),columns = ["Date","Confirmed","Deaths","Recovered","Active"])
germany.tail()
trace1 = go.Scatter( 
x = germany["Date"],
y = germany["Confirmed"],
mode = "lines",
name = "Confirmed",
line = dict(width = 2.5),
marker = dict(color = 'rgba(4,90,141, 0.8)')
)

trace2 = go.Scatter(
x = germany["Date"],
y = germany["Deaths"],
mode = "lines",
name = "Deaths",
line = dict(width = 2.5),
marker = dict(color = 'rgba(152,0,67, 0.8)')
)

trace3 = go.Scatter(
x = germany["Date"],
y = germany["Recovered"],
mode = "lines",
name = "Recovered",
line = dict(width = 2.5),    
marker = dict(color = 'rgba(1,108,89, 0.8)')
)

trace4 = go.Scatter(
x = germany["Date"],
y = germany["Active"],
mode = "lines",
name = "Active",
line = dict(width = 2.5),
marker = dict(color = 'rgba(84,39,143, 0.8)')
)

data_plt = [trace1,trace2,trace3,trace4]
layout = go.Layout(title = "Germany Patient Data",template = "plotly_white",xaxis_title="Date",yaxis_title="Value")
fig = go.Figure(data = data_plt,layout = layout)
iplot(fig)
labels = ["Recovered","Deaths","Active"]
values = [np.sum(germany["Recovered"]),np.sum(germany["Deaths"]),np.sum(germany["Active"])]

fig = go.Figure(data = [go.Pie(labels = labels, values = values,pull = [0.05,0.05,0.05],textinfo='label+percent',insidetextorientation='radial')],layout = go.Layout(title = "Germany Patient Percentage"))
fig.show()
from plotly.subplots import make_subplots
fig = make_subplots(rows=2,cols=2,subplot_titles = ("Confirmed","Deaths","Recovered","Active"))

fig.append_trace(go.Box(y = germany["Confirmed"],name = "Confirmed",marker_color = 'rgb(1,102,94)'),row = 1, col = 1)
fig.append_trace(go.Box(y = germany["Deaths"], name = "Deaths",marker_color = 'rgb(152,0,67)'),row = 1, col = 2)
fig.append_trace(go.Box(y = germany["Recovered"], name = "Recovered",marker_color = 'rgb(1,108,89)'),row = 2, col = 1)
fig.append_trace(go.Box(y = germany["Active"], name = "Active",marker_color = 'rgb(84,39,143)'),row = 2, col = 2)

fig.update_xaxes(title_text="Confirmed", row=1, col=1)
fig.update_xaxes(title_text="Deaths", row=1, col=2)
fig.update_xaxes(title_text="Recovered", row=2, col=1)
fig.update_xaxes(title_text="Active", row=2, col=2)

fig.update_yaxes(title_text="Value", row=1, col=1)
fig.update_yaxes(title_text="Value", row=1, col=2)
fig.update_yaxes(title_text="Value", row=2, col=1)
fig.update_yaxes(title_text="Value", row=2, col=2)

fig.update_layout(height = 1000,title_text="Boxplots and Outliers",template = "plotly_white")
iplot(fig)

germany = germany.tail(5)
trace1 = go.Bar(x = germany["Confirmed"],
               y = germany["Date"],
               orientation = "h",
               text = germany["Confirmed"],
               textposition = "auto",
               name = "Confirmed",
               marker = dict(color = "rgba(4,90,141,0.8)"))
                #,line = dict(color = "rgb(0,0,0)", width = 1.2)))

trace2 = go.Bar(x = germany["Deaths"],
               y = germany["Date"],
               orientation = "h",
               text = germany["Deaths"],
               textposition = "auto",
               name = "Deaths",
               marker = dict(color = "rgba(152,0,67,0.8)"))
                             #,line = dict(color = "rgb(0,0,0)", width = 1.2)))

trace3 = go.Bar(x = germany["Recovered"],
               y = germany["Date"],
               orientation = "h",
               text = germany["Recovered"],
               textposition = "auto",
               name = "Recovered",
               marker = dict(color = "rgba(1,108,89,0.8)"))
                             #,line = dict(color = "rgb(0,0,0)", width = 1.2)))

trace4 = go.Bar(x = germany["Active"],
               y = germany["Date"],
               orientation = "h",
               text = germany["Active"],
               textposition = "auto",
               name = "Active",
               marker = dict(color = "rgba(84,39,143,0.8)"))
                             #,line = dict(color = "rgb(0,0,0)", width = 1.2)))

data_bar = [trace4,trace2,trace3,trace1]
layout = go.Layout(height = 1000, title = "Last 10 Days in Australia", template = "plotly_white",yaxis_title="Date",xaxis_title="Value")
fig = go.Figure(data = data_bar, layout = layout)
iplot(fig)
italy = data[data["Country/Region"] == "Italy"]
italy.head()
date_list1 = list(italy["Date"].unique())
confirmed = []
deaths = []
recovered = []
active = []
for i in date_list1:
    x = italy[italy["Date"] == i]
    confirmed.append(sum(x["Confirmed"]))
    deaths.append(sum(x["Deaths"]))
    recovered.append(sum(x["Recovered"]))
    active.append(sum(x["Active"]))
italy = pd.DataFrame(list(zip(date_list1,confirmed,deaths,recovered,active)),columns = ["Date","Confirmed","Deaths","Recovered","Active"])
italy.tail()
trace1 = go.Scatter( 
x = italy["Date"],
y = italy["Confirmed"],
mode = "lines",
name = "Confirmed",
line = dict(width = 2.5),
marker = dict(color = 'rgba(4,90,141, 0.8)')
)

trace2 = go.Scatter(
x = italy["Date"],
y = italy["Deaths"],
mode = "lines",
name = "Deaths",
line = dict(width = 2.5),
marker = dict(color = 'rgba(152,0,67, 0.8)')
)

trace3 = go.Scatter(
x = italy["Date"],
y = italy["Recovered"],
mode = "lines",
name = "Recovered",
line = dict(width = 2.5),    
marker = dict(color = 'rgba(1,108,89, 0.8)')
)

trace4 = go.Scatter(
x = italy["Date"],
y = italy["Active"],
mode = "lines",
name = "Active",
line = dict(width = 2.5),
marker = dict(color = 'rgba(84,39,143, 0.8)')
)

data_plt = [trace1,trace2,trace3,trace4]
layout = go.Layout(title = "Italy Patient Data",template = "plotly_white",xaxis_title="Date",yaxis_title="Value")
fig = go.Figure(data = data_plt,layout = layout)
iplot(fig)
labels = ["Recovered","Deaths","Active"]
values = [np.sum(italy["Recovered"]),np.sum(italy["Deaths"]),np.sum(italy["Active"])]

fig = go.Figure(data = [go.Pie(labels = labels, values = values,pull = [0.05,0.05,0.05],textinfo='label+percent',insidetextorientation='radial')],layout = go.Layout(title = "Italy Patient Percentage"))
fig.show()
from plotly.subplots import make_subplots
fig = make_subplots(rows=2,cols=2,subplot_titles = ("Confirmed","Deaths","Recovered","Active"))

fig.append_trace(go.Box(y = italy["Confirmed"],name = "Confirmed",marker_color = 'rgb(4,90,141)'),row = 1, col = 1)
fig.append_trace(go.Box(y = italy["Deaths"], name = "Deaths",marker_color = 'rgb(152,0,67)'),row = 1, col = 2)
fig.append_trace(go.Box(y = italy["Recovered"], name = "Recovered",marker_color = 'rgb(1,108,89)'),row = 2, col = 1)
fig.append_trace(go.Box(y = italy["Active"], name = "Active",marker_color = 'rgb(84,39,143)'),row = 2, col = 2)

fig.update_xaxes(title_text="Confirmed", row=1, col=1)
fig.update_xaxes(title_text="Deaths", row=1, col=2)
fig.update_xaxes(title_text="Recovered", row=2, col=1)
fig.update_xaxes(title_text="Active", row=2, col=2)

fig.update_yaxes(title_text="Value", row=1, col=1)
fig.update_yaxes(title_text="Value", row=1, col=2)
fig.update_yaxes(title_text="Value", row=2, col=1)
fig.update_yaxes(title_text="Value", row=2, col=2)

fig.update_layout(height = 1000,title_text="Boxplots and Outliers",template = "plotly_white")
iplot(fig)
italy = italy.tail(5)
trace1 = go.Bar(x = italy["Confirmed"],
               y = italy["Date"],
               orientation = "h",
               text = italy["Confirmed"],
               textposition = "auto",
               name = "Confirmed",
               marker = dict(color = "rgba(4,90,141,0.8)"))
                #,line = dict(color = "rgb(0,0,0)", width = 1.2)))

trace2 = go.Bar(x = italy["Deaths"],
               y = italy["Date"],
               orientation = "h",
               text = italy["Deaths"],
               textposition = "auto",
               name = "Deaths",
               marker = dict(color = "rgba(152,0,67,0.8)"))
                             #,line = dict(color = "rgb(0,0,0)", width = 1.2)))

trace3 = go.Bar(x = italy["Recovered"],
               y = italy["Date"],
               orientation = "h",
               text = italy["Recovered"],
               textposition = "auto",
               name = "Recovered",
               marker = dict(color = "rgba(1,108,89,0.8)"))
                             #,line = dict(color = "rgb(0,0,0)", width = 1.2)))

trace4 = go.Bar(x = italy["Active"],
               y = italy["Date"],
               orientation = "h",
               text = italy["Active"],
               textposition = "auto",
               name = "Active",
               marker = dict(color = "rgba(84,39,143,0.8)"))
                             #,line = dict(color = "rgb(0,0,0)", width = 1.2)))

data_bar = [trace4,trace2,trace3,trace1]
layout = go.Layout(height = 1000, title = "Last 10 Days in Australia", template = "plotly_white",yaxis_title="Date",xaxis_title="Value")
fig = go.Figure(data = data_bar, layout = layout)
iplot(fig)
korea = data[data["Country/Region"] == "South Korea"]
korea.head()
date_list1 = list(korea["Date"].unique())
confirmed = []
deaths = []
recovered = []
active = []
for i in date_list1:
    x = korea[korea["Date"] == i]
    confirmed.append(sum(x["Confirmed"]))
    deaths.append(sum(x["Deaths"]))
    recovered.append(sum(x["Recovered"]))
    active.append(sum(x["Active"]))
korea = pd.DataFrame(list(zip(date_list1,confirmed,deaths,recovered,active)),columns = ["Date","Confirmed","Deaths","Recovered","Active"])
korea.tail()
trace1 = go.Scatter( 
x = korea["Date"],
y = korea["Confirmed"],
mode = "lines",
name = "Confirmed",
line = dict(width = 2.5),
marker = dict(color = 'rgba(4,90,141, 0.8)')
)

trace2 = go.Scatter(
x = korea["Date"],
y = korea["Deaths"],
mode = "lines",
name = "Deaths",
line = dict(width = 2.5),
marker = dict(color = 'rgba(152,0,67, 0.8)')
)

trace3 = go.Scatter(
x = korea["Date"],
y = korea["Recovered"],
mode = "lines",
name = "Recovered",
line = dict(width = 2.5),    
marker = dict(color = 'rgba(1,108,89, 0.8)')
)

trace4 = go.Scatter(
x = korea["Date"],
y = korea["Active"],
mode = "lines",
name = "Active",
line = dict(width = 2.5),
marker = dict(color = 'rgba(84,39,143, 0.8)')
)

data_plt = [trace1,trace2,trace3,trace4]
layout = go.Layout(title = "South Korea Patient Data",template = "plotly_white",xaxis_title="Date",yaxis_title="Value")
fig = go.Figure(data = data_plt,layout = layout)
iplot(fig)
labels = ["Recovered","Deaths","Active"]
values = [np.sum(korea["Recovered"]),np.sum(korea["Deaths"]),np.sum(korea["Active"])]

fig = go.Figure(data = [go.Pie(labels = labels, values = values,pull = [0.05,0.05,0.05],textinfo='label+percent',insidetextorientation='radial')],layout = go.Layout(title = "South Korea Patient Percentage"))
fig.show()
from plotly.subplots import make_subplots
fig = make_subplots(rows=2,cols=2,subplot_titles = ("Confirmed","Deaths","Recovered","Active"))

fig.append_trace(go.Box(y = korea["Confirmed"],name = "Confirmed",marker_color = 'rgb(4,90,141)'),row = 1, col = 1)
fig.append_trace(go.Box(y = korea["Deaths"], name = "Deaths",marker_color = 'rgb(152,0,67)'),row = 1, col = 2)
fig.append_trace(go.Box(y = korea["Recovered"], name = "Recovered",marker_color = 'rgb(1,108,89)'),row = 2, col = 1)
fig.append_trace(go.Box(y = korea["Active"], name = "Active",marker_color = 'rgb(84,39,143)'),row = 2, col = 2)

fig.update_xaxes(title_text="Confirmed", row=1, col=1)
fig.update_xaxes(title_text="Deaths", row=1, col=2)
fig.update_xaxes(title_text="Recovered", row=2, col=1)
fig.update_xaxes(title_text="Active", row=2, col=2)

fig.update_yaxes(title_text="Value", row=1, col=1)
fig.update_yaxes(title_text="Value", row=1, col=2)
fig.update_yaxes(title_text="Value", row=2, col=1)
fig.update_yaxes(title_text="Value", row=2, col=2)

fig.update_layout(height = 1000,title_text="Boxplots and Outliers",template = "plotly_white")
iplot(fig)
korea = korea.tail(5)
trace1 = go.Bar(x = korea["Confirmed"],
               y = korea["Date"],
               orientation = "h",
               text = korea["Confirmed"],
               textposition = "auto",
               name = "Confirmed",
               marker = dict(color = "rgba(4,90,141,0.8)")) #1,102,94
                #,line = dict(color = "rgb(0,0,0)", width = 1.2)))

trace2 = go.Bar(x = korea["Deaths"],
               y = korea["Date"],
               orientation = "h",
               text = korea["Deaths"],
               textposition = "auto",
               name = "Deaths",
               marker = dict(color = "rgba(152,0,67,0.8)")) #204, 0, 0
                             #,line = dict(color = "rgb(0,0,0)", width = 1.2)))

trace3 = go.Bar(x = korea["Recovered"],
               y = korea["Date"],
               orientation = "h",
               text = korea["Recovered"],
               textposition = "auto",
               name = "Recovered",
               marker = dict(color = "rgba(1,108,89,0.8)")) #16, 112, 2
                             #,line = dict(color = "rgb(0,0,0)", width = 1.2)))

trace4 = go.Bar(x = korea["Active"],
               y = korea["Date"],
               orientation = "h",
               text = korea["Active"],
               textposition = "auto",
               name = "Active",
               marker = dict(color = "rgba(84,39,143,0.8)")) #118,42,131
                             #,line = dict(color = "rgb(0,0,0)", width = 1.2)))

data_bar = [trace2,trace4,trace3,trace1]
layout = go.Layout(height = 1000, title = "Last 10 Days in Australia", template = "plotly_white",yaxis_title="Date",xaxis_title="Value")
fig = go.Figure(data = data_bar, layout = layout)
iplot(fig)
spain = data[data["Country/Region"] == "Spain"]
spain.head()
date_list1 = list(spain["Date"].unique())
confirmed = []
deaths = []
recovered = []
active = []
for i in date_list1:
    x = spain[spain["Date"] == i]
    confirmed.append(sum(x["Confirmed"]))
    deaths.append(sum(x["Deaths"]))
    recovered.append(sum(x["Recovered"]))
    active.append(sum(x["Active"]))
spain = pd.DataFrame(list(zip(date_list1,confirmed,deaths,recovered,active)),columns = ["Date","Confirmed","Deaths","Recovered","Active"])
spain.tail()
trace1 = go.Scatter( 
x = spain["Date"],
y = spain["Confirmed"],
mode = "lines",
name = "Confirmed",
line = dict(width = 2.5),
marker = dict(color = 'rgba(4,90,141, 0.8)')
)

trace2 = go.Scatter(
x = spain["Date"],
y = spain["Deaths"],
mode = "lines",
name = "Deaths",
line = dict(width = 2.5),
marker = dict(color = 'rgba(152,0,67, 0.8)')
)

trace3 = go.Scatter(
x = spain["Date"],
y = spain["Recovered"],
mode = "lines",
name = "Recovered",
line = dict(width = 2.5),    
marker = dict(color = 'rgba(1,108,89, 0.8)')
)

trace4 = go.Scatter(
x = spain["Date"],
y = spain["Active"],
mode = "lines",
name = "Active",
line = dict(width = 2.5),
marker = dict(color = 'rgba(84,39,143, 0.8)')
)

data_plt = [trace1,trace2,trace3,trace4]
layout = go.Layout(title = "Spain Patient Data",template = "plotly_white",xaxis_title="Date",yaxis_title="Value")
fig = go.Figure(data = data_plt,layout = layout)
iplot(fig)
labels = ["Recovered","Deaths","Active"]
values = [np.sum(spain["Recovered"]),np.sum(spain["Deaths"]),np.sum(spain["Active"])]

fig = go.Figure(data = [go.Pie(labels = labels, values = values,pull = [0.05,0.05,0.05],textinfo='label+percent',insidetextorientation='radial')],layout = go.Layout(title = "Spain Patient Percentage"))
fig.show()
from plotly.subplots import make_subplots
fig = make_subplots(rows=2,cols=2,subplot_titles = ("Confirmed","Deaths","Recovered","Active"))

fig.append_trace(go.Box(y = spain["Confirmed"],name = "Confirmed",marker_color = 'rgb(4,90,141)'),row = 1, col = 1)
fig.append_trace(go.Box(y = spain["Deaths"], name = "Deaths",marker_color = 'rgb(152,0,67)'),row = 1, col = 2)
fig.append_trace(go.Box(y = spain["Recovered"], name = "Recovered",marker_color = 'rgb(1,108,89)'),row = 2, col = 1)
fig.append_trace(go.Box(y = spain["Active"], name = "Active",marker_color = 'rgb(84,39,143)'),row = 2, col = 2)

fig.update_xaxes(title_text="Confirmed", row=1, col=1)
fig.update_xaxes(title_text="Deaths", row=1, col=2)
fig.update_xaxes(title_text="Recovered", row=2, col=1)
fig.update_xaxes(title_text="Active", row=2, col=2)

fig.update_yaxes(title_text="Value", row=1, col=1)
fig.update_yaxes(title_text="Value", row=1, col=2)
fig.update_yaxes(title_text="Value", row=2, col=1)
fig.update_yaxes(title_text="Value", row=2, col=2)

fig.update_layout(height = 1000,title_text="Boxplots and Outliers",template = "plotly_white")
iplot(fig)
spain = spain.tail(5)
trace1 = go.Bar(x = spain["Confirmed"],
               y = spain["Date"],
               orientation = "h",
               text = spain["Confirmed"],
               textposition = "auto",
               name = "Confirmed",
               marker = dict(color = "rgba(4,90,141,0.8)")) #1,102,94
                #,line = dict(color = "rgb(0,0,0)", width = 1.2)))

trace2 = go.Bar(x = spain["Deaths"],
               y = spain["Date"],
               orientation = "h",
               text = spain["Deaths"],
               textposition = "auto",
               name = "Deaths",
               marker = dict(color = "rgba(152,0,67,0.8)")) #204, 0, 0
                             #,line = dict(color = "rgb(0,0,0)", width = 1.2)))

trace3 = go.Bar(x = spain["Recovered"],
               y = spain["Date"],
               orientation = "h",
               text = spain["Recovered"],
               textposition = "auto",
               name = "Recovered",
               marker = dict(color = "rgba(1,108,89,0.8)")) #16, 112, 2
                             #,line = dict(color = "rgb(0,0,0)", width = 1.2)))

trace4 = go.Bar(x = spain["Active"],
               y = spain["Date"],
               orientation = "h",
               text = spain["Active"],
               textposition = "auto",
               name = "Active",
               marker = dict(color = "rgba(84,39,143,0.8)")) #118,42,131
                             #,line = dict(color = "rgb(0,0,0)", width = 1.2)))

data_bar = [trace2,trace4,trace3,trace1]
layout = go.Layout(height = 1000, title = "Last 10 Days in Australia", template = "plotly_white",yaxis_title="Date",xaxis_title="Value")
fig = go.Figure(data = data_bar, layout = layout)
iplot(fig)
turk = data[data["Country/Region"] == "Turkey"]
turk.head()
date_list1 = list(turk["Date"].unique())
confirmed = []
deaths = []
recovered = []
active = []
for i in date_list1:
    x = turk[turk["Date"] == i]
    confirmed.append(sum(x["Confirmed"]))
    deaths.append(sum(x["Deaths"]))
    recovered.append(sum(x["Recovered"]))
    active.append(sum(x["Active"]))
turk = pd.DataFrame(list(zip(date_list1,confirmed,deaths,recovered,active)),columns = ["Date","Confirmed","Deaths","Recovered","Active"])
turk.tail()
trace1 = go.Scatter( 
x = turk["Date"],
y = turk["Confirmed"],
mode = "lines",
name = "Confirmed",
line = dict(width = 2.5),
marker = dict(color = 'rgba(4,90,141, 0.8)')
)

trace2 = go.Scatter(
x = turk["Date"],
y = turk["Deaths"],
mode = "lines",
name = "Deaths",
line = dict(width = 2.5),
marker = dict(color = 'rgba(152,0,67, 0.8)')
)

trace3 = go.Scatter(
x = turk["Date"],
y = turk["Recovered"],
mode = "lines",
name = "Recovered",
line = dict(width = 2.5),    
marker = dict(color = 'rgba(1,108,89, 0.8)')
)

trace4 = go.Scatter(
x = turk["Date"],
y = turk["Active"],
mode = "lines",
name = "Active",
line = dict(width = 2.5),
marker = dict(color = 'rgba(84,39,143, 0.8)')
)

data_plt = [trace1,trace2,trace3,trace4]
layout = go.Layout(title = "Turkey Patient Data",template = "plotly_white",xaxis_title="Date",yaxis_title="Value")
fig = go.Figure(data = data_plt,layout = layout)
iplot(fig)
labels = ["Recovered","Deaths","Active"]
values = [np.sum(turk["Recovered"]),np.sum(turk["Deaths"]),np.sum(turk["Active"])]

fig = go.Figure(data = [go.Pie(labels = labels, values = values,pull = [0.05,0.05,0.05],textinfo='label+percent',insidetextorientation='radial')],layout = go.Layout(title = "Turkey Patient Percentage"))
fig.show()
from plotly.subplots import make_subplots
fig = make_subplots(rows=2,cols=2,subplot_titles = ("Confirmed","Deaths","Recovered","Active"))

fig.append_trace(go.Box(y = turk["Confirmed"],name = "Confirmed",marker_color = 'rgb(4,90,141)'),row = 1, col = 1)
fig.append_trace(go.Box(y = turk["Deaths"], name = "Deaths",marker_color = 'rgb(152,0,67)'),row = 1, col = 2)
fig.append_trace(go.Box(y = turk["Recovered"], name = "Recovered",marker_color = 'rgb(1,108,89)'),row = 2, col = 1)
fig.append_trace(go.Box(y = turk["Active"], name = "Active",marker_color = 'rgb(84,39,143)'),row = 2, col = 2)

fig.update_xaxes(title_text="Confirmed", row=1, col=1)
fig.update_xaxes(title_text="Deaths", row=1, col=2)
fig.update_xaxes(title_text="Recovered", row=2, col=1)
fig.update_xaxes(title_text="Active", row=2, col=2)

fig.update_yaxes(title_text="Value", row=1, col=1)
fig.update_yaxes(title_text="Value", row=1, col=2)
fig.update_yaxes(title_text="Value", row=2, col=1)
fig.update_yaxes(title_text="Value", row=2, col=2)

fig.update_layout(height = 1000,title_text="Boxplots and Outliers",template = "plotly_white")
iplot(fig)
turk = turk.tail(5)
trace1 = go.Bar(x = turk["Confirmed"],
               y = turk["Date"],
               orientation = "h",
               text = turk["Confirmed"],
               textposition = "auto",
               name = "Confirmed",
               marker = dict(color = "rgba(4,90,141,0.8)")) #1,102,94
                #,line = dict(color = "rgb(0,0,0)", width = 1.2)))

trace2 = go.Bar(x = turk["Deaths"],
               y = turk["Date"],
               orientation = "h",
               text = turk["Deaths"],
               textposition = "auto",
               name = "Deaths",
               marker = dict(color = "rgba(152,0,67,0.8)")) #204, 0, 0
                             #,line = dict(color = "rgb(0,0,0)", width = 1.2)))

trace3 = go.Bar(x = turk["Recovered"],
               y = turk["Date"],
               orientation = "h",
               text = turk["Recovered"],
               textposition = "auto",
               name = "Recovered",
               marker = dict(color = "rgba(1,108,89,0.8)")) #16, 112, 2
                             #,line = dict(color = "rgb(0,0,0)", width = 1.2)))

trace4 = go.Bar(x = turk["Active"],
               y = turk["Date"],
               orientation = "h",
               text = turk["Active"],
               textposition = "auto",
               name = "Active",
               marker = dict(color = "rgba(84,39,143,0.8)")) #118,42,131
                             #,line = dict(color = "rgb(0,0,0)", width = 1.2)))

data_bar = [trace2,trace4,trace3,trace1]
layout = go.Layout(height = 1000, title = "Last 10 Days in Australia", template = "plotly_white",yaxis_title="Date",xaxis_title="Value")
fig = go.Figure(data = data_bar, layout = layout)
iplot(fig)
uk = data[data["Country/Region"] == "United Kingdom"]
uk.head()
date_list1 = list(uk["Date"].unique())
confirmed = []
deaths = []
recovered = []
active = []
for i in date_list1:
    x = uk[uk["Date"] == i]
    confirmed.append(sum(x["Confirmed"]))
    deaths.append(sum(x["Deaths"]))
    recovered.append(sum(x["Recovered"]))
    active.append(sum(x["Active"]))
uk = pd.DataFrame(list(zip(date_list1,confirmed,deaths,recovered,active)),columns = ["Date","Confirmed","Deaths","Recovered","Active"])
uk.tail()
trace1 = go.Scatter( 
x = uk["Date"],
y = uk["Confirmed"],
mode = "lines",
name = "Confirmed",
line = dict(width = 2.5),
marker = dict(color = 'rgba(4,90,141, 0.8)')
)

trace2 = go.Scatter(
x = uk["Date"],
y = uk["Deaths"],
mode = "lines",
name = "Deaths",
line = dict(width = 2.5),
marker = dict(color = 'rgba(152,0,67, 0.8)')
)

trace3 = go.Scatter(
x = uk["Date"],
y = uk["Recovered"],
mode = "lines",
name = "Recovered",
line = dict(width = 2.5),    
marker = dict(color = 'rgba(1,108,89, 0.8)')
)

trace4 = go.Scatter(
x = uk["Date"],
y = uk["Active"],
mode = "lines",
name = "Active",
line = dict(width = 2.5),
marker = dict(color = 'rgba(84,39,143, 0.8)')
)

data_plt = [trace1,trace2,trace3,trace4]
layout = go.Layout(title = "UK Patient Data",template = "plotly_white",xaxis_title="Date",yaxis_title="Value")
fig = go.Figure(data = data_plt,layout = layout)
iplot(fig)
labels = ["Recovered","Deaths","Active"]
values = [np.sum(uk["Recovered"]),np.sum(uk["Deaths"]),np.sum(uk["Active"])]

fig = go.Figure(data = [go.Pie(labels = labels, values = values,pull = [0.05,0.05,0.05],textinfo='label+percent',insidetextorientation='radial')],layout = go.Layout(title = "UK Patient Percentage"))
fig.show()
from plotly.subplots import make_subplots
fig = make_subplots(rows=2,cols=2,subplot_titles = ("Confirmed","Deaths","Recovered","Active"))

fig.append_trace(go.Box(y = uk["Confirmed"],name = "Confirmed",marker_color = 'rgb(4,90,141)'),row = 1, col = 1)
fig.append_trace(go.Box(y = uk["Deaths"], name = "Deaths",marker_color = 'rgb(152,0,67)'),row = 1, col = 2)
fig.append_trace(go.Box(y = uk["Recovered"], name = "Recovered",marker_color = 'rgb(1,108,89)'),row = 2, col = 1)
fig.append_trace(go.Box(y = uk["Active"], name = "Active",marker_color = 'rgb(84,39,143)'),row = 2, col = 2)

fig.update_xaxes(title_text="Confirmed", row=1, col=1)
fig.update_xaxes(title_text="Deaths", row=1, col=2)
fig.update_xaxes(title_text="Recovered", row=2, col=1)
fig.update_xaxes(title_text="Active", row=2, col=2)

fig.update_yaxes(title_text="Value", row=1, col=1)
fig.update_yaxes(title_text="Value", row=1, col=2)
fig.update_yaxes(title_text="Value", row=2, col=1)
fig.update_yaxes(title_text="Value", row=2, col=2)

fig.update_layout(height = 1000,title_text="Boxplots and Outliers",template = "plotly_white")
iplot(fig)
uk = uk.tail(5)
trace1 = go.Bar(x = uk["Confirmed"],
               y = uk["Date"],
               orientation = "h",
               text = uk["Confirmed"],
               textposition = "auto",
               name = "Confirmed",
               marker = dict(color = "rgba(4,90,141,0.8)")) #1,102,94
                #,line = dict(color = "rgb(0,0,0)", width = 1.2)))

trace2 = go.Bar(x = uk["Deaths"],
               y = uk["Date"],
               orientation = "h",
               text = uk["Deaths"],
               textposition = "auto",
               name = "Deaths",
               marker = dict(color = "rgba(152,0,67,0.8)")) #204, 0, 0
                             #,line = dict(color = "rgb(0,0,0)", width = 1.2)))

trace3 = go.Bar(x = uk["Recovered"],
               y = uk["Date"],
               orientation = "h",
               text = uk["Recovered"],
               textposition = "auto",
               name = "Recovered",
               marker = dict(color = "rgba(1,108,89,0.8)")) #16, 112, 2
                             #,line = dict(color = "rgb(0,0,0)", width = 1.2)))

trace4 = go.Bar(x = uk["Active"],
               y = uk["Date"],
               orientation = "h",
               text = uk["Active"],
               textposition = "auto",
               name = "Active",
               marker = dict(color = "rgba(84,39,143,0.8)")) #118,42,131
                             #,line = dict(color = "rgb(0,0,0)", width = 1.2)))

data_bar = [trace3,trace2,trace4,trace1]
layout = go.Layout(height = 1000, title = "Last 10 Days in Australia", template = "plotly_white",yaxis_title="Date",xaxis_title="Value")
fig = go.Figure(data = data_bar, layout = layout)
iplot(fig)
us = data[data["Country/Region"] == "US"]
us.head()
date_list1 = list(us["Date"].unique())
confirmed = []
deaths = []
recovered = []
active = []
for i in date_list1:
    x = us[us["Date"] == i]
    confirmed.append(sum(x["Confirmed"]))
    deaths.append(sum(x["Deaths"]))
    recovered.append(sum(x["Recovered"]))
    active.append(sum(x["Active"]))
us = pd.DataFrame(list(zip(date_list1,confirmed,deaths,recovered,active)),columns = ["Date","Confirmed","Deaths","Recovered","Active"])
us.tail()
trace1 = go.Scatter( 
x = us["Date"],
y = us["Confirmed"],
mode = "lines",
name = "Confirmed",
line = dict(width = 2.5),
marker = dict(color = 'rgba(4,90,141, 0.8)')
)

trace2 = go.Scatter(
x = us["Date"],
y = us["Deaths"],
mode = "lines",
name = "Deaths",
line = dict(width = 2.5),
marker = dict(color = 'rgba(152,0,67, 0.8)')
)

trace3 = go.Scatter(
x = us["Date"],
y = us["Recovered"],
mode = "lines",
name = "Recovered",
line = dict(width = 2.5),    
marker = dict(color = 'rgba(1,108,89, 0.8)')
)

trace4 = go.Scatter(
x = us["Date"],
y = us["Active"],
mode = "lines",
name = "Active",
line = dict(width = 2.5),
marker = dict(color = 'rgba(84,39,143, 0.8)')
)

data_plt = [trace1,trace2,trace3,trace4]
layout = go.Layout(title = "US Patient Data",template = "plotly_white",xaxis_title="Date",yaxis_title="Value")
fig = go.Figure(data = data_plt,layout = layout)
iplot(fig)
labels = ["Recovered","Deaths","Active"]
values = [np.sum(us["Recovered"]),np.sum(us["Deaths"]),np.sum(us["Active"])]

fig = go.Figure(data = [go.Pie(labels = labels, values = values,pull = [0.05,0.05,0.05],textinfo='label+percent',insidetextorientation='radial')],layout = go.Layout(title = "US Patient Percentage"))
fig.show()
from plotly.subplots import make_subplots
fig = make_subplots(rows=2,cols=2,subplot_titles = ("Confirmed","Deaths","Recovered","Active"))

fig.append_trace(go.Box(y = us["Confirmed"],name = "Confirmed",marker_color = 'rgb(4,90,141)'),row = 1, col = 1)
fig.append_trace(go.Box(y = us["Deaths"], name = "Deaths",marker_color = 'rgb(152,0,67)'),row = 1, col = 2)
fig.append_trace(go.Box(y = us["Recovered"], name = "Recovered",marker_color = 'rgb(1,108,89)'),row = 2, col = 1)
fig.append_trace(go.Box(y = us["Active"], name = "Active",marker_color = 'rgb(84,39,143)'),row = 2, col = 2)

fig.update_xaxes(title_text="Confirmed", row=1, col=1)
fig.update_xaxes(title_text="Deaths", row=1, col=2)
fig.update_xaxes(title_text="Recovered", row=2, col=1)
fig.update_xaxes(title_text="Active", row=2, col=2)

fig.update_yaxes(title_text="Value", row=1, col=1)
fig.update_yaxes(title_text="Value", row=1, col=2)
fig.update_yaxes(title_text="Value", row=2, col=1)
fig.update_yaxes(title_text="Value", row=2, col=2)

fig.update_layout(height = 1000,title_text="Boxplots and Outliers",template = "plotly_white")
iplot(fig)
us = us.tail(5)
trace1 = go.Bar(x = us["Confirmed"],
               y = us["Date"],
               orientation = "h",
               text = us["Confirmed"],
               textposition = "auto",
               name = "Confirmed",
               marker = dict(color = "rgba(4,90,141,0.8)")) #1,102,94
                #,line = dict(color = "rgb(0,0,0)", width = 1.2)))

trace2 = go.Bar(x = us["Deaths"],
               y = us["Date"],
               orientation = "h",
               text = us["Deaths"],
               textposition = "auto",
               name = "Deaths",
               marker = dict(color = "rgba(152,0,67,0.8)")) #204, 0, 0
                             #,line = dict(color = "rgb(0,0,0)", width = 1.2)))

trace3 = go.Bar(x = us["Recovered"],
               y = us["Date"],
               orientation = "h",
               text = us["Recovered"],
               textposition = "auto",
               name = "Recovered",
               marker = dict(color = "rgba(1,108,89,0.8)")) #16, 112, 2
                             #,line = dict(color = "rgb(0,0,0)", width = 1.2)))

trace4 = go.Bar(x = us["Active"],
               y = us["Date"],
               orientation = "h",
               text = us["Active"],
               textposition = "auto",
               name = "Active",
               marker = dict(color = "rgba(84,39,143,0.8)")) #118,42,131
                             #,line = dict(color = "rgb(0,0,0)", width = 1.2)))

data_bar = [trace2,trace3,trace4,trace1]
layout = go.Layout(height = 1000, title = "Last 10 Days in Australia", template = "plotly_white",yaxis_title="Date",xaxis_title="Value")
fig = go.Figure(data = data_bar, layout = layout)
iplot(fig)
con = [] 

australia = australia.tail(1)
china = china.tail(1)
france = france.tail(1)
germany = germany.tail(1)
italy = italy.tail(1)
korea = korea.tail(1)
spain = spain.tail(1)
turk = turk.tail(1)
uk = uk.tail(1)
us = us.tail(1)

aus = int(australia["Confirmed"])
chi = int(china["Confirmed"])
fra = int(france["Confirmed"])
ger = int(germany["Confirmed"])
ita = int(italy["Confirmed"])
kor = int(korea["Confirmed"])
spa = int(spain["Confirmed"])
tur = int(turk["Confirmed"])
uk1 = int(uk["Confirmed"])
us1 = int(us["Confirmed"])

list1 = [aus,chi,fra,ger,ita,kor,spa,tur,uk1,us1]
list1.sort()
for i in list1:
    if i == aus:
        con.append("Australia")
    elif i == chi:
        con.append("China")
    elif i == fra:
        con.append("France")
    elif i == ger:
        con.append("Germany")
    elif i == ita:
        con.append("Italy")
    elif i == kor:
        con.append("South Korea")
    elif i == spa:
        con.append("Spain")
    elif i == tur:
        con.append("Turkey")
    elif i == uk1:
        con.append("UK")
    elif i == us1:
        con.append("US")

trace1 = go.Bar(x = con,
               y = list1,
               text = list1,
               textposition = "outside",
               name = "Confirmed",
            marker=dict(color = list1,line = dict(color = "rgb(0,0,0)", width = 1.5),colorbar=dict(
            title="Confirmed"
        ),colorscale="RdBu",))

data = [trace1]
layout = go.Layout(title = "Today Confirmed Comparison",template = "plotly_white")

fig = go.Figure(data = data, layout = layout)
fig.update_xaxes(title_text = "Country")
fig.update_yaxes(title_text = "Confirmed")
fig.show()
from sklearn.preprocessing import MinMaxScaler
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error

data = pd.read_csv("/kaggle/input/corona-virus-report/covid_19_clean_complete.csv")
data_lstm = data[["Date","Confirmed"]]
date_list1 = list(data_lstm["Date"].unique())
confirmed = []
for i in date_list1:
    x = data_lstm[data_lstm["Date"] == i]
    confirmed.append(sum(x["Confirmed"]))
data_lstm = pd.DataFrame(list(zip(date_list1,confirmed)),columns = ["Date","Confirmed"])
data_lstm.tail()

if len(data_lstm)%2 == 1:
    #data_lstm = data_lstm.iloc[1:,]
    data_lstm = data_lstm.drop([len(data_lstm)-1])
data_lstm.tail()
data_lstm = data_lstm.iloc[:,1].values
data_lstm = data_lstm.reshape(-1,1)
data_lstm = data_lstm.astype("float32")
#data_lstm.shape
df = pd.DataFrame(data_lstm)
df.head()
scaler = MinMaxScaler(feature_range = (0, 1))
data_lstm = scaler.fit_transform(data_lstm)
train_size = int(len(data_lstm)*0.50)
test_size = len(data_lstm) - train_size
train = data_lstm[0:train_size,:]
test = data_lstm[train_size:len(data_lstm),:]
print("train size: {}, test size: {}".format(len(train),len(test)))
time_step = 10 #50
datax = []
datay = []
for i in range(len(test)-time_step-1):
    a = train[i:(i+time_step),0]
    datax.append(a)
    datay.append(test[i + time_step, 0])
trainx = np.array(datax)
trainy = np.array(datay)
datax = []
datay = []
for i in range(len(test)-time_step-1):
    a = test[i:(i+time_step),0]
    datax.append(a)
    datay.append(test[i + time_step, 0])
testx = np.array(datax)
testy = np.array(datay)
trainx = np.reshape(trainx, (trainx.shape[0], 1 , trainx.shape[1]))
testx = np.reshape(testx, (testx.shape[0], 1 , testx.shape[1]))
from keras.layers import Dropout
model = Sequential()
model.add(LSTM(50,return_sequences = True, input_shape =(1,time_step)))
model.add(Dropout(0.1))
model.add(LSTM(50,return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(50,return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss = "mean_squared_error", optimizer="adam")
model.fit(trainx,trainy, epochs = 100) #, batch_size = 2)

model.summary()
trainPredict = model.predict(trainx)
testPredict = model.predict(testx)

trainPredict = scaler.inverse_transform(trainPredict)
trainy = scaler.inverse_transform([trainy])
testPredict = scaler.inverse_transform(testPredict)
testy = scaler.inverse_transform([testy])

trainScore = math.sqrt(mean_squared_error(trainy[0], trainPredict[:,0]))
print("train score: %.2f RMSE" % (trainScore))
testScore = math.sqrt(mean_squared_error(testy[0], testPredict[:,0]))
print("test score: %.2f RMSE" % (testScore))

lstm_loss = model.history.history["loss"]
plt.figure(figsize = (13,8))
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Losses-Epochs")
plt.grid(True, alpha = 0.5)
plt.plot(lstm_loss)
plt.show()
trainPredictPlot = np.empty_like(data_lstm)
trainPredictPlot[:,:] = np.nan
trainPredictPlot[time_step:len(trainPredict)+time_step, :] = trainPredict

testPredictPlot = np.empty_like(data_lstm)
testPredictPlot[:,:] = np.nan
testPredictPlot[len(trainPredict)+(time_step*2)+1:len(data_lstm)-1,:] = testPredict

plt.figure(figsize = (13,8))
plt.plot(scaler.inverse_transform(data_lstm),label = "Real Data")
plt.plot(trainPredictPlot,label = "Train Predicted")
plt.plot(testPredictPlot, label = "Test Predicted")
plt.legend()
plt.show()
