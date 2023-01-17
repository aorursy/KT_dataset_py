
import numpy as np 
import pandas as pd 


import matplotlib.pyplot as plt
import math
import csv
import os
from plotly import tools
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import seaborn as sb
import plotly.figure_factory as FF
import folium
from folium import plugins
from io import StringIO


# Importing data
schl_df = pd.read_csv('../input/data-science-for-good/2016 School Explorer.csv')
reg_df = pd.read_csv('../input/data-science-for-good/D5 SHSAT Registrations and Testers.csv')
offer_df =  pd.read_csv('../input/2017-2018-shsat-admissions-test-offers-by-schools/2017-2018 SHSAT Admissions Test Offers By Sending School.csv')
math_df =  pd.read_csv('../input/new-york-state-math-test-results/2013-2015-new-york-state-mathematics-exam.csv')
safe_df =  pd.read_csv('../input/ny-2010-2016-school-safety-report/2010-2016-school-safety-report.csv')


offer_df.rename(columns = {'School DBN':'Location Code'}, inplace = True)
del offer_df['School Name']
offer_df.set_index('Location Code', inplace=True)
schl_df.set_index('Location Code', inplace=True)

df = pd.concat([schl_df, offer_df], axis=1, join='outer')
df["School Income Estimate"] = df["School Income Estimate"].replace('[\$,]', '', regex=True).astype(float)
df["Percent of eight grade students who received offer"] = df["Percent of eight grade students who received offer"].replace('[\%,]', '', regex=True).astype(float)
df["Collaborative Teachers %"] = df["Collaborative Teachers %"].replace('[\%,]', '', regex=True).astype(float)
df["Supportive Environment %"] = df["Supportive Environment %"].replace('[\%,]', '', regex=True).astype(float)
df["Effective School Leadership %"] = df["Effective School Leadership %"].replace('[\%,]', '', regex=True).astype(float)
df["Strong Family-Community Ties %"] = df["Strong Family-Community Ties %"].replace('[\%,]', '', regex=True).astype(float)
df["Trust %"] = df["Trust %"].replace('[\%,]', '', regex=True).astype(float)
df["Percent Black / Hispanic"] = df["Percent Black / Hispanic"].replace('[\%,]', '', regex=True).astype(float)
df["Percent Asian"] = df["Percent Asian"].replace('[\%,]', '', regex=True).astype(float)
df["Percent White"] = df["Percent White"].replace('[\%,]', '', regex=True).astype(float)
df["Percentage of Black/Hispanic students"] = df["Percentage of Black/Hispanic students"].replace('[\%,]', '', regex=True).astype(float)
df["Student Attendance Rate"] = df["Student Attendance Rate"].replace('[\%,]', '', regex=True).astype(float)
df.shape





# Any results you write to the current directory are saved as output.
excluded = df[df['City'].isnull()]
excluded.shape
# 9 schools are not in the PASSNYC School Explorer


DF = df.dropna(subset=['City'])
DF.shape
DF.head(10)
schl = DF.dropna(subset=['Number of students who took test'])
gd_schl = schl.dropna(subset=['Number of students who received offer'])
ok_schl = schl[schl['Number of students who received offer'].isnull()]

tmp = DF[DF['Number of students who took test'].isnull()]
wk_schl = tmp.loc[tmp['Grade High'] == '08']
others = tmp.loc[tmp['Grade High'] != '08']

print("schls with at least 1 reg but 0 offer df \n",ok_schl.shape,"\n","schls with at least 1 offer df \n",gd_schl.shape,"\n","schls with 0 reg df \n",wk_schl.shape,"\n","schls with no 8th grade df \n",others.shape)

best = gd_schl.loc[gd_schl['Percent of eight grade students who received offer'] >= 50]


tmp = gd_schl.loc[gd_schl['Percent of eight grade students who received offer'] >= 20]
avgbest = tmp.loc[tmp['Percent of eight grade students who received offer'] < 50]
tmp = gd_schl.loc[gd_schl['Number of students who received offer'] <= 99]


tmp = gd_schl.loc[gd_schl['Percent of eight grade students who received offer'] >= 10]
lowbest = tmp.loc[tmp['Percent of eight grade students who received offer'] < 20]
shitbest = gd_schl.loc[gd_schl['Percent of eight grade students who received offer'] < 10]

labels = ['Does not offer 8th Grade','Schls with 0 takers','Schls with takers but 0 offers','Schls with offers']
values = [676,60,416,120]


# figure
fig = {
  "data": [
    {
      "values": values,
      "labels": labels,
      "domain": {"x": [0, .5]},
      "hoverinfo":"label+percent",
      "textinfo":'value',
      "hole": .3,
      "type": "pie"
    },],
  "layout": {
        "title":"Type of Schools in Dataset",
        "annotations": [
            { "font": { "size": 15},
              "showarrow": False,
              "text": "Number of Schools",
                "x": 0.14,
                "y": 1
            },
        ]
    }
}
iplot(fig)
IEgs = gd_schl.dropna(subset=['School Income Estimate'])
IEos = ok_schl.dropna(subset=['School Income Estimate'])
IEws = wk_schl.dropna(subset=['School Income Estimate'])
print("Income estimate dfs",IEgs.shape,IEos.shape,IEws.shape)
ENIgs = gd_schl.dropna(subset=['Economic Need Index'])
ENIos = ok_schl.dropna(subset=['Economic Need Index'])
ENIws = wk_schl.dropna(subset=['Economic Need Index'])
print("ENI dfs",ENIgs.shape,ENIos.shape,ENIws.shape)

trace0 = go.Box(x=IEgs["School Income Estimate"],name="Schls with Offers",boxmean=True)
trace1 = go.Box(x=IEos["School Income Estimate"],name="Schls with 0 Offers",boxmean=True)
trace2 = go.Box(x=IEws["School Income Estimate"],name="Schls with 0 Takers",boxmean=True)
data = [trace0, trace1, trace2]
layout = go.Layout(
    title = "Box Plot of School Income",
    margin = go.Margin(l=115)
)
fig = go.Figure(data=data,layout=layout)
iplot(fig)

trace0 = go.Box(x=gd_schl["Economic Need Index"],name="Schls with Offers",boxmean=True)
trace1 = go.Box(x=ok_schl["Economic Need Index"],name="Schls with 0 Offers",boxmean=True)
trace2 = go.Box(x=wk_schl["Economic Need Index"],name="Schls with 0 Takers",boxmean=True)
data = [trace0, trace1, trace2]
layout = go.Layout(
    title = "Box Plot of Economic Need Index of schools",
    margin = go.Margin(l=115)
)
fig = go.Figure(data=data,layout=layout)
iplot(fig)
sb.lmplot(x='Economic Need Index', y='Number of students who received offer', data=DF,
           fit_reg=False)

sb.lmplot(x='Economic Need Index', y='Percent of eight grade students who received offer', data=DF,
           fit_reg=False)
Eschl = DF.loc[DF['Grade High'] == '08']
sb.lmplot(x='Economic Need Index', y='Number of students who took test', data=Eschl,
           fit_reg=False)
fig = plt.figure(figsize=(15,4))
ax=sb.kdeplot(gd_schl["Percent Black / Hispanic"] , color='r',shade=True, label='Schls with Offers')
ax=sb.kdeplot(ok_schl["Percent Black / Hispanic"] , color='b',shade=True, label='Schls with 0 Offers')
ax=sb.kdeplot(wk_schl["Percent Black / Hispanic"] , color='g',shade=True, label='Schls with 0 Takers')
plt.title('Race Distribution by Schools (Black/Hispanic)')
plt.xlabel('Percent Black / Hispanic')
plt.ylabel('Frequency')
fig = plt.figure(figsize=(15,4))
ax=sb.kdeplot(gd_schl["Percent Asian"] , color='r',shade=True, label='Schls with Offers')
ax=sb.kdeplot(ok_schl["Percent Asian"] , color='b',shade=True, label='Schls with 0 Offers')
ax=sb.kdeplot(wk_schl["Percent Asian"] , color='g',shade=True, label='Schls with 0 Takers')
plt.title('Race Distribution by Schools (Asian)')
plt.xlabel('Percent Asian')
plt.ylabel('Frequency')
fig = plt.figure(figsize=(15,4))
ax=sb.kdeplot(gd_schl["Percent White"] , color='r',shade=True, label='Good Schls')
ax=sb.kdeplot(ok_schl["Percent White"] , color='b',shade=True, label='Shit Schls')
ax=sb.kdeplot(wk_schl["Percent White"] , color='g',shade=True, label='Very Shit Schls')
plt.title('Race Distribution by Schools (White)')
plt.xlabel('Percent White')
plt.ylabel('Frequency')
trace0 = go.Box(x=gd_schl["Percent Asian"],name="Asian",boxmean=True)
trace1 = go.Box(x=gd_schl["Percent White"],name="White",boxmean=True)
trace2 = go.Box(x=gd_schl["Percent Black / Hispanic"],name="Black/Hispanic",boxmean=True)
data = [trace0, trace1, trace2]
layout = go.Layout(
    title = "Box Plot of race in good schools",
    margin = go.Margin(l=115)
)
fig = go.Figure(data=data,layout=layout)
iplot(fig)
trace0 = go.Box(x=best["Percent Black / Hispanic"],name="more than 50%",boxmean=True)
trace1 = go.Box(x=avgbest["Percent Black / Hispanic"],name="20 ~ 49%",boxmean=True)
trace2 = go.Box(x=lowbest["Percent Black / Hispanic"],name="10 ~ 19%",boxmean=True)
trace3 = go.Box(x=shitbest["Percent Black / Hispanic"],name="less than 10%",boxmean=True)
data = [trace0, trace1, trace2, trace3]
layout = go.Layout(
    title = "Box Plot of Percent Black / Hispanic of Schools with Offers",
    margin = go.Margin(l=115)
)
fig = go.Figure(data=data,layout=layout)
iplot(fig)
trace0 = go.Box(x=best["Percent White"],name="more than 50%",boxmean=True)
trace1 = go.Box(x=avgbest["Percent White"],name="20 ~ 49%",boxmean=True)
trace2 = go.Box(x=lowbest["Percent White"],name="10 ~ 19%",boxmean=True)
trace3 = go.Box(x=shitbest["Percent White"],name="less than 10%",boxmean=True)
data = [trace0, trace1, trace2, trace3]
layout = go.Layout(
    title = "Box Plot of Percent White of Schools with Offers",
    margin = go.Margin(l=115)
)
fig = go.Figure(data=data,layout=layout)
iplot(fig)
trace0 = go.Box(x=best["Percent Asian"],name="more than 50%",boxmean=True)
trace1 = go.Box(x=avgbest["Percent Asian"],name="20 ~ 49%",boxmean=True)
trace2 = go.Box(x=lowbest["Percent Asian"],name="10 ~ 19%",boxmean=True)
trace3 = go.Box(x=shitbest["Percent Asian"],name="less than 10%",boxmean=True)
data = [trace0, trace1, trace2, trace3]
layout = go.Layout(
    title = "Box Plot of Percent Asian of Schools with Offers",
    margin = go.Margin(l=115)
)
fig = go.Figure(data=data,layout=layout)
iplot(fig)
gdMschl = gd_schl.loc[gd_schl["Borough"]=='Manhattan'] 
gdBkschl = gd_schl.loc[gd_schl["Borough"]=='Brooklyn']
gdBxschl = gd_schl.loc[gd_schl["Borough"]=='Bronx']
gdQschl = gd_schl.loc[gd_schl["Borough"]=='Queens']
gdSschl = gd_schl.loc[gd_schl["Borough"]=='Staten Island']
gdLschl = gd_schl.loc[gd_schl["Borough"]=='LONG ISLAND CITY']

Mschl = schl.loc[schl['Borough'] == 'Manhattan']
Bschl = schl.loc[schl['Borough'] == 'Brooklyn']
BXschl = schl.loc[schl['Borough'] == 'Bronx']
Qschl = schl.loc[schl['Borough'] == 'Queens']
Sschl = schl.loc[schl['Borough'] == 'Staten Island']
Lschl = schl.loc[schl['Borough'] == 'LONG ISLAND CITY']

Mwk_schl = wk_schl.loc[wk_schl['Borough'] == 'Manhattan']
Bwk_schl = wk_schl.loc[wk_schl['Borough'] == 'Brooklyn']
BXwk_schl = wk_schl.loc[wk_schl['Borough'] == 'Bronx']
Qwk_schl = wk_schl.loc[wk_schl['Borough'] == 'Queens']
Swk_schl = wk_schl.loc[wk_schl['Borough'] == 'Staten Island']
Lwk_schl = wk_schl.loc[wk_schl['Borough'] == 'LONG ISLAND CITY']



lat_0=40.7127
lon_0=-74.0059

maps = folium.Map([lat_0,lon_0], zoom_start=10.0,tiles='stamentoner')
for lat, long, cnt in zip(gdMschl['Latitude'], gdMschl['Longitude'], gdMschl['Percent of eight grade students who received offer']):
    folium.Circle([lat, long],radius=cnt*20, color="gold", fill=True, fill_opacity=0.5).add_to(maps)
    
for lat, long, cnt in zip(gdBkschl['Latitude'], gdBkschl['Longitude'], gdBkschl['Percent of eight grade students who received offer']):
    folium.Circle([lat, long],radius=cnt*20, color="green", fill=True, fill_opacity=0.5).add_to(maps)
    
for lat, long, cnt in zip(gdBxschl['Latitude'], gdBxschl['Longitude'], gdBxschl['Percent of eight grade students who received offer']):
    folium.Circle([lat, long],radius=cnt*20, color="red", fill=True, fill_opacity=0.5).add_to(maps)
    
for lat, long, cnt in zip(gdQschl['Latitude'], gdQschl['Longitude'], gdQschl['Percent of eight grade students who received offer']):
    folium.Circle([lat, long],radius=cnt*20, color="purple", fill=True, fill_opacity=0.5).add_to(maps)
    
for lat, long, cnt in zip(gdSschl['Latitude'], gdSschl['Longitude'], gdSschl['Percent of eight grade students who received offer']):
    folium.Circle([lat, long],radius=cnt*20, color="blue", fill=True, fill_opacity=0.5).add_to(maps)
    
for lat, long, cnt in zip(gdLschl['Latitude'], gdLschl['Longitude'], gdLschl['Percent of eight grade students who received offer']):
    folium.Circle([lat, long],radius=cnt*20, color="pink", fill=True, fill_opacity=0.5).add_to(maps)
    
maps
lat_0=40.7127
lon_0=-74.0059

maps = folium.Map([lat_0,lon_0], zoom_start=10.0,tiles='stamentoner')
for lat, long, cnt in zip(gdMschl['Latitude'], gdMschl['Longitude'], gdMschl['Number of students who received offer']):
    folium.Circle([lat, long],radius=cnt*20, color="gold", fill=True, fill_opacity=0.5).add_to(maps)
    
for lat, long, cnt in zip(gdBkschl['Latitude'], gdBkschl['Longitude'], gdBkschl['Number of students who received offer']):
    folium.Circle([lat, long],radius=cnt*20, color="green", fill=True, fill_opacity=0.5).add_to(maps)
    
for lat, long, cnt in zip(gdBxschl['Latitude'], gdBxschl['Longitude'], gdBxschl['Number of students who received offer']):
    folium.Circle([lat, long],radius=cnt*20, color="red", fill=True, fill_opacity=0.5).add_to(maps)
    
for lat, long, cnt in zip(gdQschl['Latitude'], gdQschl['Longitude'], gdQschl['Number of students who received offer']):
    folium.Circle([lat, long],radius=cnt*20, color="purple", fill=True, fill_opacity=0.5).add_to(maps)
    
for lat, long, cnt in zip(gdSschl['Latitude'], gdSschl['Longitude'], gdSschl['Number of students who received offer']):
    folium.Circle([lat, long],radius=cnt*20, color="blue", fill=True, fill_opacity=0.5).add_to(maps)
    
for lat, long, cnt in zip(gdLschl['Latitude'], gdLschl['Longitude'], gdLschl['Number of students who received offer']):
    folium.Circle([lat, long],radius=cnt*20, color="pink", fill=True, fill_opacity=0.5).add_to(maps)
    
maps
lat_0=40.7127
lon_0=-74.0059

maps = folium.Map([lat_0,lon_0], zoom_start=10.0,tiles='stamentoner')
for lat, long, cnt in zip(Mschl['Latitude'], Mschl['Longitude'], Mschl['Number of students who took test']):
    folium.Circle([lat, long],radius=cnt*5, color="gold", fill=True, fill_opacity=0.5).add_to(maps)

for lat, long, cnt in zip(Bschl['Latitude'], Bschl['Longitude'], Bschl['Number of students who took test']):
    folium.Circle([lat, long],radius=cnt*5, color="green", fill=True, fill_opacity=0.5).add_to(maps)

for lat, long, cnt in zip(BXschl['Latitude'], BXschl['Longitude'], BXschl['Number of students who took test']):
    folium.Circle([lat, long],radius=cnt*5, color="red", fill=True, fill_opacity=0.5).add_to(maps)

for lat, long, cnt in zip(Qschl['Latitude'], Qschl['Longitude'], Qschl['Number of students who took test']):
    folium.Circle([lat, long],radius=cnt*5, color="purple", fill=True, fill_opacity=0.5).add_to(maps)

for lat, long, cnt in zip(Sschl['Latitude'], Sschl['Longitude'], Sschl['Number of students who took test']):
    folium.Circle([lat, long],radius=cnt*5, color="blue", fill=True, fill_opacity=0.5).add_to(maps)

for lat, long, cnt in zip(Lschl['Latitude'], Lschl['Longitude'], Lschl['Number of students who took test']):
    folium.Circle([lat, long],radius=cnt*5, color="pink", fill=True, fill_opacity=0.5).add_to(maps)
maps
print("DF of schls in Long Island City",gdLschl.shape,Lschl.shape,Lwk_schl.shape)
lat_0=40.7127
lon_0=-74.0059

maps = folium.Map([lat_0,lon_0], zoom_start=10.5,tiles='stamentoner')
for lat, long in zip(Mwk_schl['Latitude'], Mwk_schl['Longitude'], ):
    folium.Circle([lat, long],radius=300, color="gold", fill=True, fill_opacity=0.9).add_to(maps)
for lat, long in zip(Bwk_schl['Latitude'], Bwk_schl['Longitude'], ):
    folium.Circle([lat, long],radius=300, color="green", fill=True, fill_opacity=0.9).add_to(maps)
for lat, long in zip(BXwk_schl['Latitude'], BXwk_schl['Longitude'], ):
    folium.Circle([lat, long],radius=300, color="red", fill=True, fill_opacity=0.9).add_to(maps)
for lat, long in zip(Qwk_schl['Latitude'], Qwk_schl['Longitude'], ):
    folium.Circle([lat, long],radius=300, color="purple", fill=True, fill_opacity=0.9).add_to(maps)
for lat, long in zip(Swk_schl['Latitude'], Swk_schl['Longitude'], ):
    folium.Circle([lat, long],radius=300, color="blue", fill=True, fill_opacity=0.9).add_to(maps)
for lat, long in zip(Lwk_schl['Latitude'], Lwk_schl['Longitude'], ):
    folium.Circle([lat, long],radius=300, color="pink", fill=True, fill_opacity=0.9).add_to(maps)
    
maps
tmp = safe_df.dropna(subset=['AvgOfVio N'])
tmp = tmp.dropna(subset=['Latitude'])
Msafe = tmp.loc[safe_df['Borough Name'] == 'MANHATTAN']
Bsafe = tmp.loc[safe_df['Borough Name'] == 'BROOKLYN ']
BXsafe = tmp.loc[safe_df['Borough'] == 'X']
Qsafe = tmp.loc[safe_df['Borough'] == 'Q']
Ssafe = tmp.loc[safe_df['Borough Name'] == 'STATEN ISLAND']
Lsafe = tmp.loc[safe_df['Borough Name'] == 'LONG ISLAND CITY']
print("df of Staten Island/Long Island City",Ssafe.shape,Lsafe.shape)

trace0 = go.Box(x=Msafe["AvgOfVio N"],name="Manhattan",boxmean=True)
trace1 = go.Box(x=Bsafe["AvgOfVio N"],name="Brooklyn",boxmean=True)
trace2 = go.Box(x=BXsafe["AvgOfVio N"],name="Bronx",boxmean=True)
trace3 = go.Box(x=Qsafe["AvgOfVio N"],name="Queens",boxmean=True)
data = [trace0, trace1, trace2, trace3]
layout = go.Layout(
    title = "Box Plot of Average Violent Crime",
    margin = go.Margin(l=115)
)
fig = go.Figure(data=data,layout=layout)
iplot(fig)
trace0 = go.Box(x=Msafe["AvgOfNoCrim N"],name="Manhattan",boxmean=True)
trace1 = go.Box(x=Bsafe["AvgOfNoCrim N"],name="Brooklyn",boxmean=True)
trace2 = go.Box(x=BXsafe["AvgOfNoCrim N"],name="Bronx",boxmean=True)
trace3 = go.Box(x=Qsafe["AvgOfNoCrim N"],name="Queens",boxmean=True)
data = [trace0, trace1, trace2, trace3]
layout = go.Layout(
    title = "Box Plot of Average Non-criminal Crime",
    margin = go.Margin(l=115)
)
fig = go.Figure(data=data,layout=layout)
iplot(fig)
trace0 = go.Box(x=Msafe["AvgOfMajor N"],name="Manhattan",boxmean=True)
trace1 = go.Box(x=Bsafe["AvgOfMajor N"],name="Brooklyn",boxmean=True)
trace2 = go.Box(x=BXsafe["AvgOfMajor N"],name="Bronx",boxmean=True)
trace3 = go.Box(x=Qsafe["AvgOfMajor N"],name="Queens",boxmean=True)
data = [trace0, trace1, trace2, trace3]
layout = go.Layout(
    title = "Box Plot of Major Crime",
    margin = go.Margin(l=115)
)
fig = go.Figure(data=data,layout=layout)
iplot(fig)
trace0 = go.Box(x=Msafe["AvgOfOth N"],name="Manhattan",boxmean=True)
trace1 = go.Box(x=Bsafe["AvgOfOth N"],name="Brooklyn",boxmean=True)
trace2 = go.Box(x=BXsafe["AvgOfOth N"],name="Bronx",boxmean=True)
trace3 = go.Box(x=Qsafe["AvgOfOth N"],name="Queens",boxmean=True)
data = [trace0, trace1, trace2, trace3]
layout = go.Layout(
    title = "Box Plot of Other Crime",
    margin = go.Margin(l=115)
)
fig = go.Figure(data=data,layout=layout)
iplot(fig)
trace0 = go.Box(x=gdQschl["Economic Need Index"],name="Queens Schls",boxmean=True)
trace1 = go.Box(x=gdBxschl["Economic Need Index"],name="Bronx Schls",boxmean=True)

data = [trace0, trace1]
layout = go.Layout(
    title = "Box Plot of Economic Need Index",
    margin = go.Margin(l=115)
)
fig = go.Figure(data=data,layout=layout)
iplot(fig)
trace0 = go.Box(x=gd_schl["Average ELA Proficiency"],name="Schls with Offers",boxmean=True)
trace1 = go.Box(x=ok_schl["Average ELA Proficiency"],name="Schls with 0 Offers ",boxmean=True)
trace2 = go.Box(x=wk_schl["Average ELA Proficiency"],name="Schls with 0 Takers",boxmean=True)
data = [trace0, trace1, trace2]
layout = go.Layout(
    title = "Box Plot of Average ELA Proficiency of  schools",
    margin = go.Margin(l=115)
)
fig = go.Figure(data=data,layout=layout)
iplot(fig)
trace0 = go.Box(x=gd_schl["Average Math Proficiency"],name="Schls with Offers",boxmean=True)
trace1 = go.Box(x=ok_schl["Average Math Proficiency"],name="Schls with 0 Offers",boxmean=True)
trace2 = go.Box(x=wk_schl["Average Math Proficiency"],name="Schls with 0 Takers",boxmean=True)
data = [trace0, trace1, trace2]
layout = go.Layout(
    title = "Box Plot of Average Math Proficiency of  schools",
    margin = go.Margin(l=115)
)
fig = go.Figure(data=data,layout=layout)
iplot(fig)
math_2013 = math_df[math_df['Year'] == 2013]
Asian_math_2013 = math_2013[math_2013['Category'] == 'Asian']
Black_math_2013 = math_2013[math_2013['Category'] == 'Black']
White_math_2013 = math_2013[math_2013['Category'] == 'White']
Hispanic_math_2013 = math_2013[math_2013['Category'] == 'Hispanic']

math_2014 = math_df[math_df['Year'] == 2014]
Asian_math_2014 = math_2014[math_2014['Category'] == 'Asian']
Black_math_2014 = math_2014[math_2014['Category'] == 'Black']
White_math_2014 = math_2014[math_2014['Category'] == 'White']
Hispanic_math_2014 = math_2014[math_2014['Category'] == 'Hispanic']

math_2015 = math_df[math_df['Year'] == 2015]
Asian_math_2015 = math_2015[math_2015['Category'] == 'Asian']
Black_math_2015 = math_2015[math_2015['Category'] == 'Black']
White_math_2015 = math_2015[math_2015['Category'] == 'White']
Hispanic_math_2015 = math_2015[math_2015['Category'] == 'Hispanic']

trace = go.Scatter(
    x = Asian_math_2013['Grade'],
    y = Asian_math_2013['Mean Scale Score'],
    name = 'Asian Students'
)

trace2 = go.Scatter(
    x = Hispanic_math_2013['Grade'],
    y = Hispanic_math_2013['Mean Scale Score'],
    name = 'Hispanic Students'
)

trace3 = go.Scatter(
    x = Black_math_2013['Grade'],
    y = Black_math_2013['Mean Scale Score'],
    name = 'Black Students'
)

trace4 = go.Scatter(
    x = White_math_2013['Grade'],
    y = White_math_2013['Mean Scale Score'],
    name = 'White Students'
)

layout= go.Layout(
    title= 'Student Math Performance By Grade (Race) 2013',
    xaxis= dict(
        title= 'Grade Level'
    ),
    yaxis=dict(
        title='Mean Scale Score'
    )
)

data = [trace, trace2, trace3, trace4]

fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='scatter_hover_labels')
trace = go.Scatter(
    x = Asian_math_2013['Grade'],
    y = Asian_math_2013['Number Tested'],
    name = 'Asian Students'
)

trace2 = go.Scatter(
    x = Hispanic_math_2013['Grade'],
    y = Hispanic_math_2013['Number Tested'],
    name = 'Hispanic Students'
)

trace3 = go.Scatter(
    x = Black_math_2013['Grade'],
    y = Black_math_2013['Number Tested'],
    name = 'Black Students'
)

trace4 = go.Scatter(
    x = White_math_2013['Grade'],
    y = White_math_2013['Number Tested'],
    name = 'White Students'
)

layout= go.Layout(
    title= 'Math Exam Cohort Size By Grade (Race) 2013',
    xaxis= dict(
        title= 'Grade Level'
    ),
    yaxis=dict(
        title='Number Tested`', range= [0,40000]
    )
)

data = [trace, trace2, trace3, trace4]

fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='scatter_hover_labels')
trace = go.Scatter(
    x = Asian_math_2014['Grade'],
    y = Asian_math_2014['Mean Scale Score'],
    name = 'Asian Students'
)

trace2 = go.Scatter(
    x = Hispanic_math_2014['Grade'],
    y = Hispanic_math_2014['Mean Scale Score'],
    name = 'Hispanic Students'
)

trace3 = go.Scatter(
    x = Black_math_2014['Grade'],
    y = Black_math_2014['Mean Scale Score'],
    name = 'Black Students'
)

trace4 = go.Scatter(
    x = White_math_2014['Grade'],
    y = White_math_2014['Mean Scale Score'],
    name = 'White Students'
)

layout= go.Layout(
    title= 'Student Math Performance By Grade (Race) 2014',
    xaxis= dict(
        title= 'Grade Level'
    ),
    yaxis=dict(
        title='Mean Scale Score'
    )
)

data = [trace, trace2, trace3, trace4]

fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='scatter_hover_labels')
trace = go.Scatter(
    x = Asian_math_2014['Grade'],
    y = Asian_math_2014['Number Tested'],
    name = 'Asian Students'
)

trace2 = go.Scatter(
    x = Hispanic_math_2014['Grade'],
    y = Hispanic_math_2014['Number Tested'],
    name = 'Hispanic Students'
)

trace3 = go.Scatter(
    x = Black_math_2014['Grade'],
    y = Black_math_2014['Number Tested'],
    name = 'Black Students'
)

trace4 = go.Scatter(
    x = White_math_2014['Grade'],
    y = White_math_2014['Number Tested'],
    name = 'White Students'
)

layout= go.Layout(
    title= 'Math Exam Cohort Size By Grade (Race) 2014',
    xaxis= dict(
        title= 'Grade Level'
    ),
    yaxis=dict(
        title='Number Tested`', range= [0,40000]
    )
)

data = [trace, trace2, trace3, trace4]

fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='scatter_hover_labels')
trace = go.Scatter(
    x = Asian_math_2015['Grade'],
    y = Asian_math_2015['Mean Scale Score'],
    name = 'Asian Students'
)

trace2 = go.Scatter(
    x = Hispanic_math_2015['Grade'],
    y = Hispanic_math_2015['Mean Scale Score'],
    name = 'Hispanic Students'
)

trace3 = go.Scatter(
    x = Black_math_2015['Grade'],
    y = Black_math_2015['Mean Scale Score'],
    name = 'Black Students'
)

trace4 = go.Scatter(
    x = White_math_2015['Grade'],
    y = White_math_2015['Mean Scale Score'],
    name = 'White Students'
)

layout= go.Layout(
    title= 'Student Math Performance By Grade (Race) 2015',
    xaxis= dict(
        title= 'Grade Level'
    ),
    yaxis=dict(
        title='Mean Scale Score'
    )
)

data = [trace, trace2, trace3, trace4]

fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='scatter_hover_labels')
trace = go.Scatter(
    x = Asian_math_2015['Grade'],
    y = Asian_math_2015['Number Tested'],
    name = 'Asian Students'
)

trace2 = go.Scatter(
    x = Hispanic_math_2015['Grade'],
    y = Hispanic_math_2015['Number Tested'],
    name = 'Hispanic Students'
)

trace3 = go.Scatter(
    x = Black_math_2015['Grade'],
    y = Black_math_2015['Number Tested'],
    name = 'Black Students'
)

trace4 = go.Scatter(
    x = White_math_2015['Grade'],
    y = White_math_2015['Number Tested'],
    name = 'White Students'
)

layout= go.Layout(
    title= 'Math Exam Cohort Size By Grade (Race) 2015',
    xaxis= dict(
        title= 'Grade Level'
    ),
    yaxis=dict(
        title='Number Tested`', range= [0,40000]
    )
)

data = [trace, trace2, trace3, trace4]

fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='scatter_hover_labels')
fig = plt.figure(figsize=(15,4))

ax=sb.kdeplot(gd_schl["Student Attendance Rate"] , color='r',shade=True, label='Schls with Offers')
ax=sb.kdeplot(ok_schl["Student Attendance Rate"] , color='b',shade=True, label='Schls with 0 Offers')
ax=sb.kdeplot(wk_schl["Student Attendance Rate"] , color='g',shade=True, label='Schls with 0 Takers')
plt.title('Student Attendance Rate')
plt.xlabel('Student Attendance Rate')
plt.ylabel('Frequency')
trace0 = go.Box(x=gd_schl["Percent of Students Chronically Absent"],name="Schls with Offers",boxmean=True)
trace1 = go.Box(x=ok_schl["Percent of Students Chronically Absent"],name="Schls with 0 Offers",boxmean=True)
trace2 = go.Box(x=wk_schl["Percent of Students Chronically Absent"],name="Schls with 0 Takers",boxmean=True)
data = [trace0, trace1, trace2]
layout = go.Layout(
    title = "Box Plot of Percent of Students Chronically Absent",
    margin = go.Margin(l=115)
)
fig = go.Figure(data=data,layout=layout)
iplot(fig)
trace0 = go.Box(x=gd_schl["Collaborative Teachers %"],name="Schls with Offers",boxmean=True)
trace1 = go.Box(x=ok_schl["Collaborative Teachers %"],name="Schls with 0 Offers",boxmean=True)
trace2 = go.Box(x=wk_schl["Collaborative Teachers %"],name="Schls with 0 Takers",boxmean=True)
data = [trace0, trace1, trace2]
layout = go.Layout(
    title = "Box Plot of Collaborative Teachers  of  schools",
    margin = go.Margin(l=115)
)
fig = go.Figure(data=data,layout=layout)
iplot(fig)
fig = plt.figure(figsize=(15,4))
ax=sb.kdeplot(gd_schl["Collaborative Teachers %"] , color='b',shade=True, label='Schls with Offers')
ax=sb.kdeplot(ok_schl["Collaborative Teachers %"] , color='orange',shade=True, label='Schls with 0 Offers')
ax=sb.kdeplot(wk_schl["Collaborative Teachers %"] , color='g',shade=True, label='Schls with 0 Takers')
plt.title('Collaborative Teaching Distribution by Schools')
plt.xlabel('Collaborative Teachers %')
plt.ylabel('Frequency')
trace0 = go.Box(x=best["Collaborative Teachers %"],name="more than 50%",boxmean=True)
trace1 = go.Box(x=avgbest["Collaborative Teachers %"],name="20 ~ 49%",boxmean=True)
trace2 = go.Box(x=lowbest["Collaborative Teachers %"],name="10 ~ 19%",boxmean=True)
trace3 = go.Box(x=shitbest["Collaborative Teachers %"],name="less than 10%",boxmean=True)
data = [trace0, trace1, trace2, trace3]
layout = go.Layout(
    title = "Box Plot of Collaborative Teachers % of Schls with Offers",
    margin = go.Margin(l=115)
)
fig = go.Figure(data=data,layout=layout)
iplot(fig)
trace0 = go.Box(x=gd_schl["Supportive Environment %"],name="Schls with Offers",boxmean=True)
trace1 = go.Box(x=ok_schl["Supportive Environment %"],name="Schls with 0 Offers",boxmean=True)
trace2 = go.Box(x=wk_schl["Supportive Environment %"],name="Schls with 0 Takers",boxmean=True)
data = [trace0, trace1, trace2]
layout = go.Layout(
    title = "Box Plot of Supportive Environment  of  schools",
    margin = go.Margin(l=115)
)
fig = go.Figure(data=data,layout=layout)
iplot(fig)
fig = plt.figure(figsize=(15,4))
ax=sb.kdeplot(gd_schl["Supportive Environment %"] , color='b',shade=True, label='Schls with Offers')
ax=sb.kdeplot(ok_schl["Supportive Environment %"] , color='orange',shade=True, label='Schls with 0 Offers')
ax=sb.kdeplot(wk_schl["Supportive Environment %"] , color='g',shade=True, label='Schls with 0 Takers')
plt.title('Supportive Environment Distribution by Schools')
plt.xlabel('Collaborative Teachers %')
plt.ylabel('Frequency')
trace0 = go.Box(x=best["Supportive Environment %"],name="more than 50%",boxmean=True)
trace1 = go.Box(x=avgbest["Supportive Environment %"],name="20 ~ 49%",boxmean=True)
trace2 = go.Box(x=lowbest["Supportive Environment %"],name="10 ~ 19%",boxmean=True)
trace3 = go.Box(x=shitbest["Supportive Environment %"],name="less than 10%",boxmean=True)
data = [trace0, trace1, trace2, trace3]
layout = go.Layout(
    title = "Box Plot of Supportive Environment % of Schls with Offers",
    margin = go.Margin(l=115)
)
fig = go.Figure(data=data,layout=layout)
iplot(fig)
trace0 = go.Box(x=gd_schl["Effective School Leadership %"],name="Schls with Offers",boxmean=True)
trace1 = go.Box(x=ok_schl["Effective School Leadership %"],name="Schls with 0 Offers",boxmean=True)
trace2 = go.Box(x=wk_schl["Effective School Leadership %"],name="Schls with 0 Takers",boxmean=True)
data = [trace0, trace1, trace2]
layout = go.Layout(
    title = "Box Plot of Effective School Leadership of schools",
    margin = go.Margin(l=115)
)
fig = go.Figure(data=data,layout=layout)
iplot(fig)
fig = plt.figure(figsize=(15,4))
ax=sb.kdeplot(gd_schl["Effective School Leadership %"] , color='b',shade=True, label='Schls with Offers')
ax=sb.kdeplot(ok_schl["Effective School Leadership %"] , color='orange',shade=True, label='Schls with 0 Offers')
ax=sb.kdeplot(wk_schl["Effective School Leadership %"] , color='g',shade=True, label='Schls with 0 Takers')
plt.title('Effective School Leadership  Distribution by Schools')
plt.xlabel('Effective School Leadership %')
plt.ylabel('Frequency')
trace0 = go.Box(x=best["Effective School Leadership %"],name="more than 50%",boxmean=True)
trace1 = go.Box(x=avgbest["Effective School Leadership %"],name="20 ~ 49%",boxmean=True)
trace2 = go.Box(x=lowbest["Effective School Leadership %"],name="10 ~ 19%",boxmean=True)
trace3 = go.Box(x=shitbest["Effective School Leadership %"],name="less than 10%",boxmean=True)
data = [trace0, trace1, trace2, trace3]
layout = go.Layout(
    title = "Box Plot of Effective School Leadership of Schools with Offers",
    margin = go.Margin(l=115)
)
fig = go.Figure(data=data,layout=layout)
iplot(fig)
trace0 = go.Box(x=gd_schl["Strong Family-Community Ties %"],name="Schls with Offers",boxmean=True)
trace1 = go.Box(x=ok_schl["Strong Family-Community Ties %"],name="Schls with 0 Offers",boxmean=True)
trace2 = go.Box(x=wk_schl["Strong Family-Community Ties %"],name="Schls with 0 Takers",boxmean=True)
data = [trace0, trace1, trace2]
layout = go.Layout(
    title = "Box Plot of Strong Family-Community Ties",
    margin = go.Margin(l=115)
)
fig = go.Figure(data=data,layout=layout)
iplot(fig)
fig = plt.figure(figsize=(15,4))
ax=sb.kdeplot(gd_schl["Strong Family-Community Ties %"] , color='b',shade=True, label='Schls with Offers')
ax=sb.kdeplot(ok_schl["Strong Family-Community Ties %"] , color='orange',shade=True, label='Schls with 0 Offers')
ax=sb.kdeplot(wk_schl["Strong Family-Community Ties %"] , color='g',shade=True, label='Schls with 0 Takers')
plt.title('Strong Family-Community Ties  Distribution by Schools')
plt.xlabel('Strong Family-Community Ties %')
plt.ylabel('Frequency')
trace0 = go.Box(x=best["Strong Family-Community Ties %"],name="more than 50%",boxmean=True)
trace1 = go.Box(x=avgbest["Strong Family-Community Ties %"],name="20 ~ 49%",boxmean=True)
trace2 = go.Box(x=lowbest["Strong Family-Community Ties %"],name="10 ~ 19%",boxmean=True)
trace3 = go.Box(x=shitbest["Strong Family-Community Ties %"],name="less than 10%",boxmean=True)
data = [trace0, trace1, trace2, trace3]
layout = go.Layout(
    title = "Box Plot of Strong Family-Community Ties in Schls with Offers",
    margin = go.Margin(l=115)
)
fig = go.Figure(data=data,layout=layout)
iplot(fig)
trace0 = go.Box(x=gd_schl["Trust %"],name="Schls with Offers",boxmean=True)
trace1 = go.Box(x=ok_schl["Trust %"],name="Schls with 0 Offers",boxmean=True)
trace2 = go.Box(x=wk_schl["Trust %"],name="Schls with 0 Takers",boxmean=True)
data = [trace0, trace1, trace2]
layout = go.Layout(
    title = "Box Plot of Trust %",
    margin = go.Margin(l=115)
)
fig = go.Figure(data=data,layout=layout)
iplot(fig)
fig = plt.figure(figsize=(15,4))
ax=sb.kdeplot(gd_schl["Trust %"] , color='b',shade=True, label='Schls with Offers')
ax=sb.kdeplot(ok_schl["Trust %"] , color='orange',shade=True, label='Schls with 0 Offers')
ax=sb.kdeplot(wk_schl["Trust %"] , color='g',shade=True, label='Schls with 0 Takers')
plt.title('Trust %  Distribution by Schools')
plt.xlabel('Trust %')
plt.ylabel('Frequency')
trace0 = go.Box(x=best["Trust %"],name="more than 50%",boxmean=True)
trace1 = go.Box(x=avgbest["Trust %"],name="20 ~ 49%",boxmean=True)
trace2 = go.Box(x=lowbest["Trust %"],name="10 ~ 19%",boxmean=True)
trace3 = go.Box(x=shitbest["Trust %"],name="less than 10%",boxmean=True)
data = [trace0, trace1, trace2, trace3]
layout = go.Layout(
    title = "Box Plot of Trust of Schls with Offers",
    margin = go.Margin(l=115)
)
fig = go.Figure(data=data,layout=layout)
iplot(fig)
rec1 = gd_schl.loc[gd_schl['Percent of eight grade students who received offer'] >= 20]
print(rec1.shape)

rec1.head(27)
trace = go.Scatter(
    x = Asian_math_2015['Grade'],
    y = Asian_math_2015['Mean Scale Score'],
    name = 'Asian Students'
)

trace2 = go.Scatter(
    x = Hispanic_math_2015['Grade'],
    y = Hispanic_math_2015['Mean Scale Score'],
    name = 'Hispanic Students'
)

trace3 = go.Scatter(
    x = Black_math_2015['Grade'],
    y = Black_math_2015['Mean Scale Score'],
    name = 'Black Students'
)

trace4 = go.Scatter(
    x = White_math_2015['Grade'],
    y = White_math_2015['Mean Scale Score'],
    name = 'White Students'
)

layout= go.Layout(
    title= 'Student Math Performance By Grade (Race) 2015',
    xaxis= dict(
        title= 'Grade Level'
    ),
    yaxis=dict(
        title='Mean Scale Score'
    )
)

data = [trace, trace2, trace3, trace4]

fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='scatter_hover_labels')