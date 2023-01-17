# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from datetime import date
import seaborn as sns
import random 
import warnings
import operator
warnings.filterwarnings("ignore")
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as ff
df_school = pd.read_csv("../input/2016 School Explorer.csv")
df_reg = pd.read_csv("../input/D5 SHSAT Registrations and Testers.csv")
df_school.head(10)
from collections import Counter
city_names = []
city_count = []
city_dict = dict(Counter(df_school.City))
city_dict = sorted(city_dict.items(), key=operator.itemgetter(1))
for tup in city_dict:
    city_names.append(tup[0].lower())
    city_count.append(tup[1])

dataa = [go.Bar(
            y= city_names,
            x = city_count,
            width = 0.9,
            opacity=0.6, 
            orientation = 'h',
            marker=dict(
                color='rgb(158,202,225)',
                line=dict(
                    color='rgb(8,48,107)',
                    width=1.5,
                )
            )
        )]
layout = go.Layout(
    title='Distribution of Schools ',
    autosize = False,
    width=800,
    height=800,
    margin=go.Margin(
        l=250,
        r=50,
        b=100,
        t=100,
        pad=10
    ),
)

fig = go.Figure(data=dataa, layout = layout)
py.iplot(fig, filename='School-City-Bar')

fig2 = {
  "data": [
    {
      "values": city_count,
      "labels": city_names,
      "hoverinfo":"label+percent",
      "hole": .3,
      "type": "pie"
    }],
  "layout": {
        "title":"Percentage of Schools in each City",
        "paper_bgcolor":'rgb(243, 243, 243)',"plot_bgcolor":'rgb(243, 243, 243)'
        
    }
}
py.iplot(fig2, filename='School-City-Pie')
df_school['School Income Estimate'] = df_school['School Income Estimate'].str.replace(',', '')
df_school['School Income Estimate'] = df_school['School Income Estimate'].str.replace('$', '')
df_school['School Income Estimate'] = df_school['School Income Estimate'].str.replace(' ', '')
df_school['School Income Estimate'] = df_school['School Income Estimate'].astype(float)

trace1 = go.Histogram(
    x = df_school['School Income Estimate'],
    name = 'School Income Estimate'
)
dat = [trace1]

layout = go.Layout(
    title='School Income Estimate',paper_bgcolor='rgb(243, 243, 243)',plot_bgcolor='rgb(243, 243, 243)'
)

fig = go.Figure(data=dat, layout = layout)
py.iplot(fig, filename='School-Income-Hist')
cc_names, cc_count = list(), list()
cc = dict(Counter(df_school['Community School?']))
cc = sorted(cc.items(), key=operator.itemgetter(1))
for tup in cc:
    cc_names.append(tup[0].upper())
    cc_count.append(tup[1])

dataa = [go.Bar(
            y= cc_names,
            x = cc_count,
            width = 0.9,
            opacity=0.6, 
            orientation = 'h',
            marker=dict(
                color='rgb(158,202,225)',
                line=dict(
                    color='rgb(8,48,107)',
                    width=1.5,
                )
            ),
        )]
layout = go.Layout(
    title='Community School or Not?',
    margin=go.Margin(
        l=250,
        r=50,
        b=100,
        t=100,
        pad=10
    ),paper_bgcolor='rgb(243, 243, 243)',plot_bgcolor='rgb(243, 243, 243)'
)

fig = go.Figure(data=dataa, layout = layout)
py.iplot(fig, filename='Community-School-Bar')

df_school['Grade High'] = df_school['Grade High'].map({'09': '9th Grade', '10': '10th Grade', '07': '7th Grade', '02': '2nd Grade', '0K': 'Kindergarten', '04': '4th Grade', '03': '3rd Grade', '06': '6th Grade', '12': '12th Grade', '08': '8th Grade', '05': '5th Grade'})

cc = dict(Counter(df_school['Grade High']))


dataa = [go.Bar(
            y= list(cc.keys()),
            x = list(cc.values()),
            width = 0.9,
            opacity=0.6, 
            orientation = 'h',
            marker=dict(
                color='rgb(158,202,225)',
                line=dict(
                    color='rgb(8,48,107)',
                    width=1.5,
                )
            ),
        )]
layout = go.Layout(
    title='Highest Grades in the Given Schools',
    margin=go.Margin(
        l=250,
        r=50,
        b=100,
        t=100,
        pad=10
    ),paper_bgcolor='rgb(243, 243, 243)',plot_bgcolor='rgb(243, 243, 243)'
)

fig = go.Figure(data=dataa, layout = layout)
py.iplot(fig, filename='Community-School-Bar')
trace1 = go.Histogram(
    x = df_school['Economic Need Index'],
    name = 'Economic Need Index'
)
dat = [trace1]

layout = go.Layout(
    title='Economic Need Index Histogram',paper_bgcolor='rgb(243, 243, 243)',plot_bgcolor='rgb(243, 243, 243)'
)

fig = go.Figure(data=dat, layout = layout)
py.iplot(fig, filename='School-Income-Hist')
Margin_common = go.Margin(
        l=450,
        r=50,
        b=100,
        t=100,
        pad=10)
marker_common = dict(
                color='rgb(158,202,225)',
                line=dict(
                    color='rgb(8,48,107)',
                    width=1.5,
                )
            )
rir = dict(Counter(df_school['Rigorous Instruction Rating']))
rir_bar = go.Bar(
            y= list(rir.values()),
            x = list(rir.keys()),
            width = 0.9,
            opacity=0.6, 
            orientation = 'v',
            name = 'Rigorous Instruction Rating',
            marker= marker_common
        )

rir2_hist = go.Histogram(
    x = df_school['Rigorous Instruction %'],
    name = 'Rigorous Instruction %'
)


ctr = dict(Counter(df_school['Collaborative Teachers Rating']))
ctr_bar = go.Bar(
            y= list(ctr.values()),
            x = list(ctr.keys()),
            width = 0.9,
            opacity=0.6, 
            orientation = 'v',
            name = 'Collaborative Teachers Rating',
            marker= marker_common
        )

ctr2_hist = go.Histogram(
    x = df_school['Collaborative Teachers %'],
    name = 'Collaborative Teachers %'
)


ser = dict(Counter(df_school['Supportive Environment Rating']))
ser_bar = go.Bar(
            y= list(rir.values()),
            x = list(rir.keys()),
            width = 0.9,
            opacity=0.6, 
            orientation = 'v',
            name = 'Supportive Environment Rating',
            marker= marker_common
        )

ser2_hist = go.Histogram(
    x = df_school['Supportive Environment %'],
    name = 'Supportive Environment %'
)


eslr = dict(Counter(df_school['Effective School Leadership Rating']))
eslr_bar = go.Bar(
            y= list(eslr.values()),
            x = list(eslr.keys()),
            width = 0.9,
            opacity=0.6, 
            orientation = 'v',
            name = 'Effective School Leadership Rating',
            marker= marker_common
        )

eslr2_hist = go.Histogram(
    x = df_school['Effective School Leadership %'],
    name = 'Effective School Leadership %'
)


sfct = dict(Counter(df_school['Strong Family-Community Ties Rating']))
sfct_bar = go.Bar(
            y= list(sfct.values()),
            x = list(sfct.keys()),
            width = 0.9,
            opacity=0.6, 
            orientation = 'v',
            name = 'Strong Family-Community Rating',
            marker= marker_common
        )

sfct2_hist = go.Histogram(
    x = df_school['Strong Family-Community Ties %'],
    name = 'Strong Family-Community Ties %'
)


tr = dict(Counter(df_school['Trust Rating']))
tr_bar = go.Bar(
            y= list(tr.values()),
            x = list(tr.keys()),
            width = 0.9,
            opacity=0.6, 
            orientation = 'v',
            name = 'Trust Rating',
            marker= marker_common
        )

tr2_hist = go.Histogram(
    x = df_school['Trust %'],
    name = 'Trust %'
)


fig = tls.make_subplots(rows=6, cols=2, subplot_titles=('Rigorous Instruction Rating', 'Rigorous Instruction %',
                                                        'Collaborative Teachers Rating', 'Collaborative Teachers %',
                                                        'Supportive Environment Rating', 'Supportive Environment %',
                                                       'Effective School Leadership Rating', 'Effective School Leadership %',
                                                        'Strong Family-Community Ties Rating', 'Strong Family-Community Ties %',
                                                       'Trust Rating', 'Trust %'));
fig.append_trace(rir_bar, 1, 1);
fig.append_trace(rir2_hist, 1, 2);
fig.append_trace(ctr_bar, 2, 1);
fig.append_trace(ctr2_hist, 2, 2);
fig.append_trace(ser_bar, 3, 1);
fig.append_trace(ser2_hist, 3, 2);
fig.append_trace(eslr_bar, 4, 1);
fig.append_trace(eslr2_hist, 4, 2);
fig.append_trace(sfct_bar, 5, 1);
fig.append_trace(sfct2_hist, 5, 2);
fig.append_trace(tr_bar, 6, 1);
fig.append_trace(tr2_hist, 6, 2);

fig['layout'].update(height=2400,title='School Quality Report Charts', showlegend=False, paper_bgcolor='rgb(243, 243, 243)',plot_bgcolor='rgb(243, 243, 243)');
py.iplot(fig, filename='simple-subplot')
SAR = dict(Counter(df_school['Student Achievement Rating']))

fig2 = {
  "data": [
    {
      "values": list(SAR.values()),
      "labels": list(SAR.keys()),
      "hoverinfo":"label+percent",
      "hole": .3,
      "type": "pie"
    }],
  "layout": {
        "title":"Student Achievement Rating"
        ,"paper_bgcolor":'rgb(243, 243, 243)',"plot_bgcolor":'rgb(243, 243, 243)'
    }
}
py.iplot(fig2, filename='School-City-Pie')

amp_hist = go.Histogram(
    x = df_school['Average Math Proficiency'],
    name = 'Average Math Proficiency'
)

aep_hist = go.Histogram(
    x = df_school['Average ELA Proficiency'],
    name = 'Average ELA Proficiency'
)
print("Average Math Proficiency is : " + str(np.mean(df_school['Average Math Proficiency'])))
print("Average ELA Proficiency is : " + str(np.mean(df_school['Average ELA Proficiency'])))
fig = tls.make_subplots(rows=1, cols=2, subplot_titles=('Average-Math-Proficiency-Histogram', 'Average ELA Proficiency'));
fig.append_trace(amp_hist, 1, 1);
fig.append_trace(aep_hist, 1, 2);

fig['layout'].update(height=400,title='Average Proficiency Plot', showlegend=False,paper_bgcolor='rgb(243, 243, 243)',plot_bgcolor='rgb(243, 243, 243)');
py.iplot(fig, filename='Proficiency-subplot')
PCA_hist = go.Histogram(
    x = df_school['Percent of Students Chronically Absent'],
    name = 'Percent of Students Chronically Absent'
)

dat = [PCA_hist]

layout = go.Layout(
    title='Percent of Students Chronically Absent Histogram',paper_bgcolor='rgb(243, 243, 243)',plot_bgcolor='rgb(243, 243, 243)'
)

fig = go.Figure(data=dat, layout = layout)
py.iplot(fig, filename='Percent-of-Students-Chronically-Absent-Hist')
def p2f(x):
    return float(x.strip('%'))/100

df_school['Percent Asian'] = df_school['Percent Asian'].apply(p2f)
df_school['Percent Black'] = df_school['Percent Black'].apply(p2f)
df_school['Percent Hispanic'] = df_school['Percent Hispanic'].apply(p2f)
df_school['Percent White'] = df_school['Percent White'].apply(p2f)
df_school['Percent Black / Hispanic'] = df_school['Percent Black / Hispanic'].apply(p2f)
df_school['Percent ELL'] = df_school['Percent ELL'].apply(p2f)
d3 = pd.DataFrame(df_school.groupby(['City']).mean())
d3[['Economic Need Index','School Income Estimate','Percent Asian','Percent Black','Percent Hispanic','Percent Black / Hispanic','Percent White']]
#d3.head(25)
plt.figure(figsize=(12,12))
sns.jointplot(x=df_school.Latitude.values, y=df_school.Longitude.values, size=10, color = 'red')
#sns.swarmplot(x="Latitude", y="Longitude", hue="Percent Asian" data=df_school)
plt.ylabel('Longitude', fontsize=12)
plt.xlabel('Latitude', fontsize=12)
plt.show()
import folium
from folium import plugins
from io import StringIO
import folium 

#colors = ['hsl('+str(h)+',50%'+',50%)' for h in np.linspace(0, 360, 12)]
colors = ['red', 'yellow', 'dusty purple', 'blue']
d = (df_school['Percent Asian']*100).astype('int')
cols = [colors[int(i/25)] for i in d]

map_osm2 = folium.Map([df_school['Latitude'][0], df_school['Longitude'][0]], zoom_start=10.2,tiles='cartodbdark_matter')

for lat, long, col in zip(df_school['Latitude'], df_school['Longitude'], cols):
    #rown = list(rown)
    folium.CircleMarker([lat, long], color=col, fill=True, radius=2).add_to(map_osm2)

map_osm2
d1 = (df_school['Percent ELL']*100).astype('int')
cols = [colors[int(i/25)] for i in d1]

map_osm2 = folium.Map([df_school['Latitude'][0], df_school['Longitude'][0]], zoom_start=10.2,tiles='cartodbdark_matter')

for lat, long, col in zip(df_school['Latitude'], df_school['Longitude'], cols):
    folium.CircleMarker([lat, long], color=col, fill=True, radius=2).add_to(map_osm2)

map_osm2
d3 = (df_school['Percent Black']*100).astype('int')
cols = [colors[int(i/25)] for i in d1]

map_osm2 = folium.Map([df_school['Latitude'][0], df_school['Longitude'][0]], zoom_start=10.2,tiles='cartodbdark_matter')

for lat, long, col in zip(df_school['Latitude'], df_school['Longitude'], cols):
    folium.CircleMarker([lat, long], color=col, fill=True, radius=2).add_to(map_osm2)

map_osm2
d3 = (df_school['Percent Hispanic']*100).astype('int')
cols = [colors[int(i/25)] for i in d1]

map_osm2 = folium.Map([df_school['Latitude'][0], df_school['Longitude'][0]], zoom_start=10.2,tiles='cartodbdark_matter')

for lat, long, col in zip(df_school['Latitude'], df_school['Longitude'], cols):
    folium.CircleMarker([lat, long], color=col, fill=True, radius=2).add_to(map_osm2)

map_osm2
d3 = (df_school['Percent White']*100).astype('int')
cols = [colors[int(i/25)] for i in d1]

map_osm2 = folium.Map([df_school['Latitude'][0], df_school['Longitude'][0]], zoom_start=10.2,tiles='cartodbdark_matter')

for lat, long, col in zip(df_school['Latitude'], df_school['Longitude'], cols):
    folium.CircleMarker([lat, long], color=col, fill=True, radius=2).add_to(map_osm2)

map_osm2
df_school.head(10)
# Create a trace
colors = ['hsl('+str(h)+',50%'+',50%)' for h in np.linspace(0, 360, 12)]
race = ['All Students','American Indian or Alaska Native','Black or African American',
          'Hispanic or Latino','Asian or Pacific Islander','White',
          'Multiracial','Limited English Proficient','Economically Disadvantaged']

g3_ela_count = list()
for i in race:
    g3_ela_count.append( len( df_school['Grade 3 ELA 4s - ' + i][df_school['Grade 3 ELA 4s - ' + i] > 0] ) )

total = np.sum(g3_ela_count)
trace0 = go.Scatter(
    x=race,
    y=g3_ela_count,
    mode='markers',
    marker=dict(
        size=[((x/total)*150) + 20 for x in g3_ela_count],
        color=colors[:len(race)],
    )
)

g3_math_count = list()
for i in race:
    g3_math_count.append( len( df_school['Grade 3 Math 4s - ' + i][df_school['Grade 3 Math 4s - ' + i] > 0] ) )

total2 = np.sum(g3_math_count)
trace1 = go.Scatter(
    x=race,
    y=g3_math_count,
    mode='markers',
    marker=dict(
        size=[((x/total)*150) + 20 for x in g3_math_count],
        color=colors[:len(race)],
    )
)

fig = tls.make_subplots(rows=1, cols=2, subplot_titles=('Count of Students scoring 4s in Grade 3 ELA', 'Count of Students scoring 4s in Grade 3 Math'));
fig.append_trace(trace0, 1, 1);
fig.append_trace(trace1, 1, 2);

fig['layout'].update(height=400,title='Count of Students scoring 4s in Grade 3', showlegend=False,paper_bgcolor='rgb(243, 243, 243)',plot_bgcolor='rgb(243, 243, 243)' );
py.iplot(fig, filename='Proficiency-subplot')
#---------------------Grade 4 -----------------------------------


g4_ela_count = list()
for i in race:
    g4_ela_count.append( len( df_school['Grade 4 ELA 4s - ' + i][df_school['Grade 4 ELA 4s - ' + i] > 0] ) )


total = np.sum(g4_ela_count)
trace0 = go.Scatter(
    x=race,
    y=g4_ela_count,
    mode='markers',
    marker=dict(
        size=[((x/total)*150) + 20 for x in g4_ela_count],
        color=colors[:len(race)],
    )
)

g4_math_count = list()
for i in race:
    g4_math_count.append( len( df_school['Grade 4 Math 4s - ' + i][df_school['Grade 4 Math 4s - ' + i] > 0] ) )

total2 = np.sum(g4_math_count)
trace1 = go.Scatter(
    x=race,
    y=g4_math_count,
    mode='markers',
    marker=dict(
        size=[((x/total)*150) + 20 for x in g4_math_count],
        color=colors[:len(race)],
    )
)

fig = tls.make_subplots(rows=1, cols=2, subplot_titles=('Count of Students scoring 4s in Grade 4 ELA', 'Count of Students scoring 4s in Grade 4 Math'));
fig.append_trace(trace0, 1, 1);
fig.append_trace(trace1, 1, 2);

fig['layout'].update(height=400,title='Count of Students scoring 4s in Grade 4', showlegend=False, paper_bgcolor='rgb(243, 243, 243)',plot_bgcolor='rgb(243, 243, 243)');
py.iplot(fig, filename='Proficiency-subplot')
#---------------------- Grade 5 --------------------------------------

g5_ela_count = list()
for i in race:
    g5_ela_count.append( len( df_school['Grade 5 ELA 4s - ' + i][df_school['Grade 5 ELA 4s - ' + i] > 0] ) )


total = np.sum(g5_ela_count)
trace0 = go.Scatter(
    x=race,
    y=g5_ela_count,
    mode='markers',
    marker=dict(
        size=[((x/total)*150) + 20 for x in g5_ela_count],
        color=colors[:len(race)],
    )
)

g5_math_count = list()
for i in race:
    g5_math_count.append( len( df_school['Grade 5 Math 4s - ' + i][df_school['Grade 5 Math 4s - ' + i] > 0] ) )

total2 = np.sum(g5_math_count)
trace1 = go.Scatter(
    x=race,
    y=g5_math_count,
    mode='markers',
    marker=dict(
        size=[((x/total2)*150) + 20 for x in g5_math_count],
        color=colors[:len(race)],
    )
)

fig = tls.make_subplots(rows=1, cols=2, subplot_titles=('Count of Students scoring 4s in Grade 5 ELA', 'Count of Students scoring 4s in Grade 5 Math'));
fig.append_trace(trace0, 1, 1);
fig.append_trace(trace1, 1, 2);

fig['layout'].update(height=400,title='Count of Students scoring 4s in Grade 5', showlegend=False, paper_bgcolor='rgb(243, 243, 243)',plot_bgcolor='rgb(243, 243, 243)');
py.iplot(fig, filename='Proficiency-subplot')
# ---------------- Grade 6 ---------------------------


g6_ela_count = list()
for i in race:
    g6_ela_count.append( len( df_school['Grade 6 ELA 4s - ' + i][df_school['Grade 6 ELA 4s - ' + i] > 0] ) )


total = np.sum(g6_ela_count)
trace0 = go.Scatter(
    x=race,
    y=g6_ela_count,
    mode='markers',
    marker=dict(
        size=[((x/total)*150) + 20 for x in g6_ela_count],
        color=colors[:len(race)],
    )
)

g6_math_count = list()
for i in race:
    g6_math_count.append( len( df_school['Grade 6 Math 4s - ' + i][df_school['Grade 6 Math 4s - ' + i] > 0] ) )

total2 = np.sum(g6_math_count)
trace1 = go.Scatter(
    x=race,
    y=g6_math_count,
    mode='markers',
    marker=dict(
        size=[((x/total2)*150) + 20 for x in g6_math_count],
        color=colors[:len(race)],
    )
)


fig = tls.make_subplots(rows=1, cols=2, subplot_titles=('Count of Students scoring 4s in Grade 6 ELA', 'Count of Students scoring 4s in Grade 6 Math'));
fig.append_trace(trace0, 1, 1);
fig.append_trace(trace1, 1, 2);

fig['layout'].update(height=400,title='Count of Students scoring 4s in Grade 6', showlegend=False, paper_bgcolor='rgb(243, 243, 243)',plot_bgcolor='rgb(243, 243, 243)');
py.iplot(fig, filename='Proficiency-subplot')
# -------------------------- Grade 7 ---------------------------

g7_ela_count = list()
for i in race:
    g7_ela_count.append( len( df_school['Grade 7 ELA 4s - ' + i][df_school['Grade 7 ELA 4s - ' + i] > 0] ) )

total = np.sum(g7_ela_count)
trace0 = go.Scatter(
    x=race,
    y=g7_ela_count,
    mode='markers',
    marker=dict(
        size=[((x/total)*150) + 20 for x in g7_ela_count],
        color=colors[:len(race)],
    )
)

g7_math_count = list()
for i in race:
    g7_math_count.append( len( df_school['Grade 7 Math 4s - ' + i][df_school['Grade 7 Math 4s - ' + i] > 0] ) )

total2 = np.sum(g7_math_count)
trace1 = go.Scatter(
    x=race,
    y=g7_math_count,
    mode='markers',
    marker=dict(
        size=[((x/total2)*150) + 20 for x in g7_math_count],
        color=colors[:len(race)],
    )
)


fig = tls.make_subplots(rows=1, cols=2, subplot_titles=('Count of Students scoring 4s in Grade 7 ELA', 'Count of Students scoring 4s in Grade 7 Math'));
fig.append_trace(trace0, 1, 1);
fig.append_trace(trace1, 1, 2);

fig['layout'].update(height=400,title='Count of Students scoring 4s in Grade 7', showlegend=False, paper_bgcolor='rgb(243, 243, 243)',plot_bgcolor='rgb(243, 243, 243)');
py.iplot(fig, filename='Proficiency-subplot')
#------------------------- Grade 8 -----------------------------

g8_ela_count = list()
for i in race:
    g8_ela_count.append( len( df_school['Grade 8 ELA 4s - ' + i][df_school['Grade 8 ELA 4s - ' + i] > 0] ) )

total = np.sum(g8_ela_count)
trace0 = go.Scatter(
    x=race,
    y=g8_ela_count,
    mode='markers',
    marker=dict(
        size=[((x/total)*150) + 20 for x in g8_ela_count],
        color=colors[:len(race)],
    )
)

g8_math_count = list()
for i in race:
    g8_math_count.append( len( df_school['Grade 8 Math 4s - ' + i][df_school['Grade 8 Math 4s - ' + i] > 0] ) )

total2 = np.sum(g8_math_count)
trace1 = go.Scatter(
    x=race,
    y=g8_math_count,
    mode='markers',
    marker=dict(
        size=[((x/total2)*150) + 20 for x in g8_math_count],
        color=colors[:len(race)],
        
    )
)

fig = tls.make_subplots(rows=1, cols=2, subplot_titles=('Count of Students scoring 4s in Grade 8 ELA', 'Count of Students scoring 4s in Grade 8 Math'));
fig.append_trace(trace0, 1, 1);
fig.append_trace(trace1, 1, 2);

fig['layout'].update(height=400,title='Count of Students scoring 4s in Grade 8', showlegend=False, paper_bgcolor='rgb(243, 243, 243)',plot_bgcolor='rgb(243, 243, 243)');
py.iplot(fig, filename='Proficiency-subplot')
d4 = df_reg[df_reg['Year of SHST'] == 2013]
d4 = pd.DataFrame(d4.groupby(['School name']).sum()).reset_index()
d5 = df_reg[df_reg['Year of SHST'] == 2014]
d5 = pd.DataFrame(d5.groupby(['School name']).sum()).reset_index()
d6 = df_reg[df_reg['Year of SHST'] == 2015]
d6 = pd.DataFrame(d6.groupby(['School name']).sum()).reset_index()
d7 = df_reg[df_reg['Year of SHST'] == 2016]
d7 = pd.DataFrame(d7.groupby(['School name']).sum()).reset_index()

d4['Number of students who did not take the SHSAT after registering'] = d4['Number of students who registered for the SHSAT'] - d4['Number of students who took the SHSAT']
d5['Number of students who did not take the SHSAT after registering'] = d5['Number of students who registered for the SHSAT'] - d5['Number of students who took the SHSAT']
d6['Number of students who did not take the SHSAT after registering'] = d6['Number of students who registered for the SHSAT'] - d6['Number of students who took the SHSAT']
d7['Number of students who did not take the SHSAT after registering'] = d7['Number of students who registered for the SHSAT'] - d7['Number of students who took the SHSAT']

trace1 = go.Bar(
    y=df_reg['School name'],
    x=df_reg['Number of students who registered for the SHSAT'],
    name='Number of students who registered for the SHSAT',
    orientation = 'h'
)
trace2 = go.Bar(
    y=df_reg['School name'],
    x=df_reg['Number of students who took the SHSAT'],
    name='Number of students who took the SHSAT',
    orientation = 'h'
)

data = [trace1, trace2]
layout = go.Layout(
    barmode='stack',
    showlegend = False,
    margin=go.Margin(
        l=350,
        r=50,
        b=100,
        t=100,
        pad=4
    ),
    height = 800,
    
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='marker-h-bar')

label_common = ['Number of students who did not take the SHSAT after registering', 'Number of students who took the SHSAT']

values_13 = [np.sum(d4['Number of students who did not take the SHSAT after registering']), np.sum(d4['Number of students who took the SHSAT'])]
values_14 = [np.sum(d5['Number of students who did not take the SHSAT after registering']), np.sum(d5['Number of students who took the SHSAT'])]
values_15 = [np.sum(d6['Number of students who did not take the SHSAT after registering']), np.sum(d6['Number of students who took the SHSAT'])]
values_16 = [np.sum(d7['Number of students who did not take the SHSAT after registering']), np.sum(d7['Number of students who took the SHSAT'])]


labels1 = ['Number of students who registered for the SHSAT','Number of students who took the SHSAT','Number of students who did not take the SHSAT']
val_2013 = [[np.sum(d4['Number of students who registered for the SHSAT']), 
             np.sum(d4['Number of students who took the SHSAT']),
             np.sum(d4['Number of students who did not take the SHSAT after registering'])]]

val_2014 = [[np.sum(d5['Number of students who registered for the SHSAT']), 
             np.sum(d5['Number of students who took the SHSAT']),
             np.sum(d5['Number of students who did not take the SHSAT after registering'])]]

val_2015 = [[np.sum(d6['Number of students who registered for the SHSAT']), 
             np.sum(d6['Number of students who took the SHSAT']),
             np.sum(d6['Number of students who did not take the SHSAT after registering'])]]

val_2016 = [[np.sum(d7['Number of students who registered for the SHSAT']), 
             np.sum(d7['Number of students who took the SHSAT']),
             np.sum(d7['Number of students who did not take the SHSAT after registering'])]]
trace0 = go.Bar(
    y=labels1,
    x=val_2013[0],
    marker=dict(color=['blue', 'yellow','red']),
    orientation ='h'
)

trace1 = go.Bar(
    x=val_2014[0],
    y=labels1,
    marker=dict(color=['blue', 'yellow','red']),
    orientation ='h'
)
fig = tls.make_subplots(rows=2, cols=1, subplot_titles=('Number of students who registered for the SHSAT for 2013', 'Number of students who registered for the SHSAT for 2014'));
fig.append_trace(trace0, 1, 1);
fig.append_trace(trace1, 2, 1);

fig['layout'].update(title = 'Number of students who registered for the SHSAT for 2013 and 2014', height=600, showlegend=False, margin=go.Margin(
        l=350,
        r=50,
        b=100,
        t=100,
        pad=4
    ),paper_bgcolor='rgb(243, 243, 243)',plot_bgcolor='rgb(243, 243, 243)');
py.iplot(fig, filename='Proficiency-subplot')

fig = {
  "data": [
    {
      "values": values_13,
      "labels": label_common,
      "domain": {"x": [0, .48]},
      "name": "2013",
      "hoverinfo":"label+percent+name",
      "hole": .4,
      "type": "pie"
    },
    {
      "values": values_14,
      "labels": label_common,
      "text":["2013"],
      "textposition":"inside",
      "domain": {"x": [.52, 1]},
      "name": "2014",
      "hoverinfo":"label+percent+name",
      "hole": .4,
      "type": "pie"
    }],
  "layout": {
        "title":"Number of students who registered for the SHSAT for 2013 and 2014",
        "showlegend": False,
        "annotations": [
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "2013",
                "x": 0.20,
                "y": 0.5
            },
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "2014",
                "x": 0.8,
                "y": 0.5
            }
        ],
        "paper_bgcolor": 'rgb(243, 243, 243)',"plot_bgcolor":'rgb(243, 243, 243)',
    }
}
py.iplot(fig, filename='donut')

trace0 = go.Bar(
    y=labels1,
    x=val_2015[0],
    marker=dict(color=['blue', 'yellow','red']),
    orientation ='h'
)

trace1 = go.Bar(
    x=val_2016[0],
    y=labels1,
    marker=dict(color=['blue', 'yellow','red']),
    orientation ='h'
)
fig = tls.make_subplots(rows=2, cols=1, subplot_titles=('Number of students who registered for the SHSAT for 2015', 'Number of students who registered for the SHSAT for 2016'));
fig.append_trace(trace0, 1, 1);
fig.append_trace(trace1, 2, 1);

fig['layout'].update(title = 'Number of students who registered for the SHSAT', height=600, showlegend=False, margin=go.Margin(
        l=350,
        r=50,
        b=100,
        t=100,
        pad=4
    ),paper_bgcolor='rgb(243, 243, 243)',plot_bgcolor='rgb(243, 243, 243)');
py.iplot(fig, filename='Proficiency-subplot')

fig = {
  "data": [
    {
      "values": values_15,
      "labels": label_common,
      "domain": {"x": [0, .48]},
      "name": "2015",
      "hoverinfo":"label+percent+name",
      "hole": .4,
      "type": "pie"
    },
    {
      "values": values_16,
      "labels": label_common,
      "text":["2016"],
      "textposition":"inside",
      "domain": {"x": [.52, 1]},
      "name": "2016",
      "hoverinfo":"label+percent+name",
      "hole": .4,
      "type": "pie"
    }],
  "layout": {
        "title":"Number of students who registered for the SHSAT for 2015 and 2016",
        "showlegend": False,
        "annotations": [
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "2015",
                "x": 0.20,
                "y": 0.5
            },
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "2016",
                "x": 0.8,
                "y": 0.5
            }
        ],
      "paper_bgcolor": 'rgb(243, 243, 243)',"plot_bgcolor":'rgb(243, 243, 243)',
    }
}
py.iplot(fig, filename='donut')


trace1 = go.Bar(
    x=[np.sum(d4['Enrollment on 10/31']), np.sum(d5['Enrollment on 10/31']), np.sum(d6['Enrollment on 10/31']), np.sum(d7['Enrollment on 10/31'])],
    y=['2013', '2014', '2015', '2016'],
    marker=dict(color=['blue', 'yellow','red', 'orange']),
    orientation ='h'
)
data = [trace1]

layout = go.Layout(
    title = "Enrollment on 10/31 Bar Plot",
    barmode='stack',
    height = 400,
     xaxis=dict(
        title='Count',
    ),
    yaxis=dict(
        title='Year',
    ),paper_bgcolor='rgb(243, 243, 243)',plot_bgcolor='rgb(243, 243, 243)')


fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='marker-h-bar')