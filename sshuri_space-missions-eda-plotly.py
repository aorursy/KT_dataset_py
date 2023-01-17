import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

import plotly

from plotly.subplots import make_subplots

import plotly.graph_objects as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

from PIL import Image

from wordcloud import WordCloud, ImageColorGenerator



init_notebook_mode(connected=True)
df = pd.read_csv("../input/all-space-missions-from-1957/Space_Corrected.csv")

df.head()
df.drop(['Unnamed: 0','Unnamed: 0.1'],axis=1,inplace=True)
companies = df.groupby(['Company Name'])['Detail'].count().sort_values(ascending=False).reset_index()

len(companies)
bar = px.bar(companies[::-1],x='Detail',y='Company Name',labels={'Detail':'No of Missions'})

bar.update_layout(title="Company and total missions")

bar.show()
plt.figure(figsize=(25,7))

bar = sns.barplot(x='Company Name',y='Detail',data=companies[1:],palette='rocket')

b = bar.set_xticklabels(bar.get_xticklabels(), rotation=30, horizontalalignment='right')

plt.ylabel('No of launches')

t=plt.title('Comapany vs launches')
df['Country'] = df['Location'].apply(lambda x:x.split(',')[-1])

df['year'] = df['Datum'].apply(lambda x:x.split()[3])
year_wise = df.groupby(['Company Name','year']).count()['Detail'].reset_index()

year_wise = year_wise[year_wise['Company Name'].isin(companies['Company Name'][:20])]



fig = go.Figure(data=go.Heatmap(

        z=year_wise['Detail'],

        x=year_wise['year'],

        y=year_wise['Company Name'],

        colorscale='Viridis'))



fig.update_layout(

    title='Company wise launches  per year',

    xaxis_nticks=36)



fig.show()



status = df['Status Rocket'].value_counts()



fig = make_subplots(rows=1, cols=2, specs=[[{"type": "xy"}, {"type": "domain"}]])

fig.add_trace(go.Bar( x=status.keys(), y=status.values, text=status.values.tolist(), textposition='auto',marker_color='#003786',name='Status'), row=1, col=1)

fig.add_trace(go.Pie(labels=status.keys(),values=status.values,textposition='inside', textinfo='percent+label',marker={'colors':['rgb(178,24,43)','rgb(253,219,199)']}), row=1, col=2)

fig.update_layout(title_text='Status of Rockets', font_size=10, autosize=False, width=800, height=400)

fig.show()
df_ussr = df[df['Company Name']=="RVSN USSR"]

plt.subplot(1,2,1)

bar=df_ussr.groupby('Status Rocket').count()['Detail'].plot(kind='bar',figsize=(6,3),width=0.1)

bar.set_xticklabels(bar.get_xticklabels(), rotation=0)

plt.title('Status of the mission')

plt.subplot(1,2,2)

bar=df_ussr['Status Mission'].value_counts().plot(kind='bar',figsize=(13,4))

bar.set_xticklabels(bar.get_xticklabels(), rotation=0)

for p in bar.patches:

  bar.annotate(int(p.get_height()), 

               (p.get_x() + p.get_width()/2, p.get_height()), ha='center', va='center', 

               xytext=(0,5), textcoords = 'offset points')

plt.xlabel('Status')

t=plt.title('Status of RocketLauch')



x=df_ussr.groupby('year').count()['Detail'].plot(kind='bar',figsize=(14,4))

plt.ylabel('Number of missions')

t=plt.title('number of missions by RVSN USSR per year')
df_active = df[df['Status Rocket'] == "StatusActive"]

df_active = df_active.groupby('Company Name').count()['Detail'].sort_values(ascending=False).reset_index()

len(df_active)
top_20 = companies[1:40]

cmp = df.groupby(['Company Name','Status Rocket']).count()['Detail'].reset_index()

cmp = cmp[cmp['Company Name'].isin(top_20['Company Name'])]

active = cmp[cmp['Status Rocket']=="StatusActive"].sort_values('Detail')

retired = cmp[cmp['Status Rocket']!="StatusActive"]

fig = go.Figure()

fig.add_bar(y=active['Detail'],x=active['Company Name'],name='Status Active')

fig.add_bar(y=retired['Detail'],x=retired['Company Name'],name='Status Retired')

fig.update_layout(barmode="stack",title="Comapnies and Mission Status",yaxis_title="No of Missions")

fig.show()
m = df['Status Mission'].value_counts()

mf = go.Figure([go.Pie(labels=m.keys(),values=m.values,textposition='inside', textinfo='percent+label',marker={'colors':["0e58a8","rgb(215,48,39)","rgb(112,164,148)","e2d9e2"]})])

mf.update_layout(title_text='Status of Launch', font_size=10, autosize=False, width=700, height=400)
plt.figure(figsize=(20,5))

cmp = df.groupby(['Company Name','Status Mission']).count()['Detail'].reset_index()

cmp = cmp[cmp['Status Mission']=="Success"].sort_values('Detail',ascending=False)

sns.barplot(x='Company Name',y='Detail',data=cmp[1:20])

plt.ylabel('No of successful missions')

t=plt.title('company vs sucessful missions')
plt.figure(figsize=(20,5))

cmp = df[df['Status Mission']!="Success"].groupby('Company Name').count().sort_values('Detail',ascending=False).reset_index()

sns.barplot(x='Company Name',y='Detail',data=cmp[1:20])

plt.ylabel('No of unsuccessful missions')

t=plt.title('company vs unsucessful missions')
year_wise = df.groupby(['Country','year']).count()['Detail'].reset_index()



fig = go.Figure(data=go.Heatmap(

        z=year_wise['Detail'],

        x=year_wise['year'],

        y=year_wise['Country'],

        colorscale='Viridis'))



fig.update_layout(

    title='Country wise launches  per year',

    xaxis_nticks=36)



fig.show()
country = df.groupby('Country').count()['Detail'].sort_values(ascending=False).reset_index()

country.rename(columns={"Detail":"No of Launches"},inplace=True)

country.head(10).style.background_gradient(cmap='Blues').hide_index()
# df.groupby(['Country','Status Mission']).count()

r = df.copy()

r = r.groupby(['Country','Status Mission'])['Detail'].count().unstack(fill_value=0).stack().reset_index()

cou = pd.DataFrame({"Country":r['Country'].unique(),"Success":r[r['Status Mission']=="Success"][0].values,

                    "Failure":r[r['Status Mission']=="Failure"][0].values,

                    "Partial Failure":r[r['Status Mission']=="Partial Failure"][0].values,

                    "Prelaunch Failure":r[r['Status Mission']=="Prelaunch Failure"][0].values},

                    columns=["Country","Success","Failure","Partial Failure","Prelaunch Failure"])

country_colors=["#01579b","#0277bd","#0288D1","#039BE5","#03A9F4","#29B6F6","#4FC3F7","#81D4FA","#B3E5FC","E1F5FE","E1F5FE"]

country_colors = sum([[i]*2 for i in country_colors],[])  



from collections import OrderedDict

from io import StringIO

from math import log, sqrt



import warnings

warnings.filterwarnings('ignore')

from bokeh.plotting import figure, output_file, show

from bokeh.io import output_notebook

from bokeh.resources import INLINE

output_notebook(INLINE)



status_color = OrderedDict([

    ("Success",   "#0d3362"),

    ("Failure", "#c64737"),

    ("Partial Failure", "white"  ),

    ("Prelaunch Failure","black")

])



width = 800

height = 800

inner_radius = 90

outer_radius = 300 - 10



maxr = sqrt(log(1* 1E4))

minr = sqrt(log(1395 * 1E4))

a = (outer_radius - inner_radius) / (minr - maxr)

b = inner_radius - a * maxr



def rad(mic):

    return a * np.sqrt(np.log(mic * 1E4)) + b



big_angle = 2.0 * np.pi / (len(cou)+1)

small_angle = big_angle/7 



p = figure(plot_width=width, plot_height=height, title="",

    x_axis_type=None, y_axis_type=None,

    x_range=(-420, 420), y_range=(-420, 420),

    min_border=0, outline_line_color="black",

    background_fill_color="#f0e1d2")



p.xgrid.grid_line_color = None

p.ygrid.grid_line_color = None



# annular wedges

angles = np.pi/2 - big_angle/2 - cou.index.to_series()*big_angle

colors = country_colors

p.annular_wedge(

    0, 0, inner_radius, outer_radius, -big_angle+angles, angles, color=colors,

)



# small wedges

p.annular_wedge(0, 0, inner_radius, rad(cou['Success']),

                -big_angle+angles+7*small_angle, -big_angle+angles+8*small_angle,

                color=status_color['Success'])

p.annular_wedge(0, 0, inner_radius, rad(cou['Failure']),

                -big_angle+angles+5*small_angle, -big_angle+angles+6*small_angle,

                color=status_color['Failure'])

p.annular_wedge(0, 0, inner_radius, rad(cou["Partial Failure"]),

                -big_angle+angles+3*small_angle, -big_angle+angles+4*small_angle,

                color=status_color['Partial Failure'])

p.annular_wedge(0, 0, inner_radius, rad(cou["Prelaunch Failure"]),

                -big_angle+angles+1*small_angle, -big_angle+angles+2*small_angle,

                color=status_color['Prelaunch Failure'])



# circular axes and lables

labels = np.power(10.0, np.arange(1,4))

radii = a * np.sqrt(np.log(labels * 1E4)) + b

p.circle(0, 0, radius=radii, fill_color=None, line_color="#afc1ce")

p.text(0, radii[:], [str(r) for r in labels[:]],

       text_font_size="11px", text_align="center", text_baseline="middle")



# radial axes

p.annular_wedge(0, 0, inner_radius-10, outer_radius+10,

                -big_angle+angles, -big_angle+angles, color="white")



# #  labels

xr = radii[-1]*np.cos(np.array(-big_angle/2+ angles))+0.8

yr = radii[-1]*np.sin(np.array(-big_angle/2 + angles))+0.8

label_angle=np.array(-big_angle/2+angles)

label_angle[label_angle < -np.pi/2] += np.pi 

p.text(xr, yr, cou.Country, angle=label_angle,

       text_font_size="11px", text_align="center", text_baseline="middle")





p.rect([-40, -40, -40,-40], [36, 18,0,-18], width=20, height=13,

       color=list(status_color.values()))

p.text([-15, -15,-15, -15], [36, 18,0,-18], text=list(status_color),

       text_font_size="10px", text_align="left", text_baseline="middle")



show(p)
x = df.groupby(['Country']).count().reset_index()

x.rename(columns={'Detail':'Missions'},inplace=True)

fig = px.sunburst(x, path=['Country','Missions'],values='Missions',

                  color='Missions',

                  color_continuous_scale='RdBu') #

fig.show()
map_data = [go.Choropleth( 

           locations = country['Country'],

           locationmode = 'country names',

           z = country["No of Launches"], 

           text = country['Country'],

           colorbar = {'title':'No of Launches'},

           colorscale='ylorrd')]



layout = dict(title = 'Countries wise Rocket Launches', 

             geo = dict(showframe = False, 

                       projection = dict(type = 'equirectangular')))



world_map = go.Figure(data=map_data, layout=layout)

iplot(world_map)

df['LV'] = df['Detail'].apply(lambda x:x.split()[0])

ro = df['LV'].value_counts().sort_values(ascending=True)[-50:]

fig = go.Figure(go.Bar(x=ro.values, y=ro.keys(),orientation='h'))

fig.update_layout(title="Launch Vehicle used for number of missions ")
z = df.groupby(['Country','LV']).count().reset_index()

text = z['LV']





mask = np.array(Image.open("../input/rocket/LV.jpg"))

wordcloud_fra = WordCloud(background_color="white", mode="RGBA", max_words=1000, mask=mask,contour_color='firebrick').generate(" ".join(text))





image_colors = ImageColorGenerator(mask)

plt.figure(figsize=[24,8])

plt.imshow(wordcloud_fra.recolor(color_func=lambda *args, **kwargs: "black"), interpolation="bilinear")

pt=plt.axis("off")
success = df[df['Status Mission']=='Success'].groupby('LV').count()['Detail'].reset_index()

not_s = df[df['Status Mission']!='Success'].groupby('LV').count()['Detail'].reset_index()

not_s.rename(columns={'Detail':'Fails'},inplace=True)

total = pd.merge(success,not_s,on='LV').sort_values('Detail',ascending=True)[-50:]

fig = go.Figure()

fig.add_bar(x=total['Detail'],y=total['LV'],orientation='h',name="Success")

fig.add_bar(x=total['Fails'],y=total['LV'],orientation='h',name="Failure")



fig.update_layout(barmode="stack",title="Launch vehicle number of success and Failures",xaxis_title="No of Missions",yaxis_title="Launch Vehicle")

fig.show()
days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

df['day'] = df['Datum'].apply(lambda x:x.split()[0])

df_days = df.groupby('day').count()['Detail'].reset_index()



df_days['day'] = pd.Categorical(df_days['day'], categories=days, ordered=True)

df_days = df_days.sort_values('day')

plt.figure(figsize=(11,4))

sns.barplot(x='day', y='Detail', data=df_days)

plt.ylabel('No of launches')

b=plt.title(' day vs no of launches')
from calendar import month_abbr

plt.figure(figsize=(12,6))

df['month'] = df['Datum'].apply(lambda x: x.split()[1])

df_month = df.groupby('month').count()['Detail'].reset_index()

df_month['month'] = pd.Categorical(df_month['month'], categories=list(month_abbr)[1:], ordered=True)

df_month = df_month.sort_values('month')

bar=sns.barplot(x='month',y='Detail',data=df_month)

for p in bar.patches:

    bar.annotate(int(p.get_height()), 

               (p.get_x() + p.get_width()/2, p.get_height()), ha='center', va='center', 

               xytext=(0,7), textcoords = 'offset points')

plt.ylabel('No of launches')

_ = plt.title('no of rockets launched per month')
df['year'] = df['Datum'].apply(lambda x:x.split()[3])

date= df.groupby('year').count()['Detail'].reset_index()

plt.figure(figsize=(20,6))

b=sns.barplot(x='year', y='Detail', data=date)

plt.ylabel('no of launches')

plt.title(' No of launches per year')

_=b.set_xticklabels(b.get_xticklabels(), rotation=90, horizontalalignment='right')
colors=['#0072b2','#000000','#d55e00']

status= df['Status Mission'].unique()

j=0

for s in status[1:]:

    df[df['Status Mission']==s].groupby('year').count()['Detail'].plot(kind='bar',figsize=(18,5),color=colors[j])

    j+=1

plt.ylabel('No of unsuccessful Missions')

t = plt.title("Number of unsuccessful missions per year")

t=plt.legend(status[1:])
budget = df.copy().dropna()

budget.loc[:, ' Rocket'] = budget[' Rocket'].apply(lambda x:float(x.replace(',','')))

b = budget.groupby('Company Name').sum().sort_values(' Rocket', ascending=False).reset_index()
plt.figure(figsize=(14,4))

bar = sns.barplot(x='Company Name',y=' Rocket',data=b)

bar.set_xticklabels(bar.get_xticklabels(), rotation=30)

plt.ylabel('Money spent on Missions in million $')

t=plt.title('Comapny Budget')

budget.groupby('year').mean().plot(kind='bar',figsize=(14,4))

plt.ylabel('Avg money spent in $')

plt.legend('Money spent')

t=plt.title('Average money spent on Missions per year')

b = budget.groupby('Country').sum()[' Rocket'].reset_index()

c= budget.groupby('Country').count()['Detail'].sort_values(ascending=False).reset_index()

c = c.merge(b,on='Country')





y_count=c['Detail'][::-1]

y_net_worth = c[' Rocket'][::-1]

x=c['Country'][::-1]



fig = make_subplots(rows=1, cols=2, specs=[[{}, {}]], shared_xaxes=True,

                    shared_yaxes=False, vertical_spacing=0.001)



fig.append_trace(go.Bar(

    x=y_count,

    y=x,

    marker=dict(

        color='rgba(50, 171, 96, 0.6)',

        line=dict(

            color='rgba(50, 171, 96, 1.0)',

            width=1),

    ),

    name='Countries and number of missions',

    orientation='h',

), 1, 1)



fig.append_trace(go.Scatter(

    x=y_net_worth, y=x,

    mode='lines+markers',

    line_color='rgb(128, 0, 128)',

    name='Mission budget, Million USD',

), 1, 2)



fig.update_layout(

    title='Number of missions per country & Money spent in million dollars',

    yaxis=dict(

        showgrid=False,

        showline=False,

        showticklabels=True,

        domain=[0, 0.85],

    ),

    yaxis2=dict(

        showgrid=False,

        showline=True,

        showticklabels=False,

        linecolor='rgba(102, 102, 102, 0.8)',

        linewidth=2,

        domain=[0, 0.85],

    ),

    xaxis=dict(

        zeroline=False,

        showline=False,

        showticklabels=True,

        showgrid=True,

        domain=[0, 0.42],

    ),

    xaxis2=dict(

        zeroline=False,

        showline=False,

        showticklabels=True,

        showgrid=True,

        domain=[0.47, 1],

        side='top',

        dtick=25000,

    ),

    legend=dict(x=0.029, y=1.038, font_size=10),

    margin=dict(l=100, r=20, t=70, b=70),

    paper_bgcolor='rgb(248, 248, 255)',

    plot_bgcolor='rgb(248, 248, 255)',

)



annotations = []



y_s = np.round(y_count, decimals=2)

y_nw = np.rint(y_net_worth)



# Adding labels

for ydn, yd, xd in zip(y_nw, y_s, x):

 

    annotations.append(dict(xref='x2', yref='y2',

                            y=xd, x=ydn - 20000,

                            text='{:,}'.format(ydn) + 'M',

                            font=dict(family='Arial', size=12,

                                      color='rgb(128, 0, 128)'),

                            showarrow=False))



    annotations.append(dict(xref='x1', yref='y1',

                            y=xd, x=yd + 22,

                            text=str(yd),

                            font=dict(family='Arial', size=12,

                                      color='rgb(50, 171, 96)'),

                            showarrow=False))

# Source

annotations.append(dict(xref='paper', yref='paper',

                        x=-0.2, y=-0.109,

                        text="Country with number of missions and money spent on all missions",

                        font=dict(family='Arial', size=10, color='rgb(150,150,150)'),

                        showarrow=False))



fig.update_layout(annotations=annotations)



fig.show()
df['center'] = df['Location'].apply(lambda x:x.split(',')[1])

df.groupby('center').count()['Detail'].sort_values()[-10:].plot(kind='barh')

plt.xlabel('Number of Launches')

t=plt.title('center with number of launches')
top_10 = country['Country'].head(10)

x= df.groupby(['year','Country']).count().reset_index()

x=x[x['Country'].isin(top_10)]

px.scatter(x,x='year', y='Detail', color='Country')

df['Hour']=df['Datum'].apply(lambda datum: int(datum.split()[-2][:2]) if datum.split()[-1]=='UTC' else np.nan)

hr = df.groupby('Hour').count()['Detail'].reset_index()

px.bar(hr, x='Hour',y='Detail',labels={'Detail':'No of Missions','Hour':"Time(24hrs)"},title="No of Launches in time",width=700,height=400)
df['Date'] = df['Datum'].apply(lambda x:" ".join(x.split()[:4]))

df['Date'] = df['Date'].apply(pd.to_datetime)

df.set_index('Date',inplace=True)
r = df[[' Rocket','Status Mission']]

r.dropna(inplace=True)

r[' Rocket'] = r[' Rocket'].apply(lambda x:float(x.replace(',','')))

x=r['2020-08-07':'2010-12-31'].plot(figsize=(12,4),title="Money spent on the missions in the last 10 years",ylabel="Money in million $")


plt.figure(figsize=(10,4))

g=sns.kdeplot(r[' Rocket'],shade=True)

plt.title('Density plot')

for l in g.lines:

    plt.setp(l,linewidth=2.5)
v = df.reset_index().groupby(['Date','Status Mission']).count()['Detail'].reset_index()

succ = v[v['Status Mission']!="Success"].set_index('Date').resample('M').count().reset_index()

unsuccess = go.Scatter(x=succ.Date,y=succ['Detail'],yaxis='y2',name='Unucessful Launches')

unsucc = v[v['Status Mission']=="Success"].set_index('Date').resample('M').count().reset_index()

success = go.Scatter(x=unsucc.Date,y=unsucc['Detail'],name='Successful Launches')





layout = go.Layout(height=450, width=1000,

                   title='Successful and Unsuccessful Launches Plot',

                   # Same x and first y

                   xaxis=dict(title='Date'),

                   yaxis=dict(title='Successful', color='blue'),

                   # Add a second yaxis to the right of the plot

                   yaxis2=dict(title='Unsuccessful', color='red',

                               overlaying='y', side='right')

                   )

fig = go.Figure(data=[success, unsuccess],layout=layout)

fig.update_xaxes(rangeslider_visible=True)

fig.show()