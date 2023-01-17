import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
from plotly import tools
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot

import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as ff
import math
import warnings
warnings.filterwarnings('ignore')

color = sns.color_palette()
%matplotlib inline

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

import folium

pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 9999

df = pd.read_csv('../input/2016 School Explorer.csv')
df1 = pd.read_csv('../input/D5 SHSAT Registrations and Testers.csv')


df.head(2)

df.shape
df.describe(include='all')
# Define Function

def floater(x):
    return float(x.strip('%'))

# Apply Function

df["Percent Asian"] = df["Percent Asian"].astype(str).apply(floater) 
df["Percent Black"] = df["Percent Black"].astype(str).apply(floater)
df["Percent Hispanic"] = df["Percent Hispanic"].astype(str).apply(floater)
df["Percent White"] = df["Percent White"].astype(str).apply(floater)
df["Percent Others"] = (df["Percent Black"] + df["Percent Hispanic"] + df["Percent White"] + df["Percent Asian"]).sub(100).mul(-1) # Define Percent Other
df["Rigorous Instruction %"] = df["Rigorous Instruction %"].astype(str).apply(floater)
df["Collaborative Teachers %"] = df["Collaborative Teachers %"].astype(str).apply(floater)
df['School Income Estimate'] = df['School Income Estimate'].str.replace(',', '')
df['School Income Estimate'] = df['School Income Estimate'].str.replace('$', '')
df['School Income Estimate'] = df['School Income Estimate'].str.replace(' ', '')
df['School Income Estimate'] = df['School Income Estimate'].astype(str).apply(floater)
df["Supportive Environment %"] = df["Supportive Environment %"].astype(str).apply(floater)
df["Effective School Leadership %"] = df["Effective School Leadership %"].astype(str).apply(floater)
df["Strong Family-Community Ties %"] = df["Strong Family-Community Ties %"].astype(str).apply(floater)
df["Trust %"] = df["Trust %"].astype(str).apply(floater)
df["Student Attendance Rate"] = df["Student Attendance Rate"].astype(str).apply(floater)
df["Percent of Students Chronically Absent"] = df["Percent of Students Chronically Absent"].astype(str).apply(floater)
#Overview of the data
d3 = pd.DataFrame(df.groupby(['City']).mean())
d3[['Economic Need Index','School Income Estimate','Percent Asian','Percent Black','Percent Hispanic','Percent White', 'Percent Others']]
missing_df = df.isnull().sum(axis=0).reset_index()
missing_df.columns = ['column_name', 'missing_count']
missing_df = missing_df.ix[missing_df['missing_count']>0]
missing_df = missing_df.sort_values(by='missing_count')

ind = np.arange(missing_df.shape[0])
width = 0.9
fig, ax = plt.subplots(figsize=(12,18))
rects = ax.barh(ind, missing_df.missing_count.values, color='blue')
ax.set_yticks(ind)
ax.set_yticklabels(missing_df.column_name.values, rotation='horizontal')
ax.set_xlabel("Count of missing values")
ax.set_title("Number of missing values in each column")
plt.show()
total = df.isnull().sum().sort_values(ascending=False)
percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent Missing'])
missing_data.head(25)
df["Economic Need Index"] = df["Economic Need Index"].fillna(df["Economic Need Index"].mean())
df["School Income Estimate"] = df["School Income Estimate"].fillna(df["School Income Estimate"].mean())
df["Student Attendance Rate"] = df["Student Attendance Rate"].fillna(df["Student Attendance Rate"].mean())
df["Percent of Students Chronically Absent"] = df["Percent of Students Chronically Absent"].fillna(df["Percent of Students Chronically Absent"].mean())
df["Rigorous Instruction %"] = df["Rigorous Instruction %"].fillna(df["Rigorous Instruction %"].mean())
df["Collaborative Teachers %"] = df["Collaborative Teachers %"].fillna(df["Collaborative Teachers %"].mean())
df["Average ELA Proficiency"] = df["Average ELA Proficiency"].fillna(df["Average ELA Proficiency"].mean())
df["Average Math Proficiency"] = df["Average Math Proficiency"].fillna(df["Average Math Proficiency"].mean())
df["Percent Asian"] = df["Percent Asian"].fillna(df["Percent Asian"].mean())
df["Percent Black"] = df["Percent Black"].fillna(df["Percent Black"].mean())
df["Percent Hispanic"] = df["Percent Hispanic"].fillna(df["Percent Hispanic"].mean())
df["Percent White"] = df["Percent White"].fillna(df["Percent White"].mean())
df["Percent Others"] = df["Percent Others"].fillna(df["Percent Others"].mean())
df["Rigorous Instruction %"] = df["Rigorous Instruction %"].fillna(df["Rigorous Instruction %"].mean())
df["Collaborative Teachers %"] = df["Collaborative Teachers %"].fillna(df["Collaborative Teachers %"].mean())
df["Supportive Environment %"] = df["Supportive Environment %"].fillna(df["Supportive Environment %"].mean())
df["Effective School Leadership %"] = df["Effective School Leadership %"].fillna(df["Effective School Leadership %"].mean())
df["Strong Family-Community Ties %"] = df["Strong Family-Community Ties %"].fillna(df["Strong Family-Community Ties %"].mean())
df["Trust %"] = df["Trust %"].fillna(df["Trust %"].mean())
trace1 = go.Scatter(
    y = df["Latitude"].values,
    x = df["Longitude"].values,
    mode='markers',
    marker=dict(
        size=10,
        color = df["District"].values, #set color equal to a variable
        colorscale='Reds',
        #showscale=True
    ),
    text = df["School Name"].values
)
layout = go.Layout(
    autosize=False,
    #plot_bgcolor='rgba(240,240,240,1)',
    plot_bgcolor='rgba(255,160,122,0.1)',
    width=1000,
    height=900,
    title = "Location of schools (district)"
)
data = [trace1]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='scatter-plot-with-colorscale')
df["School Income Estimate"] = df["School Income Estimate"].apply(lambda x: float(str(x).replace("$","").replace(",","")))

trace1 = go.Scatter(
    y = df["Latitude"].values,
    x = df["Longitude"].values,
    mode='markers',
    marker=dict(
        size=10,
        color = df["School Income Estimate"].values, #set color equal to a variable
        colorscale='YlOrRd',
        showscale=True,
        reversescale=True
    ),
    text = df["School Name"].values
)
layout = go.Layout(
    autosize=False,
    plot_bgcolor='rgba(255,160,122,0.1)',
    width=1000,
    height=900,
    title = "Location of schools (Income)"
)
data = [trace1]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='scatter-plot-with-colorscale')
f, ax = plt.subplots(1, 2, figsize = (15, 7))
f.suptitle("Community School?", fontsize = 18.)
_ = df['Community School?'].value_counts().plot.bar(ax = ax[0], rot = 0, color = (sns.color_palette()[0], sns.color_palette()[2])).set(xticklabels = ["No", "Yes"])
_ = df['Community School?'].value_counts().plot.pie(labels = ("No", "Yes"), autopct = "%.2f%%", label = "", fontsize = 13., ax = ax[1],\
colors = (sns.color_palette()[0], sns.color_palette()[2]), wedgeprops = {"linewidth": 1.5, "edgecolor": "#F7F7F7"}), ax[1].texts[1].set_color("#F7F7F7"), ax[1].texts[3].set_color("#F7F7F7")
target_list = ['Average ELA Proficiency','Average Math Proficiency']

fig,ax = plt.subplots(nrows=3, ncols=2, figsize=(24,24))
plt.suptitle('Community vs. Non-community schools')

sns.boxplot(x="Community School?",y="Economic Need Index",
            data=df,ax=ax[0,0],palette='hls')
sns.boxplot(x="Community School?",y="School Income Estimate",
            data=df,ax=ax[0,1],palette='hls')

sns.distplot(df[target_list[0]][df['Community School?']=='Yes'],
             ax=ax[1,0],label='Yes')
sns.distplot(df[target_list[0]][df['Community School?']=='No'],
             ax=ax[1,0],label='Yes')
ax[1,0].legend()

sns.distplot(df[target_list[1]][df['Community School?']=='Yes'],
             ax=ax[1,1],color='#3636ff',label='Yes')
sns.distplot(df[target_list[1]][df['Community School?']=='No'],
             ax=ax[1,1],label='No')
ax[1,0].legend()

sns.countplot(x="Community School?",hue="Student Achievement Rating", 
              data=df,ax=ax[2,0],palette='hls')
sns.countplot(x="Community School?",hue="Trust Rating", 
              data=df,ax=ax[2,1],palette='bwr');
plt.figure(figsize=(14,8))
sns.countplot(df['Grade Low'])
plt.show()
plt.figure(figsize=(14,8))
sns.countplot(df['Grade High'])
plt.show()
plt.figure(figsize=(25,12))
sns.barplot(x=df['City'].value_counts().index, y=df['City'].value_counts().values, data=df, 
            order=pd.value_counts(df['City']).index)
plt.xticks(fontsize = 16, rotation='vertical')
plt.show()
fig,ax = plt.subplots(1, figsize=(14,7))
df[['Average ELA Proficiency','Average Math Proficiency']]\
.plot.hist(bins=100,alpha=0.8,figsize=(14,5),ax=ax,
           color=['skyblue','lightcoral'],
           title='Average Proficiency for Schools')
ax.set_xlabel('Average Proficiency Levels')

fig,ax = plt.subplots(1, figsize=(14,7))
ax.scatter(df['Economic Need Index'],
           df['Average Math Proficiency'],
           label='Math',color='lightcoral')
ax.scatter(df['Economic Need Index'],
           df['Average ELA Proficiency'],
           label='ELA',color='skyblue')
ax.set_xlabel('Economic Need Index')
ax.set_ylabel('Average ELA Proficiency')
ax.legend()
plt.suptitle('Economic Need Index & Education');
f, axes = plt.subplots(2, 2, figsize=(19, 9), sharex=True)
sns.despine(left=True)

sns.regplot(x=df["Economic Need Index"], y=df["Percent Asian"], color='goldenrod', ax=axes[0, 0], line_kws={"color": "black"})
sns.regplot(x=df["Economic Need Index"], y=df["Percent White"], color='c', ax=axes[0, 1], line_kws={"color": "black"})
sns.regplot(x=df["Economic Need Index"], y=df["Percent Black"], color='khaki', ax=axes[1, 0], line_kws={"color": "black"})
sns.regplot(x=df["Economic Need Index"], y=df["Percent Hispanic"], color='lightcoral', ax=axes[1, 1], line_kws={"color": "black"})

axes[0,0].set_title('Ecnomic Need Index (Asian)')
axes[0,1].set_title('Ecnomic Need Index (White)')
axes[1,0].set_title('Ecnomic Need Index (Black)')
axes[1,1].set_title('Ecnomic Need Index (Hispanic)')

plt.subplots_adjust(hspace=0.4)
df2013 = df1[df1['Year of SHST'] == 2013]
df2014 = df1[df1['Year of SHST'] == 2014]
df2015 = df1[df1['Year of SHST'] == 2015]
df2016 = df1[df1['Year of SHST'] == 2016]

total =  [df2013['Number of students who registered for the SHSAT'].sum(), 
          df2014['Number of students who registered for the SHSAT'].sum(),
          df2015['Number of students who registered for the SHSAT'].sum(),
          df2016['Number of students who registered for the SHSAT'].sum()]


years = [2013,2014,2015,2016]

trace = go.Bar(
                x = years,
                y = total,
                marker = dict(color = 'light blue',
                             line=dict(color='black',width=1.5)),
                )

data = [trace]
layout = go.Layout(barmode = "group", xaxis= dict(title= 'Year',ticklen= 5,zeroline= False),
              yaxis= dict(title= 'Number of Students',ticklen= 5,zeroline= False), 
                  title='Total students registered')
fig = go.Figure(data = data, layout = layout)
iplot(fig)
trace = go.Bar(
                x = years,
                y = total,
                marker = dict(color = 'light blue',
                             line=dict(color='black',width=1.5)),
                )

data = [trace]
layout = go.Layout(barmode = "group", xaxis= dict(title= 'Year',ticklen= 5,zeroline= False),
              yaxis= dict(title= 'Number of Students',ticklen= 5,zeroline= False), 
                  title='Total students who took SHSAT')
fig = go.Figure(data = data, layout = layout)
iplot(fig)
plt.figure(figsize=(14,8))
plt.title('Economic Need Index')
sns.distplot(df["Economic Need Index"])
plt.show()
data = [
    {
        'x': df["Longitude"],
        'y': df["Latitude"],
        'text': df["School Name"],
        'mode': 'markers',
        'marker': {
            'color': df["Economic Need Index"].mul(100),
            'size': df["School Income Estimate"]/5000,
            'showscale': True,
            'colorscale':'Portland'
        }
    }
]

layout= go.Layout(
    title= 'Economic Need Index',
    xaxis= dict(
        title= 'Longitude'
    ),
    yaxis=dict(
        title='Latitude'
    ))
fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='scatter_hover_labels')
plt.figure(figsize=(14,8))
sns.regplot(x='School Income Estimate', y='Economic Need Index',  data =df)
plt.show()
temp = df["Supportive Environment Rating"].value_counts()
fig = {
  "data": [
    {
      "values": temp.values,
      "labels": temp.index,
      "domain": {"x": [0, 1]},
      "hole": .4,
      "type": "pie"
    },
    
    ],
  "layout": {
        "title":"Supportive Environment Rating",
        "annotations": [
            {
                "font": {
                    "size": 17
                },
                "showarrow": False,
                "text": "Rating",
                "x": 0.5,
                "y": 0.5
            }
            
        ]
    }
}
iplot(fig, filename='donut')
temp = df["Student Achievement Rating"].value_counts()
fig = {
  "data": [
    {
      "values": temp.values,
      "labels": temp.index,
      "domain": {"x": [0, 1]},
      "hole": .4,
      "type": "pie"
    },
    
    ],
  "layout": {
        "title":"Student Achievement Rating",
        "annotations": [
            {
                "font": {
                    "size": 17
                },
                "showarrow": False,
                "text": "Rating",
                "x": 0.5,
                "y": 0.5
            }
            
        ]
    }
}
iplot(fig, filename='donut')
df[target_list].plot.hist(bins=100,alpha=0.8,figsize=(14,5),
                                       title='Average Proficiency for Schools');

#plt.style.use('seaborn-whitegrid')
plt.figure(figsize=(14,17))
sns.countplot(y="School name", data=df1, palette='hls')
plt.title('D5 SHSAT. Number of Notes by School', fontsize=20);
plt.figure(figsize=(14,5))
sns.countplot(y="Year of SHST",hue="Grade level",data=df1, palette='hls')
plt.title('Year Distribution. D5 SHSAT', fontsize=20);
df2 = df.iloc[:,16:36]
#Correlation Matrix
corr = df2.corr()
corr = (corr)

sns.set(font_scale=1.15)
plt.figure(figsize=(14, 10))
sns.heatmap(corr, vmax=.8, linewidths=0.01,
            square=True,annot=True,cmap='YlGnBu',linecolor="black")
plt.title('Correlation between features');
