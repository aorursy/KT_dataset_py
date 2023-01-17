# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plotly import tools
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.figure_factory as ff
from sklearn.cluster import KMeans
import math
import warnings
warnings.filterwarnings('ignore')
# Importing data
df = pd.read_csv('../input/2016 School Explorer.csv')
high_school_df = pd.read_csv('../input/D5 SHSAT Registrations and Testers.csv')
df = df.iloc[:,3:]
df.sample(5)
df["School Income Estimate"] = df["School Income Estimate"].replace('[\$,]', '', regex=True).astype(float)
color_brewer = ['#57B8FF','#B66D0D','#009FB7','#FBB13C','#FE6847','#4FB5A5','#8C9376','#F29F60','#8E1C4A','#85809B','#515B5D','#9EC2BE','#808080','#9BB58E','#5C0029','#151515','#A63D40','#E9B872','#56AA53','#CE6786','#449339','#2176FF','#348427','#671A31','#106B26','008DD5','#034213','#BC2F59','#939C44','#ACFCD9','#1D3950','#9C5414','#5DD9C1','#7B6D49','#8120FF','#F224F2','#C16D45','#8A4F3D','#616B82','#443431','#340F09']
def floater(x):
    return float(x.strip('%'))

df["Percent Asian"] = df["Percent Asian"].astype(str).apply(floater)
df["Percent Black"] = df["Percent Black"].astype(str).apply(floater)
df["Percent Hispanic"] = df["Percent Hispanic"].astype(str).apply(floater)
df["Percent White"] = df["Percent White"].astype(str).apply(floater)
df["Percent Others"] = (df["Percent Black"] + df["Percent Hispanic"] + df["Percent White"] + df["Percent Asian"]).sub(100).mul(-1)
df["Rigorous Instruction %"] = df["Rigorous Instruction %"].astype(str).apply(floater)
df["Collaborative Teachers %"] = df["Collaborative Teachers %"].astype(str).apply(floater)
df["Supportive Environment %"] = df["Supportive Environment %"].astype(str).apply(floater)
df["Effective School Leadership %"] = df["Effective School Leadership %"].astype(str).apply(floater)
df["Strong Family-Community Ties %"] = df["Strong Family-Community Ties %"].astype(str).apply(floater)
df["Trust %"] = df["Trust %"].astype(str).apply(floater)
df["Student Attendance Rate"] = df["Student Attendance Rate"].astype(str).apply(floater)
df["Percent of Students Chronically Absent"] = df["Percent of Students Chronically Absent"].astype(str).apply(floater)
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
figure = ff.create_distplot([df["Economic Need Index"]],['ENI'],bin_size=0.01)
iplot(figure, filename='ENI distplot')
t2 = go.Box(y=df["Economic Need Index"],name="Box plot")
t1 = go.Violin(y=df["Economic Need Index"],name="Violin plot")
fig = tools.make_subplots(rows=1, cols=2, shared_yaxes=True, print_grid=False)

fig.append_trace(t1, 1, 1)
fig.append_trace(t2, 1, 2)

fig['layout'].update(height=600, width=800, title='Economic Need Index distribution')
iplot(fig, filename='ENI Box Violin')
colorscale = ['#7A4579', '#D56073', 'rgb(236,158,105)', (1, 1, 0.2), (0.98,0.98,0.98)]
fig = ff.create_2d_density(
    df["Economic Need Index"], df["School Income Estimate"], colorscale=colorscale,
    hist_color='rgb(255, 237, 222)', point_size=3, title="Economic Need Index vs. School Income Estimate"
)
fig.layout.yaxis.update({'title': 'School Income Estimate'})
fig.layout.xaxis.update({'title': 'Economic Need Index'})
iplot(fig, filename='histogram_subplots')
dframe = [('City', df["City"].value_counts(sort=True).index),('Mean ENI',[round(df["Economic Need Index"][df["City"]==i].mean(),3) for i in list(df["City"].value_counts(sort=True).index)])]
dframe = pd.DataFrame.from_items(dframe)
dframe = dframe.sort_values(['Mean ENI'],ascending=[False],axis=0)
data = [go.Bar(
            x=dframe["City"],
            y=dframe["Mean ENI"],
            text=dframe["Mean ENI"],
            textposition = 'auto',
            marker=dict(
                color='rgb(158,202,225)',
                line=dict(
                    color='rgb(8,48,107)',
                    width=1.5),
            ),
            opacity=0.6
        )]

iplot(data, filename='bar-direct-labels')
trace0 = go.Box(x=df["School Income Estimate"][df["Community School?"]=="Yes"],name="Community School",boxmean=True)
trace1 = go.Box(x=df["School Income Estimate"][df["Community School?"]=="No"],name="Private School",boxmean=True)
data = [trace0, trace1]
layout = go.Layout(
    title = "Box Plot of estimated income of Community and Private schools",
    margin = go.Margin(l=115)
)
fig = go.Figure(data=data,layout=layout)
iplot(fig)
data = [
    {
        'x': df["Latitude"],
        'y': df["Longitude"],
        'text': df["School Name"],
        'mode': 'markers',
        'marker': {
            'color': df["Economic Need Index"].mul(100),
            'size': df["School Income Estimate"].div(5000),
            'showscale': True
        }
    }
]

iplot(data, filename='scatter-colorscale')
data = []
city_list = list(df["City"].value_counts().index)
for i in city_list:
    data.append(
        go.Bar(
          y = [df["Percent Asian"][df["City"] == i].mean(), df["Percent Black"][df["City"] == i].mean(), df["Percent Hispanic"][df["City"] == i].mean(), df["Percent White"][df["City"] == i].mean(), df["Percent Others"][df["City"] == i].mean()],
          x = ['Asian','Black','Hispanic', 'White', 'Others'],
          name = i,
          opacity = 0.6
        )
    )
k=0
fig = tools.make_subplots(rows=15, cols=3, subplot_titles=city_list, print_grid=False)
for i in range(1,16):
    for j in range(1,4):
        fig.append_trace(data[k], i, j)
        k = k + 1
fig['layout'].update(height=2000, title='Average racial distribution in different cities',showlegend=False)
iplot(fig, filename='make-subplots-multiple-with-titles')

fig = ff.create_scatterplotmatrix(df.loc[:,["Economic Need Index","Percent Asian","Percent Black","Percent Hispanic","Percent White","Percent Others"]], index='Economic Need Index', diag='box', size=2, height=800, width=800)
iplot(fig, filename ='Scatterplotmatrix')
data = [
    go.Scatterpolar(
      r = [df["Percent Asian"][df["Community School?"] == "Yes"].mean(), df["Percent Black"][df["Community School?"] == "Yes"].mean(), df["Percent Hispanic"][df["Community School?"] == "Yes"].mean(), df["Percent White"][df["Community School?"] == "Yes"].mean(), df["Percent Others"][df["Community School?"] == "Yes"].mean(), df["Percent Asian"][df["Community School?"] == "Yes"].mean()],
      theta = ['Asian','Black','Hispanic', 'White', 'Others', 'Asian'],
      fill = 'toself',
      name = 'Community School'
    ),
    go.Scatterpolar(
      r = [df["Percent Asian"][df["Community School?"] == "No"].mean(), df["Percent Black"][df["Community School?"] == "No"].mean(), df["Percent Hispanic"][df["Community School?"] == "No"].mean(), df["Percent White"][df["Community School?"] == "No"].mean(), df["Percent Others"][df["Community School?"] == "No"].mean(), df["Percent Asian"][df["Community School?"] == "No"].mean()],
      theta = ['Asian','Black','Hispanic', 'White', 'Others', 'Asian'],
      fill = 'toself',
      name = 'Not Community School'
    )
]

layout = go.Layout(
  polar = dict(
    radialaxis = dict(
      visible = True,
      range = [0, 60]
    )
  ),
  showlegend = True,
  title = "Racial distribution in community and private schools"
)

fig = go.Figure(data=data, layout=layout)
iplot(fig, filename = "radar/multiple")
figure = ff.create_distplot([df["Rigorous Instruction %"]],['RI%'],bin_size=1,colors = ['#F0B100'])
iplot(figure, filename='RI distplot')
trace1 = go.Bar(
    y=df["Rigorous Instruction Rating"].value_counts(sort=True).index,
    x=df["Rigorous Instruction Rating"].value_counts(sort=True).values,
    text=df["Rigorous Instruction Rating"].value_counts(sort=True).values,
    textposition='auto',
    name='Frequency',
    orientation = 'h',
    marker = dict(
        color = 'rgba(246, 78, 139, 0.6)',
        line = dict(
            color = 'rgba(246, 78, 139, 1.0)',
            width = 3)
    )
)
trace2 = go.Bar(
    y=list(df["Rigorous Instruction Rating"].unique()),
    x=[df["Rigorous Instruction %"][df["Rigorous Instruction Rating"] == i].mean() for i in list(df["Rigorous Instruction Rating"].unique())],
    text=[df["Rigorous Instruction %"][df["Rigorous Instruction Rating"] == i].mean() for i in list(df["Rigorous Instruction Rating"].unique())],
    textposition='auto',
    name='Mean',
    orientation = 'h',
    marker = dict(
        color = 'rgba(58, 71, 80, 0.6)',
        line = dict(
            color = 'rgba(58, 71, 80, 1.0)',
            width = 3)
    )
)

trace3 = go.Bar(
    y=list(df["Rigorous Instruction Rating"].unique()),
    x=[df["Rigorous Instruction %"][df["Rigorous Instruction Rating"] == i].median() for i in list(df["Rigorous Instruction Rating"].unique())],
    text=[df["Rigorous Instruction %"][df["Rigorous Instruction Rating"] == i].median() for i in list(df["Rigorous Instruction Rating"].unique())],
    textposition='auto',
    name='Median',
    orientation = 'h',
    marker = dict(
        color = 'rgba(0, 100, 100, 0.6)',
        line = dict(
            color = 'rgba(0, 100, 100, 1.0)',
            width = 3)
    )
)

fig = tools.make_subplots(rows=1, cols=3, print_grid=False, shared_yaxes=True)

fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 2)
fig.append_trace(trace3, 1, 3)

fig['layout'].update(height=400, width=800, title='Statistical analysis of rigorous instruction rating',margin=go.Margin(l=100),yaxis=dict(tickangle=45))
iplot(fig, filename='simple-subplot-with-annotations')
figure = ff.create_distplot([df["Collaborative Teachers %"]],['CT%'],bin_size=1,colors = ['#E84855'])
iplot(figure, filename='CT distplot')
trace1 = go.Bar(
    y=df["Collaborative Teachers Rating"].value_counts(sort=True).index,
    x=df["Collaborative Teachers Rating"].value_counts(sort=True).values,
    text=df["Collaborative Teachers Rating"].value_counts(sort=True).values,
    textposition='auto',
    name='Frequency',
    orientation = 'h',
    marker = dict(
        color = 'rgba(232, 72, 85, 0.6)',
        line = dict(
            color = 'rgba(232, 72, 85, 1.0)',
            width = 3)
    )
)
trace2 = go.Bar(
    y=list(df["Collaborative Teachers Rating"].unique()),
    x=[df["Collaborative Teachers %"][df["Collaborative Teachers Rating"] == i].mean() for i in list(df["Collaborative Teachers Rating"].unique())],
    text=[df["Collaborative Teachers %"][df["Collaborative Teachers Rating"] == i].mean() for i in list(df["Collaborative Teachers Rating"].unique())],
    textposition='auto',
    name='Mean',
    orientation = 'h',
    marker = dict(
        color = 'rgba(255, 155, 113, 0.6)',
        line = dict(
            color = 'rgba(255, 155, 113, 1.0)',
            width = 3)
    )
)

trace3 = go.Bar(
    y=list(df["Collaborative Teachers Rating"].unique()),
    x=[df["Collaborative Teachers %"][df["Collaborative Teachers Rating"] == i].median() for i in list(df["Collaborative Teachers Rating"].unique())],
    text=[df["Collaborative Teachers %"][df["Collaborative Teachers Rating"] == i].median() for i in list(df["Collaborative Teachers Rating"].unique())],
    textposition='auto',
    name='Median',
    orientation = 'h',
    marker = dict(
        color = 'rgba(252, 176, 64, 0.6)',
        line = dict(
            color = 'rgba(252, 176, 64, 1.0)',
            width = 3)
    )
)

fig = tools.make_subplots(rows=1, cols=3, print_grid=False, shared_yaxes=True)

fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 2)
fig.append_trace(trace3, 1, 3)

fig['layout'].update(height=400, width=800, title='Statistical analysis of collaborative teachers rating',margin=go.Margin(l=100),yaxis=dict(tickangle=45))
iplot(fig, filename='simple-subplot-with-annotations')
figure = ff.create_distplot([df["Average ELA Proficiency"],df["Average Math Proficiency"]],['ELA proficiency','Math proficiency'],bin_size=0.05,colors = ['#093A3E','#64E9EE'])
iplot(figure, filename='CT distplot')
colorscale = ['#1D7874', '#1C2541', '#E4572E', "#B5303B", '#DCF2F2']
fig = ff.create_2d_density(
    df["Average ELA Proficiency"], df["Average Math Proficiency"], colorscale=colorscale,
    hist_color='#1D7874', point_size=3, title="ELA proficiency vs Math proficiency"
)
fig.layout.yaxis.update({'title': 'Math proficiency'})
fig.layout.xaxis.update({'title': 'ELA proficiency'})
iplot(fig, filename='histogram_subplots')
figure = ff.create_distplot([df["Supportive Environment %"]],['Supportive Environment'],bin_size=1,colors = ['#E88A20'])
iplot(figure, filename='SE distplot')
trace1 = go.Bar(
    y=df["Supportive Environment Rating"].value_counts(sort=True).index,
    x=df["Supportive Environment Rating"].value_counts(sort=True).values,
    text=df["Supportive Environment Rating"].value_counts(sort=True).values,
    textposition='auto',
    name='Frequency',
    orientation = 'h',
    marker = dict(
        color = 'rgba(62, 66, 75, 0.6)',
        line = dict(
            color = 'rgba(62, 66, 75, 1)',
            width = 3)
    )
)
trace2 = go.Bar(
    y=list(df["Supportive Environment Rating"].unique()),
    x=[df["Supportive Environment %"][df["Supportive Environment Rating"] == i].mean() for i in list(df["Supportive Environment Rating"].unique())],
    text=[df["Supportive Environment %"][df["Supportive Environment Rating"] == i].mean() for i in list(df["Supportive Environment Rating"].unique())],
    textposition='auto',
    name='Mean',
    orientation = 'h',
    marker = dict(
        color = 'rgba(232, 138, 32, 0.6)',
        line = dict(
            color = 'rgba(232, 138, 32, 1)',
            width = 3)
    )
)

trace3 = go.Bar(
    y=list(df["Supportive Environment Rating"].unique()),
    x=[df["Supportive Environment %"][df["Supportive Environment Rating"] == i].median() for i in list(df["Supportive Environment Rating"].unique())],
    text=[df["Supportive Environment %"][df["Supportive Environment Rating"] == i].median() for i in list(df["Supportive Environment Rating"].unique())],
    textposition='auto',
    name='Median',
    orientation = 'h',
    marker = dict(
        color = 'rgba(243, 186, 50, 0.6)',
        line = dict(
            color = 'rgba(243, 186, 50, 1)',
            width = 3)
    )
)

fig = tools.make_subplots(rows=1, cols=3, print_grid=False, shared_yaxes=True)

fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 2)
fig.append_trace(trace3, 1, 3)

fig['layout'].update(height=400, width=800, title='Statistical analysis of Supportive Environment rating',margin=go.Margin(l=100),yaxis=dict(tickangle=45))
iplot(fig, filename='simple-subplot-with-annotations')
figure = ff.create_distplot([df["Effective School Leadership %"]],['Effective School Leadership'],bin_size=1,colors = ['#FFE066'])
iplot(figure, filename='ESL distplot')
trace1 = go.Bar(
    y=df["Effective School Leadership Rating"].value_counts(sort=True).index,
    x=df["Effective School Leadership Rating"].value_counts(sort=True).values,
    text=df["Effective School Leadership Rating"].value_counts(sort=True).values,
    textposition='auto',
    name='Frequency',
    orientation = 'h',
    marker = dict(
        color = 'rgba(215, 38, 56, 0.6)',
        line = dict(
            color = 'rgba(215, 38, 56, 1)',
            width = 3)
    )
)
trace2 = go.Bar(
    y=list(df["Effective School Leadership Rating"].unique()),
    x=[df["Effective School Leadership %"][df["Effective School Leadership Rating"] == i].mean() for i in list(df["Effective School Leadership Rating"].unique())],
    text=[df["Effective School Leadership %"][df["Effective School Leadership Rating"] == i].mean() for i in list(df["Effective School Leadership Rating"].unique())],
    textposition='auto',
    name='Mean',
    orientation = 'h',
    marker = dict(
        color = 'rgba(244, 157, 55, 0.6)',
        line = dict(
            color = 'rgba(244, 157, 55, 1)',
            width = 3)
    )
)

trace3 = go.Bar(
    y=list(df["Effective School Leadership Rating"].unique()),
    x=[df["Effective School Leadership %"][df["Effective School Leadership Rating"] == i].median() for i in list(df["Effective School Leadership Rating"].unique())],
    text=[df["Effective School Leadership %"][df["Effective School Leadership Rating"] == i].median() for i in list(df["Effective School Leadership Rating"].unique())],
    textposition='auto',
    name='Median',
    orientation = 'h',
    marker = dict(
        color = 'rgba(65, 105, 225, 0.6)',
        line = dict(
            color = 'rgba(65, 105, 225, 1)',
            width = 3)
    )
)

fig = tools.make_subplots(rows=1, cols=3, print_grid=False, shared_yaxes=True)

fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 2)
fig.append_trace(trace3, 1, 3)

fig['layout'].update(height=400, width=800, title='Statistical analysis of Effective School Leadership rating',margin=go.Margin(l=100),yaxis=dict(tickangle=45))
iplot(fig, filename='simple-subplot-with-annotations')
figure = ff.create_distplot([df["Strong Family-Community Ties %"]],['Strong Family-Community Ties'],bin_size=1,colors = ['#59CD90'])
iplot(figure, filename='SFCT distplot')
trace1 = go.Bar(
    y=df["Strong Family-Community Ties Rating"].value_counts(sort=True).index,
    x=df["Strong Family-Community Ties Rating"].value_counts(sort=True).values,
    text=df["Strong Family-Community Ties Rating"].value_counts(sort=True).values,
    textposition='auto',
    name='Frequency',
    orientation = 'h',
    marker = dict(
        color = 'rgba(249, 12, 74, 0.6)',
        line = dict(
            color = 'rgba(249, 12, 74, 1)',
            width = 3)
    )
)
trace2 = go.Bar(
    y=list(df["Strong Family-Community Ties Rating"].unique()),
    x=[df["Strong Family-Community Ties %"][df["Strong Family-Community Ties Rating"] == i].mean() for i in list(df["Strong Family-Community Ties Rating"].unique())],
    text=[df["Strong Family-Community Ties %"][df["Strong Family-Community Ties Rating"] == i].mean() for i in list(df["Strong Family-Community Ties Rating"].unique())],
    textposition='auto',
    name='Mean',
    orientation = 'h',
    marker = dict(
        color = 'rgba(75, 183, 236, 0.6)',
        line = dict(
            color = 'rgba(75, 183, 236, 1)',
            width = 3)
    )
)

trace3 = go.Bar(
    y=list(df["Strong Family-Community Ties Rating"].unique()),
    x=[df["Strong Family-Community Ties %"][df["Strong Family-Community Ties Rating"] == i].median() for i in list(df["Strong Family-Community Ties Rating"].unique())],
    text=[df["Strong Family-Community Ties %"][df["Strong Family-Community Ties Rating"] == i].median() for i in list(df["Strong Family-Community Ties Rating"].unique())],
    textposition='auto',
    name='Median',
    orientation = 'h',
    marker = dict(
        color = 'rgba(162, 59, 114, 0.6)',
        line = dict(
            color = 'rgba(162, 59, 114, 1)',
            width = 3)
    )
)

fig = tools.make_subplots(rows=1, cols=3, print_grid=False, shared_yaxes=True)

fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 2)
fig.append_trace(trace3, 1, 3)

fig['layout'].update(height=400, width=800, title='Statistical analysis of Strong Family-Community Ties rating',margin=go.Margin(l=100),yaxis=dict(tickangle=45))
iplot(fig, filename='simple-subplot-with-annotations')
figure = ff.create_distplot([df["Trust %"]],['Trust'],bin_size=1,colors = ['#453E54'])
iplot(figure, filename='Trust distplot')
trace1 = go.Bar(
    y=df["Trust Rating"].value_counts(sort=True).index,
    x=df["Trust Rating"].value_counts(sort=True).values,
    text=df["Trust Rating"].value_counts(sort=True).values,
    textposition='auto',
    name='Frequency',
    orientation = 'h',
    marker = dict(
        color = 'rgba(255, 127, 80, 0.6)',
        line = dict(
            color = 'rgba(255, 127, 80, 1)',
            width = 3)
    )
)
trace2 = go.Bar(
    y=list(df["Trust Rating"].unique()),
    x=[df["Trust %"][df["Trust Rating"] == i].mean() for i in list(df["Trust Rating"].unique())],
    text=[df["Trust %"][df["Trust Rating"] == i].mean() for i in list(df["Trust Rating"].unique())],
    textposition='auto',
    name='Mean',
    orientation = 'h',
    marker = dict(
        color = 'rgba(240, 101, 67, 0.6)',
        line = dict(
            color = 'rgba(240, 101, 67, 1)',
            width = 3)
    )
)

trace3 = go.Bar(
    y=list(df["Trust Rating"].unique()),
    x=[df["Trust %"][df["Trust Rating"] == i].median() for i in list(df["Trust Rating"].unique())],
    text=[df["Trust %"][df["Trust Rating"] == i].median() for i in list(df["Trust Rating"].unique())],
    textposition='auto',
    name='Median',
    orientation = 'h',
    marker = dict(
        color = 'rgba(255, 180, 30, 0.6)',
        line = dict(
            color = 'rgba(255, 180, 30, 1)',
            width = 3)
    )
)

fig = tools.make_subplots(rows=1, cols=3, print_grid=False, shared_yaxes=True)

fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 2)
fig.append_trace(trace3, 1, 3)

fig['layout'].update(height=400, width=800, title='Statistical analysis of Trust rating',margin=go.Margin(l=100),yaxis=dict(tickangle=45))
iplot(fig, filename='simple-subplot-with-annotations')
d = df.loc[:,["Economic Need Index","School Income Estimate","Student Attendance Rate","Percent of Students Chronically Absent","Percent Asian","Percent Black","Percent Hispanic","Percent White","Rigorous Instruction %","Collaborative Teachers %","Supportive Environment %","Effective School Leadership %","Strong Family-Community Ties %","Trust %"]]
d = d.convert_objects(convert_numeric=True)
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(d)
    wcss.append(kmeans.inertia_)

trace = go.Scatter(
    x = [i for i in range(1,11)],
    y = wcss
)
data = [trace]
iplot(data, filename='elbow-line')
kmeans = KMeans(n_clusters=4,init='k-means++',max_iter=300,n_init=10,random_state=0) 
y_kmeans = kmeans.fit_predict(d)
d = d.as_matrix(columns=None)
trace0 = go.Scatter(
    x = d[y_kmeans == 0,0],
    y = d[y_kmeans == 0,1],
    mode = 'markers',
    name = 'Cluster 1'
)
trace1 = go.Scatter(
    x = d[y_kmeans == 1,0],
    y = d[y_kmeans == 1,1],
    mode = 'markers',
    name = 'Cluster 2'
)
trace2 = go.Scatter(
    x = d[y_kmeans == 2,0],
    y = d[y_kmeans == 2,1],
    mode = 'markers',
    name = 'Cluster 3'
)
trace3 = go.Scatter(
    x = d[y_kmeans == 3,0],
    y = d[y_kmeans == 3,1],
    mode = 'markers',
    name = 'Cluster 4'
)
data = [trace0, trace1, trace2, trace3]
iplot(data, filename='line-mode')