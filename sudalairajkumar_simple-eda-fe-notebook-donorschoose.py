import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
%matplotlib inline

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999
### Let us read the train file and look at the top few rows ###
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
train_df.head()
train_df.shape
temp_series = train_df["project_is_approved"].value_counts()

labels = (np.array(temp_series.index))
sizes = (np.array((temp_series / temp_series.sum())*100))

trace = go.Pie(labels=labels, values=sizes)
layout = go.Layout(
    title='Project Proposal is Approved'
)
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="ProjectApproval")
### Stacked Bar Chart ###
x_values = train_df["project_grade_category"].value_counts().index.tolist()
y0_values = []
y1_values = []
y_values = []
for val in x_values:
    y1_values.append(np.sum(train_df["project_is_approved"][train_df["project_grade_category"]==val] == 1))
    y0_values.append(np.sum(train_df["project_is_approved"][train_df["project_grade_category"]==val] == 0))
    y_values.append(np.mean(train_df["project_is_approved"][train_df["project_grade_category"]==val]))
    
trace1 = go.Bar(
    x = x_values,
    y = y1_values,
    name='Accepted Proposals'
)
trace2 = go.Bar(
    x = x_values,
    y = y0_values, 
    name='Rejected Proposals'
)

data = [trace1, trace2]
layout = go.Layout(
    title = "Project Grade Distribution",
    barmode='stack',
    width = 1000
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='ProjectGradeCategory')

### Bar chart ###
trace = go.Bar(
    x = x_values,
    y = y_values,
    name='Accepted Proposals'
)
data = [trace]
layout = go.Layout(
    title = "Project acceptance percentage by Project grade",
    width = 1000,
    yaxis=dict(range=[0.7, 0.95])
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='ProjectGradePerc')

### Stacked Bar Chart ###
x_values = train_df["project_subject_categories"].value_counts().index.tolist()
y0_values = []
y1_values = []
y_values = []
for val in x_values:
    y1_values.append(np.sum(train_df["project_is_approved"][train_df["project_subject_categories"]==val] == 1))
    y0_values.append(np.sum(train_df["project_is_approved"][train_df["project_subject_categories"]==val] == 0))
    y_values.append(np.mean(train_df["project_is_approved"][train_df["project_subject_categories"]==val]))
    
trace1 = go.Bar(
    x = x_values,
    y = y1_values,
    name='Accepted Proposals'
)
trace2 = go.Bar(
    x = x_values,
    y = y0_values, 
    name='Rejected Proposals'
)

data = [trace1, trace2]
layout = go.Layout(
    title = "Project Subject Category Distribution",
    barmode='stack',
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='ProjectGradeCategory')

### Bar chart ###
trace = go.Bar(
    x = x_values,
    y = y_values,
    name='Accepted Proposals'
)
data = [trace]
layout = go.Layout(
    title = "Project acceptance percentage by Project Subject Category",
    yaxis=dict(range=[0.6, 0.95])
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='ProjectGradePerc')
### Stacked Bar Chart ###
x_values = train_df["teacher_prefix"].value_counts().index.tolist()
y0_values = []
y1_values = []
y_values = []
for val in x_values:
    y1_values.append(np.sum(train_df["project_is_approved"][train_df["teacher_prefix"]==val] == 1))
    y0_values.append(np.sum(train_df["project_is_approved"][train_df["teacher_prefix"]==val] == 0))
    y_values.append(np.mean(train_df["project_is_approved"][train_df["teacher_prefix"]==val]))
    
trace1 = go.Bar(
    x = x_values,
    y = y1_values,
    name='Accepted Proposals'
)
trace2 = go.Bar(
    x = x_values,
    y = y0_values, 
    name='Rejected Proposals'
)

data = [trace1, trace2]
layout = go.Layout(
    title = "Teacher Prefix Distribution",
    barmode='stack',
    width = 1000
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='ProjectGradeCategory')

### Bar chart ###
trace = go.Bar(
    x = x_values,
    y = y_values,
    name='Accepted Proposals'
)
data = [trace]
layout = go.Layout(
    title = "Project acceptance rate by Teacher Prefix",
    width = 1000,
    yaxis=dict(range=[0.7, 0.95])
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='ProjectGradePerc')
### Stacked Bar Chart ###
x_values = train_df["school_state"].value_counts().index.tolist()
y0_values = []
y1_values = []
y_values = []
for val in x_values:
    y1_values.append(np.sum(train_df["project_is_approved"][train_df["school_state"]==val] == 1))
    y0_values.append(np.sum(train_df["project_is_approved"][train_df["school_state"]==val] == 0))
    y_values.append(np.mean(train_df["project_is_approved"][train_df["school_state"]==val]))
    
trace1 = go.Bar(
    x = x_values,
    y = y1_values,
    name='Accepted Proposals'
)
trace2 = go.Bar(
    x = x_values,
    y = y0_values, 
    name='Rejected Proposals'
)

data = [trace1, trace2]
layout = go.Layout(
    title = "School State Distribution",
    barmode='stack',
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='ProjectGradeCategory')

### Bar chart ###
trace = go.Bar(
    x = x_values,
    y = y_values,
    name='Accepted Proposals'
)
data = [trace]
layout = go.Layout(
    title = "Project acceptance rate by School state",
    yaxis=dict(range=[0.75, 0.9])
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='ProjectGradePerc')
con_df = pd.DataFrame(train_df["school_state"].value_counts()).reset_index()
con_df.columns = ['state_code', 'num_proposals']

scl = [[0.0, 'rgb(242,240,247)'],[0.2, 'rgb(218,218,235)'],[0.4, 'rgb(188,189,220)'],\
            [0.6, 'rgb(158,154,200)'],[0.8, 'rgb(117,107,177)'],[1.0, 'rgb(84,39,143)']]

data = [ dict(
        type='choropleth',
        colorscale = scl,
        autocolorscale = False,
        locations = con_df['state_code'],
        z = con_df['num_proposals'].astype(float),
        locationmode = 'USA-states',
        text = con_df['state_code'],
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            ) ),
        colorbar = dict(
            title = "Num Project Proposals")
        ) ]

layout = dict(
        title = 'Project Proposals by US States<br>(Hover for breakdown)',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
    
fig = dict( data=data, layout=layout )
py.iplot( fig, filename='d3-cloropleth-map' )


### mean acceptance rate ###
con_df = pd.DataFrame(train_df.groupby("school_state")["project_is_approved"].apply(np.mean)).reset_index()
con_df.columns = ['state_code', 'mean_proposals']

scl = [[0.0, 'rgb(242,240,247)'],[0.2, 'rgb(218,218,235)'],[0.4, 'rgb(188,189,220)'],\
            [0.6, 'rgb(158,154,200)'],[0.8, 'rgb(117,107,177)'],[1.0, 'rgb(84,39,143)']]

data = [ dict(
        type='choropleth',
        colorscale = scl,
        autocolorscale = False,
        locations = con_df['state_code'],
        z = con_df['mean_proposals'].astype(float),
        locationmode = 'USA-states',
        text = con_df['state_code'],
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            ) ),
        colorbar = dict(
            title = "Project Proposals Acceptance Rate")
        ) ]

layout = dict(
        title = 'Project Proposals Acceptance Rate by US States<br>(Hover for breakdown)',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
    
fig = dict( data=data, layout=layout )
py.iplot( fig, filename='d3-cloropleth-map' )

train_df["project_submitted_datetime"] = pd.to_datetime(train_df["project_submitted_datetime"])
train_df["date_created"] = train_df["project_submitted_datetime"].dt.date

x_values = train_df["date_created"].value_counts().index.tolist()
y0_values = []
y1_values = []
y_values = []
for val in x_values:
    y1_values.append(np.sum(train_df["project_is_approved"][train_df["date_created"]==val] == 1))
    y0_values.append(np.sum(train_df["project_is_approved"][train_df["date_created"]==val] == 0))
    y_values.append(np.mean(train_df["project_is_approved"][train_df["date_created"]==val]))

trace1 = go.Bar(
    x = x_values,
    y = y1_values,
    name='Accepted Proposals'
)
trace2 = go.Bar(
    x = x_values,
    y = y0_values, 
    name='Rejected Proposals'
)

data = [trace1, trace2]
layout = go.Layout(
    title = "Project Proposal Submission Date Distribution",
    barmode='stack',
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='ProjectGradeCategory')

### Bar chart ###
trace = go.Bar(
    x = x_values,
    y = y_values,
    name='Accepted Proposals'
)
data = [trace]
layout = go.Layout(
    title = "Project acceptance rate by Proposal Submission date",
    yaxis=dict(range=[0.7, 0.95])
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='ProjectGradePerc')

train_df["month_created"] = train_df["project_submitted_datetime"].dt.month

x_values = train_df["month_created"].value_counts().index.tolist()
y0_values = []
y1_values = []
y_values = []
for val in x_values:
    y1_values.append(np.sum(train_df["project_is_approved"][train_df["month_created"]==val] == 1))
    y0_values.append(np.sum(train_df["project_is_approved"][train_df["month_created"]==val] == 0))
    y_values.append(np.mean(train_df["project_is_approved"][train_df["month_created"]==val]))

trace1 = go.Bar(
    x = x_values,
    y = y1_values,
    name='Accepted Proposals'
)
trace2 = go.Bar(
    x = x_values,
    y = y0_values, 
    name='Rejected Proposals'
)

data = [trace1, trace2]
layout = go.Layout(
    title = "Project Proposal Submission Month Distribution",
    barmode='stack',
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='ProjectGradeCategory')

### Bar chart ###
trace = go.Bar(
    x = x_values,
    y = y_values,
    name='Accepted Proposals'
)
data = [trace]
layout = go.Layout(
    title = "Project acceptance rate by Proposal Submission Month",
    yaxis=dict(range=[0.7, 0.95])
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='ProjectGradePerc')
train_df["weekday_created"] = train_df["project_submitted_datetime"].dt.weekday

x_values = train_df["weekday_created"].value_counts().index.tolist()
y0_values = []
y1_values = []
y_values = []
for val in x_values:
    y1_values.append(np.sum(train_df["project_is_approved"][train_df["weekday_created"]==val] == 1))
    y0_values.append(np.sum(train_df["project_is_approved"][train_df["weekday_created"]==val] == 0))
    y_values.append(np.mean(train_df["project_is_approved"][train_df["weekday_created"]==val]))
x_values = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

trace1 = go.Bar(
    x = x_values,
    y = y1_values,
    name='Accepted Proposals'
)
trace2 = go.Bar(
    x = x_values,
    y = y0_values, 
    name='Rejected Proposals'
)

data = [trace1, trace2]
layout = go.Layout(
    title = "Project Proposal Submission Weekday Distribution",
    barmode='stack',
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='ProjectGradeCategory')

### Bar chart ###
trace = go.Bar(
    x = x_values,
    y = y_values,
    name='Accepted Proposals'
)
data = [trace]
layout = go.Layout(
    title = "Project acceptance rate by Proposal Submission Weekday",
    yaxis=dict(range=[0.7, 0.95])
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='ProjectGradePerc')
train_df["hour_created"] = train_df["project_submitted_datetime"].dt.hour

x_values = train_df["hour_created"].value_counts().index.tolist()
y0_values = []
y1_values = []
y_values = []
for val in x_values:
    y1_values.append(np.sum(train_df["project_is_approved"][train_df["hour_created"]==val] == 1))
    y0_values.append(np.sum(train_df["project_is_approved"][train_df["hour_created"]==val] == 0))
    y_values.append(np.mean(train_df["project_is_approved"][train_df["hour_created"]==val]))

trace1 = go.Bar(
    x = x_values,
    y = y1_values,
    name='Accepted Proposals'
)
trace2 = go.Bar(
    x = x_values,
    y = y0_values, 
    name='Rejected Proposals'
)

data = [trace1, trace2]
layout = go.Layout(
    title = "Project Proposal Submission Hour Distribution",
    barmode='stack',
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='ProjectGradeCategory')

### Bar chart ###
trace = go.Bar(
    x = x_values,
    y = y_values,
    name='Accepted Proposals'
)
data = [trace]
layout = go.Layout(
    title = "Project acceptance rate by Proposal Submission Hour",
    yaxis=dict(range=[0.7, 0.95])
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='ProjectGradePerc')
## Reading the data ##
resource_df = pd.read_csv("../input/resources.csv")

## Merging with train and test data ##
train_df = pd.merge(train_df, resource_df, on="id", how='left')
test_df = pd.merge(test_df, resource_df, on="id", how='left')

resource_df.head()
trace = go.Histogram(
    x = np.log1p(train_df["price"]),
    nbinsx = 50,
    opacity=0.75
)
data = [trace]
layout = go.Layout(
    title = "Log Histogram of the prices of project proposal",
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='ProjectGradePerc')
x1_values = np.log1p(train_df["price"][train_df["project_is_approved"]==1])
x0_values = np.log1p(train_df["price"][train_df["project_is_approved"]==0])

trace0 = go.Histogram(
    x = x0_values,
    nbinsx = 50,
    opacity = 0.75,
    name = "Rejected Proposals"
)
trace1 = go.Histogram(
    x = x1_values,
    nbinsx = 50,
    opacity = 0.75,
    name = "Approved Proposal"
)
data = [trace1, trace0]
layout = go.Layout(
    title = "Proposal Approval at different price levels",
    barmode = "overlay"
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='ProjectGradePerc')
x_values = np.sort(train_df["quantity"].value_counts().index.tolist())[:20]
y0_values = []
y1_values = []
y_values = []
for val in x_values:
    y1_values.append(np.sum(train_df["project_is_approved"][train_df["quantity"]==val] == 1))
    y0_values.append(np.sum(train_df["project_is_approved"][train_df["quantity"]==val] == 0))
    y_values.append(np.mean(train_df["project_is_approved"][train_df["quantity"]==val]))

trace1 = go.Bar(
    x = x_values,
    y = y1_values,
    name='Accepted Proposals'
)
trace2 = go.Bar(
    x = x_values,
    y = y0_values, 
    name='Rejected Proposals'
)

data = [trace1, trace2]
layout = go.Layout(
    title = "Project Proposal - Item Quantity Distribution (Till 20)",
    barmode='stack',
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='ProjectGradeCategory')