import numpy as np
import pandas as pd
import seaborn as sns
import math

sns.set()
import colorlover as cl

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly as py
import plotly.graph_objs as go
from plotly import tools

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import networkx as nx

init_notebook_mode(connected=True)
pd.options.mode.chained_assignment = None
df_multiple_choice = pd.read_csv('../input/multipleChoiceResponses.csv',dtype='str',skiprows=[1])
dds = df_multiple_choice[df_multiple_choice['Q26']=='Definitely yes']
df_reference = pd.read_csv('../input/multipleChoiceResponses.csv',dtype='str',nrows=2,header=None)
print ("""Out of the""",len(df_multiple_choice),"""respondents,""", len(dds), 
       """says that they are DEFINITELY DATA SCIENTISTS, \n that is""", 
       np.round(len(dds)*100/len(df_multiple_choice),2),"""% of the respondents""")
definitely_dds = (pd.DataFrame(pd.value_counts(df_multiple_choice.Q26)))
data = [go.Bar(
            x=definitely_dds.index,
            y=np.round((definitely_dds['Q26'].values)*100/len(df_multiple_choice),2),
            marker=dict(
            color=cl.scales['5']['seq']['RdPu'])
    )]
layout = go.Layout(
            title='Do you consider yourself a Data Scientist?',
            yaxis=dict(
                title='% of DDS'),
            autosize=False,
            width=700,
            height=500,
            annotations=[
                dict(
                    x=1,
                    y=19.63,
                    xref='x',
                    yref='y',
                    text='DEFINITELY DATA SCIENTISTS - Our Target Group',
                    showarrow=True,
                    arrowhead=7,
                    ax=0,
                    ay=-40
                )]
)

fig = go.Figure(data=data, layout=layout)
py.offline.iplot(fig)
age_distribution = (pd.DataFrame(pd.value_counts(dds.Q1)))
data = [go.Bar(
            x=age_distribution.index,
            y=np.round((age_distribution['Q1'].values)*100/len(dds),2),
            marker=dict(
            color=cl.scales['4']['div']['Spectral'])
    )]
layout = go.Layout(
            title='Gender',
            yaxis=dict(
                title='% of DDS'),
            xaxis=dict(
            title = 'Gender'
            ),
            autosize=False,
            width=700,
            height=500
)

fig = go.Figure(data=data, layout=layout)
py.offline.iplot(fig)
df_usa = dds[dds['Q3']=='United States of America']

age_distribution = (pd.DataFrame(pd.value_counts(df_usa.Q1)))
data = [go.Bar(
            x=age_distribution.index,
            y=np.round((age_distribution['Q1'].values)*100/len(df_usa),2),
            marker=dict(
            color=cl.scales['4']['div']['Spectral'])
    )]
layout = go.Layout(
            title='Gender - DDS from USA',
            yaxis=dict(
                title='% of respondents'),
            xaxis=dict(
            title = 'Gender'
            ),
            autosize=False,
            width=700,
            height=500
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
print ('We have respondents from',len(set(dds['Q3'])), 'countries')
location = (pd.DataFrame(pd.value_counts(dds.Q3)))
data = [ dict(
        type = 'choropleth',
        locations = location.index,
        locationmode = 'country names',
        z = np.round(location['Q3'].values*100/len(dds),2),
        text = location.index,
        colorscale = 'Virdis',
        autocolorscale = False,
        reversescale = True,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick = False,
            ticksuffix = '%',
            title = '% of DDS'),
      ) ]

layout = dict(
    title = 'Kaggle 2018 Survey - DefinitelyDataScientists',
    geo = dict(
        showframe = False,
        showcoastlines = False,
        projection = dict(
            type = 'Mercator'
        )
    ),
    autosize=True
)

fig = dict( data=data, layout=layout )
py.offline.iplot(fig, validate=False)

percentage_us = np.round((len(df_multiple_choice[df_multiple_choice['Q3']=='United States of America'])*100/len(df_multiple_choice)),2)
percentage_indians = np.round((len(df_multiple_choice[df_multiple_choice['Q3']=='India'])*100/len(df_multiple_choice)),2)
age = dds.groupby(['Q2','Q1']).size().reset_index()

data = []

color_list = cl.scales['4']['div']['Spectral']
n = 0
for x in ['Male','Female','Prefer not to say','Prefer to self-describe']:
    tracen = go.Bar(
    x=age[age['Q1']==x]['Q2'].values,
    y=np.round(age[age['Q1']==x][0].values*100/len(dds),2),
    name=x,
    marker=dict(color=color_list[n])
)
    n = n+1
    data.append(tracen)

layout = go.Layout(
    barmode='group',
    title='Age Range ',
            yaxis=dict(
                title='% of DDS'),
            xaxis=dict(
            title = 'Age Range'
            ),
            autosize=False,
            width=800,
            height=500
)

fig = go.Figure(data=data, layout=layout)
py.offline.iplot(fig, filename='grouped-bar')
current_role = (pd.DataFrame(pd.value_counts(dds.Q6)))
data = [go.Bar(
            x=current_role.index,
            y=np.round((current_role['Q6'].values)*100/len(dds),2),
            marker=dict(
            color=cl.scales['11']['div']['Spectral']),
    )]
layout = go.Layout(
            title='Current Role',
            yaxis=dict(
                title='% of DDS',
                showticklabels=True,
                tickangle=0,
                tickfont=dict(
                    family='Old Standard TT, serif',
                    size=12,
                    color='black'
                )),
            xaxis=dict(
                title = '',
                showticklabels=True,
                tickangle=30,
                tickfont=dict(
                    size=10,
                    color='black'
                )
            ),
            autosize=False,
            width=700,
            height=500
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
industry = (pd.DataFrame(pd.value_counts(dds.Q7)))
data = [go.Bar(
            x=industry.index,
            y=np.round((industry['Q7'].values)*100/len(dds),2),
            marker=dict(
            color=cl.scales['11']['div']['Spectral']),
    )]
layout = go.Layout(
            title='Industry',
            yaxis=dict(
                title='% of DDS',
                showticklabels=True,
                tickangle=0,
                tickfont=dict(
                    family='Old Standard TT, serif',
                    size=12,
                    color='black'
                )),
            xaxis=dict(
                title = '',
                showticklabels=True,
                tickangle=30,
                tickfont=dict(
                    size=10,
                    color='black'
                )
            ),
            autosize=False,
            width=700,
            height=500
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
top_5_roles = (list(current_role.index))[0:5]
top_5_industry = (list(industry.index))[0:5]
dds_role_indus = dds[(dds['Q6'].isin(top_5_roles)) & (dds['Q7'].isin(top_5_industry))]
role_indus = dds_role_indus.groupby(['Q6','Q7']).size().reset_index()

data = []

color_list = cl.scales['10']['div']['Spectral']
n = 0
for x in top_5_industry:
    tracen = go.Bar(
    x=role_indus[role_indus['Q7']==x]['Q6'].values,
    y=np.round(role_indus[role_indus['Q7']==x][0].values*100/len(dds_role_indus),2),
    name=x,
    marker=dict(color=color_list[n])
)
    n = n+1
    data.append(tracen)

layout = go.Layout(
    barmode='group',
    title='Job Role and Industry of DDS',
            yaxis=dict(
                title='% of DDS'),
            xaxis=dict(
                showticklabels=True,
                tickangle=10
            ),
            autosize=False,
            width=800,
            height=500
)

fig = go.Figure(data=data, layout=layout)
py.offline.iplot(fig, filename='grouped-bar')
def getstartsal(x):
    if (x=='I do not wish to disclose my approximate yearly compensation') | (x=='Do not wish to disclose'):
        return -1
    else:
        y = str(str(x).split('-')[0].replace('+','').replace(',',''))
        if y=='nan':
            return None
        else:
            return int(y)

def renamedisclose(x):
    if x=='I do not wish to disclose my approximate yearly compensation':
        return 'Do not wish to disclose'
    else:
        return x
dds['salary_start'] = dds['Q9'].apply(getstartsal)
dds['Q9']  = dds['Q9'].apply(renamedisclose)
dds['experience_start'] = dds['Q8'].apply(getstartsal)
dds_ds_ct = dds[(dds['Q6']=='Data Scientist') & (dds['Q7']=='Computers/Technology')]
dds_ds_ct = dds_ds_ct.sort_values(by=['experience_start'])

trace1 = go.Box(
    y=dds_ds_ct.Q8.values,
    name = 'Experience',
    marker = dict(
        color = cl.scales['5']['seq']['RdPu'][4]
    ),
    showlegend=False
)

dds_ds_ct = dds_ds_ct.sort_values(by=['salary_start'])

trace2 = go.Box(
    y=dds_ds_ct.Q9.values,
    name = 'Compensation',
    marker = dict(
         color = cl.scales['5']['seq']['RdPu'][2]
    ),
    showlegend=False
)

fig = tools.make_subplots(rows=1, cols=2, print_grid=False)
fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 2)

fig['layout']['yaxis2'].update(title = 'Compensation in USD',tickfont=dict(
                    family='Old Standard TT, serif',
                    size=11,
                    color='black'
                ))
fig['layout']['yaxis1'].update(title = 'Experience in Years',tickfont=dict(
                    family='Old Standard TT, serif',
                    size=11,
                    color='black'
                ))
fig['layout'].update(autosize=False, width=800, height=500)
iplot(fig)
def getsal_category(x):
    if (x == 'Do not wish to disclose'):
        return 'Undisclosed'
    else:
        sal_start = (str(x).split('-')[0])
        sal_start = sal_start.replace(',','')
        sal_start = sal_start.replace('+','')
        if sal_start=='nan':
            return None
        else:
            sal_start = int(sal_start)
            if(sal_start<=20):
                return 'Low Compensation'
            elif(sal_start>20) & (sal_start<=50):
                return 'Moderate Compensation'
            elif(sal_start>50):
                return 'High Compensation'

def getexp_category(x):
    exp_start = str(x).split('-')[0]
    exp_start = exp_start.replace('+','')
    if exp_start=='nan':
        return None
    else:
        exp_start = int(exp_start)
        if(exp_start<2):
            return 'Beginner'
        elif(exp_start>=2) & (exp_start<=4):
            return 'Intermediate'
        elif(exp_start>4):
            return 'Experienced'
        
dds_ds_ct['experience'] = dds_ds_ct['Q8'].apply(getexp_category)
dds_ds_ct['compensation'] = dds_ds_ct['Q9'].apply(getsal_category)
degree = dds_ds_ct.groupby(['Q4','compensation']).size().reset_index()

color_list = cl.scales['6']['div']['RdYlBu']
n=0
data = []
for x in set(degree['compensation']):
    tracen = go.Bar(
    x=degree[degree['compensation']==x]['Q4'].values,
    y=np.round(degree[degree['compensation']==x][0].values*100/len(dds_ds_ct),2),
    name=x,
    marker=dict(color=color_list[n])
)
    n = n+1
    data.append(tracen)

layout = go.Layout(
            title='Highest degree attained and the Compensation<br> of DDS from Computers/Technology Sector',
            yaxis=dict(
                title='% of DDS'),
            xaxis=dict(
                showticklabels=True,
                tickangle=15
            ),
            autosize=False,
            width=750,
            height=500
)
    
fig = go.Figure(data=data, layout=layout)
py.offline.iplot(fig, filename='grouped-bar')
degree = dds_ds_ct.groupby(['experience','compensation']).size().reset_index()

color_list = cl.scales['6']['div']['RdYlBu']
n=0
data = []
for x in set(degree['compensation']):
    tracen = go.Bar(
    x=degree[degree['compensation']==x]['experience'].values,
    y=np.round(degree[degree['compensation']==x][0].values*100/len(dds_ds_ct),2),
    name=x,
    marker=dict(color=color_list[n])
)
    n = n+1
    data.append(tracen)

layout = go.Layout(
            yaxis=dict(
                title='% of DDS'),
            xaxis=dict(
                showticklabels=True,
                tickangle=0
            ),
            title = 'Experience and the Compensation<br> of DDS from Computers/Technology Sector',
            autosize=False,
            width=750,
            height=500
)
    
fig = go.Figure(data=data, layout=layout)
py.offline.iplot(fig, filename='grouped-bar')
source = []
target = []
value = []
label = []
from bokeh.palettes import viridis

label=list(dds_ds_ct['Q1'].unique())+list(dds_ds_ct['Q4'].unique())+list(dds_ds_ct['experience'].unique())+list(['High Compensation','Moderate Compensation','Low Compensation','Undisclosed'])
colors_sankey = viridis(len(label))
for gender in dds_ds_ct['Q1'].unique():
    value_count  = dds_ds_ct[dds_ds_ct['Q1']==gender].groupby(['Q4']).size().reset_index()[0].values
    degree = dds_ds_ct[dds_ds_ct['Q1']==gender].groupby(['Q4']).size().reset_index()['Q4'].values
    
    for d in range(0,len(degree)):
        gender_val = label.index(gender)
        degree_val = label.index(degree[d])
        source.append(gender_val)
        target.append(degree_val)
        value.append(value_count[d])

for degree in dds_ds_ct['Q4'].unique():
    experience = dds_ds_ct[dds_ds_ct['Q4']==degree].groupby(['experience']).size().reset_index()['experience'].values
    value_count  = dds_ds_ct[dds_ds_ct['Q4']==degree].groupby(['experience']).size().reset_index()[0].values
    
    for e in range(0,len(experience)):
        degree_val = label.index(degree)
        exp_val = label.index(experience[e])
        source.append(degree_val)
        target.append(exp_val)
        value.append(value_count[e])

for experience in dds_ds_ct['experience'].unique():
    compensation = dds_ds_ct[dds_ds_ct['experience']==experience].groupby(['compensation','salary_start']).size().reset_index()['compensation'].values
    value_count  = dds_ds_ct[dds_ds_ct['experience']==experience].groupby(['compensation','salary_start']).size().reset_index()[0].values
    
    for c in range(0,len(compensation)):
        exp_val = label.index(experience)
        comp_val = label.index(compensation[c])
        source.append(exp_val)
        target.append(comp_val)
        value.append(value_count[c])
data = dict(
    type='sankey',
    node = dict(
        pad = 10,
        thickness = 15,
        line = dict(
        color = "black",
        width = 0.5
      ),
        label = label,
        color=colors_sankey,
    ),
    link = dict(
      source = source,
      target = target,
      value = value
  ))

layout =  dict(
    font = dict(
      size = 10
    )
)

fig = dict(data=[data], layout=layout)

iplot(fig, validate=False)
def get_question(x):
    col_list = []
    for cols in dds_ds_ct.columns:
        array_names = cols.split('_')
        if (array_names[0]==x):
            if (array_names[len(array_names)-1]!='TEXT'):
                col_list.append(cols)
    return col_list

def get_count_question(x):
    dds_question = dds_ds_ct[get_question(x)]
    count_dds = pd.DataFrame(dds_question.count()).reset_index()
    options = []
    for c in dds_question.columns:
        list_options = (set(dds_question[c]))
        for opt in list_options:
            if str(opt)!='nan':
                options.append(opt)
    count_dds['question'] = options
    count_dds['count'] = count_dds[0]
    return count_dds[['question','count']]

def get_combo_answers(x):
    df_q11_dummies = pd.DataFrame(pd.get_dummies(dds_ds_ct[get_question(x)])).reset_index()
    cols_dummies1 = list(df_q11_dummies.columns)
    if "index" in str(cols_dummies1):
        cols_dummies1.remove("index")
    df_question = pd.DataFrame(df_q11_dummies.groupby(cols_dummies1).count()).reset_index().sort_values(by='index',ascending=False)
    df_question['count'] = df_question['index']
    df_question = df_question.drop(['index'],axis=1)
    return df_question
df_q11 = get_count_question('Q11').sort_values(by='count',ascending=False)
data = [go.Bar(
            x=['Build <br> prototypes to <br> explore applying <br> machine learning <br>to new areas', 
               'Analyze <br> and understand <br> data to <br>influence product <br>or business <br>decisions', 
               'Build <br> and/or run a <br> machine learning <br> service that <br> operationally improves <br> my product or workflows', 
               'Do research <br> that advances the <br> state of the art <br> of machine learning', 
               'Build and/or <br> run the data <br> infrastructure <br> that my business uses for <br> storing, analyzing, and <br> operationalizing data', 
               'Other', 'None of these <br>  activities <br> are an important <br> part of my role at work'],
            y=df_q11['count'],
             marker=dict(
                color='MidnightBlue',
                opacity=0.9
            ),
    
    )]

layout = go.Layout(
    title = 'Activities that are important part of the DefinitelyDataScientists role at work!',
    xaxis=dict(
        tickangle=0,
        tickfont=dict(
            size=9,
            color='black'
        )
    ),
    yaxis=dict(
        title='Number of respondents',
        titlefont=dict(
            size=12,
            color='black'
        ),
        tickfont=dict(
            size=12,
            color='rgb(107, 107, 107)'
        )
    )
)

fig = go.Figure(data=data, layout=layout)
py.offline.iplot(fig)
df_q11_groups = get_combo_answers('Q11')
df_q11_groups.head(10)
type_data = (pd.DataFrame(pd.value_counts(dds_ds_ct.Q32)))
data = [go.Bar(
            x=type_data.index,
            y=np.round((type_data['Q32'].values)*100/len(dds_ds_ct),2),
            marker=dict(
            color=cl.scales['11']['div']['Spectral']),
    )]
layout = go.Layout(
            title='Type of Data used',
            yaxis=dict(
                title='% of DDS',
                showticklabels=True,
                tickangle=0,
                
                tickfont=dict(
                    family='Old Standard TT, serif',
                    size=12,
                    color='black'
                )),
            xaxis=dict(
                title = '',
                showticklabels=True,
                tickangle=15,
                tickfont=dict(
                    size=11,
                    color='black'
                )
            ),
            autosize=False,
            width=850,
            height=500
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
#get_question('Q17')
dds_tools = dds_ds_ct[['Q32','Q17','Q20']].reset_index()
dds_tools_count = pd.DataFrame(dds_tools.groupby(['Q32','Q17','Q20']).count().sort_values(by=['index'],ascending=False)).reset_index()
dds_tools_count['count'] = dds_tools_count['index']
dds_tools_count = dds_tools_count.drop(['index'],axis=1)
dds_tools_count[dds_tools_count['Q32']=='Time Series Data']