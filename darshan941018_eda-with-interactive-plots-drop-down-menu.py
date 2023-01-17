import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import missingno as msno

pd.options.mode.chained_assignment = None



from IPython.display import HTML



import plotly.plotly as py

import plotly.graph_objs as go

from plotly import tools

from plotly.graph_objs import *

from plotly.offline import iplot, init_notebook_mode

init_notebook_mode()
df=pd.read_csv('../input/Reveal_EEO1_for_2016.csv')

print("Lets take a look at what our data  looks like")

df.head()
df['count'].replace(to_replace='na',value=0,inplace=True)

df['count']=df['count'].astype(int)
d=df.groupby(['company']).agg({'count': lambda x: sum((x).astype(int))})

plt.figure(figsize=(10,8))

sns.set_style('white')

sns.barplot(x=d.index.get_values(),y=d['count'],palette='viridis')

plt.title('Number of employees by company',size=16)

plt.ylabel('Number of employees',size=14)

plt.xlabel('Company',size=14)

plt.yticks(size=14)

plt.xticks(size=14,rotation=90)

sns.despine()

plt.show()
labels = df.groupby(['gender']).agg({'count':sum}).index.get_values()

values = df.groupby(['gender']).agg({'count':sum})['count'].values

trace = go.Pie(labels=labels, values=values,textinfo='label+value+percent')

layout=go.Layout(title='Distribution of Female and Male Employee')

data=[trace]





fig = dict(data=data,layout=layout)

iplot(fig, filename='Distribution of Female and Male Employee')
d=df.groupby(['gender','company']).agg({'count':sum}).reset_index()

trace1 = go.Bar(

    x=d[d.gender=='male']['company'],

    y=d[d.gender=='male']['count'],

    name='Males'

)

trace2 = go.Bar(

    x=d[d.gender=='female']['company'],

    y=d[d.gender=='female']['count'],

    name='Females'

)

data = [trace1, trace2]

layout = go.Layout(

    barmode='group',title='Distribution of Male and Female Employees by Company')





fig = dict(data=data, layout=layout)

iplot(fig, filename='Distribution of Male and Female Employees by Company')
d=d.groupby(['company','gender']).agg({'count':sum})

d=d.unstack()

d=d['count']

d=d.iloc[:,:].apply(lambda x: (x/x.sum())*100,axis=1)

d['Disparity']=d['male']/d['female']

d.sort_values(by='Disparity',inplace=True,ascending=False)

d.columns=['Female %','Male %','Disparity']
d
trace1 = go.Bar(

    x=d.index.get_values(),

    y=d['Disparity'],text=np.round(d['Disparity'],2),textposition='auto'

)









data = [trace1]

layout = go.Layout(

    barmode='group',title='Disparity in Number of Male and Female Employees')



fig = dict(data=data, layout=layout)

iplot(fig, filename='Disparity in Number of Male and Female Employees')
y=df.groupby(['company','race','gender']).agg({'count':sum}).reset_index()
from collections import deque



## We have to build a list of true/false, in order to support a dropdown menu

## The for loop right shifts True by 1 position

visibility=[True]+[False]*((y.company.nunique()+1))

d=[0]*((y.company.nunique()+3))

for i in range(0,len(d)):

    a=deque(visibility)

    a.rotate(i) 

    d[i]=list(a)



a={}

al=[]

data=[]



unique_company=y.company.unique()

for i in range(0,22):

    m=y[y.company==unique_company[i]]

    data.append(Bar(x=m['race'].unique(),y=m[m.gender=='male']['count'].values+m[m.gender=='female']['count'].values))

    max_annotations=[]

    xcord=np.arange(0,m['race'].nunique())

    for j in range(0,len(m['race'].unique())):

        max_annotations.append([dict(x=xcord[j], y=m[m.gender=='male']['count'].values[j]+m[m.gender=='female']['count'].values[j],

                       xref='x', yref='y',text='<b>Males:</b> '+str(m[m.gender=='male']['count'].values[j])

                       +"<br><b>Females:</b> "+str(m[m.gender=='female']['count'].values[j]))]) 

    al.append(dict(label=unique_company[i],method='update',args=[{'visible':d[i]},{'title':"<b>"+unique_company[i]+"</b><br>Total Employees:"+str(sum(m['count'])),

                                                                                   'annotations':max_annotations[0]+max_annotations[1]+max_annotations[2]+max_annotations[3]

                                                                                  +max_annotations[4]+max_annotations[5]+max_annotations[6]}]))

    



    



data=Data(data)

updatemenus=list([

        dict(

            x=0,

            y=1,

            xanchor='left',

            active=0,

            yanchor='top',

            buttons=al,

        )

    ])



layout = Layout(updatemenus=updatemenus,showlegend=False,)



fig = dict(data=data,layout=layout)

iplot(fig,filename='Number of employees by Company, Race and Gender')
d=df.groupby(['gender','race']).agg({'count':sum}).reset_index()
d=df.groupby(['gender','race']).agg({'count':sum}).reset_index()

trace1 = go.Bar(

    x=d[d.gender=='male']['race'],

    y=d[d.gender=='male']['count'],

    name='Males'

)

trace2 = go.Bar(

    x=d[d.gender=='female']['race'],

    y=d[d.gender=='female']['count'],

    name='Females'

)



xcord=np.arange(0,7)

annotations_1=[]

annotations_2=[]

for i in range(0,7):

    annotations_1.append(dict(x=xcord[i]-0.2,y=d[d.gender=='male']['count'].values[i]+100,text='%d' %d[d.gender=='male']['count'].values[i],

                             font=dict(family='Arial', size=10,

                                  color='rgba(0,0,0,1)'),

                                  showarrow=True,))

    

for i in range(0,7):

    annotations_2.append(dict(x=xcord[i]+0.3,y=d[d.gender=='female']['count'].values[i]+100,text='%d' %d[d.gender=='female']['count'].values[i],

                             font=dict(family='Arial', size=10,

                                  color='rgba(0,0,0,1)'),

                                  showarrow=True,))

    

annotations=annotations_1+annotations_2

data = [trace1, trace2]

layout = go.Layout(

    barmode='group',title='Distribution of Male and Female Employees by Race')

layout['annotations'] = annotations



fig = dict(data=data, layout=layout)

iplot(fig, filename='Distribution of Male and Female Employees by Race')
d=df.groupby(['gender','race','job_category']).agg({'count':sum}).reset_index()
plt.figure(figsize=(15,12))

sns.set_style('white')

sns.barplot(x='job_category',y='count',hue='gender',data=d, palette="muted",ci=None)

plt.title('Number of employee by Job Category and Gender',size=16)

plt.yticks(size=14)

plt.ylabel('Number of Employees',size=14)

plt.xlabel('Job Category',size=14)

plt.xticks(rotation=90,size=14)

plt.show()
HTML('''<script>

code_show=true; 

function code_toggle() {

 if (code_show){

 $('div.input').hide();

 } else {

 $('div.input').show();

 }

 code_show = !code_show

} 

$( document ).ready(code_toggle);

</script>

To toggle code, click <a href="javascript:code_toggle()">here</a>.''')