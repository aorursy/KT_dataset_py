import pandas as pd

import numpy as np

import matplotlib.pyplot as plt 

import plotly_express as px

import plotly.graph_objects as go

import seaborn as sns
data = pd.read_csv('../input/students-performance-in-exams/StudentsPerformance.csv')

data.head(15)
data.isnull().sum()
data.describe()
plt.figure(figsize=(10,5))

sns.heatmap(data.corr(),annot=True)
grouped = data['gender'].value_counts().reset_index()

fig = px.pie(data_frame = grouped,names = grouped['index'], values = grouped['gender'],  color_discrete_sequence=px.colors.sequential.RdBu, title = 'Gender' )

fig.update_traces(textposition='inside', textinfo='percent+label', hoverinfo = 'label+percent')
female = data[data['gender'] == 'female']

male = data[data['gender'] == 'male']

female = female['race/ethnicity'].value_counts().reset_index()

male = male['race/ethnicity'].value_counts().reset_index()



# magic bar chart

fig = go.Figure(data=[

    go.Bar(name='Female', x= female['index'], y= female['race/ethnicity']),

    go.Bar(name='Male', x= male['index'], y= male['race/ethnicity'])])

fig.update_layout(barmode='group')

fig.show()
plt.figure(figsize=(20,8))

plt.subplot(1, 3, 1)

plt.title('MATH SCORES')

sns.barplot(x='race/ethnicity',y='math score',data=data,hue='gender',palette='gist_heat')

plt.subplot(1, 3, 2)

plt.title('READING SCORES')

sns.barplot(x='race/ethnicity',y='reading score',data=data,hue='gender',palette='gist_heat')

plt.subplot(1, 3, 3)

plt.title('WRITING SCORES')

sns.barplot(x='race/ethnicity',y='writing score',data=data,hue='gender',palette='gist_heat')

plt.show()
# create pivot table

gen2 = data.pivot_table(index=['test preparation course'],values=['math score','reading score','writing score'], aggfunc= np.mean)

gen2 = gen2.reset_index()



# draw chart

fig = go.Figure(data=[

    go.Bar(name='Math', y=gen2['math score'], x=gen2['test preparation course']),

    go.Bar(name='Reading', y=gen2['reading score'], x=gen2['test preparation course']),

go.Bar(name='Writing', y=gen2['writing score'], x=gen2['test preparation course'])])

fig.update_layout(barmode='group')

fig.show()
# pivot table

gen3 = data.pivot_table(index=['lunch','test preparation course'],values=['math score','reading score','writing score'], aggfunc= np.mean)

gen3 = gen3.reset_index()

gen3['lunch and test preparation'] = gen3['lunch'] + ' '+'lunch'+'/'+ gen3['test preparation course'] + ' '+'course'

gen3.drop

# chart

fig = go.Figure(data=[

    go.Bar(name='Math', y=gen3['math score'], x=gen3['lunch and test preparation']),

    go.Bar(name='Reading', y=gen3['reading score'], x=gen3['lunch and test preparation']),

go.Bar(name='Writing', y=gen3['writing score'], x=gen3['lunch and test preparation'])])

fig.update_layout(barmode='group')

fig.show()

data['Total score']=data['math score']+data['reading score']+data['writing score']

# chart drawing

fig,ax=plt.subplots()

sns.barplot(x=data['parental level of education'],y=data['Total score'],data=data,palette='Paired')

fig.autofmt_xdate()