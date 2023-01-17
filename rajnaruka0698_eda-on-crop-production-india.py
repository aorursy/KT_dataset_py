import numpy as np ## Linear Algebra

import pandas as pd ## To work with data

import plotly.express as px ## Visualization

import plotly.graph_objects as go ## Visualization

import matplotlib.pyplot as plt ## Visualization

import plotly as py ## Visuaization

from plotly import tools ## Visualization

import os
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

path = os.path.join(dirname, filename)



df = pd.read_csv(path)
df.shape
df.isnull().sum()
df.dropna(inplace=True) # looking at the data we can drop the null values as they are less in number.
df.head()
temp = df.groupby(by='State_Name')['Production'].sum().reset_index().sort_values(by='Production')

px.bar(temp, 'State_Name', 'Production')
temp = df.groupby('State_Name')['Area', 'Production'].sum().reset_index()

temp['Production_Per_Unit_Area'] = temp['Production']/temp['Area']

temp = temp.sort_values(by='Production_Per_Unit_Area')

px.bar(temp, 'State_Name', 'Production_Per_Unit_Area', color='Production_Per_Unit_Area', )
temp = df.groupby(by='Crop_Year')['Production'].sum().reset_index()

px.line(temp, 'Crop_Year', 'Production')
temp = df.groupby(by='Crop_Year')['Area'].mean().reset_index()

px.scatter(temp, 'Crop_Year', 'Area', color='Area', size='Area')
temp = df.groupby('State_Name')['Area', 'Production'].sum().reset_index()

temp['Production_Per_Unit_Area'] = temp['Production']/temp['Area']

temp = temp.sort_values(by='Production_Per_Unit_Area')

px.bar(temp, 'State_Name', 'Production_Per_Unit_Area', color='Production_Per_Unit_Area')
fig = py.subplots.make_subplots(rows=1,cols=2,

                    subplot_titles=('Highest crop producing districts', 'Least overall crop producing districts'))



temp = df.groupby(by='District_Name')['Production'].sum().reset_index().sort_values(by='Production')

temp1 = temp.tail()

trace1 = go.Bar(x= temp1['District_Name'], y=temp1['Production'])



temp1=temp.head()

trace2 = go.Bar(x= temp1['District_Name'], y=temp1['Production'])



fig.append_trace(trace1,1,1)

fig.append_trace(trace2,1,2)

fig.show()

del temp,temp1
temp = df.groupby(by='Crop')['Production'].sum().reset_index().sort_values(by='Production')

px.bar(temp.tail(), 'Crop', 'Production')
temp[temp['Production']==0]
coconut = df[df['Crop']=='Coconut ']



fig = py.subplots.make_subplots(rows=1,cols=2,

                               subplot_titles=('Coconut production in different states', 'Coconut crop area in states'))



temp = coconut.groupby(by='State_Name')['Production'].sum().reset_index().sort_values(by='Production')

trace0 = go.Bar(x=temp['State_Name'], y=temp['Production'])



temp = coconut.groupby(by='State_Name',)['Area'].mean().reset_index().sort_values(by='Area')

trace1 = go.Bar(x=temp['State_Name'], y=temp['Area'])



fig.append_trace(trace0, 1,1)

fig.append_trace(trace1, 1,2)

fig.show()
temp = coconut.groupby(by='Crop_Year')['Production'].sum().reset_index()

px.line(temp, 'Crop_Year', 'Production', title='Coconut production over the years')
kerala = df[df['State_Name']=='Kerala']

temp = kerala.groupby(by='Crop')['Production'].mean().reset_index().sort_values(by='Production')

px.bar(temp, 'Crop', 'Production', title = 'Avg. Crop Production')
kerala = kerala[~(kerala['Crop']=='Coconut ')]

temp = kerala.groupby(by='Crop')['Production'].sum().reset_index().sort_values(by='Production')

px.bar(temp, 'Crop', 'Production', title='AVG. Crop Production excluding coconut')
df1 = df[~((df['State_Name']=='Kerala') | (df['Crop']=='Coconut '))]
temp=df1.groupby('Crop')['Production'].sum().reset_index().sort_values(by='Production')

px.bar(temp, 'Crop', 'Production', title='Overall production of crops')