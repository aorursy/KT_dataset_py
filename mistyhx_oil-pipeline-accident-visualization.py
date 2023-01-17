import pandas as pd 
import numpy as np 
from pandas import Series,DataFrame

import matplotlib.pyplot as plt 
import seaborn as sns 
sns.set_style('whitegrid')
%matplotlib inline 

import plotly
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

oil = pd.read_csv("../input/database.csv")
oil.head()
oil.info()
#Clean up the data, remove the columns which lack informaiton
oil = oil.drop(columns=['Operator Employee Injuries', 'Operator Contractor Injuries','Emergency Responder Injuries','Other Injuries','Public Injuries','All Injuries','Operator Employee Fatalities','Emergency Responder Fatalities','Other Fatalities','Public Fatalities','All Fatalities','Operator Contractor Fatalities','Liquid Name'])
oil = oil.drop([2793,2794])
main_causes = oil['Cause Category'].value_counts()
main_causes = go.Bar(
   y = main_causes.values, 
   x = main_causes.index.values, 
   name = 'Causes Category Count', 
     marker=dict(
        color=main_causes.values,
        colorscale = 'Viridis',
        reversescale = True
        )

)

data = [main_causes]

layout = go.Layout (
title = 'Cause category', 
width = 700, 
margin=go.Margin(b=140, r=150)
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='basic-bar')

#most failure resulted from material/weld/equipment failure, while all other reasons together only 1358. So need
#to pay particular attention to Material/Weld/Equipment failure. 
main_causes_sub = oil['Cause Subcategory'].value_counts()
main_causes_sub = go.Bar(
   y = main_causes_sub.values, 
   x = main_causes_sub.index.values, 
   name = 'Causes Subcategory Count', 
     marker=dict(
        color=main_causes_sub.values,
        colorscale = 'Viridis',
        reversescale = True
        )
    
)

data = [main_causes_sub]


layout = go.Layout (
title = 'Cause category', 
width = 800, 
margin=go.Margin(b=140, r=150)
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='basic-bar')
#create a crosstab to dig deeper into the reasons. 
cm = sns.light_palette("green", as_cmap=True)
pd.crosstab(oil['Cause Category'],oil['Cause Subcategory'], margins=True).style.background_gradient(cmap = cm)
Material_Weld_Equipment_Failure = oil[oil['Cause Category'] == 'MATERIAL/WELD/EQUIP FAILURE']
CORROSION = oil[oil['Cause Category'] == 'CORROSION']
INCORRECT_OPERATION = oil[oil['Cause Category'] == 'INCORRECT OPERATION']
NATURAL_FORCE_DAMAGE = oil[oil['Cause Category'] == 'NATURAL FORCE DAMAGE']
EXCAVATION_DAMAGE = oil[oil['Cause Category'] == 'EXCAVATION DAMAGE']
OTHER_OUTSIDE_FORCE_DAMAGE = oil[oil['Cause Category'] == 'OTHER OUTSIDE FORCE DAMAGE']
ALL_OTHER_CAUSES = oil[oil['Cause Category'] == 'ALL OTHER CAUSES']
#Material_Weld_Equipment_Failure 
trace0 = Material_Weld_Equipment_Failure ['Cause Subcategory'].value_counts()
trace0 = go.Bar(

     y = trace0.values, 
     x = trace0.index.values, 
   name = 'Material/Weld/Equipment Failure Count', 
  
     marker=dict(
        color=trace0.values,
        colorscale = 'Viridis',
        reversescale = True
        )



)

#Corrision 
trace1 = CORROSION['Cause Subcategory'].value_counts()
trace1 = go.Bar(
       y = trace1.values, 
       x = trace1.index.values, 
       name = 'Corroison Count', 
        marker=dict(
        color=trace1.values,
        colorscale = 'Viridis',
        reversescale = True
        )
)

#Corrision 
trace1 = CORROSION['Cause Subcategory'].value_counts()
trace1 = go.Bar(
       y = trace1.values, 
       x = trace1.index.values, 
       name = 'Corroison Count', 
        marker=dict(
        color=trace1.values,
        colorscale = 'Viridis',
        reversescale = True
        )
)

#Corrision 
trace2 = INCORRECT_OPERATION['Cause Subcategory'].value_counts()
trace2 = go.Bar(
       y = trace2.values, 
       x = trace2.index.values, 
       name = 'INCORRECT OPERATION COUNT', 
    
        marker=dict(
        color=trace2.values,
        colorscale = 'Viridis',
        reversescale = True
        )
)

#Natural Force Damage
trace3 = NATURAL_FORCE_DAMAGE['Cause Subcategory'].value_counts()
trace3 = go.Bar(
       y = trace3.values, 
       x = trace3.index.values, 
       name = 'NATURAL FORCE DAMAGE COUNT', 
     
        marker=dict(
        color=trace3.values,
        colorscale = 'Viridis',
        reversescale = True
        )
      
)

#EXCAVATION DAMAGE
trace4 = EXCAVATION_DAMAGE['Cause Subcategory'].value_counts()
trace4 = go.Bar(
       y = trace4.values, 
       x = trace4.index.values, 
       name = 'EXCAVATION DAMAGE COUNT', 
      
        marker=dict(
        color=trace4.values,
        colorscale = 'Viridis',
        reversescale = True
        )
)


#OTHER OUTSIDE FORCE DAMAGE 
trace5 = OTHER_OUTSIDE_FORCE_DAMAGE['Cause Subcategory'].value_counts()
trace5 = go.Bar(
       y = trace5.values, 
       x = trace5.index.values, 
       name = 'OTHER OUTSIDE FORCE DAMAGE COUNT', 
      
        marker=dict(
        color=trace5.values,
        colorscale = 'Viridis',
        reversescale = True
        )
)

#ALL OTHER CAUSES
trace6 = ALL_OTHER_CAUSES['Cause Subcategory'].value_counts()
trace6 = go.Bar(
       y = trace6.values, 
       x = trace6.index.values, 
       name = 'ALL OTHER CAUSES COUNT', 
        
        marker=dict(
        color=trace6.values,
        colorscale = 'Viridis',
        reversescale = True
        )
)

#Creating the grid 
fig = tls.make_subplots(rows=3, cols=3,
                          subplot_titles=('MATERIAL/WELD/EQUIPMENT FAILURE',
                                          'CORROSION','INCORRECT OPERATION', 'NATURAL FORCE DAMAGE','EXCAVATION DAMAGE','OTHER OUTSIDE FORCE DAMAGE','ALL OTHER CAUSES' ), 
                     )


#setting the figs 
fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 1, 2)
fig.append_trace(trace2, 1, 3)
fig.append_trace(trace3, 2, 1)
fig.append_trace(trace4, 2, 2)
fig.append_trace(trace5, 2, 3)
fig.append_trace(trace6, 3, 1)

fig['layout']['xaxis3'].update( tickangle=90)
fig['layout']['xaxis4'].update( tickangle=90)
fig['layout']['xaxis5'].update( tickangle=90)

fig['layout']['yaxis1'].update(range=[0, 400])
fig['layout']['yaxis2'].update(range=[0, 400],showticklabels=False)
fig['layout']['yaxis3'].update(range=[0, 400],showticklabels=False)
fig['layout']['yaxis4'].update(range=[0, 400])
fig['layout']['yaxis5'].update(range=[0, 400],showticklabels=False)
fig['layout']['yaxis6'].update(range=[0, 400],showticklabels=False)
fig['layout']['yaxis7'].update(range=[0, 400])



fig['layout'].update( title='CAUSES FOR OIL PIPELINE PILLS', 
                     height=2600, width=900,showlegend=False, 
                     
    )

py.iplot(fig, filename='customizing-subplot-axes')
