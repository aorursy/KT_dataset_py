from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)

import pandas as pd

import plotly.graph_objects as go

from plotly.subplots import make_subplots

import plotly.express as px



df_merge = pd.read_csv('../input/graph-data/graph_data_merged.csv')
fig = go.Figure(go.Scattermapbox(

                name = 'Path Sailed',

                mode = "markers+lines",

                hoverinfo='all',

                lon = [-1.4044, -1.6221, -8.2943, -49.9469],

                lat = [50.9097, 49.6337, 51.8503, 41.7325],

                 text = ['Southampton', 'Cherbourg', 'Queenstown', 'Point where Titanic sank'],

#                 textposition = 'bottom right',

                marker = {'size': 10}

            )

      )



fig.add_trace(go.Scattermapbox(

                name = 'Path Not Sailed',

                mode = "markers+lines",

                hoverinfo='all',

                lon = [-49.9469, -74.0060],

                lat = [41.7325, 40.7128],

                text = ['Point where Titanic sank', 'New York'],

                marker = {'size': 10}

               )

    

)

fig.update_layout(

    title={

        'text': 'Route of the RMS Titanic',

        'y':0.9999,

        'x':0.4},

    margin ={'l':0,'t':35,'b':0,'r':0},

    mapbox = {

        'style': "stamen-terrain",

        'center': {'lon': -40, 'lat': 40},

        'zoom': 2},

    width=800,

    height=400

)

fig.show()
port_fare = (df_merge

             .groupby('Embarked')

             .agg({'PerPersonFare': 'sum'})

             .rename(columns={'PerPersonFare':'PortFare'})

             .reset_index()

            )

embarked_details = (df_merge

                    .groupby('Embarked')

                    .agg({'PassengerId': 'count'})

                    .rename(columns={'PassengerId':'PassengersEmbarked'})

                    .reset_index()

                   )

totals = pd.DataFrame([['Total', port_fare['PortFare'].sum()]], columns=['Embarked', 'PortFare'])

port_fare = port_fare.append(totals)

totals = pd.DataFrame([['Total', embarked_details['PassengersEmbarked'].sum()]], columns=['Embarked', 'PassengersEmbarked'])

embarked_details = embarked_details.append(totals)



embarked_pclass = (df_merge.groupby(['Embarked','Pclass']).agg({'PassengerId': 'count'})

                   .rename(columns={'PassengerId':'Count'})

                   .reset_index()

                  )



color=px.colors.sequential.GnBu

pclass = embarked_pclass['Pclass'].unique()



fig = make_subplots(

    rows=2, cols=2,

    specs=[[{'type':'domain'}, {'type':'domain'}],

           [{"colspan": 2}, None]],

    horizontal_spacing=0.10, 

    vertical_spacing=0.10,

    subplot_titles=['Passengers embarked from each Port', 

                    'Revenue collected from each port',

                    'Passengers embarked by class from each Port'

                   ])



fig1 = go.Sunburst(

    labels= embarked_details['Embarked'],

    parents=['Total', 'Total', 'Total', ''],

    values= embarked_details['PassengersEmbarked'],

    branchvalues='total',

    hoverinfo='all',

    marker=dict(

        colors=[color[3], color[5], color[7], '#FFFFFF']

    )

)



fig2 = go.Sunburst(

    labels= port_fare['Embarked'],

    parents=['Total', 'Total', 'Total', ''],

    values= port_fare['PortFare'],

    branchvalues='total',

    hoverinfo='all',

    marker=dict(

        colors=[color[3], color[5], color[7], '#FFFFFF']

    )

)



fig.add_trace(fig1, 1, 1)

fig.add_trace(fig2, 1, 2)



fig.add_trace(go.Bar(name='Cherbourg', 

                     x=pclass, 

                     y=embarked_pclass[embarked_pclass['Embarked']=='Cherbourg']['Count'], 

                     marker_color=color[3]),

              2,1)

fig.add_trace(go.Bar(name='Queenstown', 

                     x=pclass, 

                     y=embarked_pclass[embarked_pclass['Embarked']=='Queenstown']['Count'], 

                     marker_color=color[5]),

              2,1)

fig.add_trace(go.Bar(name='Southampton', 

                     x=pclass, 

                     y=embarked_pclass[embarked_pclass['Embarked']=='Southampton']['Count'], 

                     marker_color=color[7]),

              2,1)



fig.update_layout(margin = dict(t=40, l=0, r=0, b=0),

                  autosize=False,

                  width=800,

                  height=500,

                  xaxis_title='Pclass',

                  yaxis_title='Number of Passengers'

                 )



fig.show()
colors=px.colors.sequential.GnBu



fig = go.Figure()

fig = make_subplots(1,3,

                    x_title='Pclass',

                    y_title='Per Passenger Fare')



fig.add_trace(go.Box(

    y=df_merge[df_merge['Embarked']=='Cherbourg']['PerPersonFare'],

    x=df_merge[df_merge['Embarked']=='Cherbourg']['Pclass'],

    name='Cherbourg',

    marker_color=colors[3]

), 1, 1)



fig.add_trace(go.Box(

    y=df_merge[df_merge['Embarked']=='Queenstown']['PerPersonFare'],

    x=df_merge[df_merge['Embarked']=='Queenstown']['Pclass'],

    name='Queenstown',

    marker_color=colors[5]

), 1,2)



fig.add_trace(go.Box(

    y=df_merge[df_merge['Embarked']=='Southampton']['PerPersonFare'],

    x=df_merge[df_merge['Embarked']=='Southampton']['Pclass'],

    name='Southampton',

    marker_color=colors[7]

), 1, 3)



fig.update_layout(

    autosize=False,

    width=800,

    height=600,

    title={

        'text': 'Per Passenger Fare From Each Port By Class',

        'y':0.9999,

        'x':0.4,

        'xanchor': 'center',

        'yanchor': 'top'},

    margin=dict(l=60, r=0, t=40, b=50)

)



fig.show()
#data for single passengers on the ship

singles = (df_merge[((df_merge['SibSp']==0) & (df_merge['Parch']==0))]

                  .groupby('Embarked')

                  .agg({'PassengerId':'count'})

                  .rename(columns={'PassengerId':'Count'})

                  .reset_index()

                 )

#data for passengers with family on the ship

with_family = (df_merge[(~((df_merge['SibSp']==0) & (df_merge['Parch']==0)))]

               .groupby('Embarked')

               .agg({'PassengerId':'count'})

               .rename(columns={'PassengerId':'Count'})

               .reset_index()

              )

#data for passengers travelling on same ticket 

on_same_ticket = (df_merge

                  .groupby('Ticket')

                  .agg({'PassengerId':'count'})

                  .rename(columns={'PassengerId':'PeopleOnSameTicket'})

                  .reset_index()

                 )

ppl_same_ticket = df_merge.merge(on_same_ticket, on='Ticket')

ppl_same_ticket = (ppl_same_ticket[ppl_same_ticket['PeopleOnSameTicket']!=1]

                   .groupby('Embarked')

                   .agg({'PassengerId':'count'})

                   .rename(columns={'PassengerId':'MultiplePeopleOnATicket'})

                   .reset_index()

)



#plot graph

names = singles['Embarked']

colors = px.colors.sequential.GnBu



fig = go.Figure(data=[

    go.Bar(name='singles', x=names, y=singles['Count'], marker_color=colors[3]),

    go.Bar(name='with family', x=names, y=with_family['Count'], marker_color=colors[5]),

    go.Bar(name='Multiple people on same ticket', x=names, y=ppl_same_ticket['MultiplePeopleOnATicket'], marker_color=colors[7])

])



# Change the bar mode and set title

fig.update_layout(barmode='group',

                  margin=dict(l=0, r=0, t=40, b=0),

                  width=800,

                  height=400,

                  yaxis_title='Number of Passengers',

                  xaxis_title='Embarked Port',

                  title={

                      'text': 'Distribution of Singles/People With Family',

                      'y':0.9999,

                      'x':0.4,

                      'xanchor': 'center',

                      'yanchor': 'top'},

                 )

fig.show()
#prepare dataset for graph

children = df_merge.copy()

children.loc[children['Age'] <= 14.0, ['Sex']] = 'child'

graph_data = (children

              .groupby(['Sex', 'Survived'])

              .agg({'PassengerId':'count'})

              .rename(columns={'PassengerId':'PassengerCount'})

              .reset_index()

)



#plot graph

colors = px.colors.sequential.GnBu

x = graph_data['Sex'].unique()



fig = go.Figure()

fig = make_subplots(

    rows=1, 

    cols=2,

    specs=[[{'type': 'domain'}, {'type': 'xy'}]],

    subplot_titles=['Passenger data by Sex', 

                    'Passenger Data by Age']

)



#add subplot at (1,1)

fig.add_trace(go.Sunburst(

    labels= ['Total','Females', 'Females Dead', 'Females Survived', 'Females In Test', 'Males', 'Males Dead', 'Males Survived', 'Males In Test'],

    parents=['','Total', 'Females', 'Females', 'Females', 'Total', 'Males', 'Males', 'Males'],

    values= [1309, 466, 81, 233, 152, 843, 468, 109, 266],

    branchvalues='total',

    hoverinfo='all',

    opacity=1,

    marker=dict(

        colors=['#FFFFFF','#FD99C8','#f72346', '#4be36a', '#a3a0a0', '#25A9D1', '#f72346', '#4be36a', '#a3a0a0']

    )

),1,1)



#add subplot at (1,2)

fig.add_trace(go.Bar(x=x, y=graph_data[graph_data['Survived']==0]['PassengerCount'], name='Dead', marker_color='#F3526D'), 1,2)

fig.add_trace(go.Bar(x=x, y=graph_data[graph_data['Survived']==1]['PassengerCount'], name='Survived', marker_color='#76E28C'), 1,2)

fig.add_trace(go.Bar(x=x, y=graph_data[graph_data['Survived']==-1]['PassengerCount'], name='In Test', marker_color='#c2c0c0'),1,2)



fig.update_layout(barmode='stack',

                  xaxis={'categoryorder':'category ascending'},

                  yaxis_title = 'Number of Passengers',

                  margin = dict(t=40, l=0, r=0, b=0),

                  autosize=False,

                  width=800,

                  height=400

                 )



fig.show()
#plot graph

fig =go.Figure(go.Sunburst(

    labels= ['Total','Singles', 'Singles Dead', 'Singles Survived', 'Singles In Test',

             'Family Members', 'Family Members Dead', 'Family Members Survived', 'Family Members In Test'],

    parents=['','Total', 'Singles', 'Singles', 'Singles', 'Total', 'Family Members', 'Family Members', 'Family Members'],

    values= [1309, 692, 294, 234, 164, 617, 255, 108, 254],

    branchvalues='total',

    hoverinfo='all',

    marker=dict(

        colors=['#FFFFFF','rgb(8,104,172)','#f72346', '#4be36a', '#a3a0a0', 

                'rgb(78,179,211)', '#f72346', '#4be36a', '#a3a0a0']

    ),

))



fig.update_layout(margin = dict(t=25, l=0, r=0, b=0),

                  title={

                      'text': 'Distribution of Singles/People With Family',

                      'y':0.9999,

                      'x':0.5,

                      'xanchor': 'center',

                      'yanchor': 'top'},

                  width=800

                 )



fig.show()