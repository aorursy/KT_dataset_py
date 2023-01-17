import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
!pip install pywaffle
import pandas as pd

import numpy as np

from plotly.offline import init_notebook_mode, iplot, plot

from plotly.subplots import make_subplots

import plotly as py

init_notebook_mode(connected=True)

import plotly.graph_objs as go

import matplotlib.pyplot as plt

import seaborn as sns

from pywaffle import Waffle

%matplotlib inline



survey18 = pd.read_csv("../input/kaggle-survey-2018/multipleChoiceResponses.csv", skiprows=[1], low_memory=False)

survey19 = pd.read_csv("../input/kaggle-survey-2019/multiple_choice_responses.csv", skiprows=[1])



production = [

    'We have well established ML methods (i.e., models in production for more than 2 years)',

    'We recently started using ML methods (i.e., models in production for less than 2 years)',

]

explorers = [

    'We are exploring ML methods (and may one day put a model into production)',

    'We use ML methods for generating insights (but do not put working models into production)',

]

nop = [

    'I do not know',

    'No (we do not use ML methods)',

]
title = 'ML status in organizations'

labels = ['We have models in production', 'We\'re exploring it', 'Not now']



def get_data(keys):

    return np.array([

        int(round(survey18.Q10.value_counts(normalize=True)[keys].sum() * 100)),

        int(round(survey19.Q8.value_counts(normalize=True)[keys].sum() * 100)),

    ])



y_data = np.array([

    get_data(production),

    get_data(explorers),

    get_data(nop),   

])



x = ['2o18', '2o19']

colors = ['rgb(49,130,189)', 'rgb(67,67,67)', 'rgb(115,115,115)']



mode_size = [12, 8, 8]

line_size = [4, 2, 2]

annotations = []



fig = go.Figure()



for i in range(0, 3):

    # lines

    fig.add_trace(go.Scatter(x=x, y=y_data[i], mode='lines',

        name=labels[i],

        line=dict(color=colors[i], width=line_size[i]),

        connectgaps=True,

    ))

    # dots

    fig.add_trace(go.Scatter(

        x=[x[0], x[-1]],

        y=[y_data[i][0], y_data[i][-1]],

        mode='markers',

        marker=dict(color=colors[i], size=mode_size[i])

    ))



fig.update_layout(

    xaxis=dict(

        showline=True,

        showgrid=False,

        showticklabels=True,

        linecolor='rgb(204, 204, 204)',

        linewidth=2,

        ticks='outside',

        tickfont=dict(

            family='Arial',

            size=12,

            color='rgb(82, 82, 82)',

        ),

    ),

    yaxis=dict(

        showgrid=False,

        zeroline=False,

        showline=False,

        showticklabels=False,

    ),

    autosize=False,

    margin=dict(

        autoexpand=False,

        l=250,

        r=20,

        t=110,

    ),

    showlegend=False,

    plot_bgcolor='white'

)



annotations = []



# Adding labels

for y_trace, label, color in zip(y_data, labels, colors):

    # labeling the left_side of the plot

    annotations.append(dict(xref='paper', x=0.05, y=y_trace[0],

                                  xanchor='right', yanchor='middle',

                                  text=label + ' {}%'.format(y_trace[0]),

                                  font=dict(family='Arial',

                                            size=16),

                                  showarrow=False))

    # labeling the right_side of the plot

    annotations.append(dict(xref='paper', x=0.95, y=y_trace[1],

                                  xanchor='left', yanchor='middle',

                                  text='{}%'.format(y_trace[1]),

                                  font=dict(family='Arial',

                                            size=16),

                                  showarrow=False))

# Title

annotations.append(dict(xref='paper', yref='paper', x=0.1, y=1.05,

                              xanchor='left', yanchor='bottom',

                              text=title,

                              font=dict(family='Arial',

                                        size=25,

                                        color='rgb(37,37,37)'),

                              showarrow=False))



fig.update_layout(annotations=annotations)

fig.show()
usa18 = survey18[(survey18.Q10.isin(production)) & (survey18.Q3 == 'United States of America')].Q3.count()

india18 = survey18[(survey18.Q10.isin(production)) & (survey18.Q3 == 'India')].Q3.count()

others18 = survey18[survey18.Q10.isin(production)].Q3.value_counts()[2:].sum()



usa19 = survey19[(survey19.Q8.isin(production)) & (survey19.Q3 == 'United States of America')].Q3.count()

india19 = survey19[(survey19.Q8.isin(production)) & (survey19.Q3 == 'India')].Q3.count()

others19 = survey19[(survey19.Q8.isin(production))].Q3.value_counts()[2:].sum()



data = pd.DataFrame(

    {

        'labels': ['United States of America', 'India', 'Rest of World'],

        '2018': [usa18, india18, others18],

        '2019': [usa19, india19, others19],

    },

).set_index('labels')



fig = plt.figure(

    FigureClass=Waffle,

    plots={

        '311': {

            'values': data['2018'],

            'labels': ["{0} ({1})".format(n, v) for n, v in data['2018'].items()],

            'legend': {

                'loc': 'upper left',

                'bbox_to_anchor': (1.05, 1),

                'fontsize': 10

            },

            'title': {

                'label': 'Respondents residence in 2018 (working with production ML)',

                'loc': 'left',

                'fontsize': 14

            }

        },

        '312': {

            'values': data['2019'],

            'labels': ["{0} ({1})".format(n, v) for n, v in data['2019'].items()],

            'legend': {

                'loc': 'upper left',

                'bbox_to_anchor': (1.05, 1),

                'fontsize': 10

            },

            'title': {

                'label': 'Respondents residence in 2019 (working with production ML)',

                'loc': 'left',

                'fontsize': 14

            }

        },

    },

    rows=5,

    columns=12,

    colors=("#2196f3", "#ff5252", "#999999"),

    figsize=(14, 10),

)

fig.set_facecolor('#ffffff')

fig.set_tight_layout(False)

plt.show()
prod_set = survey19[survey19.Q8.isin(production)]

prod_set = prod_set.rename(

    columns={

        'Q3': 'country',

        'Q5': 'role',

        'Q6': 'company_size',

        'Q7': 'company_ds_size',

    }

)
team_size = prod_set.groupby(['company_size', 'company_ds_size'])['company_ds_size'].count().unstack(fill_value=0)



colors = [

    'rgba(190, 192, 213, 0.90)',

    'rgba(190, 192, 213, 0.95)',

    'rgba(190, 192, 213, 1)',

    'rgba(164, 163, 204, 0.85)',

    'rgba(122, 120, 168, 0.8)',

    'rgba(71, 58, 131, 0.8)',

    'rgba(38, 24, 74, 0.8)',

]

company_size_order = [

    '0-49 employees',

    '50-249 employees',

    '250-999 employees',

    '1000-9,999 employees',

    '> 10,000 employees',

]

ds_dep_size = [

    '0', '1-2', '3-4', '5-9', '10-14', '15-19', '20+',

]



fig = go.Figure()



for i in range(0, len(ds_dep_size)):

    fig.add_trace(go.Bar(

        name=ds_dep_size[i],

        x=team_size[ds_dep_size[i]].values,

        y=team_size.index,

        orientation='h',

        marker=dict(

            color=colors[i],

        )

    ))

    

fig.update_layout(

    plot_bgcolor='white',

    yaxis=dict(

        categoryorder='array',

        categoryarray=company_size_order,

    ),

    xaxis=dict(

        title='# respondents'

    )

)



annotations = []



annotations.append(dict(xref='paper', yref='paper', x=0.1, y=1.12,

                              xanchor='left', yanchor='bottom',

                              text='Company size / Data Science Department Size',

                              font=dict(family='Arial',

                                        size=25,

                                        color='rgb(37,37,37)'),

                              showarrow=False))



fig.update_layout(barmode='stack', annotations=annotations)

fig.show()
dominant_roles = prod_set.role.value_counts(normalize=True)



fig = go.Figure(go.Pie(

    labels=dominant_roles.index,

    values=dominant_roles.values,

    textinfo='label+percent',

    hoverinfo='label',

))



annotations = []



annotations.append(dict(xref='paper', yref='paper', x=0.49, y=1.20,

                          xanchor='center', yanchor='top',

                          text='Roles in teams doing Production ML',

                          font=dict(family='Arial',

                                    size=25,

                                    color='rgb(37,37,37)'),

                          showarrow=False))



fig.update_layout(

    annotations=annotations

)

fig.show()

ml_xp = prod_set.Q23.value_counts()

all_ml_xp = survey19.Q23.value_counts()



fig = go.Figure()



fig.add_trace(go.Bar(

    x=ml_xp.index,

    y=ml_xp,

    name='In production teams'

))



fig.add_trace(go.Bar(

    x=all_ml_xp.index,

    y=all_ml_xp,

    name='All',

    marker=dict(

        color='#999'

    )

))



annotations = []



annotations.append(dict(xref='paper', yref='paper', x=0.5, y=1.05,

                          xanchor='center', yanchor='top',

                          text='ML methods experience in years',

                          font=dict(family='Arial',

                                    size=25,

                                    color='rgb(37,37,37)'),

                          showarrow=False))



fig.update_layout(

    barmode='stack',

    plot_bgcolor='white',

    annotations=annotations,

    xaxis=dict(

        categoryorder='array',

        categoryarray=['< 1 years', '1-2 years', '2-3 years', '3-4 years', '4-5 years', '5-10 years', '10-15 years', '20+ years'],

    ),

    yaxis=dict(

        title='# respondents',

    )

)

fig.show()
def count_multi_choice(df, question):

    columns = []

    for q in df.columns:

        if question in q:

            columns.append(q)

    

    count = []

    for c in columns:

        if 'OTHER_TEXT' not in c:

            total = df[c].value_counts()

            count.append({ 'name': total.index[0], 'total': total.tolist()[0] })

    

    countdf = pd.DataFrame(count, columns=['name', 'total']).sort_values(by='total', ascending=False)

    return countdf



libs_prod = count_multi_choice(prod_set, 'Q28')

all_libs = libs = count_multi_choice(survey19, 'Q28')



fig = go.Figure(go.Bar(

    x=libs_prod.name,

    y=libs_prod.total,

    name='In production',

))

fig.add_trace(go.Bar(

    x=all_libs.name,

    y=all_libs.total,

    name='All',

    marker=dict(

        color='#999'

    )

))



annotations = []



annotations.append(dict(xref='paper', yref='paper', x=0.25, y=1.05,

                              xanchor='left', yanchor='bottom',

                              text='Most used ML frameworks 2019',

                              font=dict(family='Arial',

                                        size=25,

                                        color='rgb(37,37,37)'),

                              showarrow=False))



fig.update_layout(

    annotations=annotations,

    barmode='stack',

    plot_bgcolor='white',

    yaxis=dict(

        title='# respondents',

    )

)

fig.show()