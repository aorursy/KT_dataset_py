import numpy as np

import pandas as pd

import plotly.express as px

import seaborn as sb

import matplotlib.pyplot as plt





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
stressed = pd.DataFrame({'Stressed':['Strongly disgree', 'Disagree', 'Neutral', 'Agree', 'Strongly agree'],

                         'Percentage':[5, 14, 30, 32, 19]

                        })
stressed
fig = px.pie(stressed, stressed['Stressed'],

             stressed['Percentage'],

             color='Stressed',

             hole=0.4,

            )

fig.update_traces(showlegend=False,

                  textposition='inside',

                  textinfo='percent+label'

                 )

fig.update_layout(title={

                        'text': "Survey Question: Is my stress level normal?",

                        'y':0.95,

                        'x':0.5,

                        'xanchor': 'center',

                        'yanchor': 'top'},

                         showlegend=False,

                         height=600

                        )



fig.data[0].marker.line.width = 1

fig.data[0].marker.line.color = "black"



fig.show()
concerns = pd.DataFrame({'Concerns':['Health and economic<br> impact on the<br> country', 

                                     'A member of my<br> family might contract<br> COVID-19',

                                     'Cost-cutting measures<br> that might impact<br> my compensation'],

                         'Percentage':[72, 61, 40]

                        })
concerns
fig = px.bar(concerns, y='Concerns', x='Percentage', color='Concerns')

fig.update_layout(showlegend=False, title='Top Concerns among Employees', template='ggplot2')

fig.show()
productivity = pd.DataFrame({'Productivity':['Less Productive', 

                                     'Neutral',

                                     'More Productive'],

                         'Percentage':[36, 45, 18]

                        })
productivity_inc = productivity.sort_values(by='Percentage', ascending=False)



productivity_inc.style.background_gradient(cmap='Blues', subset=['Percentage'])
fig = px.bar(productivity_inc, y='Productivity', x='Percentage', color='Productivity')

fig.update_layout(showlegend=False, title='Productivity afte work from home', template='ggplot2')

fig.show()
wfh = pd.DataFrame({'WFH Preference':['100% WFH', 

                                     '75% WFH',

                                     '50% WFH',

                                     '25% WFH',

                                     'No WFH'],

                         'Percentage':[16, 35, 31, 9, 9]

                        })
fig = px.bar(wfh, x='WFH Preference', y='Percentage', color='WFH Preference', text='Percentage')



fig.update_layout(showlegend=False, title='% of time they want to WFH', template='ggplot2')

fig.update_traces(textposition='outside')

fig.show()
confidence = pd.DataFrame({'Confidence':['Strongly Disagree',

                                         'Disagree',

                                         'Neutral',

                                         'Agree',

                                         'Strongly Agree'

                                        ],

                         'Percentage':[1, 3, 16, 36, 44]

                        })
fig = px.pie(confidence, confidence['Confidence'],

             confidence['Percentage'],

             color='Confidence',

             hole=0.4,

            )

fig.update_traces(showlegend=True,

                  textposition='inside',

                  textinfo='percent+label'

                 )

fig.update_layout(title={

                        'text': "Confident in the future of the organisation?",

                        'y':0.95,

                        'x':0.46,

                        'xanchor': 'center',

                        'yanchor': 'top'},

                         height=600

                        )



fig.data[0].marker.line.width = 1

fig.data[0].marker.line.color = "black"



fig.show()
top_concern = pd.DataFrame({'Concerns':['Resurgence of<br> Covid-19 infection',

                                         'Co-workers compliance<br> to measures',

                                         'Self-compliance<br> to measures',

                                         'Cleanliness of<br> physical workplace',

                                         'Making homecare<br> arrangements for<br> family members',

                                         'Reduces productivity<br> when back to workplace'

                                        ],

                         'Percentage':[70, 56, 46, 32, 17, 16]

                        })
top_concern_sort = top_concern.sort_values(by='Percentage', ascending=False)

fig = px.bar(top_concern_sort, y='Concerns', x='Percentage', color='Concerns')



fig.update_layout(showlegend=False, title='Top concerns returning to workplace', template='ggplot2')



fig.show()
safe = pd.DataFrame({'Feel safe on returning to workplace':['Strongly Disagree',

                                         'Disagree',

                                         'Neutral',

                                         'Agree',

                                         'Strongly Agree'

                                        ],

                         'Percentage':[9, 17, 33, 26, 15]

                        })
# lets use seaborn to plot this one

plt.style.use('ggplot')

plt.subplots(figsize=(12,8))

fig = sb.barplot(x=safe['Feel safe on returning to workplace'], y=safe['Percentage'], data=safe)
social_distance = pd.DataFrame({'Social Distancing':['For the<br> next 3 months',

                                         'For the<br> next 6 months',

                                         'For the<br> next 1 year',

                                         'Social distancing<br> should be put<br> in place permanently',

                                        ],

                         'Percentage':[28, 42, 21, 9]

                        })
fig = px.bar(social_distance, y='Social Distancing', x='Percentage', color='Social Distancing')

fig.update_layout(showlegend=False, title='Social distancing should be observerd for', template='ggplot2')

fig.show()