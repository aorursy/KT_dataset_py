import numpy as np

import pandas as pd

pd.options.mode.chained_assignment = None



import plotly.plotly as py

import plotly.graph_objs as go

from plotly import tools

from plotly.offline import iplot, init_notebook_mode

init_notebook_mode()



tobacco_data = pd.read_csv('../input/tobacco.csv', usecols=[0, 1, 2, 3, 4, 5])

tobacco_data = tobacco_data.rename(

    columns={'Smoke everyday':'daily_smoker', 'Smoke some days':'weekly_smoker',

             'Former smoker':'former_smoker', 'Never smoked':'never_smoker'})

tobacco_data.columns = tobacco_data.columns.str.lower()

for percents in tobacco_data.columns[2:]:

    tobacco_data[percents] = tobacco_data[percents].str.rstrip('%')

    tobacco_data[percents] = pd.to_numeric(tobacco_data[percents])



# tobacco use in US states only, territories excluded (812 rows)

mask = tobacco_data['state'].isin(

    ['Guam', 'Puerto Rico', 'Virgin Islands', 'Nationwide (States and DC)',

     'Nationwide (States, DC, and Territories)'])

tobacco_usa = tobacco_data[~mask].sort_values(['year', 'state'])
# tobacco use by year in United States

tobacco_total = tobacco_data[

                  tobacco_data.state == 'Nationwide (States and DC)'].sort_values('year')



labels = ['Daily', 'Weekly', 'Past', 'Never']

colors = ['rgb(0, 142, 194)', 'rgb(128, 199, 225)',

          'rgb(242, 130, 128)', 'rgb(229, 5, 0)']

x_data = np.asarray(tobacco_total['year'].values)

y_data = np.asarray([tobacco_total['daily_smoker'].values,

                     tobacco_total['weekly_smoker'].values,

                     tobacco_total['former_smoker'].values,

                     tobacco_total['never_smoker'].values])



traces = []

for i in range(0, 4):

    traces.append(go.Scatter(

        x = x_data,

        y = y_data[i],

        mode = 'lines',

        name = labels[i],

        line = dict(color = colors[i], width = 3)

    ))

    traces.append(go.Scatter(

        x = [x_data[0], x_data[15]],

        y = [y_data[i][0], y_data[i][15]],

        mode = 'markers',

        hoverinfo = 'none',

        marker = dict(color = colors[i], size = 7)

    ))



layout = go.Layout(

         title = 'Tobacco Use by Year in United States (1995-2010)',

         showlegend = False,

         xaxis = dict(

             showline = True,

             showgrid = False

         ),

         yaxis = dict(

             ticksuffix = '%',

             showline = False,

             zeroline = False,

             showgrid = False,

             showticklabels = False

         ))



annotations = []

for y_trace, label in zip(y_data, labels):

    annotations.append(dict(xref='paper', x=0.0485, y=y_trace[0],

                            xanchor='right', yanchor='middle',

                            text=label + ' {}%'.format(y_trace[0]),

                            showarrow=False))

    annotations.append(dict(xref='paper', x=0.9515, y=y_trace[15],

                            xanchor='left', yanchor='middle',

                            text='{}%'.format(y_trace[15]),

                            showarrow=False))

layout['annotations'] = annotations



figure = dict(data = traces, layout = layout)

iplot(figure)
# tobacco use in United States in 1995

tobacco_1995 = tobacco_usa[tobacco_usa.year == 1995]

# values for District of Columbia and Utah missing from 1995 data

tobacco_1995.loc[0] = [1996, 'District of Columbia', 14.9, 5.6, 17.8, 61.7]

tobacco_1995.loc[1] = [1997, 'Utah', 11.1, 2.6, 17.2, 69.0]

tobacco_1995 = tobacco_1995.sort_values('state')

tobacco_1995.index = range(51)



# tobacco use in United States in 2010

tobacco_2010 = tobacco_usa[tobacco_usa.year == 2010].sort_values('state')

tobacco_2010.index = range(51)



# change in percent daily smokers between 1995 and 2010

tobacco_dailychg = np.asarray(tobacco_2010['daily_smoker'].subtract(

                                           tobacco_1995['daily_smoker'], axis=0))



us_states = np.asarray(['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'DC', 'FL', 'GA',

                        'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA',

                        'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY',

                        'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX',

                        'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY'])



tobacco_scale = [[0, 'rgb(0, 142, 194)'], [1, 'rgb(229, 243, 248)']]



data = [dict(

        type = 'choropleth',

        autocolorscale = False,

        colorscale = tobacco_scale,

        showscale = False,

        locations = us_states,

        locationmode = 'USA-states',

        z = tobacco_dailychg,

        marker = dict(

            line = dict(

                color = 'rgb(255, 255, 255)',

                width = 2)

            )

        )]



layout = dict(

         title = 'Change in Daily Tobacco Use by State (1995-2010)',

         geo = dict(

             scope = 'usa',

             projection = dict(type = 'albers usa'),

             countrycolor = 'rgb(255, 255, 255)',

             showlakes = True,

             lakecolor = 'rgb(255, 255, 255)')

         )



figure = dict(data = data, layout = layout)

iplot(figure)
# change in percent weekly smokers between 1995 and 2010

tobacco_weeklychg = np.asarray(tobacco_2010['weekly_smoker'].subtract(

                                            tobacco_1995['weekly_smoker'], axis=0))



tobacco_scale = [[0, 'rgb(252, 230, 229)'], [1, 'rgb(229, 5, 0)']]



data = [dict(

        type = 'choropleth',

        autocolorscale = False,

        colorscale = tobacco_scale,

        showscale = False,

        locations = us_states,

        locationmode = 'USA-states',

        z = tobacco_weeklychg,

        marker = dict(

            line = dict(

                color = 'rgb(255, 255, 255)',

                width = 2)

            )

        )]



layout = dict(

        title = 'Change in Weekly Tobacco Use by State (1995-2010)',

        geo = dict(

            scope = 'usa',

            projection = dict(type = 'albers usa'),

            countrycolor = 'rgb(255, 255, 255)',

            showlakes = True,

            lakecolor = 'rgb(255, 255, 255)')

        )



figure = dict(data = data, layout = layout)

iplot(figure)
# change in percent former smokers between 1995 and 2010

tobacco_formerchg = np.asarray(tobacco_2010['former_smoker'].subtract(

                                            tobacco_1995['former_smoker'], axis=0))



tobacco_scale = [[0.0, 'rgb(0, 142, 194)'], [0.425, 'rgb(255, 255, 255)'],

                 [1.0, 'rgb(229, 5, 0)']]



data = [dict(

        type = 'choropleth',

        autocolorscale = False,

        colorscale = tobacco_scale,

        showscale = False,

        locations = us_states,

        locationmode = 'USA-states',

        z = tobacco_formerchg,

        marker = dict(

            line = dict(

                color = 'rgb(255, 255, 255)',

                width = 2)

            )

        )]



layout = dict(

         title = 'Change in Past Tobacco Use by State (1995-2010)',

         geo = dict(

             scope = 'usa',

             projection = dict(type = 'albers usa'),

             countrycolor = 'rgb(255, 255, 255)',

             showlakes = True,

             lakecolor = 'rgb(255, 255, 255)')

         )



figure = dict(data = data, layout = layout)

iplot(figure)
# change in percent never smokers between 1995 and 2010

tobacco_neverchg = np.asarray(tobacco_2010['never_smoker'].subtract(

                                           tobacco_1995['never_smoker'], axis=0))



tobacco_scale = [[0.0, 'rgb(0, 142, 194)'], [0.5, 'rgb(255, 255, 255)'],

                 [1.0, 'rgb(229, 5, 0)']]



data = [dict(

        type = 'choropleth',

        autocolorscale = False,

        colorscale = tobacco_scale,

        showscale = False,

        locations = us_states,

        locationmode = 'USA-states',

        z = tobacco_neverchg,

        marker = dict(

            line = dict(

                color = 'rgb(255, 255, 255)',

                width = 2)

            )

        )]



layout = dict(

         title = 'Change in No Tobacco Use by State (1995-2010)',

         geo = dict(

             scope = 'usa',

             projection = dict(type = 'albers usa'),

             countrycolor = 'rgb(255, 255, 255)',

             showlakes = True,

             lakecolor = 'rgb(255, 255, 255)')

         )



figure = dict(data = data, layout = layout)

iplot(figure)