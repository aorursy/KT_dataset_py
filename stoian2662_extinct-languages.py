import math



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import plotly.plotly as py

import plotly.graph_objs as go

#from plotly import tools

from plotly.offline import iplot, init_notebook_mode

init_notebook_mode()
data = pd.read_csv('../input/data.csv')

data.columns.values
def filter_by_area(ds, start_lon, end_lon, start_lat, end_lat):

    return ds[(ds['Longitude'] >= start_lon)

             & (ds['Longitude'] <= end_lon)

             & (ds['Latitude'] >= start_lat)

             & (ds['Latitude'] <= end_lat)]



def plot_data_on_map(df, title, max_mark_size=1024, start_lon=-80, end_lon=80, start_lat=-180,end_lat=180):

    colors = ['#C53333', '#A21616', '#800000', '#5A0000', '#330000']

    degrees_of_endangerment = ['Vulnerable', 

                               'Severely endangered', 

                               'Definitely endangered', 

                               'Critically endangered', 

                               'Extinct']

    traces = []

    max_numb_of_speakers = df['Number of speakers'].max() ** 0.5

    coeff = max_mark_size / max_numb_of_speakers



    for i in range(0, 5):

        data_for_degree = df[df['Degree of endangerment'] == degrees_of_endangerment[i]]

        traces.append(dict(

            type = 'scattergeo',

            lon = data_for_degree['Longitude'],

            lat = data_for_degree['Latitude'],

            text = data_for_degree['Name in English'] + ', speakers: ' + data_for_degree['Number of speakers'].astype(str),

            name = degrees_of_endangerment[i],

            hoverinfo = 'text+name',

            mode = 'markers',

            marker = dict( 

                size = 1 + ((data_for_degree['Number of speakers'] ** 0.5) * coeff),

                opacity = 0.85,

                color = colors[i],

                line = dict(color = 'rgb(255, 255, 255)', width = 0.5),

                sizemode = 'area'

            )

        ))



    geo_dict = dict(

             showland = True,

             showframe = False,

             landcolor = 'rgb(239, 239, 239)',

             subunitwidth = 1,

             subunitcolor = 'rgb(196, 196, 196)',

             countrywidth = 1,

             countrycolor = 'rgb(196, 196, 196)',

             showlakes = True,

             lakecolor = 'rgb(255, 255, 255)'

        )

    if start_lon != -80 and end_lon != 80 and start_lat != -180 and end_lat != 180:

        geo_dict['lonaxis'] = dict( range= [ start_lon, end_lon ] )

        geo_dict['lataxis'] = dict( range= [ start_lat, end_lat ] )

            

    layout = dict(

             title = title,

             showlegend = True,

             legend = dict(

                 x = 0.85, y = 0.4

             ),

             geo = geo_dict

        )



    figure = dict(data = traces, layout = layout)

    iplot(figure, filename='d3-cloropleth-map')

    

def plot_bivariate_distribution(df, x_size=4, y_size=3, start_lon=-180, end_lon=180, start_lat=-80, end_lat=80):    

    x = df['Longitude'].tolist()

    y = df['Latitude'].tolist()



    plt_data = [

        go.Histogram2d(x=x, y=y, histnorm='probability',

            autobinx=False,

            xbins=dict(start=start_lon, end=end_lon, size=x_size),

            autobiny=False,

            ybins=dict(start=start_lat, end=end_lat, size=y_size),

            colorscale=[[0.006, 'rgb(12,51,131)'], 

                        [0.012, 'rgb(10,136,186)'], 

                        [0.018, 'rgb(242,211,56)'], 

                        [0.024, 'rgb(242,143,56)'], 

                        [0.030, 'rgb(217,30,30)']]

        )

    ]



    layout = go.Layout(

        title='Distribution of all languages over the world',

        hovermode='closest',

    )



    fig = go.Figure(data=plt_data, layout=layout)

    iplot(plt_data)
plot_data_on_map(data, 'Threatened languages across the world<br>'

                     '<sub>Click on category to Show/Hide its languages</sub>')
data['Languages count'] = data.groupby('Degree of endangerment')['Name in English'].transform('count')

data['Total number of speakers'] = data.groupby('Degree of endangerment')['Number of speakers'].transform('sum')

data['Mean number of speakers'] = data.groupby('Degree of endangerment')['Number of speakers'].transform('mean')

data['Total number of speakers (logarithmic)'] = np.log(data['Total number of speakers']) ** 3 / 10

data['Mean number of speakers (logarithmic)'] = np.log(data['Mean number of speakers']) ** 3 / 10

data = data.sort_values(['Languages count', 'Total number of speakers'])



degrees_of_endangerment = data['Degree of endangerment'].value_counts().index.tolist()

languages_counts = data['Languages count'].value_counts().index.tolist()

speakers_counts = data['Total number of speakers (logarithmic)'].value_counts().index.tolist()

speakers_means = data['Mean number of speakers (logarithmic)'].value_counts().index.tolist()



trace1 = go.Bar(

    x=degrees_of_endangerment,

    y=languages_counts,

    name='Total number of languages'

)

trace2 = go.Bar(

    x=degrees_of_endangerment,

    y=speakers_counts,

    name='Total number of speakers',

    text = [round(count) for count in np.exp([(count * 10) ** (1/3) for count in speakers_counts])],

    hoverinfo = 'text+name',

)

trace3 = go.Bar(

    x=degrees_of_endangerment,

    y=speakers_means,

    name='Mean number of speakers',

    text = [round(mean, 3) for mean in np.exp([(mean * 10) ** (1/3) for mean in speakers_means])],

    hoverinfo = 'text+name',

)



plt_data = [trace1, trace2, trace3]



layout = go.Layout(

    title='Number of languages and speakers by degree of endangerment<br>'

            '<sub>Click on item in the legend to show/hide its values</sub>',

    barmode='group'

)



fig = go.Figure(data=plt_data, layout=layout)

iplot(fig)
plot_bivariate_distribution(data)