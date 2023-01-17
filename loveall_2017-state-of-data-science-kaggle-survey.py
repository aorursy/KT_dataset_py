import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', 500)
mcr = pd.read_csv('../input/multipleChoiceResponses.csv', encoding="ISO-8859-1", low_memory=False)

ffr = pd.read_csv('../input/freeformResponses.csv', low_memory=False)

conversion = pd.read_csv('../input/conversionRates.csv')
mcr.head(3)
# Plotting 2010 and 2014 visuals

metricscale1=[[0, 'rgb(102,194,165)'], [0.05, 'rgb(102,194,165)'], [0.15, 'rgb(171,221,164)'], 

              [0.2, 'rgb(230,245,152)'], [0.25, 'rgb(255,255,191)'], [0.35, 'rgb(254,224,139)'], 

              [0.45, 'rgb(253,174,97)'], [0.55, 'rgb(213,62,79)'], [1.0, 'rgb(158,1,66)']]

data = [ dict(

        type = 'choropleth',

        autocolorscale = False,

        colorscale = metricscale1,

        showscale = True,

        locations = mcr.Country.value_counts().index,

        z = mcr.Country.value_counts().values,

        locationmode = 'country names',

        text = mcr.Country.value_counts().index,

        marker = dict(

            line = dict(color = 'rgb(250,250,225)', width = 0.5)),

            colorbar = dict(autotick = True, tickprefix = '', 

            title = 'Number of Respondees')

            )

       ]



layout = dict(

    title = 'Number of People Surveyed\n (Countries in white represent no survey participation)',

    geo = dict(

        showframe = True,

        showocean = True,

        oceancolor = 'rgb(28,107,160)',

        #oceancolor = 'rgb(222,243,246)',

        projection = dict(

        type = 'orthographic',

            rotation = dict(

                    lon = 60,

                    lat = 10),

        ),

        lonaxis =  dict(

                showgrid = False,

                gridcolor = 'rgb(102, 102, 102)'

            ),

        lataxis = dict(

                showgrid = False,

                gridcolor = 'rgb(102, 102, 102)'

                )

            ),

        )

fig = dict(data=data, layout=layout)

py.iplot(fig, validate=False, filename='worldmap')
data = [go.Bar(

            x = mcr.GenderSelect.value_counts().index,

            y = mcr.GenderSelect.value_counts().values,

            marker= dict(colorscale='Jet',

                         color =  mcr.GenderSelect.value_counts().values

                        ),

            text='Number of respondees'

    )]



layout = go.Layout(

    title='Gender distribution of Survey participants'

)



fig = go.Figure(data=data, layout=layout)



py.iplot(fig, filename='basic-bar')
mcr.EmploymentStatus.value_counts()
mcr.LanguageRecommendationSelect.value_counts()
plt.style.use('dark_background')

x_sort = mcr.groupby(['LanguageRecommendationSelect','MLMethodNextYearSelect']).size().sort_values(ascending= False)

x_sort.unstack().plot(kind='bar',stacked=True, colormap= 'tab20', grid=False,  figsize=(13,11), )

plt.title('Stacked Barplot of Language with Machine Learning method')

plt.ylabel('Global Sales')