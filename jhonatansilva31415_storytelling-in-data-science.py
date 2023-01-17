import os

import pandas as pd 

pd.set_option('display.max_columns', 10)



base_dir = '../input/'

multiple_path = 'multipleChoiceResponses.csv'

freeForm_path = 'freeFormResponses.csv'

schema_path = 'SurveySchema.csv'



multiple_choices_df = pd.read_csv(os.path.join(base_dir,multiple_path))

free_form_df = pd.read_csv(os.path.join(base_dir,freeForm_path))

schema_df = pd.read_csv(os.path.join(base_dir,schema_path))
multiple_choices_df[0:2]
free_form_df[0:2]
schema_df[0:2]
len(multiple_choices_df.columns)
information_we_need = multiple_choices_df[

    ['Q3','Q37']

]

information_we_need.head()
multiple_choices_df.groupby(['Q3','Q37']).size()
multiple_choices_df.groupby(['Q3','Q37']).size().reset_index()
information_we_need = multiple_choices_df.groupby(['Q3', 'Q37']).size().reset_index()

information_we_need.columns = ['Country', 'Platform', 'How many votes']

information_we_need.head()
information_we_need = information_we_need.sort_values(by=['Country','How many votes'])

information_we_need[:10]
information_we_need.drop_duplicates(subset='Country', keep='first').head()
information_we_need = information_we_need.drop_duplicates(subset='Country', keep='last')

information_we_need.head()
import matplotlib.pyplot as plt

import numpy as np 

from plotly.offline import init_notebook_mode, iplot

from sklearn import preprocessing

import warnings

warnings.filterwarnings("ignore")

init_notebook_mode(connected=True)
def plot_geomap(dataframe):

    """ Function by SKR, the kernel is in the references section """

    colorscale = [[0, 'rgb(102,194,165)'], [0.33, 'rgb(253,174,97)'], [0.66, 'rgb(213,62,79)'], [1.0, 'rgb(158,1,66)']]

    data = [ dict(

            type = 'choropleth',

            autocolorscale = False,

            colorscale = colorscale,

            showscale = False,

            locations = dataframe['Country'],

            z = dataframe['How many votes'].values,

            locationmode = 'country names',

            text = dataframe['Platform'],

            marker = dict(

                line = dict(color = '#fff', width = 2)) )           ]



    layout = dict(

        height=800,

        title = 'Most Used Online Platform by Country',

        geo = dict(

            showframe = True,

            showocean = True,

            oceancolor = '#222',

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

    iplot(fig)
plot_geomap(information_we_need)
from sklearn import preprocessing

lbl = preprocessing.LabelEncoder()

information_we_need['How many votes'] = lbl.fit_transform(information_we_need['Platform'].values)

information_we_need.head()
plot_geomap(information_we_need)