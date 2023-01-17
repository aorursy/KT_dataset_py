import os

import pandas as pd 

pd.set_option('display.max_columns', 10)



base_dir = '../input/'

multiple_path = 'multipleChoiceResponses.csv'

freeForm_path = 'freeFormResponses.csv'

schema_path = 'SurveySchema.csv'



multipla_escolha_df = pd.read_csv(os.path.join(base_dir,multiple_path))

perguntas_livres_df = pd.read_csv(os.path.join(base_dir,freeForm_path))

quantos_responderam_df = pd.read_csv(os.path.join(base_dir,schema_path))
multipla_escolha_df.head()
perguntas_livres_df.head()
quantos_responderam_df.head()
info_necessarias = multipla_escolha_df[

    ['Q3','Q37']

]

info_necessarias.head()
multipla_escolha_df.groupby(['Q3','Q37']).size()
multipla_escolha_df.groupby(['Q3','Q37']).size().reset_index()
info_necessarias = multipla_escolha_df.groupby(['Q3', 'Q37']).size().reset_index()

info_necessarias.columns = ['País', 'Plataforma', 'Quantos votos?']

info_necessarias.head()
info_necessarias = info_necessarias.sort_values(by=['País','Quantos votos?'])

info_necessarias[:10]
info_necessarias.drop_duplicates(subset='País', keep='first').head()
info_necessarias = info_necessarias.drop_duplicates(subset='País', keep='last')

info_necessarias.head()
import matplotlib.pyplot as plt

import numpy as np 

from plotly.offline import init_notebook_mode, iplot

from sklearn import preprocessing

import warnings

warnings.filterwarnings("ignore")

init_notebook_mode(connected=True)
def plotar_geomap(dataframe):

    """ Function by SKR, the kernel is in the references section """

    colorscale = [[0, 'rgb(102,194,165)'], [0.33, 'rgb(253,174,97)'], [0.66, 'rgb(213,62,79)'], [1.0, 'rgb(158,1,66)']]

    data = [ dict(

            type = 'choropleth',

            autocolorscale = False,

            colorscale = colorscale,

            showscale = False,

            locations = dataframe['País'],

            z = dataframe['Quantos votos?'].values,

            locationmode = 'country names',

            text = dataframe['Plataforma'],

            marker = dict(

                line = dict(color = '#fff', width = 2)) )           ]



    layout = dict(

        height=800,

        title = 'Plataforma Online Mais utilizada por País',

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
plotar_geomap(info_necessarias)
from sklearn import preprocessing

lbl = preprocessing.LabelEncoder()

info_necessarias['Quantos votos?'] = lbl.fit_transform(info_necessarias['Plataforma'].values)

info_necessarias.head()
plotar_geomap(info_necessarias)