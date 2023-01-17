import numpy as np

import pandas as pd

import pandas_profiling as pp

import matplotlib.pyplot as plt

import seaborn as sns

import altair as alt

import urllib

import io

from io import StringIO

import requests

import os

from PIL import Image



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        file = os.path.join(dirname, filename)

top50 = pd.read_csv(file, encoding='ISO-8859-1')

top50.head(2)
top50.rename(columns={'Unnamed: 0' : 'No', 'Track.Name': 'Track_Name', 'Artist.Name' : 'Artist_Name',

                      'Beats.Per.Minute' : 'Beats_Per_Min', 'Loudness..dB..' : 'Loudness_DB', 'Valence.' : 'Valence',

                      'Length.' : 'Length','Acousticness..' : 'Acousticness','Speechiness.' : 'Speechiness'},

                      inplace = True

            )
'''

SKEWNESS. 



In statistics, skewness is a measure of the asymmetry of the probability distribution of a random variable about its mean.

If skewness is between -1 and -0.5 or between 0.5 and 1, the distribution is moderately skewed.

If skewness is between -0.5 and 0.5, the distribution is approximately symmetric.

'''

skew = pd.DataFrame(top50.skew(axis=0))

skew.style.bar(align='mid', color=['#2aff00'])
# SELECTION BASED ON THE VALUES OF THE GENRE



input_dropdown = alt.binding_select(options=list(set(top50.Genre)))



selected_points = alt.selection_single(fields=['Genre'], bind=input_dropdown, name='Select')



color = alt.condition(selected_points,



                    alt.Color('Genre:N'),



                    alt.value('lightgray'))



alt.Chart(top50).mark_circle().encode(



    x='Length:Q',



    y='Popularity:Q',



    color=color,



    tooltip='Genre:N'



).add_selection(



    selected_points



)

pts = alt.selection(type="single", encodings=['x'])



rect = alt.Chart(top50).mark_rect().encode(

    alt.X('Loudness_DB:Q', bin=True),

    alt.Y('Danceability:Q', bin=True),

    alt.Color('count()',

        scale=alt.Scale(scheme='greenblue'),

        legend=alt.Legend(title='Total Records')

    )

)



circ = rect.mark_point().encode(

    alt.ColorValue('grey'),

    alt.Size('count()',

        legend=alt.Legend(title='Records in Selection')

    )

).transform_filter(

    pts

)



bar = alt.Chart(top50).mark_bar().encode(

    x='Genre:N',

    y='count()',

    color =alt.condition(pts, alt.ColorValue("steelblue"), alt.ColorValue("grey"))

).properties(

    width=550,

    height=200

).add_selection(pts)



alt.vconcat(

    rect + circ,

    bar

).resolve_legend(

    color="independent",

    size="independent"

)