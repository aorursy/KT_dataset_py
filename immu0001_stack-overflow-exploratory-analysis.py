# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import chart_studio.plotly as py

import plotly.graph_objs as go

from plotly import tools

from plotly.offline import iplot, init_notebook_mode

init_notebook_mode()

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

pd.set_option('display.max_columns', None)

data = pd.read_csv("../input/so-survey-2017/survey_results_public.csv")

data.head(n=2)
data['Country'].value_counts()[0:10].plot(kind='pie',figsize=(12,10))
codes = [data['Country'] if country != 'I prefer not to say' else None for country in data['Country']]
data['Code']=codes

data['Code']
gdata = [ dict(

        type = 'choropleth',

        locations = data['Code'].value_counts().index,

        z = data['Code'].value_counts(),

        text = data['Code'].value_counts().index,

        colorscale = [[0,"rgb(5, 10, 172)"],[0.35,"rgb(40, 60, 190)"],[0.5,"rgb(70, 100, 245)"],\

            [0.6,"rgb(90, 120, 245)"],[0.7,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"]],

        autocolorscale = False,

        reversescale = True,

        marker = dict(

            line = dict (

                color = 'rgb(180,180,180)',

                width = 0.5

            ) ),

        colorbar = dict(

            tickprefix = '#',

            title = '# Devs'),

      ) ]



layout = dict(

    title = 'Developers Distribution',

    geo = dict(

            projection = dict(

                type = 'Mercator'

            ),

            showframe=False

            )

)









figure = dict( data=gdata, layout=layout )

iplot(figure)

languages = [lang for sublist in [str(langs).replace(" ", "").split(";") for langs in data['HaveWorkedLanguage']] for lang in sublist]

languages = pd.Series(languages)
Java = [1 if 'Java' in str(languages) else 0 for languages in data['HaveWorkedLanguage'] ]

Python = [1 if 'Python' in str(languages) else 0 for languages in data['HaveWorkedLanguage'] ]

data['Java']=Java

data['Python'] = Python
data[['Code','JobSatisfaction']].groupby('Code').mean().index
gdata = [ dict(

        type = 'choropleth',

        locations = data[['Code','JobSatisfaction']].groupby('Code').mean().index,

        z = data[['Code','JobSatisfaction']].groupby('Code').mean()['JobSatisfaction'],

        text = data[['Code','JobSatisfaction']].groupby('Code').mean().index,

        colorscale = [[0,"rgb(5, 10, 172)"],[0.35,"rgb(40, 60, 190)"],[0.5,"rgb(70, 100, 245)"],\

            [0.6,"rgb(90, 120, 245)"],[0.7,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"]],

        autocolorscale = False,

        reversescale = True,

        marker = dict(

            line = dict (

                color = 'rgb(180,180,180)',

                width = 0.5

            ) ),

        colorbar = dict(

            tickprefix = '#',

            title = '# Devs'),

      ) ]



layout = dict(

    title = 'Developers Satisfaction',

    geo = dict(

            projection = dict(

                type = 'Mercator'

            ),

            showframe=False

            )

)









figure = dict( data=gdata, layout=layout )

iplot(figure)

data['Professional'].value_counts().plot(kind='pie',figsize=(12,10))
data['FormalEducation'].value_counts().plot(kind='pie',figsize=(12,10))
data[data['Gender']=='Female']['Salary'].describe()
data[data['Gender']=='Male']['Salary'].describe()
