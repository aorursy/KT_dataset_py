import numpy as np

import pandas as pd



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls
# Importing the dataset

dataset = pd.read_csv('../input/survey_results_public.csv', header = 0)
dataset.info()
interesting_columns = [

    'Professional',

    'Country',

    'EmploymentStatus',

    'HomeRemote',

    'DeveloperType',

    'YearsProgram',

    'JobSatisfaction',

    'PronounceGIF',

    'ClickyKeys',

    'TabsSpaces',

    'EducationTypes',

    'HaveWorkedLanguage', 

    'WantWorkLanguage',

    'HaveWorkedFramework',

    'WantWorkFramework',

    'IDE',

    'VersionControl'

]

data = pd.DataFrame(dataset, columns=interesting_columns)
plt.figure(figsize=(16,8))

sns.set_style("whitegrid")

plt.title("It's GIF! okay?", fontsize=30, fontweight='bold', y=1.05,)

sns.countplot(y="PronounceGIF", data=data, palette=sns.color_palette("Reds_d"));

plt.xlabel('', fontsize=15)

plt.ylabel('', fontsize=15)

plt.show()
plt.figure(figsize=(16,8))

sns.set_style("whitegrid")

plt.title('Of course that tabs.', fontsize=30, fontweight='bold', y=1.05,)

sns.countplot(y="TabsSpaces", data=data, palette=sns.color_palette("Reds_d"));

plt.xlabel('', fontsize=15)

plt.ylabel('', fontsize=15)

plt.show()
plt.figure(figsize=(16,8))

sns.set_style("whitegrid")

plt.title("Don't worry. The mechanical keyboard is ok.", fontsize=30, fontweight='bold', y=1.05,)

sns.countplot(y="ClickyKeys", data=data, palette=sns.color_palette("Reds_d"));

plt.xlabel('', fontsize=15)

plt.ylabel('', fontsize=15)

plt.show()
import operator

from collections import OrderedDict

languages = {}

for index, row in data.iterrows():

    if (row['HaveWorkedLanguage'] != None and type(row['HaveWorkedLanguage']) != float):

        languages_list = row['HaveWorkedLanguage'].split('; ')

        for lang in languages_list:

            if lang in languages.keys():

                languages[lang][0] += 1

            else:

                languages[lang] = [1,0]

                

    if (row['WantWorkLanguage'] != None and type(row['WantWorkLanguage']) != float):

        languages_list = row['WantWorkLanguage'].split('; ')

        for lang in languages_list:

            if lang in languages.keys():

                languages[lang][1] += 1

            else:

                languages[lang] = [0,1]

                

languages = OrderedDict(sorted(languages.items(), key=operator.itemgetter(1), reverse=True))



languages_names = list(languages.keys())

languages_know = [row[0] for row in list(languages.values())]

languages_learn = [row[1] for row in list(languages.values())]
trace1 = go.Bar(

    x = languages_names,

    y = languages_know,

    name='Most used',

    marker=dict(

        color='rgba(55, 128, 191, 0.7)',

        line=dict(

            color='rgba(55, 128, 191, 1.0)',

            width=2,

        )

    )

)

trace2 = go.Bar(

    x = languages_names,

    y = languages_learn,

    name='Most wanted',

    marker=dict(

        color='rgba(219, 64, 82, 0.7)',

        line=dict(

            color='rgba(219, 64, 82, 1.0)',

            width=2,

        )

    )

)



data = [trace1, trace2]

layout = go.Layout(

    barmode='group',

    title = 'Top languages in 2017',

    width=800,

    height=500,

     margin=go.Margin(

        l=75,

        r=20,

        b=130,

        t=80,

        pad=20

    ),

    paper_bgcolor='rgb(244, 238, 225)',

    plot_bgcolor='rgb(244, 238, 225)',

    yaxis = dict(

        title= 'Number of developers',

        anchor = 'x',

        rangemode='tozero'

    ),

    xaxis = dict(title= 'Languages'),

    legend=dict(x=0.70, y=0.45)

)



fig = go.Figure(data=data, layout=layout)

py.iplot(fig)