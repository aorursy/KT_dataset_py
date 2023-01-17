# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
free = pd.read_csv(r'/kaggle/input/kaggle-survey-2018/freeFormResponses.csv', sep=',')

mc = pd.read_csv(r'/kaggle/input/kaggle-survey-2018/multipleChoiceResponses.csv', sep=',') 

schema = pd.read_csv(r'/kaggle/input/kaggle-survey-2018/SurveySchema.csv', sep=',')
print(f'freeFormResponses: {free.shape}')

print(f'freeFormResponses: {free.columns}')

print(f'multipleChoiceResponses: {mc.shape}')

print(f'multipleChoiceResponses: {mc.columns}')

print(f'SurveySchema: {schema.shape}')

print(f'SurveySchema: {schema.columns}')

schema.describe()
mc.describe()
import math

import matplotlib.pyplot as plt

import seaborn as sns

from IPython.core import display as ICD

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode, iplot



def plotMultipleChoice(questionNumber):

    yAxisTitle='# of Respondents'

    begin=0

    end=10

    print(questionNumber,':',schema[questionNumber][0])

    counts = mc[questionNumber].value_counts()

    countsDf = pd.DataFrame(counts)

    trace1 = go.Bar(

                    x = countsDf.index,

                    y = countsDf[questionNumber][begin:end],

                    name = "Kaggle",

                    marker = dict(color = 'rgba(0, 0, 255, 0.8)',

                                 line=dict(color='rgb(0,0,0)',width=1.5)),

                    text = countsDf.index)

    data = [trace1]

    layout = go.Layout(barmode = "group",title=schema[questionNumber][0], yaxis= dict(title=yAxisTitle),showlegend=False)

    fig = go.Figure(data = data, layout = layout)

    iplot(fig)

    

plotMultipleChoice('Q1')
plotMultipleChoice('Q2')