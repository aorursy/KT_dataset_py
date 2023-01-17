import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

plt.style.use('fivethirtyeight')

import warnings

warnings.filterwarnings('ignore')

import numpy as np

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
answers=pd.read_csv('../input/multipleChoiceResponses.csv',encoding='ISO-8859-1')
response = answers

response[:10]
cols = list(response)
for column in response:

    response[column] = response[column].astype('category')

    response[column] = response[column].cat.codes
response[:10]
train = response



train = train[train.CodeWriter == 1]

colormap = plt.cm.jet

plt.figure(figsize=(40,40))

plt.title('Pearson correlation of All the features', y=1.05, size=25)

sns.heatmap(train.corr(),linewidths=0.05,vmax=.6, square=True, cmap=colormap, linecolor='black', annot=False)
train = response



train = train[train.CodeWriter == 0]

colormap = plt.cm.jet

plt.figure(figsize=(40,40))

plt.title('Pearson correlation of All the features', y=1.05, size=25)

sns.heatmap(train.corr(),linewidths=0.05,vmax=.6, square=True, cmap=colormap, linecolor='black', annot=False)