# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# Basic Libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Interactive Visualization

import plotly.graph_objects as go

import plotly.express as px

from plotly.subplots import make_subplots



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# read the data

cm=pd.read_excel("/kaggle/input/Celebrity Marriages.xlsx")



#view the first 10 items in the dataset

cm.head(10)
# Are celebrities more likely to marry their fellow celebs or not



# This counts the values of the Spouse_Celebrity Column

marriages=cm.Spouse_Celebrity.value_counts().head(2).reset_index()

marriages.columns=["Celebrity or Not","Total"]

marriages
# Time to Visualize this shit

fig = px.pie(marriages, values='Total', names='Celebrity or Not', title='Do Celebs Marry Their Fellow Celebs?')

fig.update_traces(textposition='inside', textinfo='percent+label')

fig.show()
# Is there any correlation between Celeb/Celeb Marriages and divorces



# First get the data on celebrity-celebrity marriage

celebs_only=cm[cm['Spouse_Celebrity']=='Yes']

celebs_only.head(5)
# What is the divorce rate in celebs only marriages

celeb_divorces=celebs_only.Divorced.value_counts().head(2).reset_index()

celeb_divorces.columns=['Divorced','Total']

celeb_divorces
# Visualize this shit

fig = px.pie(celeb_divorces, values='Total', names='Divorced', title='Do Celeb-Celeb Marriages Lead to Divorce?')

fig.update_traces(textposition='inside', textinfo='percent+label')

fig.show()