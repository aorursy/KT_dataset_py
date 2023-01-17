# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# import plotly
import plotly.plotly as py
import plotly.graph_objs as go

# these two lines are what allow your code to show up in a notebook!
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
# print(os.listdir("../input"))
path = "../input/"

# Any results you write to the current directory are saved as output. 
users = pd.read_csv(path+"Users.csv")
user_count = users.shape[0]

# Lets convert reg_date to datetime
users['date_parsed'] = pd.to_datetime(users['RegisterDate'], format = "%m/%d/%Y")

year_of_registration = users['date_parsed'].dt.year

competitions = pd.read_csv(path+"Competitions.csv")

competitions['date_parsed'] = pd.to_datetime(competitions['EnabledDate'], format = "%m/%d/%Y %I:%M:%S %p")

year_of_competition = competitions['date_parsed'].dt.year

# plotly histogram chart
user_data = [go.Histogram(x=year_of_registration)]

# specify the layout of our figure
layout = dict(title = "Number of Users per Year",
              xaxis= dict(title= 'Date',ticklen= 5,zeroline= False))

# create and show our figure
fig = dict(data = user_data, layout = layout)
iplot(fig)

competition_data = [go.Histogram(x=year_of_competition)]
# specify the layout of our figure
layout2 = dict(title = "Number of Competitions per Year",
              xaxis= dict(title= 'Date',ticklen= 5,zeroline= False))

# create and show our figure
comp_fig = dict(data = competition_data, layout = layout2)
iplot(comp_fig)

