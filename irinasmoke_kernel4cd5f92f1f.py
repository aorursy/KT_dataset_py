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
import plotly.graph_objects as go

import time,sys



percent=.13

rate=275

lower_bound=rate-(rate*.13)

upper_bound=rate*1.13



steps = [{'range': [200, lower_bound], 'color': "lightgray"},

        {'range': [lower_bound, upper_bound], 'color': "green"}, 

         {'range': [upper_bound, 300], 'color': "lightgray"}]





fig_gague = go.Figure(go.Indicator(

    domain = {'x': [0, 1], 'y': [0, 1]},

    value = rate,

    mode = "gauge+number",

    title = {'text': "Rate"},

    delta = {'reference': 380},

    gauge = {'shape':'angular',

            'axis': {'range': [None, 500]},

             'steps': steps,

             'bar': {'color': "darkblue",

                    'thickness':0, 'line':{'width': 0}},

             'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 1, 'value': rate}}))



fig_gague.show()





fig_bullet = go.Figure(go.Indicator(

    domain = {'x': [0, 1], 'y': [0, 1]},

    value = rate,

    mode = "gauge+number+delta",

    title = {'text': "Rate"},

    delta = {'reference': 380},

    gauge = {'shape':'bullet',

            'axis': {'range': [None, 500]},

             'steps' : [

                 {'range': [0, 250], 'color': "lightgray"},

                 {'range': [250, 400], 'color': "orange"}],

 

             'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 490}}))



fig_bullet.show()







    