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
import plotly.graph_objects as go

from plotly import __version__

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import pandas as pd

import numpy as mp
dataset = pd.read_csv('/kaggle/input/foreign-exchange-rates-per-dollar-20002019/Foreign_Exchange_Rates.csv')
dataset.head(10)
fig = go.Figure()

fig.add_trace(go.Scatter(

x=rates['Time Serie'],

y=rates['CANADA - CANADIAN DOLLAR/US$'],

name="CAD/USD",

line_color='red'))





fig.add_trace(go.Scatter(

x=rates['Time Serie'],

y=rates['CHINA - YUAN/US$'],

name="China/USD",

line_color='blue'))



fig.add_trace(go.Scatter(

x=rates['Time Serie'],

y=rates['EURO AREA - EURO/US$'],

name="EUR/USD",

line_color='Orange'))



fig.add_trace(go.Scatter(

x=rates['Time Serie'],

y=rates['UNITED KINGDOM - UNITED KINGDOM POUND/US$'],

name="GPB/USD",

line_color='Green'))



fig.add_trace(go.Scatter(

x=rates['Time Serie'],

y=rates['NORWAY - NORWEGIAN KRONE/US$'],

name="NRW/USD",

line_color='red'))











fig.update_layout(title_text="Daily Exchange Rates (2000 - 2019)")

fig.show()
fig = go.Figure()

fig.add_trace(go.Scatter(

x=rates['Time Serie'],

y=rates['CANADA - CANADIAN DOLLAR/US$'],

name="CAD/USD",

line_color='red'))





fig.add_trace(go.Scatter(

x=rates['Time Serie'],

y=rates['CHINA - YUAN/US$'],

name="China/USD",

line_color='blue'))



fig.add_trace(go.Scatter(

x=rates['Time Serie'],

y=rates['EURO AREA - EURO/US$'],

name="EUR/USD",

line_color='Orange'))



fig.add_trace(go.Scatter(

x=rates['Time Serie'],

y=rates['UNITED KINGDOM - UNITED KINGDOM POUND/US$'],

name="GPB/USD",

line_color='Green'))



fig.add_trace(go.Scatter(

x=rates['Time Serie'],

y=rates['NORWAY - NORWEGIAN KRONE/US$'],

name="NRW/USD",

line_color='red'))











fig.update_layout(xaxis_range=['2000-01-01','2010-12-31'],

                   title_text="Daily Exchange Rates (2000 - 2010)",

                 xaxis_rangeslider_visible=True)



fig.show()