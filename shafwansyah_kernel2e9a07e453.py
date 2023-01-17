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


import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.express as px

import plotly.graph_objects as go

from plotly.subplots import make_subplots

import math
hotel_bookings = pd.read_csv("../input/hotel-booking-demand/hotel_bookings.csv")

hotel_bookings.head(2)
hotel_bookings.columns
from plotly.subplots import make_subplots

import plotly.graph_objects as go



fig = make_subplots(rows=2, cols=2, shared_yaxes=True)



distribution_channel = hotel_bookings.groupby(['distribution_channel']).is_canceled.mean().round(2) * 100

arrival_date_month = hotel_bookings.groupby(['arrival_date_month']).is_canceled.mean().round(2) * 100

hotel = hotel_bookings.groupby(['hotel']).is_canceled.mean().round(2) * 100



# Plots

fig.add_trace(go.Bar(x=distribution_channel.index, y=distribution_channel, text=distribution_channel, textposition='auto'),1, 1)

fig.add_trace(go.Bar(x=arrival_date_month.index, y=arrival_date_month, text=arrival_date_month, textposition='auto'),1, 2)

fig.add_trace(go.Bar(x=hotel.index, y=hotel, text=hotel, textposition='auto'),2, 1)



fig.update_layout(height=800, width=1000, title_text="Cancel rate by column")



# Update xaxis properties

fig.update_xaxes(title_text="Distribution Channel", row=1, col=1)

fig.update_xaxes(title_text="arrival date months", row=1, col=2)

fig.update_xaxes(title_text="Hotel Type", row=2, col=1)



# Update yaxis properties

fig.update_yaxes(title_text="Cancel rate in percent (%)", row=1, col=1)

fig.update_yaxes(title_text="Cancel rate in percent (%)", row=1, col=2)

fig.update_yaxes(title_text="Cancel rate in percent (%)", row=2, col=1)



fig.show()