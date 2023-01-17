# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import matplotlib.pyplot as plt

import plotly.graph_objects as go

import plotly.express as px

import seaborn as sns

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
Sales = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv')

Calendar = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/calendar.csv')

SapmleSUB = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sample_submission.csv')

SellPrice = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sell_prices.csv')
Sales.info()
Calendar.info()
SellPrice.info()
Calendar.head()
SellPrice.head()
Sales.head()
print("Number of States \n{}".format(Sales['state_id'].value_counts())) 





fig = go.Figure(data=[go.Pie(labels=Sales['state_id'],hole=.4)])

fig.update_layout(

    title="Item Distribution",

    font=dict(

        family="Courier New, monospace",

        size=18

    ))

fig.show()