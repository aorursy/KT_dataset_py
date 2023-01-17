# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
import seaborn as sn
sn.set(color_codes=True, style="white")
import matplotlib.pyplot as ml
import numpy as np 
import pandas as pd 
import statsmodels.formula.api as stats
from statsmodels.formula.api import ols
import sklearn
from sklearn import linear_model, datasets
from sklearn.metrics import mean_squared_error
import plotly.plotly as py 
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
import warnings
warnings.filterwarnings("ignore")
hap=pd.read_csv("../input/2017.csv",sep=",",header=0)


print(hap.shape)
print(hap.head(11))
hap_cor=hap.corr()
print(hap_cor)
sn.heatmap(hap_cor, 
        xticklabels=hap_cor.columns,
        yticklabels=hap_cor.columns)
data6 = dict(type = 'choropleth', 
           locations = hap['Country'],
           locationmode = 'country names',
           z = hap['Happiness.Rank'], 
           text = hap['Country'],
          colorscale = 'Viridis', reversescale = False)
layout = dict(title = 'Happiness Rank World Map', 
             geo = dict(showframe = False, 
                       projection = {'type': 'Mercator'}))
choromap6 = go.Figure(data = [data6], layout=layout)
iplot(choromap6)
trace4 = go.Scatter(
    x = hap["Economy..GDP.per.Capita."],
    y = hap["Happiness.Rank"],
    mode = 'markers'
)
data4 = [trace4]
layout = go.Layout(
    title='Happiness Rank Determined by Economy',
    xaxis=dict(
        title='Economy, GDP',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    ),
    yaxis=dict(
        title='Happiness Rank',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    )
)

fig4 = go.Figure(data=data4, layout=layout)
iplot(fig4)
