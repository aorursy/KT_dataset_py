import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.datasets import fetch_20newsgroups_vectorized

from sklearn.feature_selection import chi2

from sklearn.feature_selection import RFE

from sklearn.ensemble import ExtraTreesClassifier

from sklearn import datasets

from sklearn import metrics

import types

from sklearn.manifold import TSNE

import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/Energy Census and Economic Data US 2010-2014.csv")

df.head()
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Number of Rows

len(df.index)
# Column Names

col = list(df.columns)

print(col)
# Assuming: 

# CoalC{year}: Coal total consumption in billion BTU in given year

# 14,000 British thermal units (Btu) per pound

# coal carbon content of 78 percent

# 204.3 pounds of carbon dioxide per million Btu when completely burned

# Source: https://www.eia.gov/coal/production/quarterly/co2_article/co2.html



# 204.3 lb /1 million BTU

# 204300 lb /1 billion BTU

# 2000 lb/ 1 tons



df['2014'] = (df['CoalC2014']*204300)/2000

df['2014'] = round(df['2014'], -1)

df = df[df.StateCodes != 'US']

df[['StateCodes','2014']].head(52)


data = dict(type = 'choropleth',

            colorscale = 'YIOrRd',

            locations = df['StateCodes'],

            locationmode = 'USA-states',

            z = df['2014'],

            text = df['StateCodes'],

            marker = dict(line = dict(color = 'rgb(255,255,255)', width = 2)),

            colorbar = {'title': 'Tons of C02'})
layout = dict(title = 'C02 Production by Coal 2014',

             geo = dict(scope = 'usa',showlakes = True, lakecolor = 'rgb(85, 173, 240)'))
choromap = go.Figure(data = [data], layout=layout)

iplot(choromap)