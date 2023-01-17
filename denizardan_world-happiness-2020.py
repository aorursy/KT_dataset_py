# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # viz 

import matplotlib.pyplot as plt #viz

import plotly.express as px

import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

import warnings

warnings.filterwarnings('ignore') 







# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/world-happiness-report-2020/WHR20_DataForFigure2.1.csv',index_col="Country name")
# At glance the data

df.head()
# We need target and predictors



data = df.loc[:,"Regional indicator":"Perceptions of corruption"].drop(df.loc[:,"Standard error of ladder score":"lowerwhisker"], axis=1)

data.head()
#check data types and missing variables



data.info()
# Lets see the summary statistics on numeric(float) variables



data.describe()
# Summarize Regional indicator



reg_cnt = data["Regional indicator"].nunique()

reg_name = data["Regional indicator"].unique()

reg_cnt_val = data["Regional indicator"].value_counts()



print("Number of Regions: " + str(reg_cnt),"\n")

print("Name of Regions:\n " + str(reg_name),"\n")

print("Value Count of Regions:\n", reg_cnt_val)

#calculating correlation

cor = data.corr()



#heatmap as below

sns.heatmap(cor, square = True, cmap="coolwarm",annot=True,linewidths=0.5)



plt.show()
# for loop for each regions 



for i in reg_name:

    ax = plt.axes()

    corc = data[data["Regional indicator"]==i].corr()

    sns.heatmap(corc, square = True, cmap="coolwarm",annot=True,linewidths=0.5)

    ax.set_title(i)

    plt.show()
sns.clustermap(data.corr(), center=0, cmap="vlag", z_score=0, linewidths=.75)
sns.clustermap(data.select_dtypes(include="float"), center=0, cmap="vlag", z_score=0, linewidths=.75,figsize=(10,50))
fig = px.scatter(data, x="Logged GDP per capita", y="Ladder score", size="Social support", color="Regional indicator", hover_name=data.index, size_max=20)



fig.show()
sns.boxplot(x=data["Logged GDP per capita"], y = data["Regional indicator"],palette = "pastel")
sns.violinplot(x=data["Logged GDP per capita"], y = data["Regional indicator"], scale = "width",palette="Set3")
sns.set(style="whitegrid")

sns.boxenplot(x=data["Logged GDP per capita"], y = data["Regional indicator"],scale="linear")
sns.swarmplot(x=data["Logged GDP per capita"], y = data["Regional indicator"])
disp = sns.PairGrid(data, diag_sharey=False)

disp.map_upper(sns.scatterplot)

disp.map_lower(sns.kdeplot, colors="C0")

disp.map_diag(sns.kdeplot, lw=2)
sns.pairplot(data, hue = "Regional indicator")
data = pd.read_csv('/kaggle/input/world-happiness-report-2020/WHR20_DataForFigure2.1.csv')



data = dict(type = 'choropleth', 

           locations = data['Country name'],

           locationmode = 'country names',

           z = data['Ladder score'], 

           text = data['Country name'],

           colorbar = {'title':'Happiness'})

layout = dict(title = 'Happiness Score 2020', 

             geo = dict(showframe = False,

                       showocean = False,

                       showlakes = True,

                       showcoastlines = True,

                       projection = {'type': 'natural earth'}))

map_ = go.Figure(data = data, layout=layout)

iplot(map_)