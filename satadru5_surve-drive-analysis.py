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

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

color = sns.color_palette()

%matplotlib inline



import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls



pd.options.mode.chained_assignment = None

pd.options.display.max_columns = 999
sc=pd.read_csv("../input/conversionRates.csv", encoding="ISO-8859-1", low_memory=False)
mcr_df = pd.read_csv("../input/multipleChoiceResponses.csv", encoding="ISO-8859-1", low_memory=False)

mcr_df.shape
mcr_df['CurrentJobTitleSelect'].unique()
(np.array((temp_series / temp_series.sum())*100))
temp_series = mcr_df['GenderSelect'].value_counts()

labels = (np.array(temp_series.index))

sizes = (np.array((temp_series / temp_series.sum())*100))



trace = go.Pie(labels=labels, values=temp_series)

layout = go.Layout(

    title='Gender distribution'

)

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="gender")
temp_series = mcr_df['Age'].value_counts()

labels = (np.array(temp_series.index))

sizes = (np.array((temp_series / temp_series.sum())*100))



trace = go.Pie(labels=labels, values=temp_series)

layout = go.Layout(

    title='Age distribution'

)

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="gender")
temp_series = mcr_df['Age'].value_counts()

x1=temp_series.index

y1=temp_series.values

trace=go.Bar(

x=x1,

y=y1   

)

go.Layout(title="Age Distribution")

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="AgeDistribution")
temp_series = mcr_df['EmploymentStatus'].value_counts()

labels = (np.array(temp_series.index))

sizes = (np.array((temp_series / temp_series.sum())*100))



trace = go.Pie(labels=labels, values=temp_series)

layout = go.Layout(

    title='EmploymentStatus distribution'

)

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="EmploymentStatus")
curr=mcr_df['CurrentJobTitleSelect'].value_counts()

trace=go.Bar(

x=curr.index,

    y=curr.values

)

go.Layout(title="CurrentJobTitleSelect Distribution")

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="CurrentJobTitleSelect")
mcr_df.head(3)
em = mcr_df['EmploymentStatus'].value_counts()





trace = go.Bar(

x=em.index,

    y=em.values

)

layout = go.Layout(

    title='EmploymentStatus distribution'

)

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="EmploymentStatus")



ln = mcr_df['LanguageRecommendationSelect'].value_counts()

size = [20, 40, 60, 80, 100, 80, 60, 40, 20, 40,45,89,45]

trace0 = go.Scatter(

    x=ln.index,

    y=ln.values,

    mode='markers',

    marker=dict(

        size=size,

        sizemode='area',

        sizeref=2.*max(size)/(40.**2),

        sizemin=4

    )

)



data = [trace0]

py.iplot(data, filename='bubblechart-color')