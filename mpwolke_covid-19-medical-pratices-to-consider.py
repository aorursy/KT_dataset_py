#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'data:image/png;base64,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',width=400,height=400)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import feature_extraction, linear_model, model_selection, preprocessing

import plotly.graph_objs as go

import plotly.offline as py

import plotly.graph_objs as go

import plotly.figure_factory as ff



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df1 = pd.read_csv('../input/diabetes/diabetes.csv', encoding='ISO-8859-2')



df1.head()
df = pd.read_csv('../input/cancer-data-2017/cancer2017.csv', encoding='ISO-8859-2')



df.head()
#The evolution of results From Tawej https://www.kaggle.com/tawejssh/covid-19 Graph China/italy

cindicators = df.groupby('State').sum()[['Brain/ nervous system', 'Female breast', 'Colon & rectum', 'Leukemia', 'Liver', 'Lung & bronchus', 'Non-Hodgkin Lymphoma', 'Ovary', 'Pancreas', 'Prostate']]

#evolution['Expiration Rate'] = (evolution['Expired'] / evolution['Cumulative']) * 100

#evolution['Discharging Rate'] = (evolution['Discharged'] / evolution['Cumulative']) * 100

cindicators.head()
plt.figure(figsize=(20,7))

plt.plot(cindicators['Lung & bronchus'], label='Lung cancer')

plt.plot(cindicators['Female breast'], label='Breast cancer')

plt.plot(cindicators['Colon & rectum'], label='Colon cancer')

plt.plot(cindicators['Prostate'], label='Prostate cancer')

plt.plot(cindicators['Liver'], label='Liver cancer')

plt.legend()

#plt.grid()

plt.title('Types of Cancer')

plt.xticks(cindicators.index,rotation=45)

plt.xlabel('States')

plt.ylabel('Cancer Types')

plt.show()
#What about the evolution of China Diagnosed Worldometer rate ?

plt.figure(figsize=(20,7))

plt.plot(cindicators['Lung & bronchus'], label='Lung cancer')

plt.legend()

plt.grid()

plt.title('Lung cancer')

plt.xticks(cindicators.index,rotation=45)

plt.ylabel('Rate %')

plt.show()
y=list(df.columns)

#bdf=df.copy()

#for col in range(1,len(y)):

 #   bdf[y[col]].fillna((bdf[y[col]].mean()), inplace=True)

#bdf.head()
# Code from Mohamed Sharbudeen  https://www.kaggle.com/iamsharbu/cancer2017

x='State'

i=1

z=["prostate","brain","breast","colon","leukemia","liver","lung","lymphoma","ovary","pancreas"]

fig,ax=plt.subplots(nrows=5,ncols=2,figsize=(20,20))

fig.suptitle('Incomplete Data Set')

for row in ax:

    for col in row:

        col.plot(df[x],df[y[i]])

        i=i+1

i=0

for ax in fig.axes:

    plt.xlabel('States')

    plt.ylabel("no of people affected")

    plt.title(z[i])

    i=i+1

    plt.sca(ax)

    plt.xticks(rotation=90)

    plt.grid()

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=1)

fig.tight_layout()

fig.subplots_adjust(top=0.95)

plt.show()
i=1

fig,ax=plt.subplots(nrows=5,ncols=2,figsize=(20,20))

fig.suptitle('Incomplete Data Set')



for row in ax:

    for col in row:

        col.bar(df[x],df[y[i]])

        i=i+1

i=0

for ax in fig.axes:

    plt.xlabel('States')

    plt.ylabel("no of people affected")

    plt.title(z[i])

    i=i+1

    plt.sca(ax)

    plt.xticks(rotation=90)

    plt.grid()

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=1)



fig.tight_layout()

fig.subplots_adjust(top=0.95)

plt.show()
cnt_srs = df['Lung & bronchus'].value_counts().head()

trace = go.Bar(

    y=cnt_srs.index[::-1],

    x=cnt_srs.values[::-1],

    orientation = 'h',

    marker=dict(

        color=cnt_srs.values[::-1],

        colorscale = 'Greens',

        reversescale = True

    ),

)



layout = dict(

    title='Lung Cancer',

    )

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="Lung & bronchus")
cnt_srs = df['Female breast'].value_counts().head()

trace = go.Bar(

    y=cnt_srs.index[::-1],

    x=cnt_srs.values[::-1],

    orientation = 'h',

    marker=dict(

        color=cnt_srs.values[::-1],

        colorscale = 'armyrose',

        reversescale = True

    ),

)



layout = dict(

    title='Breast Cancer',

    )

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="Female breast")
cnt_srs = df['Colon & rectum'].value_counts().head()

trace = go.Bar(

    y=cnt_srs.index[::-1],

    x=cnt_srs.values[::-1],

    orientation = 'h',

    marker=dict(

        color=cnt_srs.values[::-1],

        colorscale = 'magenta',

        reversescale = True

    ),

)



layout = dict(

    title='Colon & rectum Cancer',

    )

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="Colon & rectum")
cnt_srs = df['Prostate'].value_counts().head()

trace = go.Bar(

    y=cnt_srs.index[::-1],

    x=cnt_srs.values[::-1],

    orientation = 'h',

    marker=dict(

        color=cnt_srs.values[::-1],

        colorscale = 'teal',

        reversescale = True

    ),

)



layout = dict(

    title='Prostate Cancer',

    )

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="Prostate")
#Code from Prashant Banerjee @Prashant111

labels = df['State'].value_counts().index

size = df['State'].value_counts()

colors=['#BF3F3F','#3F3FBF']

plt.pie(size, labels = labels, colors = colors, shadow = True, autopct='%1.1f%%',startangle = 90)

plt.title('State', fontsize = 20)

plt.legend()

plt.show()
#Codes from Saurav Anand https://www.kaggle.com/saurav9786/eda-makes-sense



fig=plt.figure(figsize=(20,3))



for i in np.arange(1,7):

    data=plt.subplot(1,7,i,title=df1.columns[i])

    sns.boxplot(df1[df1.columns[i]])
#Codes Nasir Islam Sujam https://www.kaggle.com/nasirislamsujan/intro-to-plotly/data

import plotly
#Codes Nasir Islam Sujam https://www.kaggle.com/nasirislamsujan/intro-to-plotly/data

trace0 = go.Box(

    name = 'Pregnancies',

    y = df1["Pregnancies"]

)



trace1 = go.Box(

    name = "Glucose",

    

    y = df1["Glucose"]

)



trace2 = go.Box(

    name = "BloodPressure",

    y = df1["BloodPressure"]

)



trace3 = go.Box(

    name = "SkinThickness",

    y = df1["SkinThickness"]

)



trace4 = go.Box(

    name = "Insulin",

    y = df1["Insulin"]

)



trace5 = go.Box(

    name = "DiabetesPedigreeFunction",

    y = df1["DiabetesPedigreeFunction"]

)



trace6 = go.Box(

    name = "Age",

    y = df1["Age"]

)



trace7 = go.Box(

    name = "Outcome",

    y = df1["Outcome"]

)



data = [trace0, trace1, trace2, trace3, trace4, trace5, trace6, trace7]

plotly.offline.iplot(data)
#Codes Nasir Islam Sujam https://www.kaggle.com/nasirislamsujan/intro-to-plotly/data

column_names = df1.columns



y_data = df1[df1.columns].values



colors = ['rgba(93, 164, 214, 0.5)', 'rgba(255, 144, 14, 0.5)', 'rgba(44, 160, 101, 0.5)',

          'rgba(255, 65, 54, 0.5)', 'rgba(207, 114, 255, 0.5)', 'rgba(127, 96, 0, 0.5)',

          'rgba(255, 140, 184, 0.5)', 'rgba(79, 90, 117, 0.5)', 'rgba(222, 223, 0, 0.5)']



traces = []



for col_name, yd, color in zip(column_names, y_data, colors):

        traces.append(go.Box(

            y = yd,

            name = col_name,

            boxpoints = 'all',

            jitter = 0.5,

            whiskerwidth = 0.2,

            fillcolor = color,

            marker = dict(

                size = 2,

            ),

            line = dict(width = 1),

        ))



data=traces

plotly.offline.iplot(data)
N = 2000



random_x = np.random.randn(N)

random_y = np.random.randn(N)

# Create a trace

trace = go.Scatter(

    x = random_x,

    y = random_y,

    mode = 'markers'

)



data = [trace]



#plotly.offline.iplot(data, filename='basic-scatter')
#Codes Nasir Islam Sujam https://www.kaggle.com/nasirislamsujan/intro-to-plotly/data

x_data = df1["Glucose"]

y_data = df1["Insulin"]

colors = np.random.rand(2938)

sz = np.random.rand(N)*30



fig = go.Figure()

fig.add_scatter(x = x_data,

                y = y_data,

                mode = 'markers',

                marker = {'size': sz,

                         'color': colors,

                         'opacity': 0.6,

                         'colorscale': 'Portland'

                       })

plotly.offline.iplot(fig)
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'data:image/jpeg;base64,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',width=400,height=400)