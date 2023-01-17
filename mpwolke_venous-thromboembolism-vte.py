#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcShHIpdubHvpUTuFU8w_GEhFBfZh04L_boM3psdecQiPg0wNnv2&usqp=CAU',width=400,height=400)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.graph_objs as go

import plotly.offline as py

import plotly.express as px



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcSJeIJf932CO0xINkzXfWW6mleSVLWP7vm67p5Y6GCKV7VF0jn-&usqp=CAU',width=400,height=400)
nRowsRead = 1000 # specify 'None' if want to read whole file

df = pd.read_csv('../input/cusersmarildownloadsthromboembolismcsv/thromboembolism.csv', delimiter=';', encoding = "ISO-8859-1", nrows = nRowsRead)

df.dataframeName = 'thromboembolism.csv'

nRow, nCol = df.shape

print(f'There are {nRow} rows and {nCol} columns')
df.head()
fig,axes = plt.subplots(1,1,figsize=(20,5))

sns.heatmap(df.isna(),yticklabels=False,cbar=False,cmap='viridis')

plt.show()
# Distribution graphs (histogram/bar graph) of column data

def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):

    nunique = df.nunique()

    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values

    nRow, nCol = df.shape

    columnNames = list(df)

    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow

    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')

    for i in range(min(nCol, nGraphShown)):

        plt.subplot(nGraphRow, nGraphPerRow, i + 1)

        columnDf = df.iloc[:, i]

        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):

            valueCounts = columnDf.value_counts()

            valueCounts.plot.bar()

        else:

            columnDf.hist()

        plt.ylabel('counts')

        plt.xticks(rotation = 90)

        plt.title(f'{columnNames[i]} (column {i})')

    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)

    plt.show()
plotPerColumnDistribution(df, 6, 3)
fig = plt.figure(figsize=[10,7])

sns.countplot(df[' Region '], color=sns.xkcd_rgb['greenish cyan'])

plt.title('Venous Thromboembolism')

plt.show()
# filling missing values with NA

df[[' Region ', ' Org Name ', ' VTE Risk Assessed Admissions ', '  Total Admissions  ', 'Percent of admitted patients risk-assessed for VTE']] = df[[' Region ', ' Org Name ', ' VTE Risk Assessed Admissions ', '  Total Admissions  ', 'Percent of admitted patients risk-assessed for VTE']].fillna('NA')
fig = px.bar(df[[' Region ', ' VTE Risk Assessed Admissions ']].sort_values(' VTE Risk Assessed Admissions ', ascending=False), 

             y=" VTE Risk Assessed Admissions ", x=" Region ", color=' Region ', 

             log_y=True, template='ggplot2', title='Venous Thromboembolism')

fig.show()
fig = px.bar(df, 

             x=' Region ', y='  Total Admissions  ', color_discrete_sequence=['#D63230'],

             title='Venous Thromboembolism', text='Percent of admitted patients risk-assessed for VTE')

fig.show()
fig = px.bar(df,

             y=' Region ',

             x='  Total Admissions  ',

             orientation='h',

             color=' Org Name ',

             title='Venous Thromboembolism',

             opacity=0.8, 

             color_discrete_sequence=px.colors.diverging.Armyrose,

             template='plotly_dark'

            )

fig.update_xaxes(range=[0,35])

fig.show()
fig = px.line(df, x=" Region ", y="  Total Admissions  ", color_discrete_sequence=['green'], 

              title="Venous Thromboembolism")

fig.show()
fig = px.pie(df, values=df[' VTE Risk Assessed Admissions '], names=df[' Region '],

             title='Venous Thromboembolism',

            )

fig.update_traces(textposition='inside', textinfo='percent+label')

fig.show()
fig = px.bar(df[[' Org Name ','Percent of admitted patients risk-assessed for VTE']].sort_values('Percent of admitted patients risk-assessed for VTE', ascending=False), 

                        y = "Percent of admitted patients risk-assessed for VTE", x= " Org Name ", color='Percent of admitted patients risk-assessed for VTE', template='ggplot2')

fig.update_xaxes(tickangle=45, tickfont=dict(family='Rockwell', color='crimson', size=14))

fig.update_layout(title_text="Venous Thromboembolism")



fig.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQNcXKdfN8Xyh5VTRYD2Aj4R2JVqaD5gOFSSOCGDpIpxQ7wuTGY&usqp=CAU',width=400,height=400)