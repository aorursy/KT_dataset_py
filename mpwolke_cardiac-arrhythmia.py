#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcRzAk7IsrvoQWoyrbU_x8MEeL8aV75K1NoaIY3OQm1HHaZI7Gez&usqp=CAU',width=400,height=400)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

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
nRowsRead = 1000 # specify 'None' if want to read whole file

df = pd.read_csv('../input/arrhythmia1/data_arrhythmia1.csv', delimiter=';', encoding = "ISO-8859-1", nrows = nRowsRead)

df.dataframeName = 'data_arrhythmia1.csv'

nRow, nCol = df.shape

print(f'There are {nRow} rows and {nCol} columns')
df.head()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcS-dwLKHrSCAifcISmdcRmCksOSdw_cOt_IUZLgjB1OuDP3py3M&usqp=CAU',width=400,height=400)
df.plot.area(y=['age','sex','height','weight','qrs_duration','q-t_interval','t_interval','p_interval', 'diagnosis'],alpha=0.4,figsize=(12, 6));
df["diagnosis"].plot.hist()

plt.show()
fig = px.bar(df,

             y='t_interval',

             x='diagnosis',

             orientation='h',

             color='age',

             title='Cardiac Arrhythmia',

             opacity=0.8,

             color_discrete_sequence=px.colors.diverging.Armyrose,

             template='plotly_dark'

            )

fig.update_xaxes(range=[0,35])

fig.show()
fig = px.bar(df, x= "diagnosis", y= "qrs_duration", color_discrete_sequence=['crimson'],)

fig.show()
fig = px.bar(df, x= "age", y= "diagnosis", color_discrete_sequence=['crimson'],)

fig.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcTgTkM4HE8PssA8gdwOOS5imUl0KIDNvcd3JLduQ2mlmCXhnK3T&usqp=CAU',width=400,height=400)
fig = px.line(df, x="age", y="t_interval", color_discrete_sequence=['green'], 

              title="Cardiac Arrhythmia")

fig.show()
fig = px.bar(df, y="diagnosis", x="age", color="weight", orientation="h",

             color_continuous_scale='Bluered_r', hover_name="t_interval")



fig.show()
fig = px.scatter(df, x="age", y="diagnosis", color="qrs_duration",

                 color_continuous_scale=["red", "green", "blue"])



fig.show()
fig = px.parallel_coordinates(df, color="t_interval",

                             color_continuous_scale=[(0.00, "red"),   (0.33, "red"),

                                                     (0.33, "green"), (0.66, "green"),

                                                     (0.66, "blue"),  (1.00, "blue")])

fig.show()
fig = px.pie(df, values=df['q-t_interval'], names=df['age'],

             title='Cardiac Arrhythmia',

            )

fig.update_traces(textposition='inside', textinfo='percent+label')

fig.show()
fig = px.bar(df[['t_interval','diagnosis']].sort_values('diagnosis', ascending=False), 

                        y = "diagnosis", x= "t_interval", color='diagnosis', template='ggplot2')

fig.update_xaxes(tickangle=45, tickfont=dict(family='Rockwell', color='crimson', size=14))

fig.update_layout(title_text="Cardiac Arrhythmia")



fig.show()
fig = px.scatter(df, x="age", y="diagnosis", color="t_interval", marginal_y="rug", marginal_x="histogram")

fig
def plot_bar(df, feature, title='Cardiac Arrhythmia', show_percent = False, size=2):

    f, ax = plt.subplots(1,1, figsize=(4*size,3*size))

    total = float(len(df))

    sns.barplot(np.round(df[feature].value_counts().index).astype(int), df[feature].value_counts().values, alpha=0.8, palette='Set2')



    plt.title(title)

    if show_percent:

        for p in ax.patches:

            height = p.get_height()

            ax.text(p.get_x()+p.get_width()/2.,

                    height + 3,

                    '{:1.2f}%'.format(100*height/total),

                    ha="center", rotation=45) 

    plt.xlabel(feature, fontsize=12, )

    plt.ylabel('Number of Occurrences', fontsize=12)

    plt.xticks(rotation=90)

    plt.show()
plot_bar(df, 'age', 'age  count and %age', show_percent=True, size=4)
hist = df[['age','diagnosis']]

bins = range(hist.age.min(), hist.age.max()+10, 5)

ax = hist.pivot(columns='diagnosis').age.plot(kind = 'hist', stacked=True, alpha=0.5, figsize = (10,5), bins=bins, grid=False)

ax.set_xticks(bins)

ax.grid('on', which='major', axis='x')
arrhythmia = df.groupby('age').count()['diagnosis'].reset_index().sort_values(by='diagnosis',ascending=False)

arrhythmia

fig = go.Figure(go.Funnelarea(

    text =arrhythmia.diagnosis,

    values = arrhythmia.diagnosis,

    title = {"position": "top center", "text": "Funnel-Chart of Cardiac Arrhythmia Distribution"}

    ))

fig.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcTF3WxrWMNn2aY0FZplkChlvXGp7JLIo16r5_5wJcQUVHTDE8g4&usqp=CAU',width=400,height=400)