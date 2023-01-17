# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.graph_objs as go

import plotly.offline as py

import plotly.express as px



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/covid19-healthy-diet-dataset/Food_Supply_kcal_Data.csv")

df.head()
targets = list(df.columns[0:])

targets
def plot_bar(df, feature, title='Food Supply', show_percent = False, size=2):

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
plot_bar(df, 'Obesity', 'Obesity  count and %Obesity', show_percent=True, size=4)
fig = px.area(df,

            x='Population',

            y='Sugar & Sweeteners',

            template='plotly_dark',

            color_discrete_sequence=['rgb(18, 115, 117)'],

            title='Food Supply kcal',

           )



fig.update_yaxes(range=[0,2])

fig.show()
fig = px.bar(df, 

             x='Country', y='Population', color_discrete_sequence=['#27F1E7'],

             title='Food Supply kcal', text='Sugar & Sweeteners')

fig.show()
fig = px.density_contour(df, x="Country", y="Population", color_discrete_sequence=['purple'])

fig.show()
fig = px.line(df, x="Country", y="Population", color_discrete_sequence=['darkseagreen'], 

              title="Country & Population")

fig.show()
food = df.groupby('Country').count()['Obesity'].reset_index().sort_values(by='Obesity',ascending=False)

food.style.background_gradient(cmap='summer')
fig = go.Figure(go.Funnelarea(

    text =food.Country,

    values = food.Obesity,

    title = {"position": "top center", "text": "Funnel-Chart of Country Distribution"}

    ))

fig.show()