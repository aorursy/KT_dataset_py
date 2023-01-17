#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcTbf_Ez8tp1GIDUZdWF0z4H5wV4MRvtmaqzvXLYNsZP3FfMZ3ZT&usqp=CAU',width=400,height=400)
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
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcSO-KW-UQQ6KwIJLNR_QBb2ActgYRGGTpkiun3fPjQqO9PAqVMe&usqp=CAU',width=400,height=400)
df = pd.read_csv("../input/darwins-finches-evolution-dataset/finch_beaks_2012.csv")

df.head()
finch = df.groupby('blength').count()['bdepth'].reset_index().sort_values(by='bdepth',ascending=False)

finch.style.background_gradient(cmap='summer')
fig = go.Figure(go.Funnelarea(

    text =finch.blength,

    values = finch.bdepth,

    title = {"position": "top center", "text": "Funnel-Chart of Blength Distribution"}

    ))

fig.show()
px.histogram(df, x='band', color='species')
import seaborn

seaborn.set(rc={'axes.facecolor':'purple', 'figure.facecolor':'purple'})

sns.countplot(df["species"])

plt.xticks(rotation=90)

plt.show()
hist = df[['band','species']]

bins = range(hist.band.min(), hist.band.max()+10, 5)

ax = hist.pivot(columns='species').band.plot(kind = 'hist', stacked=True, alpha=0.5, figsize = (10,5), bins=bins, grid=False)

ax.set_xticks(bins)

ax.grid('on', which='major', axis='x')
fig = px.bar(finch[['blength', 'bdepth']].sort_values('bdepth', ascending=False), 

             y="bdepth", x="blength", color='blength', 

             log_y=True, template='ggplot2', title='Darwins Finches')

fig.show()
finch = finch.sort_values(by=['bdepth'],ascending = False)
plt.figure(figsize=(40,15))

plt.bar(finch.blength, finch.bdepth,label="bdepth")

plt.bar(finch.blength, finch.blength,label="blength")

#plt.bar(finch.blength, finch.yearend,label="yearend")

plt.xlabel('blength')

plt.ylabel("bdepth")

plt.xticks(fontsize=13)

plt.yticks(fontsize=15)



plt.legend(frameon=True, fontsize=12)

plt.title('Darwins Finches',fontsize=30)

plt.show()



f, ax = plt.subplots(figsize=(40,15))

ax=sns.scatterplot(x="blength", y="bdepth", data=finch,

             color="black",label = "bdepth")

#ax=sns.scatterplot(x="ID", y="topic", data=df_grp_rl20,

#             color="red",label = "topic")

#ax=sns.scatterplot(x="blength", y="yearend", data=finch,

 #            color="blue",label = "yearend")

plt.plot(finch.blength,finch.bdepth,zorder=1,color="black")

plt.plot(finch.blength,finch.blength,zorder=1,color="red")

#plt.plot(finch.blength,finch.yearend,zorder=1,color="blue")

plt.xticks(fontsize=13)

plt.yticks(fontsize=15)

plt.legend(frameon=True, fontsize=12)
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcSElSbo5IOvp1jJIltXq7OWJ0cOmklWa6C0x5jRNvDRCnsslkMI&usqp=CAU',width=400,height=400)