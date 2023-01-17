# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2

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
df = pd.read_csv("../input/covid19-latent-period-computational-experiment/simTables/Italy-0_1.csv")

df.head()
from PIL import Image

im = Image.open("../input/covid19-latent-period-computational-experiment/fitFigures/Italy-2.png")

#tlabel = np.asarray(Image.open("../input/train_label/170908_061523257_Camera_5_instanceIds.png")) // 1000

#tlabel[tlabel != 0] = 255

# plt.imshow(Image.blend(im, Image.fromarray(tlabel).convert('RGB'), alpha=0.4))

plt.imshow(im)

display(plt.show())
df = df.rename(columns={'infected.obs':'infobs', 'removed.obs': 'remobs', 'susceptible.pred': 'suscep', 'exposed.pred': 'expos', 'infected.pred': 'infpred', 'removed.pred': 'remov'})
import seaborn

import matplotlib.pyplot as plt

import numpy as np



t = np.linspace(0.0, 2.0, 201)

s = np.sin(2 * np.pi * t)
seaborn.set(rc={'axes.facecolor':'magenta', 'figure.facecolor':'magenta'})

df["infobs"].plot.hist()

plt.show()
seaborn.set(rc={'axes.facecolor':'#27F1E7', 'figure.facecolor':'27F1E7'})

ax = df.plot.barh(x= 'infobs', y = 'days')
seaborn.set(rc={'axes.facecolor':'crimson', 'figure.facecolor':'crimson'})

#sns.set(style="darkgrid")

fig=plt.gcf()

fig.set_size_inches(10,7)

fig = sns.swarmplot(x="days", y="infobs", data=df)
from PIL import Image

im = Image.open("../input/covid19-latent-period-computational-experiment/fitFigures/Italy-0_167.png")

#tlabel = np.asarray(Image.open("../input/train_label/170908_061523257_Camera_5_instanceIds.png")) // 1000

#tlabel[tlabel != 0] = 255

# plt.imshow(Image.blend(im, Image.fromarray(tlabel).convert('RGB'), alpha=0.4))

plt.imshow(im)

display(plt.show())
import seaborn

seaborn.set(rc={'axes.facecolor':'magenta', 'figure.facecolor':'magenta'})

t = np.linspace(0.0, 2.0, 201)

s = np.sin(2 * np.pi * t)



#fig, ax = plt.subplots()



fig=sns.lmplot(x="days", y="remobs",data=df)

#ax.set_facecolor('#7F3FBF')

#ax.plot(t, s, 'xkcd:crimson')
seaborn.set(rc={'axes.facecolor':'magenta', 'figure.facecolor':'magenta'})

df.plot.area(y=['days','infobs','remobs'],alpha=0.4,figsize=(12, 6));
im = Image.open("../input/covid19-latent-period-computational-experiment/fitFigures/Italy-0_1.png")

plt.imshow(im)

display(plt.show())
fig = px.scatter(df, x= "days", y= "infobs")

fig.show()
fig = px.bar(df, x= "days", y= "remobs")

fig.show()
px.histogram(df, x='remobs', color='days')
seaborn.set(rc={'axes.facecolor':'magenta', 'figure.facecolor':'magenta'})

# Set the width and height of the figure

plt.figure(figsize=(10,6))



# Add title

plt.title("Covid-19 Computational Experiment")



# Bar chart showing average arrival delay for Spirit Airlines flights by month

sns.barplot(x=df.index, y=df['expos'])



# Add label for vertical axis

plt.ylabel("Covid-19 Computational Experiment ")
fig = px.line(df, x="days", y="remobs", 

              title="Covid-19 Computational Experiment")

fig.show()
im = Image.open("../input/covid19-latent-period-computational-experiment/fitFigures/Italy-0_071.png")

plt.imshow(im)

display(plt.show())