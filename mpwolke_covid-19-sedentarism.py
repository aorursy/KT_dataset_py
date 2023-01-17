#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQ6I1KwX4nch95kH9YnrPPMjWLbyBxYK3jguwrIoiL2-kpxHwED&usqp=CAU',width=400,height=400)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import feature_extraction, linear_model, model_selection, preprocessing

import plotly.graph_objs as go

import plotly.offline as py

import plotly.express as px

from plotly.offline import iplot

import seaborn



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/uncover/UNCOVER/us_cdc/us_cdc/nutrition-physical-activity-and-obesity-behavioral-risk-factor-surveillance-system.csv', encoding='ISO-8859-2')

df.head()
hist = df[['locationid','yearstart']]

bins = range(hist.locationid.min(), hist.locationid.max()+10, 5)

ax = hist.pivot(columns='yearstart').locationid.plot(kind = 'hist', stacked=True, alpha=0.5, figsize = (10,5), bins=bins, grid=False)

ax.set_xticks(bins)

ax.grid('on', which='major', axis='x')
bboxtoanchor=(1.1, 1.05)

#seaborn.set(rc={'axes.facecolor':'03fc28', 'figure.facecolor':'03fc28'})

df.plot.area(y=['yearstart','yearend','locationid'],alpha=0.4,figsize=(12, 6));
from IPython.display import IFrame

IFrame('https://app.powerbi.com/view?r=eyJrIjoiZTU5ZDE5MGYtMzUzMy00ZjRmLTg4MGEtMTM3ZGJiZDNhODFkIiwidCI6IjZiOTAyNjkzLTEwNzQtNDBhYS05ZTIxLWQ4OTQ0NmEyZWJiNSIsImMiOjh9', width=800, height=500)
df_grp = df.groupby(["class","stratificationcategory1", "datasource", "topic", "stratification1"])[["yearstart","yearend","locationid"]].sum().reset_index()

df_grp.head()
plt.figure(figsize=(15, 5))

plt.title('topic')

df_grp.topic.value_counts().plot.bar();
df_grp_r = df_grp.groupby("topic")[["class","stratification1","yearstart", "yearend"]].sum().reset_index()
df_grp_r.head()
df_grp_rl20 = df_grp_r.tail(20)
fig = px.bar(df_grp_rl20[['topic', 'yearstart']].sort_values('yearstart', ascending=False), 

             y="yearstart", x="topic", color='topic', 

             log_y=True, template='ggplot2', title='Covi-19 and Sedentarism')

fig.show()
df_grp_rl20 = df_grp_rl20.sort_values(by=['yearstart'],ascending = False)
plt.figure(figsize=(40,15))

plt.bar(df_grp_rl20.topic, df_grp_rl20.yearstart,label="yearstart")

plt.bar(df_grp_rl20.topic, df_grp_rl20.topic,label="topic")

plt.bar(df_grp_rl20.topic, df_grp_rl20.yearend,label="yearend")

plt.xlabel('topic')

plt.ylabel("yearstart")

plt.xticks(fontsize=13)

plt.yticks(fontsize=15)



plt.legend(frameon=True, fontsize=12)

plt.title('Covid-19 and Sedentarism',fontsize=30)

plt.show()



f, ax = plt.subplots(figsize=(40,15))

ax=sns.scatterplot(x="topic", y="yearstart", data=df_grp_rl20,

             color="black",label = "yearstart")

#ax=sns.scatterplot(x="ID", y="topic", data=df_grp_rl20,

#             color="red",label = "topic")

ax=sns.scatterplot(x="topic", y="yearend", data=df_grp_rl20,

             color="blue",label = "yearend")

plt.plot(df_grp_rl20.topic,df_grp_rl20.yearstart,zorder=1,color="black")

plt.plot(df_grp_rl20.topic,df_grp_rl20.topic,zorder=1,color="red")

plt.plot(df_grp_rl20.topic,df_grp_rl20.yearend,zorder=1,color="blue")

plt.xticks(fontsize=13)

plt.yticks(fontsize=15)

plt.legend(frameon=True, fontsize=12)
#word cloud

from wordcloud import WordCloud, ImageColorGenerator

text = " ".join(str(each) for each in df.topic)

# Create and generate a word cloud image:

wordcloud = WordCloud(max_words=200,colormap='Set3', background_color="green").generate(text)

plt.figure(figsize=(10,6))

plt.figure(figsize=(15,10))

# Display the generated image:

plt.imshow(wordcloud, interpolation='Bilinear')

plt.axis("off")

plt.figure(1,figsize=(12, 12))

plt.show()
from IPython.display import IFrame

IFrame('https://app.powerbi.com/view?r=eyJrIjoiMjcxNDIyNjAtOGM0Yi00ZWJhLWJkNmEtNjFiOTI0MWVlYjNiIiwidCI6IjI1NmNiNTA1LTAzOWYtNGZiMi04NWE2LWEzZTgzMzI4NTU3OCIsImMiOjh9', width=800, height=500)
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://miro.medium.com/max/1400/1*ggUhr7rH00aWdMz97PKXaA.png',width=400,height=400)