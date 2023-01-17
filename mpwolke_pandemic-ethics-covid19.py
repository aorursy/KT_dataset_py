#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQRRj-tfuSzPt7o3-94o2wQ27_miKovtxRB8SU_iEbkwGH0f70iLQ&s',width=400,height=400)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

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

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcRCq1BjY4rDedVjh7a3J2v-xiu5de7zglb6QNCOicbHw8iTUgRR',width=400,height=400)
df = pd.read_csv('/kaggle/input/CORD-19-research-challenge/2020-03-13/all_sources_metadata_2020-03-13.csv')

df.head()
title = df.copy()



title = title.dropna(subset=['title'])



title.head()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQzJ1llu225VfMQ0wsvmFnBqGKQwABOFyjJt58-Hek6hOtP_7N8',width=400,height=400)
title['title'] = title['title'].str.replace('[^a-zA-Z]', ' ', regex=True)

title['title'] = title['title'].str.lower()



title.tail()
title['keyword_ethics'] = title['title'].str.find('ethics') 
title.head()
included_ethics = title.loc[title['keyword_ethics'] != -1]

included_ethics
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQpGi8_FkX2_6x41HMbV_uhwSgLg3poUo6m0KU7yKqGwQ4dwiKt',width=400,height=400)
import json

file_path = '/kaggle/input/CORD-19-research-challenge/2020-03-13/comm_use_subset/comm_use_subset/a95e3a09d229a35ff2189682c618cff5add85e4a.json'

with open(file_path) as json_file:

     json_file = json.load(json_file)

json_file
sns.countplot(df["has_full_text"])

plt.xticks(rotation=90)

plt.show()
ethics = pd.read_csv("../input/trolley-dilemma/trolley.csv")

ethics.head()
sns.distplot(ethics["age"].apply(lambda x: x**4))

plt.show()
sns.distplot(ethics["health"].apply(lambda x: x**4))

plt.show()
sns.distplot(ethics["expected_years_left"].apply(lambda x: x**4))

plt.show()
import json

file_path = '/kaggle/input/CORD-19-research-challenge/2020-03-13/comm_use_subset/comm_use_subset/f91df0d67f4d6fc329815c233f76b5ef0dd3876a.json'

with open(file_path) as json_file:

     json_file = json.load(json_file)

json_file
#word cloud

from wordcloud import WordCloud, ImageColorGenerator

text = " ".join(str(each) for each in df.source_x)

# Create and generate a word cloud image:

wordcloud = WordCloud(max_words=200,colormap='Set3', background_color="black").generate(text)

plt.figure(figsize=(10,6))

plt.figure(figsize=(15,10))

# Display the generated image:

plt.imshow(wordcloud, interpolation='Bilinear')

plt.axis("off")

plt.figure(1,figsize=(12, 12))

plt.show()
fig = px.pie( values=ethics.groupby(['smoking']).size().values,names=ethics.groupby(['smoking']).size().index)

fig.update_layout(

    title = "Smokers",

    font=dict(

        family="Arial, monospace",

        size=15,

        color="#7f7f7f"

    )

    )   

    

py.iplot(fig)
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQNP_nQM3D8Q82PqZnxjOZjNrXW776c5jAmXLcAgHcdeW-cqOO4',width=400,height=400)