#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQrgdUtLa2QfMk5nGJXqLmYymMk2yrXbmapC6uoiXCtkRGYc3b7',width=400,height=400)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcTi99ufOw_XJLtyZONDgKzUIRsUOho8dh5haQESBgUqNCWUdiZ_',width=400,height=400)
df = pd.read_csv('../input/google-trends-data/20150601_CaitlynJenner.csv', encoding='ISO-8859-2')
df.head()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQKFbA3NCmDDM0XDg9-kw_CGe-yK_b8Unr3nephIUIYzRsaDEAC',width=400,height=400)
df.dtypes
plt.figure(figsize=(16,10))

sns.scatterplot(x='Unnamed: 1',y='Unnamed: 2',data=df)

plt.xticks(rotation=90)

plt.yticks(rotation=45)

plt.show()
import plotly.offline as pyo

import plotly.graph_objs as go

lowerdf1 = df.groupby('Unnamed: 1').size()/df['Unnamed: 2'].count()*100

labels = lowerdf1.index

values = lowerdf1.values



# Use `hole` to create a donut-like pie chart

fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.6)])

fig.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcS1YR-nouuSI9YR_IiNTwcqwzpiUsBnSPfHgNo6Q7T9M7TLBbpV',width=400,height=400)
sns.countplot(df["Unnamed: 1"])

plt.xticks(rotation=90)

plt.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcT1BcUKq9hgOHAYkOmKekKDss8CPhGwfO7gsa_L7UpkcYvqNnkZ',width=400,height=400)