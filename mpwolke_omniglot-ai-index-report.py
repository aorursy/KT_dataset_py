#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTvar6Bz2aAPqIcedCp2Xm5_VWqjXEjD22OgdJBnXvjD5Weci0uOQ&s',width=400,height=400)
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

Image(url = 'https://www.omniglot.com/images/banners/aboutme.gif',width=400,height=400)
df = pd.read_excel('/kaggle/input/the-2019-ai-index-report/AI INDEX 2019 PUBLIC DATA/3. Technical Performance/Omniglot/Omniglot.xlsx')

df.head()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR2KVrg9CzgFf_3rVbHtUyXs4qlAR9zUNp3xnnYSYTayE5FZnJP0A&s',width=400,height=400)
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQFOvQu8mMY--ZVnS5q2KY2Qpy9cMVGw-cYrw3850bHaTPMlBq7&s',width=400,height=400)
df.dtypes
df = df.rename(columns={ 'Unnamed: 0':'Unnamed', 'Unnamed: 1': 'year', 'error rate': 'error_rate', 'Unnamed: 6': 'alphabet', 'Unnamed: 7': 'human', 'Unnamed: 8': 'BPL'})
print("The number of nulls in each column are \n", df.isna().sum())
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTIOMPQDENRG_EA3Nbh3jbQpyaa5uztfK-XRuTBBvmksR42ZkM_&s',width=400,height=400)
sns.countplot(df["error_rate"])

plt.xticks(rotation=45)

plt.yticks(rotation=45)

plt.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQY7YAqIqLIooPE1eiRxEPrbbBY8xEK8VArdH_dcGoKQJ5khSzg&s',width=400,height=400)
df = df.rename(columns={ 'Unnamed: 0':'Unnamed'})
import plotly.offline as pyo

import plotly.graph_objs as go

lowerdf = df.groupby('error_rate').size()/df['human'].count()*100

labels = lowerdf.index

values = lowerdf.values



# Use `hole` to create a donut-like pie chart

fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.6)])

fig.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ2jwg73ed_5A6yXTk3PWwwy8NDsWgxGacSUof_UHYVvrEscU2d4Q&s',width=400,height=400)
#word cloud

from wordcloud import WordCloud, ImageColorGenerator

text = " ".join(str(each) for each in df.error_rate)

# Create and generate a word cloud image:

wordcloud = WordCloud(max_words=200, background_color="white").generate(text)

plt.figure(figsize=(10,6))

plt.figure(figsize=(15,10))

# Display the generated image:

plt.imshow(wordcloud, interpolation='Bilinear')

plt.axis("off")

plt.show()
import plotly.offline as pyo

import plotly.graph_objs as go

lowerdf = df.groupby('Unnamed').size()/df['alphabet'].count()*100

labels = lowerdf.index

values = lowerdf.values



# Use `hole` to create a donut-like pie chart

fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.6)])

fig.show()
#word cloud

from wordcloud import WordCloud, ImageColorGenerator

text = " ".join(str(each) for each in df.Unnamed)

# Create and generate a word cloud image:

wordcloud = WordCloud(max_words=200, background_color="black").generate(text)

plt.figure(figsize=(10,6))

plt.figure(figsize=(15,10))

# Display the generated image:

plt.imshow(wordcloud, interpolation='Bilinear')

plt.axis("off")

plt.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSczyhmfI98klzegk54xX1nB4c0vv4j0bRA0qO5Kw7mIPfkPavE&s',width=400,height=400)
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS0ZT_cDittI8lCsp9uuidAb7SS6ZLVM1bsNFy06OaNPJqOIeweGQ&s',width=400,height=400)