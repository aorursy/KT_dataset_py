#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcSlxW_H3CMwIcKZb48Hu8c9w3uG5VNx5cOhG3WZBzjtX1xN2rfq',width=400,height=400)
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

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcRRffmJENxcC0fHkOcQwpSDHbJzRummirTVpbp3s3eu5cdsyu4C',width=400,height=400)
df = pd.read_excel('/kaggle/input/the-2019-ai-index-report/AI INDEX 2019 PUBLIC DATA/3. Technical Performance/Vision/Semantic Segmentation/Pascal VOC 2012.xlsx')

df.head()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQ4o6X4ArRuD0xKvGv_BcH5f0JjUGCbKrki6RY8TI_rGXrJZrE2',width=400,height=400)
df.shape
df.dtypes
sns.distplot(df["Rank"].apply(lambda x: x**4))

plt.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcRL6MG_xa-oDBV0qkhvsiYObhATzVnWDAMYmLXzkJRepHbEHHB6',width=400,height=400)
#word cloud

from wordcloud import WordCloud, ImageColorGenerator

text = " ".join(str(each) for each in df.Method)

# Create and generate a word cloud image:

wordcloud = WordCloud(max_words=200, background_color="black").generate(text)

plt.figure(figsize=(10,6))

plt.figure(figsize=(15,10))

# Display the generated image:

plt.imshow(wordcloud, interpolation='Bilinear')

plt.axis("off")

plt.show()
df.groupby(['Rank']).size().plot.bar()
# Necessary Functions: 

def pie_plot(labels, values, colors, title):

    fig = {

      "data": [

        {

          "values": values,

          "labels": labels,

          "domain": {"x": [0, .48]},

          "name": "Job Type",

          "sort": False,

          "marker": {'colors': colors},

          "textinfo":"percent+label+value",

          "textfont": {'color': '#FFFFFF', 'size': 10},

          "hole": .6,

          "type": "pie"

        } ],

        "layout": {

            "title":title,

            "annotations": [

                {

                    "font": {

                        "size": 25,



                    },

                    "showarrow": False,

                    "text": ""



                }

            ]

        }

    }

    return fig
import plotly.offline as py

value_counts = df['Rank'].value_counts()

labels = value_counts.index.tolist()

py.iplot(pie_plot(labels, value_counts,['#1B9E77', '#7570B3'], "Rank"))
df1 = pd.read_excel('/kaggle/input/the-2019-ai-index-report/AI INDEX 2019 PUBLIC DATA/3. Technical Performance/Vision/Semantic Segmentation/PASCAL Context.xlsx')

df1.head()
corrs = df.corr()

corrs
plt.figure(figsize = (20, 8))



# Heatmap of correlations

sns.heatmap(corrs, cmap = plt.cm.RdYlBu_r, vmin = -0.25, annot = True, vmax = 0.6)

plt.title('Rank');
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcRyoIYh0Dppmng9zFdht3bSnbhqm_QhsIhfpwI_WuNcqMxxE2Y6',width=400,height=400)
sns.heatmap(corrs, annot=True, cmap="Greens")

plt.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcTZ34lHxrxnwtUYrs0vahh557PGs0p1w29MQKShHWoaHG5Hlu4i',width=400,height=400)
df1.shape
df1.dtypes
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcRp_dnEZMYDm9Cu1Hd_50XYu-U5krj3bIoOHm8md6kWSJekZUJH',width=400,height=400)
sns.pairplot(df1, markers="+", diag_kind="kde")

plt.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQyGx5LP4021dw6yLDfSFavtrqxmMunb6rEzTVGdR0SUsALI7iY',width=400,height=400)
sns.heatmap(corrs, annot=True, cmap="Reds")

plt.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcSfdUmiIYut4f-zGLoo8fkxMZbp_7Z9u9nkT9V5YwIg70WKXGzl',width=400,height=400)
segmentation = df1["Method"]

print(segmentation)
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQGsErxC94shrO-JYxcNvevFvg9vga1gVLP3nMcqiATY0kTu_5o',width=400,height=400)
#word cloud

from wordcloud import WordCloud, ImageColorGenerator

text = " ".join(str(each) for each in df1.Method)

# Create and generate a word cloud image:

wordcloud = WordCloud(width=480, height=480, margin=0, background_color="white").generate(text)

plt.figure(figsize=(10,6))

plt.figure(figsize=(15,10))

# Display the generated image:

plt.imshow(wordcloud, interpolation='Bilinear')

plt.axis("off")

plt.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcToSMV0NAjWMl6RtMlYEfXr7XYFo6b3JbKStTzsdE8EXTu7Tycg',width=400,height=400)
plt.figure(figsize=(8,6))

sns.scatterplot(x='Rank',y='mean Iou',data=df)

plt.xticks(rotation=90)

plt.yticks(rotation=45)

plt.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcRgjAwRIP-XEJj92m896UP9qemqzEvlltk9Ouwl4HwUz1KHmJck',width=400,height=400)