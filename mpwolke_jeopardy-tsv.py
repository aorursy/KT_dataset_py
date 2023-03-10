#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcSIzP-3yzvrG2U5yQKm36yWkWVhptCTwxYMj8252QVm3vm2acZX',width=400,height=400)
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

Image(url = 'data:image/jpeg;base64,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',width=400,height=400)
df = pd.read_csv('../input/350000-jeopardy-questions/kids_teen.tsv', sep='\t', header=None)
df.head()
df = df.rename(columns={0:'round', 1:'value', 2:'daily_double', 3:'category', 4: 'comments', 5: 'answer', 6: 'question', 7: 'air_date', 8: 'notes'})
df.dtypes
sns.countplot(df["value"])

plt.xticks(rotation=90)

plt.show()
#word cloud

from wordcloud import WordCloud, ImageColorGenerator

text = " ".join(str(each) for each in df.answer)

# Create and generate a word cloud image:

wordcloud = WordCloud(max_words=200, background_color="black").generate(text)

plt.figure(figsize=(10,6))

plt.figure(figsize=(15,10))

# Display the generated image:

plt.imshow(wordcloud, interpolation='Bilinear')

plt.axis("off")

plt.show()
#word cloud

from wordcloud import WordCloud, ImageColorGenerator

text = " ".join(str(each) for each in df.category)

# Create and generate a word cloud image:

wordcloud = WordCloud(max_words=200, background_color="green").generate(text)

plt.figure(figsize=(10,6))

plt.figure(figsize=(15,10))

# Display the generated image:

plt.imshow(wordcloud, interpolation='Bilinear')

plt.axis("off")

plt.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcT8exlz-B8oGl86LgeL8MxfRa_tsEQXyXaUrM2Pq-PZuYzflBcF',width=400,height=400)