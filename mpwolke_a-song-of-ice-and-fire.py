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

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSdVa0JcnscPzaECds5UTWqbvs0j7NhRzETUeYFp0-jYf2M5H42XA&s',width=400,height=400)
df=pd.read_csv('/kaggle/input//game-of-thrones-twitter/gotTwitter.csv',encoding='ISO-8859-1')
df.head()
print("Shape of the dataframe is",df.shape)

print("The number of nulls in each column are \n", df.isna().sum())
print("Percentage null or na values in df")

((df.isnull() | df.isna()).sum() * 100 / df.index.size).round(2)
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'data:image/jpeg;base64,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',width=400,height=400)
categorical_cols = [cname for cname in df.columns if

                    df[cname].nunique() < 10 and 

                    df[cname].dtype == "object"]





# Select numerical columns

numerical_cols = [cname for cname in df.columns if 

                df[cname].dtype in ['int64', 'float64']]

print(numerical_cols)
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRaUiIbMiTHJdIo2uEt1B0AVvDYuDJvhITMQv0pghO-utMsgKPiGw&s',width=400,height=400)
#word cloud

from wordcloud import WordCloud, ImageColorGenerator

text = " ".join(str(each) for each in df.reply_to_user_id)

# Create and generate a word cloud image:

wordcloud = WordCloud(max_words=200, background_color="black").generate(text)

plt.figure(figsize=(10,6))

plt.figure(figsize=(15,10))

# Display the generated image:

plt.imshow(wordcloud, interpolation='Bilinear')

plt.axis("off")

plt.figure(1,figsize=(12, 12))

plt.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSVC0hFfYrlXiW7TsIrD9nZGY1H7KGVXNAkR8s1m-WceCSHXHcO&s',width=400,height=400)
#word cloud

from wordcloud import WordCloud, ImageColorGenerator

text = " ".join(str(each) for each in df.reply_to_screen_name)

# Create and generate a word cloud image:

wordcloud = WordCloud(max_words=200, background_color="black").generate(text)

plt.figure(figsize=(10,6))

plt.figure(figsize=(15,10))

# Display the generated image:

plt.imshow(wordcloud, interpolation='Bilinear')

plt.axis("off")

plt.figure(1,figsize=(12, 12))

plt.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'data:image/jpeg;base64,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',width=400,height=400)
f, ax = plt.subplots(figsize=(8,6))

x = df['followers_count']

ax = sns.distplot(x, bins=10)

plt.show()

f, ax = plt.subplots(figsize=(8,6))

x = df['retweet_count']

ax = sns.distplot(x, bins=10)

plt.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRdXrP4lbnExEQsp96PFMZUpvXJsUhiJz_lMfg3eZwb5jj5ILmm&s',width=400,height=400)
f, ax = plt.subplots(figsize=(8,6))

x = df['quoted_followers_count']

ax = sns.distplot(x, kde=False, rug=True, bins=10)

plt.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRtmoqLTFxgyOislIXmX_83UfMbf89MoRet7mszjEReftuO57b6&s',width=400,height=400)
f, ax = plt.subplots(figsize=(8,6))

x = df['quoted_followers_count']

ax = sns.distplot(x, bins=10)

plt.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSXmRDPSk7bBA2f-gpTuC7OjwpX4Fpr341RcV30W0MPBCQp0iEE&s',width=400,height=400)
f, ax = plt.subplots(figsize=(8,6))

x = df['quoted_retweet_count']

ax = sns.distplot(x, bins=10, vertical = True)

plt.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTj_alXfjND52rfR4WPJ-qbHUcI4pbAKJEIWrZYuWorLEQ5VLf93w&s',width=400,height=400)
f, ax = plt.subplots(figsize=(8,6))

x = df['quoted_favorite_count']

ax = sns.distplot(x, bins=10)

plt.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR2oebnzSzsOCgYG9H6FdOQuJTdLyk2sK-XgCirNbhHCX9MjYSG&s',width=400,height=400)
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSsweo1smXKxNPTh7us5K0aQPU2tppijX1cue4isYSQROX8OgFG&s',width=400,height=400)