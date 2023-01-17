# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/goodreadsbooks/books.csv', error_bad_lines = False)
print("Number of total books: {}".format(df['title'].count()))

print("Number of unique books: {}".format(df['title'].value_counts().count()))
most_text_reviews = df.groupby(['title'])['text_reviews_count'].sum().sort_values(ascending=False).to_frame()

most_text_reviews = most_text_reviews.reset_index()

most_text_reviews = most_text_reviews.head()

most_text_reviews
# sns.set_context('poster')

plt.figure(figsize=(12,6))



x=most_text_reviews['text_reviews_count']

y=most_text_reviews['title']



sns.barplot(x=x, y=y, palette='twilight')



plt.title("BOOKS WITH MOST TEXT REVIEWS")

plt.xlabel("TEXT REVIEWS COUNT")

plt.ylabel("TITLES")

plt.show()