# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("/kaggle/input/consumercomplaintsdata/Consumer_Complaints.csv")

df.head()
df = df[['Consumer complaint narrative','Product']]

df.shape
# removing null rows

df = df[df["Consumer complaint narrative"].notnull() == True]
# converting label into intergar value 

df["category_id"] = df["Product"].factorize()[0]
category_id_df = df[['Product', 'category_id']].drop_duplicates().sort_values('category_id')

category_id_df
category_to_id = dict(category_id_df.values)

id_to_category = dict(category_id_df[['category_id', 'Product']].values)
df.head()
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(8,6))

df.groupby('Product')["Consumer complaint narrative"].count().plot.bar(ylim=0)

plt.show()
from sklearn.feature_extraction.text import TfidfVectorizer



tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')