# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df=pd.read_csv("/kaggle/input/google-play-store-apps/googleplaystore_user_reviews.csv")

df2=pd.read_csv("/kaggle/input/google-play-store-apps/googleplaystore.csv")

df.head()
df2.head()
df.tail()
df.info()
df2.info()
df2.shape
df2.isnull()
df2.dropna(how='any')
df2["Reviews"].describe()
df2["Reviews"]=pd.to_numeric(df2["Reviews"],errors='coerce')
count_=sns.countplot(x="Category",data=df2, palette = "Set1",saturation=5)

count_.set_xticklabels(count_.get_xticklabels(), rotation=90, ha="right")

count_ 

plt.title('Count of app in each category',size = 20)
sns.scatterplot( x="Rating", y="Reviews",data=df2,color="b")
a=sns.catplot(x="Category", y="Reviews", jitter=False, data=df2)
