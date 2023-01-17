# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.







df = pd.read_json(open("../input/train.json", "r"))

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df.head()
df.describe()
ax = sns.countplot(x="interest_level", data=df)
x = df['price'].quantile([0.5,0.75,0.9,0.95] + [0.99 + i*0.001 for i in range(10)])

x.index = [(1 - i) * df.shape[0]  for i in x.index]

x
df_train = df[df['price']<10000]

sns.distplot(df_train['price'])
sns.violinplot(y="interest_level", x="price", data=df_train);
x = df['longitude'].quantile([0.01 + i*0.002 for i in range(5)] + [0.99 + i*0.002 for i in range(5)])

#x.index = [(1 - i) * df.shape[0]  for i in x.index]

x
x = df['latitude'].quantile([0.01 + i*0.002 for i in range(5)] + [0.99 + i*0.002 for i in range(5)])

#x.index = [(1 - i) * df.shape[0]  for i in x.index]

x
df_train = df_train[(df_train['longitude'] > -74.4) & (df_train['longitude'] < -73.80)]

df_train = df_train[(df_train['latitude'] > 40.6) & (df_train['latitude'] < 40.9)]
sns.violinplot(y="interest_level", x="longitude", data=df_train);
sns.violinplot(y="interest_level", x="latitude", data=df_train);
sns.pairplot(df_train[['longitude','latitude','interest_level']],hue = 'interest_level',markers='.')
sns.countplot(y="bedrooms", hue = "interest_level", data=df_train)
sns.countplot(y="bathrooms", hue = "interest_level", data=df_train)
sns.set(style="white")

# Compute the correlation matrix

corr = df_train.corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True,n=100)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1,

            square=True, xticklabels=True, yticklabels=True,

            linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
df_train['created'] = pd.to_datetime(df_train['created'],infer_datetime_format=True)



df_train['month'] = df_train['created'].apply(lambda x: x.month)
sns.countplot(y="month", hue = "interest_level", data=df_train)
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer()



tfidf.fit_transform(df_train['features'].apply(lambda x: " ".join(x)))



response = tfidf.transform(df_train['features'].apply(lambda x: " ".join(x)))

feature_array = np.array(tfidf.get_feature_names())

tfidf_sorting = np.argsort(response.toarray()).flatten()[::-1]



n = 100

top_n = feature_array[tfidf_sorting][:n]

top_n