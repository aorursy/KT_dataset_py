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
df = pd.read_excel('/kaggle/input/covid19-outbreak-epidemiological-line-list-data/nCoV2019_2020_line_list_open.xlsx')

df.head()
df.dtypes
df = df.rename(columns={'wuhan(0)_not_wuhan(1)':'wuhan'})
df["wuhan"].plot.hist()

plt.show()
dfcorr=df.corr()

dfcorr
sns.heatmap(dfcorr,annot=True,cmap='pink')

plt.show()
#word cloud

from wordcloud import WordCloud, ImageColorGenerator

text = " ".join(str(each) for each in df.country)

# Create and generate a word cloud image:

wordcloud = WordCloud(max_words=200,colormap='Set3', background_color="black").generate(text)

plt.figure(figsize=(10,6))

plt.figure(figsize=(15,10))

# Display the generated image:

plt.imshow(wordcloud, interpolation='Bilinear')

plt.axis("off")

plt.figure(1,figsize=(12, 12))

plt.show()
def plot_feature(df,col):

    plt.figure(figsize=(14,6))

    plt.subplot(1,2,1)

    if df[col].dtype == 'int64':

        df[col].value_counts().sort_index().plot()

    else:

        mean = df.groupby(col)['wuhan'].mean()

        df[col] = df[col].astype('chronic_disease_binary')

        levels = mean.sort_values().index.tolist()

        df[col].cat.reorder_categories(levels,inplace=True)

        df[col].value_counts().plot()

    plt.xticks(rotation=45)

    plt.xlabel(col)

    plt.ylabel('Counts')

    plt.subplot(1,2,2)

    

    if df[col].dtype == 'int64' or col == 'wuhan':

        mean = df.groupby(col)['wuhan'].mean()

        std = df.groupby(col)['wuhan'].std()

        mean.plot()

        plt.fill_between(range(len(std.index)),mean.values-std.values,mean.values + std.values, \

                        alpha=0.1)

    else:

        sns.boxplot(x = col,y='wuhan',data=df)

    plt.xticks(rotation=45)

    plt.ylabel('wuhan')

    plt.show()
plot_feature(df,'wuhan')
plt.style.use('ggplot')

df['wuhan'].value_counts().plot()

plt.show()
df['wuhan'].mean()
plt.figure(figsize=(10,5))

df['wuhan'].plot(kind='hist',bins=50)

plt.show()
sns.pairplot(df)

plt.show()