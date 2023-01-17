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
df = pd.read_excel('/kaggle/input/covid19-corona-dataset-worldwide/COVID-19-geographic-disbtribution-worldwide-2020-03-07.xls')

df.head().style.background_gradient(cmap='PRGn')
df.dtypes
df["NewConfCases"].plot.hist()

plt.show()
df["NewDeaths"].plot.box()

plt.show()
sns.pairplot(df, x_vars=['NewConfCases'], y_vars='NewDeaths', markers="+", size=4)

plt.show()
df.corr()
plt.figure(figsize=(10,4))

sns.heatmap(df.corr(),annot=True,cmap='Blues')

plt.show()
fig, axes = plt.subplots(1, 1, figsize=(14, 6))

sns.boxplot(x='NewConfCases', y='NewDeaths', data=df, showfliers=False);
fig=sns.lmplot(x="NewConfCases", y="NewDeaths",data=df)
df.plot.area(y=['NewConfCases','NewDeaths'],alpha=0.4,figsize=(12, 6));
import matplotlib.style



import matplotlib as mpl



mpl.style.use('classic')
fig=sns.lmplot(x="NewConfCases", y="NewDeaths",data=df)
for col in df.columns:

    plt.figure(figsize=(18,9))

    sns.factorplot(x=col,y='NewConfCases',data=df)

    plt.tight_layout()

    plt.show()
for col in df.columns:

    plt.figure(figsize=(18,9))

    sns.lineplot(x=col,y='NewConfCases',data=df)

    plt.tight_layout()

    plt.xlabel(col)

    plt.ylabel('NewConfCases')

    plt.show()
for col in df.columns:

    plt.figure(figsize=(18,9))

    sns.barplot(x=col,y='NewConfCases',data=df)

    sns.pointplot(x=col,y='NewConfCases',data=df,color='Black')

    plt.tight_layout()

    plt.show()
sns.pairplot(df)

plt.show()
import plotly.express as px
fig = px.histogram(df, x='NewConfCases', y='NewDeaths', color='CountryExp',

                   marginal="box",

                   hover_data=df.columns)

fig.show()
cat = []

num = []

for col in df.columns:

    if df[col].dtype=='O':

        cat.append(col)

    else:

        num.append(col)
num
cat
plt.style.use('dark_background')

for col in df[num].drop(['NewConfCases'],axis=1):

    plt.figure(figsize=(12,7))

    plt.plot(df[col].value_counts(),color='Red')

    plt.xlabel(col)

    plt.ylabel('New Confirmed Cases')

    plt.tight_layout()

    plt.show()
#word cloud

from wordcloud import WordCloud, ImageColorGenerator

text = " ".join(str(each) for each in df.CountryExp)

# Create and generate a word cloud image:

wordcloud = WordCloud(max_words=200,colormap='Set3', background_color="black").generate(text)

plt.figure(figsize=(10,6))

plt.figure(figsize=(15,10))

# Display the generated image:

plt.imshow(wordcloud, interpolation='Bilinear')

plt.axis("off")

plt.figure(1,figsize=(12, 12))

plt.show()