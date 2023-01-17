#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcSsEMRQB2TGXYHFc4FFtKO_SHYRiPmFC0bN47P2TkJZBkl7trkT',width=400,height=400)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import feature_extraction, linear_model, model_selection, preprocessing

import plotly.graph_objs as go

import plotly.offline as py

import plotly.express as px



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input//covid19-data-for-spain-at-regional-level/ccaa_covid19_fallecidos.csv', encoding='ISO-8859-2')

df.head().style.background_gradient(cmap='Wistia')
df1 = pd.read_csv('../input//covid19-data-for-spain-at-regional-level/ccaa_covid19_casos.csv', encoding='ISO-8859-2')

df1.head().style.background_gradient(cmap='Wistia')
df2 = pd.read_csv('../input//covid19-data-for-spain-at-regional-level/ccaa_covid19_altas.csv', encoding='ISO-8859-2')

df2.head().style.background_gradient(cmap='Wistia')
df3 = pd.read_csv('../input//covid19-data-for-spain-at-regional-level/ccaa_covid19_uci.csv', encoding='ISO-8859-2')

df3.head().style.background_gradient(cmap='Wistia')
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcTrDLHCyB2fHVLHb9llmT-HU5zSpBWBb8oVv0UtAg1XZod2Zbhk',width=400,height=400)
df["20/03/2020"].plot.hist()

plt.show()
df["20/03/2020"].plot.box()

plt.show()
sns.pairplot(df, x_vars=['20/03/2020'], y_vars='cod_ine', markers="+", size=4)

plt.show()
import matplotlib.style



import matplotlib as mpl



mpl.style.use('classic')
fig=sns.jointplot(x='20/03/2020',y='cod_ine',kind='hex',data=df)
fig=sns.lmplot(x="20/03/2020", y="cod_ine",data=df)
df.plot.area(y=['cod_ine','17/03/2020','18/03/2020','21/03/2020'],alpha=0.4,figsize=(12, 6));
#word cloud

from wordcloud import WordCloud, ImageColorGenerator

text = " ".join(str(each) for each in df.CCAA)

# Create and generate a word cloud image:

wordcloud = WordCloud(max_words=200,colormap='Set3', background_color="black").generate(text)

plt.figure(figsize=(10,6))

plt.figure(figsize=(15,10))

# Display the generated image:

plt.imshow(wordcloud, interpolation='Bilinear')

plt.axis("off")

plt.figure(1,figsize=(12, 12))

plt.show()
cat = []

num = []

for col in df.columns:

    if df[col].dtype=='O':

        cat.append(col)

    else:

        num.append(col)  

        

        

num 
plt.style.use('dark_background')

for col in df[num].drop(['cod_ine'],axis=1):

    plt.figure(figsize=(8,5))

    plt.plot(df[col].value_counts(),color='Orange')

    plt.xlabel(col)

    plt.ylabel('cod_ine')

    plt.tight_layout()

    plt.show()
df.corr()

plt.figure(figsize=(10,4))

sns.heatmap(df.corr(),annot=True,cmap='Wistia')

plt.show()
sns.barplot(x=df['cod_ine'].value_counts().index,y=df['cod_ine'].value_counts())
plt.figure(figsize=(18,9))

sns.factorplot(x=col,y='cod_ine',data=df)

plt.tight_layout()

plt.show()
for col in df.columns:

    plt.figure(figsize=(18,9))

    sns.barplot(x=col,y='cod_ine',data=df)

    sns.pointplot(x=col,y='cod_ine',data=df,color='Black')

    plt.tight_layout()

    plt.show()
sns.pairplot(df)

plt.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQxnxOkjSNxs5N-0QAA4-WC8dgUvfdfKY5pPgS26Yj7FtH-Av0a',width=400,height=400)