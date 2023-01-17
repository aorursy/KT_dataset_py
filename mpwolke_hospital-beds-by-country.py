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
df = pd.read_csv("../input/hospital-beds-by-country/API_SH.MED.BEDS.ZS_DS2_en_csv_v2_887506.csv")
df.head()
df.dtypes
df = df.rename(columns={'Country Name':'CountryName', 'Country code': 'code'})
fig,axes = plt.subplots(1,1,figsize=(20,5))

sns.heatmap(df.isna(),yticklabels=False,cbar=False,cmap='viridis')

plt.show()
df.dropna(how = 'all',inplace = True)

df.drop(['1961','1962','1963', '1964', '1966', '1967', '1968', '1969', '2016', '2017', '2018','2019'],axis=1,inplace = True)

df.shape
sns.distplot(df["1990"].apply(lambda x: x**4))

plt.show()
sns.barplot(x=df['1990'].value_counts().index,y=df['1990'].value_counts())
#word cloud

from wordcloud import WordCloud, ImageColorGenerator

text = " ".join(str(each) for each in df.CountryName)

# Create and generate a word cloud image:

wordcloud = WordCloud(max_words=200,colormap='Set3', background_color="green").generate(text)

plt.figure(figsize=(10,6))

plt.figure(figsize=(15,10))

# Display the generated image:

plt.imshow(wordcloud, interpolation='Bilinear')

plt.axis("off")

plt.figure(1,figsize=(12, 12))

plt.show()
cnt_srs = df['CountryName'].value_counts().head()

trace = go.Bar(

    y=cnt_srs.index[::-1],

    x=cnt_srs.values[::-1],

    orientation = 'h',

    marker=dict(

        color=cnt_srs.values[::-1],

        colorscale = 'Blues',

        reversescale = True

    ),

)



layout = dict(

    title='Country distribution',

    )

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="CountryName")
plt.figure(figsize=(10,8))

ax=sns.countplot(df['CountryName'])

ax.set_xlabel(xlabel="",fontsize=17)

ax.set_ylabel(ylabel='CountryName',fontsize=17)

ax.axes.set_title('CountryName',fontsize=17)

ax.tick_params(labelsize=13)

plt.xticks(rotation=90)

plt.yticks(rotation=90)
fig = px.pie( values=df.groupby(['CountryName']).size().values,names=df.groupby(['CountryName']).size().index)

fig.update_layout(

    title = "Country Name",

    font=dict(

        family="Arial, monospace",

        size=15,

        color="#7f7f7f"

    )

    )   

    

py.iplot(fig)