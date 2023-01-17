#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQtz2gKmmUJcM5SPcE6PSl2eL5KchSivd7_lBk7BvcpVG-CZ8Eo',width=400,height=400)
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



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/corona-details/corona.csv")
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcR4cUM904wVorgJ_kQz_HDQ6wKxrc-onAovOphV5LrpcpKJ4hA3',width=400,height=400)
df.head().style.background_gradient(cmap='summer')
df.dtypes
fig,axes = plt.subplots(1,1,figsize=(20,5))

sns.heatmap(df.isna(),yticklabels=False,cbar=False,cmap='viridis')

plt.show()
df.dropna(how = 'all',inplace = True)

df.drop(['Publications','Geo_Location','Isolation_Source'],axis=1,inplace = True)

df.shape
fig,axes = plt.subplots(1,1,figsize=(20,5))

sns.heatmap(df.isna(),yticklabels=False,cbar=False,cmap='viridis')

plt.show()
sns.distplot(df["Length"].apply(lambda x: x**4))

plt.show()
sns.barplot(x=df['Length'].value_counts().index,y=df['Length'].value_counts())
from wordcloud import WordCloud, STOPWORDS

stopwords = set(STOPWORDS)



def show_wordcloud(data, title = None):

    wordcloud = WordCloud(

        background_color='red',

        stopwords=stopwords,

        max_words=200,

        max_font_size=40, 

        scale=3,

        random_state=1 # chosen at random by flipping a coin; it was heads

).generate(str(data))



    fig = plt.figure(1, figsize=(15, 15))

    plt.axis('off')

    if title: 

        fig.suptitle(title, fontsize=20)

        fig.subplots_adjust(top=2.3)



    plt.imshow(wordcloud)

    plt.show()

    

show_wordcloud(df['Genus'])


from wordcloud import WordCloud, STOPWORDS

stopwords = set(STOPWORDS)



def show_wordcloud(data, title = None):

    wordcloud = WordCloud(

        background_color='green',

        stopwords=stopwords,

        max_words=200,

        max_font_size=40, 

        scale=3,

        random_state=1 # chosen at random by flipping a coin; it was heads

).generate(str(data))



    fig = plt.figure(1, figsize=(15, 15))

    plt.axis('off')

    if title: 

        fig.suptitle(title, fontsize=20)

        fig.subplots_adjust(top=2.3)



    plt.imshow(wordcloud)

    plt.show()

    

show_wordcloud(df['Species'])


from wordcloud import WordCloud, STOPWORDS

stopwords = set(STOPWORDS)



def show_wordcloud(data, title = None):

    wordcloud = WordCloud(

        background_color='black',

        stopwords=stopwords,

        max_words=200,

        max_font_size=40, 

        scale=3,

        random_state=1 # chosen at random by flipping a coin; it was heads

).generate(str(data))



    fig = plt.figure(1, figsize=(15, 15))

    plt.axis('off')

    if title: 

        fig.suptitle(title, fontsize=20)

        fig.subplots_adjust(top=2.3)



    plt.imshow(wordcloud)

    plt.show()

    

show_wordcloud(df['Family'])
cnt_srs = df['Protein'].value_counts().head()

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

    title='Protein distribution',

    )

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="Protein")
plt.figure(figsize=(10,8))

ax=sns.countplot(df['Protein'])

ax.set_xlabel(xlabel="Protein",fontsize=17)

ax.set_ylabel(ylabel='Protein',fontsize=17)

ax.axes.set_title('Protein',fontsize=17)

ax.tick_params(labelsize=13)

plt.xticks(rotation=90)

plt.yticks(rotation=90)
for col in df.columns:

    plt.figure(figsize=(18,9))

    sns.factorplot(x=col,y='Length',data=df)

    plt.tight_layout()

    plt.show()
sns.barplot(x=df['Length'].value_counts().index,y=df['Length'].value_counts())

plt.xticks(rotation=90)

plt.yticks(rotation=90)
for col in df.columns:

    plt.figure(figsize=(18,9))

    sns.factorplot(x=col,y='Length',data=df)

    plt.tight_layout()

    plt.show()
for col in df.columns:

    plt.figure(figsize=(18,9))

    sns.barplot(x=col,y='Length',data=df)

    sns.pointplot(x=col,y='Length',data=df,color='Black')

    plt.tight_layout()

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

for col in df[num].drop(['Length'],axis=1):

    plt.figure(figsize=(8,5))

    plt.plot(df[col].value_counts(),color='Blue')

    plt.xlabel(col)

    plt.ylabel('Length')

    plt.tight_layout()

    plt.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://quotestats.com/topic/955595-biogenetic-quotes-92331.jpg',width=400,height=400)