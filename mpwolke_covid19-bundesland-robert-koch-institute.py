#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQLgWJw3rljC-eNTQFQ1nc5VBGUheEdXi8mSUQa-h7iFdizUPKH',width=400,height=400)
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

from wordcloud import WordCloud



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/covid19-data-germany-robert-koch-institute/dd4580c810204019a7b8eb3e0b329dd6_0.csv")

df.head().style.background_gradient(cmap='afmhot')
df1 = pd.read_csv("../input/covid19-data-germany-robert-koch-institute/ef4b445a53c1406892257fe63129a8ea_0.csv")

df1.head().style.background_gradient(cmap='afmhot')
df2 = pd.read_csv("../input/covid19-data-germany-robert-koch-institute/917fc37a709542548cc3be077a786c17_0.csv")

df2.head().style.background_gradient(cmap='afmhot')
cnt_srs = df2['cases_per_100k'].value_counts().head()

trace = go.Bar(

    y=cnt_srs.index[::-1],

    x=cnt_srs.values[::-1],

    orientation = 'h',

    marker=dict(

        color=cnt_srs.values[::-1],

        colorscale = 'Reds',

        reversescale = True

    ),

)



layout = dict(

    title='cases_per_100k',

    )

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="cases_per_100k")        
fig=sns.lmplot(x="cases", y="cases_per_population",data=df2)
df2.plot.area(y=['cases','deaths','OBJECTID','ADE', 'GF', 'BSG'],alpha=0.4,figsize=(12, 6));
def nonan(x):

    if type(x) == str:

        return x.replace("\n", "")

    else:

        return ""



text = ' '.join([nonan(abstract) for abstract in df2["GEN"]])

wordcloud = WordCloud(max_font_size=None,colormap='Set3', background_color='black', collocations=False,

                      width=1200, height=1000).generate(text)

fig = px.imshow(wordcloud)

fig.update_layout(title_text='GEN')
fig = px.pie( values=df1.groupby(['LAN_ew_GEN']).size().values,names=df1.groupby(['LAN_ew_GEN']).size().index)

fig.update_layout(

    title = "Cities",

    font=dict(

        family="Arial, monospace",

        size=10,

        color="#7f7f7f"

    )

    )   

    

py.iplot(fig)
cat = []

num = []

for col in df.columns:

    if df[col].dtype=='O':

        cat.append(col)

    else:

        num.append(col)  

        

        

num 
plt.style.use('dark_background')

for col in df[num].drop(['AnzahlTodesfall'],axis=1):

    plt.figure(figsize=(8,5))

    plt.plot(df[col].value_counts(),color='Red')

    plt.xlabel(col)

    plt.ylabel('AnzahlTodesfall')

    plt.tight_layout()

    plt.show()
df2.corr()

plt.figure(figsize=(10,4))

sns.heatmap(df.corr(),annot=True,cmap='afmhot')

plt.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcT0eqFg5_wEfybGZy2BTrjCrInUFsw3BC4ZuPLlqwmgslZqW3QG',width=400,height=400)