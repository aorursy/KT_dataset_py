#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://quincy-network.s3.ca-central-1.amazonaws.com/wp-content/uploads/sites/8/2020/03/Screen-Shot-2020-03-29-at-9.12.05-AM.png',width=400,height=400)
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcR-oTjGB8rBCWCZyN7Pfr6fujX_0f8MdCCbP0zh-Qo4ii34OmDB&usqp=CAU',width=400,height=400)
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

from plotly.offline import iplot





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcTknnBmg_gO_vxD0lMfgvS7RytozXMJOlex3nXhyOFSARiAYgL9&usqp=CAU',width=400,height=400)
df = pd.read_csv("../input/uncover/nextstrain/covid-19-genetic-phylogeny.csv")

df.head()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcTjPpIRHE28xGGaN-8_FwZqMTEFmT5hml7zStJa5AuwPmfGqa39&usqp=CAU',width=400,height=400)
fig,axes = plt.subplots(1,1,figsize=(20,5))

sns.heatmap(df.isna(),yticklabels=False,cbar=False,cmap='viridis')

plt.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQxPDSWhfcZIcoUHLAuJuApC9xt9I5H4U2ZCWR6FiRQq1rqWKPt&usqp=CAU',width=400,height=400)
import plotly.express as px



# Grouping it by job title and country

plot_data = df.groupby(['age', 'virus'], as_index=False).country.sum()



fig = px.bar(plot_data, x='age', y='country', color='virus')

fig.show()
plot_data = df.groupby(['age'], as_index=False).country.sum()



fig = px.line(plot_data, x='age', y='country')

fig.show()
plot_data = df.groupby(['age'], as_index=False).virus.sum()



fig = px.line(plot_data, x='age', y='virus')

fig.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcT4W2yWRgAxzBAXy2vKWVLNabc0ERjHcAgV8hff37WWqRANoai6&usqp=CAU',width=400,height=400)
plot_data = df.groupby(['sex'], as_index=False).strain.sum()



fig = px.line(plot_data, x='sex', y='strain')

fig.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcSo0VSaiDXjGYXIHbJhPOXcv3LtQpL8lNffD4oOyDp0FKDPWv0J&usqp=CAU',width=400,height=400)
fig = px.scatter(df, x= "age", y= "country")

fig.show()
fig = px.bar(df, x= "location", y= "genbank_accession")

fig.show()
fig = px.scatter(df, x= "date", y= "originating_lab")

fig.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcRvYy-EVKVyypAtP_zrzkPxAY_xWK_FH-ERDyw-RI4TcJg6rQKq&usqp=CAU',width=400,height=400)
fig = px.bar(df, x= "segment", y= "host")

fig.show()
cnt_srs = df['age'].value_counts().head()

trace = go.Bar(

    y=cnt_srs.index[::-1],

    x=cnt_srs.values[::-1],

    orientation = 'h',

    marker=dict(

        color=cnt_srs.values[::-1],

        colorscale = 'Purples',

        reversescale = True

    ),

)



layout = dict(

    title='Age',

    )

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="age")
cnt_srs = df['host'].value_counts().head()

trace = go.Bar(

    y=cnt_srs.index[::-1],

    x=cnt_srs.values[::-1],

    orientation = 'h',

    marker=dict(

        color=cnt_srs.values[::-1],

        colorscale = 'Greens',

        reversescale = True

    ),

)



layout = dict(

    title='host',

    )

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="host")
df1 = pd.read_csv("../input/corona-virus-capillary-and-liver-tumor-samples/both_clean_liver_capillary_CoV.csv")

df1.head().style.background_gradient(cmap='PuBuGn')
cat = []

num = []

for col in df1.columns:

    if df1[col].dtype=='O':

        cat.append(col)

    else:

        num.append(col)  

        

        

num 
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQAbAnhnwdkpHdbm-Xv0Aq_O0ecKClTfNHGauEwPOBdc3x5afEE&usqp=CAU',width=400,height=400)
plt.style.use('dark_background')

for col in df1[num].drop(['LiverTumorSamples GSM2359853_CoV2'],axis=1):

    plt.figure(figsize=(8,5))

    plt.plot(df1[col].value_counts(),color='Orange')

    plt.xlabel(col)

    plt.ylabel('LiverTumorSamples GSM2359851_CoV1')

    plt.tight_layout()

    plt.show()
cnt_srs = df1['LiverTumorSamples GSM2359853_CoV2'].value_counts().head()

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

    title='LiverTumorSamples GSM2359853_CoV2',

    )

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="LiverTumorSamples GSM2359853_CoV2")
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcR2AHd6g77eXzQ3xIL02Mn_YLv4CqhKlAq9sYJAI5bVro25gFKM&usqp=CAU',width=400,height=400)
cnt_srs = df1['LiverTumorSamples GSM2359916_inactiveHeatCoV2'].value_counts().head()

trace = go.Bar(

    y=cnt_srs.index[::-1],

    x=cnt_srs.values[::-1],

    orientation = 'h',

    marker=dict(

        color=cnt_srs.values[::-1],

        colorscale = 'Greens',

        reversescale = True

    ),

)



layout = dict(

    title='LiverTumorSamples GSM2359916_inactiveHeatCoV2',

    )

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="LiverTumorSamples GSM2359916_inactiveHeatCoV2")
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcReKpCSfOrQTDl4rpO0Z_ivQgVbl74zBXQnk-q2wmGZhBou4qhy&usqp=CAU',width=400,height=400)