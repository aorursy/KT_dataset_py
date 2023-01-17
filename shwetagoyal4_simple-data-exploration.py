import numpy as np 

import pandas as pd 



import matplotlib.pyplot as plt

import seaborn as sns

import plotly.graph_objects as go

import plotly.express as px



from wordcloud import WordCloud



from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.offline as py

py.init_notebook_mode(connected=True)



import warnings

warnings.filterwarnings('ignore')
train = pd.read_csv("../input/nlp-getting-started/train.csv")
# Print few rows of train data



train.head()
# Basic information



train.info()
# Describing data



train.describe() 
# Data types of columns



train.dtypes
train.isnull().sum()
import missingno as msno

msno.matrix(train)
Loc = train['location'].value_counts()

fig = px.choropleth(Loc.values, locations=Loc.index,

                    locationmode='country names',

                    color=Loc.values,

                    color_continuous_scale=px.colors.sequential.OrRd)

fig.update_layout(title="Countrywise Distribution")

py.iplot(fig, filename='test')
Tar = train['target'].value_counts()



fig = go.Figure([go.Bar(x=Tar.index, y=Tar)])

fig.update_layout(title = "Target Category")

py.iplot(fig, filename='test')
wordcloud = WordCloud(width = 1000, height = 600, max_font_size = 200, max_words = 150,

                      background_color='white').generate(" ".join(train.text))



plt.figure(figsize=[10,10])

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
plt.figure(figsize=(16,8))

plt.title('Most frequent keywords',fontsize=16)

plt.xlabel('keywords')



sns.countplot(train.keyword,order=pd.value_counts(train.keyword).iloc[:15].index,palette=sns.color_palette("PuRd", 15))



plt.xticks(size=16,rotation=60)

plt.yticks(size=16)

sns.despine(bottom=True, left=True)

plt.show()