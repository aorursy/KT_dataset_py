#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcTRr1Gn-jlGX456nMVwueufhP1bICaomacgIKnBG2nF_1eVuzF0&usqp=CAU',width=400,height=400)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.graph_objs as go

import plotly.offline as py

import plotly.express as px

plt.style.use('fivethirtyeight')



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcRhcM90BIhI1N81YeMCtzjRh-M_tJZuaX-BT7iaeOGQrEHmPLvh&usqp=CAU',width=400,height=400)
df = pd.read_csv('../input/andrewng-machine-learning-tweets/AndrewNG Machine Learning Tweets.csv', encoding='ISO-8859-2')
df.head() 
df.plot(subplots=True, figsize=(10, 10), sharex=False, sharey=False)

plt.show()
sns.countplot(df['polarity'],linewidth=3,palette="Set2",edgecolor='black')

plt.show()
from category_encoders import OneHotEncoder

from sklearn.linear_model import LinearRegression, Ridge

from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler



cols_selected = ['username']

ohe = OneHotEncoder(cols=cols_selected, use_cat_names=True)

df_t = ohe.fit_transform(df[cols_selected+['id']])



#scaler = MaxAbsScaler()

X = df_t.iloc[:,:-1]

y = df_t.iloc[:, -1].fillna(df_t.iloc[:, -1].mean()) / df_t.iloc[:, -1].max()



mdl = Ridge(alpha=0.1)

mdl.fit(X,y)



pd.Series(mdl.coef_, index=X.columns).sort_values().head(10).plot.barh()
df['id'].hist(figsize=(10,5), bins=20)
sns.countplot(x="polarity",data=df,palette="GnBu_d",edgecolor="black")

plt.xticks(rotation=45)

plt.yticks(rotation=45)

# changing the font size

sns.set(font_scale=1)
ax = df['polarity'].value_counts().plot.barh(figsize=(14, 6))

ax.set_title('Polarity Distribution', size=18)

ax.set_ylabel('polarity', size=14)

ax.set_xlabel('id', size=14)
import matplotlib.ticker as ticker

ax = sns.distplot(df['polarity'])

plt.xticks(rotation=45)

ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
import warnings

warnings.filterwarnings("ignore")

sns.boxplot(x='polarity', y='id', data=df, palette='rainbow')
fig = px.bar(df[['polarity','id']].sort_values('id', ascending=False), 

                        y = "id", x= "polarity", color='id', template='ggplot2')

fig.update_xaxes(tickangle=45, tickfont=dict(family='Rockwell', color='crimson', size=14))

fig.update_layout(title_text="Andrew Ng tweets")



fig.show()
fig = px.parallel_categories(df, color="polarity", color_continuous_scale=px.colors.sequential.Viridis)

fig.show()
plt.figure(figsize=(18,6))

plt.subplot(1, 2, 1)

sns.countplot(x=df['polarity'],hue=df['id'],palette='summer',linewidth=3,edgecolor='white')

plt.title('id')

plt.subplot(1, 2, 2)

sns.countplot(x=df['polarity'],hue=df['conversation_id'],palette='hot',linewidth=3,edgecolor='white')

plt.title('conversation_id')

plt.show()
fig = px.bar(df, x= "polarity", y= "id", color_discrete_sequence=['crimson'],)

fig.show()
#word cloud

from wordcloud import WordCloud, ImageColorGenerator

text = " ".join(str(each) for each in df.tweet)

# Create and generate a word cloud image:

wordcloud = WordCloud(max_words=200,colormap='Set2', background_color="black").generate(text)

plt.figure(figsize=(10,6))

plt.figure(figsize=(15,10))

# Display the generated image:

plt.imshow(wordcloud, interpolation='Bilinear')

plt.axis("off")

plt.figure(1,figsize=(12, 12))

plt.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcTtQw2GSN_QCpC7B-Pf9SP2o93ym2WH7-c1MxjQFikDNS9oUW3L&usqp=CAU',width=400,height=400)