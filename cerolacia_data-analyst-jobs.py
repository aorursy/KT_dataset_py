# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import plotly.express as px

import plotly.graph_objects as go
df=pd.read_csv("../input/data-analyst-jobs/DataAnalyst.csv")
df.head()
df.shape
df.describe()
df.info()
df.isnull().sum()
df.dropna(subset=['Company Name'],inplace=True)
df.columns
df.drop(['Unnamed: 0'],axis=1,inplace=True)
df['Easy Apply'].value_counts()
df['Easy Apply']=df['Easy Apply'].replace({"-1":"0"})
df['Easy Apply']=df['Easy Apply'].replace({"True":"1"})
df['Size'].value_counts()
df['Size']=df['Size'].replace({'-1':'Unknown'})
df['Type of ownership']=df['Type of ownership'].replace({'-1','Unknown'})
df['Location']
df['Founded'].drop_duplicates().nsmallest(10)
df['Founded'].drop_duplicates().nlargest(10)
import matplotlib.pyplot as plt
a=df['Job Title'].value_counts()
ind=a.index[0:10]
a=a[0:10]
plt.bar(ind,a)

plt.xticks(rotation=90)

plt.show()
import seaborn as sns



def bar_plot(col):



    ax = sns.barplot(

        x = df[col].value_counts().keys(), 

        y = df[col].value_counts().values

    )

    ax.set_xticklabels(ax.get_xticklabels(), rotation = 90)

    plt.show()
bar_plot('Size')
bar_plot('Type of ownership')
bar_plot('Easy Apply')
df['State'] = df['Location'].apply(lambda x: x.split(',')[1].strip())

df['City'] = df['Location'].apply(lambda x: x.split(',')[0].strip())
state_df = pd.DataFrame(df['State'].value_counts().head(10)).reset_index()



state_df.style.background_gradient(cmap='YlGnBu', low=0, high=0, axis=0, subset=None)
state_fig = go.Figure(data=[go.Pie(labels=state_df['index'],

                             values=state_df['State'],

                             hole=.7,

                             title = 'Count by State',

                             marker_colors = px.colors.sequential.Greens_r,

                            )

                     ])

state_fig.update_layout(title = 'Job Count % by State')

state_fig.show()
city_df = pd.DataFrame(df['City'].value_counts().head(10)).reset_index()



city_df.style.background_gradient(cmap='YlGnBu', low=0, high=0, axis=0, subset=None)
city_fig = go.Figure(data=[go.Pie(labels=city_df['index'],

                             values=city_df['City'],

                             hole=.7,

                             title = 'Count % by City',

                             marker_colors = px.colors.sequential.Blues_r,

                            )

                     ])

city_fig.update_layout(title = 'Job Count % by Location City')

city_fig.show()
job_desc = ', '.join(df['Job Description'])
job_desc
from wordcloud import WordCloud, STOPWORDS

from PIL import Image



stopwords = set(STOPWORDS)

wc = WordCloud(background_color = "white", max_words = 1000, 

               stopwords = stopwords, contour_width = 0, contour_color = 'black')



wc.generate(job_desc)



# show

plt.figure(figsize = [30, 20])

plt.imshow(wc, interpolation = 'bilinear')

plt.axis("off")

plt.show()