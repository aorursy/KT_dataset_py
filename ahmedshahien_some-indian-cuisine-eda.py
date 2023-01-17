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
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv('../input/indian-food-101/indian_food.csv')

df.head()
df.info()
df['name'].count()
df['diet'].unique()
import plotly.express as px

pie_df = df.diet.value_counts().reset_index()

pie_df.columns = ['diet','count']

fig = px.pie(pie_df, values='count', names='diet', title='Proportion of Vegetarian and Non-Vegetarian dishes',

             color_discrete_sequence=['green', 'red'])

fig.show()
df['flavor_profile'].unique()
sns.set(style="darkgrid")

 

ax = sns.countplot(x="flavor_profile", data=df)
df['state'].unique()
 

sns.set(style="darkgrid")

ax = sns.countplot(x="region", data=df,facecolor=(0, 0, 0, 0),linewidth=5,edgecolor=sns.color_palette("dark",3))
ax = sns.countplot(y="course", data=df)
from wordcloud import WordCloud

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[26, 8])

wordcloud1 = WordCloud( background_color='white',

                        width=600,

                        height=400).generate(" ".join(df['ingredients']))

ax1.imshow(wordcloud1)

ax1.axis('off')

ax1.set_title('ingredients',fontsize=40);



wordcloud2 = WordCloud( background_color='white',

                        width=600,

                        height=400).generate(" ".join(df['state']))

ax2.imshow(wordcloud2)

ax2.axis('off')

ax2.set_title('state',fontsize=40);