# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns

data=pd.read_csv('../input/globalterrorismdb_0617dist.csv',encoding='ISO-8859-1')
pd.set_option('display.max_column',500)
data.head()
data.describe()
#lets see which countries are worst affected by terrorism

from wordcloud import WordCloud, STOPWORDS



wordcloud = WordCloud(

                          stopwords=STOPWORDS,

                          background_color='white',

                          width=2500,

                          height=2000

                         ).generate(" ".join(data['country_txt']))



plt.imshow(wordcloud)

plt.axis('off')
latitude = data['latitude']

longitude = data['longitude']



lat = (min(latitude),max(latitude))

long = (min(longitude),max(longitude))
sns.set(rc={'axes.facecolor':'black', 'figure.facecolor':'white'})

ax = plt.scatter(data['longitude'].values,data['latitude'].values,color='red',s=0.5,alpha=0.5)

ax.axes.set_title('Most affect terrorist countries')

ax.figure.set_size_inches(8,5)

plt.grid(False)

plt.ylim(lat)

plt.xlim(long)

plt.show()
region = data.groupby('region_txt')['nkill'].count().reset_index().sort_values(by='region_txt',ascending=False).reset_index(drop=True)
region
#lets check how many kill in which region

plt.figure(figsize=(12,12))

sns.barplot(x='region_txt',y='nkill',data=region)

plt.xticks(rotation=90)
#clearly shows that middle east, north africa,south asia worstly affected