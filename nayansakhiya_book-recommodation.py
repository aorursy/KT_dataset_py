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
import plotly.graph_objs as go

from plotly.offline import  init_notebook_mode, iplot

init_notebook_mode(connected=True)

import cufflinks as cf

cf.go_offline()

cf.set_config_file(offline=False, world_readable=True)

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

from wordcloud import WordCloud,STOPWORDS



%matplotlib inline





plt.rcParams['figure.figsize'] = (6, 4)

plt.style.use('ggplot')

%config InlineBackend.figure_formats = {'png', 'retina'}
book_data = pd.read_csv('/kaggle/input/goodbooks-10k/books.csv')

book_data = book_data.dropna()

book_data.head()
book_data.columns
book_data = book_data.drop(columns=['id', 'best_book_id', 'work_id', 'isbn', 'isbn13', 'title','work_ratings_count',

                                   'work_text_reviews_count', 'ratings_1', 'ratings_2', 'ratings_3', 'ratings_4', 'ratings_5', 

                                    'image_url','small_image_url'])

book_data.head()
rating_data = pd.read_csv('/kaggle/input/goodbooks-10k/ratings.csv')

rating_data.head()
# mean rating per user



MRPU = rating_data.groupby(['user_id']).mean().reset_index()

MRPU['mean_rating'] = MRPU['rating']



MRPU.drop(['book_id','rating'],axis=1, inplace=True)
MRPU.head()
rating_data = pd.merge(rating_data, MRPU, on=['user_id', 'user_id'])

rating_data.head()
rating_data['rating'].value_counts().iplot(kind='bar',

                                          xTitle='Rating',

                                          yTitle='Frequency of Ratings',

                                          title='Rating vs Frequency',

                                          color='blue')
stop_words=set(STOPWORDS)

authors_string = " ".join(book_data['authors'])

wc = WordCloud(width=800,height=500,mask=None,random_state=21, max_font_size=110,stopwords=stop_words).generate(authors_string)
fig=plt.figure(figsize=(16,8))

plt.axis('off')

plt.imshow(wc)
stop_words=set(STOPWORDS)

title_string = " ".join(book_data['original_title'])

wc = WordCloud(width=800,height=500,mask=None,random_state=21, max_font_size=110,stopwords=stop_words).generate(title_string)
fig=plt.figure(figsize=(16,8))

plt.axis('off')

plt.imshow(wc)
rating_data = rating_data.drop(rating_data[rating_data.rating < rating_data.mean_rating].index)
# user1's favorite books



rating_data[rating_data['user_id']== 1].head()
# user2's favorite book

rating_data[rating_data['user_id']== 2].head()
# user5's favorite book



rating_data[rating_data['user_id']== 5].head()
rating_data.shape
rating_data['user_id'].unique()
rating_data = rating_data.rename({'rating':'userRating'}, axis='columns')
# merge 2 dataset

mergedata = pd.merge(book_data,rating_data,on=['book_id','book_id'])

mergedata.head()
len(mergedata['book_id'].unique())
len(book_data['book_id'].unique())
user_book = pd.crosstab(mergedata['user_id'], mergedata['original_title'])

user_book.head()
user_book.shape
from sklearn.decomposition import PCA



pca = PCA(n_components=3)

pca.fit(user_book)

pca_samples = pca.transform(user_book)
bs = pd.DataFrame(pca_samples)

bs.head()
tocluster = pd.DataFrame(bs[[0,1,2]])
plt.rcParams['figure.figsize'] = (16, 9)





fig = plt.figure()

ax = Axes3D(fig)

ax.scatter(tocluster[0], tocluster[2], tocluster[1])



plt.title('Data points in 3D PCA axis', fontsize=20)

plt.show()
from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score



scores = []

inertia_list = np.empty(8)



for i in range(2,8):

    kmeans = KMeans(n_clusters=i)

    kmeans.fit(tocluster)

    inertia_list[i] = kmeans.inertia_

    scores.append(silhouette_score(tocluster, kmeans.labels_))
from sklearn.cluster import KMeans



clusterer = KMeans(n_clusters=4,random_state=30).fit(tocluster)

centers = clusterer.cluster_centers_

c_preds = clusterer.predict(tocluster)



print(centers)
fig = plt.figure()

ax = Axes3D(fig)

ax.scatter(tocluster[0], tocluster[2], tocluster[1], c = c_preds)

plt.title('Data points in 3D PCA axis', fontsize=20)



plt.show()
fig = plt.figure(figsize=(10,8))

plt.scatter(tocluster[1],tocluster[0],c = c_preds)

for ci,c in enumerate(centers):

    plt.plot(c[1], c[0], 'o', markersize=8, color='red', alpha=1)



plt.xlabel('x_values')

plt.ylabel('y_values')



plt.title('Data points in 2D PCA axis', fontsize=20)

plt.show()
user_book['cluster'] = c_preds





user_book.head()
user_book.info()
c0 = user_book[user_book['cluster']==0].drop('cluster',axis=1).mean()

c1 = user_book[user_book['cluster']==1].drop('cluster',axis=1).mean()

c2 = user_book[user_book['cluster']==2].drop('cluster',axis=1).mean()

c3 = user_book[user_book['cluster']==3].drop('cluster',axis=1).mean()
c0.sort_values(ascending=False)[0:10]
c1.sort_values(ascending=False)[0:10]
c2.sort_values(ascending=False)[0:10]
c3.sort_values(ascending=False)[0:15]