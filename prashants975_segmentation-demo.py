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
df  = pd.read_csv('/kaggle/input/developers-and-programming-languages/user-languages.csv')
df.head()
# Delete skills without users

df = df.loc[:, (df != 0).any(axis=0)]

try: 

    del(df['user_id'])

except Exception:

    print ("Error", Exception)



df.head()
from sklearn.metrics import silhouette_samples, silhouette_score

from sklearn.cluster import KMeans





df_reduced = df



scores = []

for n_clusters in range(3,13):

    kmeans = KMeans(n_clusters = n_clusters, random_state = 11 ).fit(df_reduced)

    labels = kmeans.labels_

    silhouette_avg = silhouette_score(df_reduced, labels)

    print("n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)

    scores.append(silhouette_avg)

    

scores
import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns



plt.plot(np.arange(3, 13), scores)

plt.title('Silhouette scores, for number of clusters') 



plt.show()
import pandas as pd

from sklearn.cluster import KMeans



n_clusters = 7 #choosing the n_clusters

kmeans = KMeans(n_clusters = n_clusters, random_state = 11).fit(df)

labels = kmeans.labels_#





df_cluster = pd.DataFrame()

df_cluster['clusters'] = labels





clusters_all = []





for cluster in range(n_clusters):

    sub_df = df[df_cluster['clusters'] == cluster]

    print(sub_df.shape)

    dict_tags = {}

    for column in sub_df.columns:

        if sub_df[column].sum() > 0: dict_tags[column] = sub_df[column].sum()#

    

    print("Segment/Cluster Number:", cluster, "and", sub_df.shape[0]/df.shape[0]*100 ," % of users")

    df_temp = pd.DataFrame(sorted(dict_tags.items(), key=lambda x: x[1], reverse=True)[:10], columns=['Skill', '  Weightage']) #choosing top 10 only

    print(df_temp)

    print("**"*30)

    

    clusters_all.append(sorted(dict_tags.items(), key=lambda x: x[1], reverse=True)[:10])

# Import packages

import matplotlib.pyplot as plt

from wordcloud import WordCloud, STOPWORDS

%matplotlib inline

# Define a function to plot word cloud

def plot_cloud(wordcloud):

    # Set figure size

    plt.figure(figsize=(7, 7))

    # Display image

    plt.imshow(wordcloud) 

    # No axis details

    plt.axis("off");
def convert_to_text(df_temp):

    final_text = ""

    

    for i in range(df_temp.shape[0]):

        final_text += str(" " +  df_temp.loc[i,'Skill']) * int(df_temp.iloc[i,1]/5)

    

    return final_text
# Generate word cloud



for i,clusters in enumerate(clusters_all):

    df_temp = pd.DataFrame(clusters, columns=['Skill', '  Weightage'])

    cluster_text = convert_to_text(df_temp)

    wordcloud = WordCloud(width = 3000, height = 2000, random_state=2, background_color='salmon', colormap='Pastel1', collocations=False).generate(cluster_text)

    # Plot

    plot_cloud(wordcloud)