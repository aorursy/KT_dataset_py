import pandas as pd

import numpy as np

import re

import nltk

from sklearn.metrics.pairwise import cosine_similarity

from nltk.stem import PorterStemmer

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer

import warnings

warnings.filterwarnings('ignore')
data = pd.read_csv('../input/the-office-dataset/the_office_series.csv')

df = data[['EpisodeTitle','About']]
df['Index'] = np.arange(0,188)
df.head()
stop_words = stopwords.words('english')

stemmer = PorterStemmer()
summary = []

for i in range(0,len(df)):

    about = re.sub('[^a-zA-Z]', ' ', df['About'][i])

    about = about.lower()

    about = about.split()

    about = [stemmer.stem(word) for word in about if not word in stop_words]

    about = ' '.join(about)

    summary.append(about)
summary
tf = TfidfVectorizer()

X = tf.fit_transform(summary)

cos_sim = cosine_similarity(X, X)
cos_sim.shape
cos_sim_df = pd.DataFrame(data=cos_sim,columns=df.EpisodeTitle)

cos_sim_df.head()
def show_recommendation(episode_name):

    for i in range(0,len(df)):

        if df.values[i][0] == episode_name:

            episode_index = df.values[i][2]

            sorted_data = cos_sim_df.iloc[episode_index].sort_values(ascending=False)

            sorted_df = pd.DataFrame(sorted_data)

            print(sorted_df.index[1:4],sorted_df.iloc[1:4])
show_recommendation('Basketball')