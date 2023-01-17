import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
plt.style.use('ggplot')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df = pd.read_csv('../input/google-job-skills/job_skills.csv')
df.head(3)
# I modify the column name so that I can use df dot column name more easily
df = df.rename(columns={'Minimum Qualifications': 'Minimum_Qualifications', 'Preferred Qualifications': 'Preferred_Qualifications'})
df.Company.value_counts()
df.Category.value_counts()
df.Location.value_counts()[:10]
df['Country'] = df['Location'].apply(lambda x : x.split(',')[-1])
df.Country.value_counts()[:10]
pd.isnull(df).sum()
df = df.dropna(how='any',axis='rows')
# Perform the necessary imports for similarity
from sklearn.decomposition import NMF
from sklearn.preprocessing import Normalizer, MaxAbsScaler
from sklearn.pipeline import make_pipeline


scaler = MaxAbsScaler()

model = NMF(n_components=100)

normalizer = Normalizer()

# Create a pipeline: pipeline
pipeline = make_pipeline(scaler,model,normalizer)
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
vectors_Responsibilities = vectorizer.fit_transform(df['Responsibilities'])
Responsibilities = pipeline.fit_transform(vectors_Responsibilities)
df_Responsibilities = pd.DataFrame(Responsibilities,index=df['Title'])
df_Responsibilities.head(2)
pd.set_option('display.max_colwidth', -1)
print(df[df.Title.str.contains('Data Scientist')]['Title'])
Position = df_Responsibilities.loc['Customer Experience Data Scientist, Google Cloud Support']
similarities_1 = df_Responsibilities.dot(Position)
similarities_1[:3]
print(similarities_1.nlargest())
df[np.isin(df['Title'],similarities_1.nlargest().index.tolist())].head()
type(similarities_1)
vectorizer_Requirements = TfidfVectorizer()
vectors_Requirements = vectorizer_Requirements.fit_transform(df['Minimum_Qualifications'])
Requirements = pipeline.fit_transform(vectors_Requirements)
df_Requirementss = pd.DataFrame(Requirements,index=df['Title'])
Position = df_Requirementss.loc['Customer Experience Data Scientist, Google Cloud Support']
similarities_2 = df_Responsibilities.dot(Position)
print(similarities_2.nlargest())
similarities_1
similarities_1.rename("similarity")
similarities_2.rename("similarity")

similarities_1.to_frame().join(similarities_2.to_frame(),lsuffix='1')
similarities_overall = (2 * similarities_1) + similarities_2
print(similarities_overall.nlargest())
df[np.isin(df['Title'],similarities_overall.nlargest(3).index.tolist())].head()
from scipy.cluster.vq import kmeans, vq
from numpy import random

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import string

from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from spacy.lang.en.stop_words import STOP_WORDS
string.punctuation
stop_words_0 = set(stopwords.words('english')) 
stop_words = ['and', 'in', 'of', 'or', 'with','to','on','a']

def remove_noise(text):
    tokens = word_tokenize(text)
    clean_tokens = []
    lemmatizer=WordNetLemmatizer()
    for token in tokens:
        token = re.sub('[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+', '', token)
        token = lemmatizer.lemmatize(token.lower())
        if len(token) > 1 and token not in stop_words_0 and token not in stop_words:
            clean_tokens.append(token)
            
    return clean_tokens
# Initialize TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=100,tokenizer=remove_noise)

# Use the .fit_transform() method on the list plots
tfidf_matrix = tfidf_vectorizer.fit_transform(df['Minimum_Qualifications'])
random.seed = 123
distortions = []
num_clusters = range(2, 25)

# Create a list of distortions from the kmeans function
for i in num_clusters:
    cluster_centers, distortion = kmeans(tfidf_matrix.todense(),i)
    distortions.append(distortion)

# Create a data frame with two lists - num_clusters, distortions
elbow_plot = pd.DataFrame({'num_clusters': num_clusters, 'distortions': distortions})

# Creat a line plot of num_clusters and distortions
sns.lineplot(x='num_clusters', y='distortions', data = elbow_plot)
plt.xticks(num_clusters)
plt.title('Clusters and Distortions')
plt.show()
cluster_centers, distortion = kmeans(tfidf_matrix.todense(),13)

# Generate terms from the tfidf_vectorizer object
terms = tfidf_vectorizer.get_feature_names()

for i in range(13):
    # Sort the terms and print top 10 terms
    center_terms = dict(zip(terms, list(cluster_centers[i])))
    sorted_terms = sorted(center_terms, key=center_terms.get, reverse=True)
    print(sorted_terms[:5])
