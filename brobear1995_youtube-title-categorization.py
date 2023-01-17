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
!pip install sentence_transformers
import pandas as pd
df = pd.read_csv('/kaggle/input/youtube-video-categories/title_category.csv')
columns = df.columns.tolist()
columns[0] = 'Index'
df.columns = columns
df.head()
df[pd.isnull(df['Index'])]
for i in range(df.shape[0]):
    if pd.isnull(df.iloc[i,2]):
        df.iloc[i,2] = df.iloc[i+1,1]
df.dropna(subset=['Index'], inplace=True)
df.index = df['Index']
df.drop(['Index'], axis = 1, inplace=True)
df['Type of Video'] = df['Type of Video'].apply(lambda x: str(x).replace('__##__',','))
df.head()
from sklearn.feature_extraction.text import TfidfVectorizer
tf_idf = TfidfVectorizer(min_df=2, max_features = None, strip_accents = 'unicode', norm='l2',
                            analyzer = 'char', token_pattern = r'\w{1,}',ngram_range=(1,5),
                            use_idf = 1, smooth_idf = 1, sublinear_tf = 1, stop_words = 'english')
features = tf_idf.fit_transform(df['Title of the video']).toarray()
features.shape
from sentence_transformers import SentenceTransformer
embedder = SentenceTransformer('bert-base-nli-mean-tokens')
bert_features = embedder.encode(df['Title of the video'].tolist())
semantic_embedder = SentenceTransformer('bert-base-nli-stsb-mean-tokens')
semantic_bert_features = semantic_embedder.encode(df['Title of the video'].tolist())
bert_features = np.array(bert_features)
semantic_bert_features = np.array(semantic_bert_features)
import numpy as np
final_features = np.hstack((features, bert_features, semantic_bert_features))
final_features.shape
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(tokenizer=lambda x: x.split(','))
y = vectorizer.fit_transform(df['Type of Video'])
final_features.shape
y.shape
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import SGDClassifier
clf = MultiOutputClassifier(SGDClassifier(max_iter=4000)).fit(final_features, y.toarray())
def generate_embedding(text):
    word_transform = tf_idf.transform([text]).toarray()[0]
    bert_transform = embedder.encode([text], show_progress_bar=False)[0]
    semantic_bert_transform = semantic_embedder.encode([text], show_progress_bar=False)[0]
    embedding = np.hstack((word_transform, bert_transform, semantic_bert_transform))
    return embedding

def get_terms(pred_list):
    return [w.title() for w in vectorizer.inverse_transform([pred_list])[0]]

def get_topics(text):
    text_embedding = generate_embedding(text)
    pred_list = clf.predict([text_embedding])[0]
    return get_terms(pred_list)

def increment_learn(text, topics):
    available_topics = vectorizer.get_feature_names()
    for topic in topics.split(','):
        if topic.lower() not in available_topics:
            return -1
    text_embedding = generate_embedding(text)
    topics = vectorizer.transform([topics]).toarray()[0]
    clf.partial_fit([text_embedding], [topics])

title = 'Eric Weinstein: Revolutionary Ideas in Science, Math, and Society | Artificial Intelligence Podcast'
get_topics(title)
title = 'Healing the Nervous System From Trauma- Somatic Experiencing'
get_topics(title)
title = 'Consciousness -- the final frontier | Dada Gunamuktananda | TEDxNoosa 2014'
get_topics(title)
title = 'The art of being yourself | Caroline McHugh | TEDxMiltonKeynesWomen'
get_topics(title)
title = "What is Borderline Personality Disorder?"
get_topics(title)
title = "Feelings: Handle them before they handle you | Mandy Saligari | TEDxGuildford"
get_topics(title)
title = 'The Science of Lucid Dreaming'
get_topics(title)
title = 'Your Brain on LSD and Acid'
get_topics(title)
title = 'How to stop screwing yourself over | Mel Robbins | TEDxSF'
get_topics(title)
title = 'Power of Breakup | Onkar Kishan Khullar | TEDxRamanujanCollege'
get_topics(title)
title = 'What is True Love? By Sandeep Maheshwari (Dubbed in English)'
get_topics(title)
title = 'But what is a Fourier series? From heat flow to circle drawings | DE4'
get_topics(title)
increment_learn('But what is a Fourier series? From heat flow to circle drawings | DE4','Mathematics')
get_topics(title)