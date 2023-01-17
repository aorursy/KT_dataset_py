import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tqdm import trange

import random

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.decomposition import NMF, LatentDirichletAllocation



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
df = pd.read_csv('../input/rajyasabha_questions_and_answers_2009.csv', encoding='latin1')

df.head()
def print_top_words(model, feature_names, n_top_words):

    parts = []

    for topic_idx, topic in enumerate(model.components_):

        parts.append([feature_names[i]

                             for i in topic.argsort()[:-n_top_words - 1:-1]])

    return parts



def get_topics(df, n_components=20, n_top_words=1):

    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,

                                       stop_words='english')

    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,

                                    stop_words='english')

    questions = df.question_title

    tfidf = tfidf_vectorizer.fit_transform(questions)

    tf = tf_vectorizer.fit_transform(questions)

    

    nmf = NMF(n_components=n_components, random_state=1,

              alpha=.1, l1_ratio=.5).fit(tfidf)

    

    tfidf_feature_names = tfidf_vectorizer.get_feature_names()

    nmf = NMF(n_components=n_components, random_state=1,

          beta_loss='kullback-leibler', solver='mu', max_iter=1000, alpha=.1,

          l1_ratio=.5).fit(tfidf)

    return pd.DataFrame(print_top_words(nmf, tfidf_feature_names, n_top_words))



data = {}

for year in trange(2009, 2018):

    df = pd.read_csv('../input/rajyasabha_questions_and_answers_{}.csv'.format(year), encoding='latin1')

    topics = get_topics(df)

    data[str(year)] = topics.values
d = {k: [i[0] for i in v] for k, v in data.items()}

df = pd.DataFrame(d)

df = df[list(reversed(df.columns))]

def randomcolor(used=[]):

    r = lambda: random.randint(100,255)

    c = ('#%02X%02X%02X' % (r(),r(),r()))

    while c in used:

        c = ('#%02X%02X%02X' % (r(),r(),r()))

    used.append(c)

    return c

unique_words = set(df.values.flatten())

cmap = {w: randomcolor() for w in unique_words}



def highlight_same_words(s):

    return ['background-color: {}'.format(cmap[w]) for w in s]



df.style.apply(highlight_same_words)