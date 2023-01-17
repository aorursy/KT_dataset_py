import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")



train.head()
test.head()
train.info()
test.info()
print (train['is_duplicate'].mean())
train.groupby('is_duplicate')['id'].count().plot.bar()
qid1 = train['qid1'].tolist()

qid2 = train['qid2'].tolist()

qid = pd.Series(qid1+qid2)

plt.figure(figsize=(12,5))

plt.hist(qid.value_counts(), bins= 50)

plt.yscale('log', nonposy='clip')

from nltk.corpus import stopwords

stop = set(stopwords.words("english"))

def word_share(row):

    q1 = {}

    q2 = {}

    for word in str(row['question1']).lower().split():

        if word not in stop:

            q1[word] = 1

    for word in str(row['question2']).lower().split():

        if word not in stop:

            q2[word] = 1

    if len(q1) == 0 or len(q2) == 0:

        return 0

    shared_in_q1 = [word for word in q1.keys() if word in q2]

    shared_in_q2 = [word for word in q2.keys() if word in q1]

    Ratio = (len(shared_in_q1)+len(shared_in_q2))/(len(q1)+len(q2))

    return Ratio

    

train_word_match = train.apply(word_share, axis=1, raw=True)

#train_word_match
def tfidf_word_match_share(row):

    q1words = {}

    q2words = {}

    for word in str(row['question1']).lower().split():

        if word not in stop:

            q1words[word] = 1

    for word in str(row['question2']).lower().split():

        if word not in stop:

            q2words[word] = 1

    if len(q1words) == 0 or len(q2words) == 0:

        # The computer-generated chaff includes a few questions that are nothing but stopwords

        return 0

    

    shared_weights = [weights.get(w, 0) for w in q1words.keys() if w in q2words] + [weights.get(w, 0) for w in q2words.keys() if w in q1words]

    total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]

    

    R = np.sum(shared_weights) / np.sum(total_weights)

    return R
corpus = (train['question1'].str.lower().astype('U').tolist() + train['question2'].str.lower().astype('U').tolist())



from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(min_df = 50,max_features = 3000000,ngram_range = (1,10))

X = vectorizer.fit_transform(corpus)

idf = vectorizer.idf_

weights = (dict(zip(vectorizer.get_feature_names(), idf)))
weights['filled']
import nltk

from sklearn.neighbors import DistanceMetric

from sklearn.preprocessing import MinMaxScaler

def jaccard_similarity_coefficient(row):

    if (type(row['question1']) is str) and (type(row['question2']) is str):

        words_1 = row['question1'].lower().split()

        words_2 = row['question2'].lower().split()

    else:

        words_1 = nltk.word_tokenize(str(row['question1']))

        words_2 = nltk.word_tokenize(str(row['question2']))

   

minkowski_dis = DistanceMetric.get_metric('minkowski')

mms_scale_mink = MinMaxScaler()



def get_similarity_values(words_1, words_2):

    minkowsk_dis = []

    for i,j in zip(words_1, words_2):

        i_ = i.toarray()

        j_ = j.toarray()

        sim = minkowski_dis.pairwise(i_,j_)

        minkowsk_dis.append(sim[0][0])

    return minkowsk_dis



minkowsk_dis = get_similarity_values(words_1, words_2)

print ("minkowsk_dis sample = \n", minkowsk_dis[0:2])

minkowsk_dis_array = np.array(minkowsk_dis).reshape(-1,1)

minkowsk_dis_array = mms_scale_mink.fit_transform(minkowsk_dis_array)

minkowsk_dis = minkowsk_dis_array.flatten()