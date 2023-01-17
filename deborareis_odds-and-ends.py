from IPython.display import YouTubeVideo



YouTubeVideo("kYB8IZa5AuE")
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn import decomposition

from glob import glob

import os
np.set_printoptions(suppress=True)
#filenames = []

#for folder in ["british-fiction-corpus"]: #, "french-plays", "hugo-les-misérables"]:

#    filenames.extend(glob("data/literature/" + folder + "/*.txt"))
### Alteração para trazer o dataset do Kaggle

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
len(filenames)
vectorizer = TfidfVectorizer(input='filename', stop_words='english')

dtm = vectorizer.fit_transform(filenames).toarray()

vocab = np.array(vectorizer.get_feature_names())

dtm.shape, len(vocab)
[f.split("/")[3] for f in filenames]
clf = decomposition.NMF(n_components=10, random_state=1)



W1 = clf.fit_transform(dtm)

H1 = clf.components_
num_top_words=8



def show_topics(a):

    top_words = lambda t: [vocab[i] for i in np.argsort(t)[:-num_top_words-1:-1]]

    topic_words = ([top_words(t) for t in a])

    return [' '.join(t) for t in topic_words]
def get_all_topic_words(H):

    top_indices = lambda t: {i for i in np.argsort(t)[:-num_top_words-1:-1]}

    topic_indices = [top_indices(t) for t in H]

    return sorted(set.union(*topic_indices))
ind = get_all_topic_words(H1)
vocab[ind]
show_topics(H1)
W1.shape, H1[:, ind].shape
from IPython.display import FileLink, FileLinks
np.savetxt("britlit_W.csv", W1, delimiter=",", fmt='%.14f')

FileLink('britlit_W.csv')
np.savetxt("britlit_H.csv", H1[:,ind], delimiter=",", fmt='%.14f')

FileLink('britlit_H.csv')
np.savetxt("britlit_raw.csv", dtm[:,ind], delimiter=",", fmt='%.14f')

FileLink('britlit_raw.csv')
[str(word) for word in vocab[ind]]
U, s, V = decomposition.randomized_svd(dtm, 10)
ind = get_all_topic_words(V)
len(ind)
vocab[ind]
show_topics(H1)
np.savetxt("britlit_U.csv", U, delimiter=",", fmt='%.14f')

FileLink('britlit_U.csv')
np.savetxt("britlit_V.csv", V[:,ind], delimiter=",", fmt='%.14f')

FileLink('britlit_V.csv')
np.savetxt("britlit_raw_svd.csv", dtm[:,ind], delimiter=",", fmt='%.14f')

FileLink('britlit_raw_svd.csv')
np.savetxt("britlit_S.csv", np.diag(s), delimiter=",", fmt='%.14f')

FileLink('britlit_S.csv')
[str(word) for word in vocab[ind]]