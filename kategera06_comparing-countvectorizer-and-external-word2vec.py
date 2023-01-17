import pandas as pd 
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.decomposition import PCA, TruncatedSVD

import matplotlib.patches as mpatches
from nltk import word_tokenize

%pylab inline
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train_data = pd.read_csv("../input/nlp-getting-started/train.csv", header=0, delimiter=",")
test_data = pd.read_csv("../input/nlp-getting-started/test.csv", header=0, delimiter=",")
print('Shape of train data: ', train_data.shape)
print('Shape of test data: ', test_data.shape)
train_data.head()
train_data.text[0]
plt.hist(train_data.target)
plt.show()

percentage = train_data.target.sum()/train_data.target.count()
print('1 class percentage in train set', percentage)
train_corpus = train_data.text
train_labels = train_data.target

test_corpus = test_data.text

corpus = pd.concat([train_corpus, test_corpus])
pipe = Pipeline([('vect', CountVectorizer()),
                 ('tfidf', TfidfTransformer())
                ])

pipe.fit(corpus)
print('Vocabulary lenght is', len(pipe['vect'].vocabulary_), 'words.')

processed_train_corpus = pipe.transform(train_corpus)
processed_test_corpus = pipe.transform(test_corpus)
cv_results = cross_validate(LogisticRegression(), processed_train_corpus, train_labels, cv=5)
cv_results['test_score'].mean()
def plot_LSA(test_data, test_labels, savepath="PCA_demo.csv", plot=True):
        lsa = TruncatedSVD(n_components=2)
        lsa.fit(test_data)
        lsa_scores = lsa.transform(test_data)
        color_mapper = {label:idx for idx,label in enumerate(set(test_labels))}
        color_column = [color_mapper[label] for label in test_labels]
        colors = ['orange','blue','blue']
        if plot:
            plt.scatter(lsa_scores[:,0], lsa_scores[:,1], s=8, alpha=.8, c=test_labels, cmap=matplotlib.colors.ListedColormap(colors))
            red_patch = mpatches.Patch(color='orange', label='Irrelevant')
            green_patch = mpatches.Patch(color='blue', label='Disaster')
            plt.legend(handles=[red_patch, green_patch], prop={'size': 15})
fig = plt.figure(figsize=(10, 10))          
plot_LSA(processed_train_corpus, train_labels)
plt.show()
from gensim.scripts.glove2word2vec import glove2word2vec

word2vec_path = "../input/glove6b100dtxt/glove.6B.100d.txt"
glove2word2vec(glove_input_file=word2vec_path, word2vec_output_file="gensim_glove_vectors.txt")

from gensim.models.keyedvectors import KeyedVectors
glove_model = KeyedVectors.load_word2vec_format("gensim_glove_vectors.txt", binary=False)
glove_model.wv.most_similar(positive=["best"])
glove_model.wv.most_similar(positive=["cat"])
tokenized_train_corpus = [word_tokenize(sent) for sent in train_corpus]
tokenized_test_corpus = [word_tokenize(sent) for sent in test_corpus]
def get_average_word2vec(tokens_list, vectors, generate_missing=False, k=100):
    if len(tokens_list)<1:
        return np.zeros(k)
    if generate_missing:
        vectorized = [vectors[word] if word in vectors else np.random.rand(k) for word in tokens_list]
    else:
        vectorized = [vectors[word] if word in vectors else np.zeros(k) for word in tokens_list]
    length = len(vectorized)
    summed = np.sum(vectorized, axis=0)
    averaged = np.divide(summed, length)
    return averaged

def get_word2vec_embeddings(vectors, data, generate_missing=False):
    embeddings = data.apply(lambda x: get_average_word2vec(x, vectors, generate_missing=generate_missing))
    return list(embeddings)

embeddings_train_corpus = get_word2vec_embeddings(glove_model, train_corpus)
print('Check length of resulted list of embeddings', len(embeddings_train_corpus))
cv_results = cross_validate(LogisticRegression(), embeddings_train_corpus, train_labels, cv=5)
cv_results['test_score'].mean()
fig = plt.figure(figsize=(10, 10))          
plot_LSA(embeddings_train_corpus, train_labels)
plt.show()
clf = LogisticRegression()
clf.fit(processed_train_corpus, train_labels)
results = clf.predict(processed_test_corpus)
submission = pd.DataFrame({'id': test_data.id, 'target': results})
submission.to_csv('submission.csv', index=False)
