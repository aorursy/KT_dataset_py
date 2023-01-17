import numpy as np
np.random.seed(13)
from matplotlib import pyplot as plt
import itertools
import scipy
import sklearn.decomposition
import sklearn.manifold
import json
import gc
EMBEDDING_SIZE = 8
MIN_OCCURRENCES = 10
from kaggle.competitions import twosigmanews
env = twosigmanews.make_env()
(market_data, news_data) = env.get_training_data()
%%time
news_data['subjects_tuples'] = news_data['subjects'].copy()
subjects_cats = news_data['subjects_tuples'].cat.categories
subjects_cats = [eval(c.replace("{", "(").replace("}", ")")) for c in subjects_cats]
news_data['subjects_tuples'].cat.categories = subjects_cats

del subjects_cats
news_data[['assetCodes', 'time', 'subjects', 'subjects_tuples']].head(3)
%%time
news_data['audiences_tuples'] = news_data['audiences'].copy()
audiences_cats = news_data['audiences_tuples'].cat.categories
audiences_cats = [eval(c.replace("{", "(").replace("}", ")")) for c in audiences_cats]
news_data['audiences_tuples'].cat.categories = audiences_cats

del audiences_cats
news_data[['assetCodes', 'time', 'audiences', 'audiences_tuples']].head(3)
def pd_categorical_to_dummies(series, min_occurrences=0):
    
    features_cats_evals = series.cat.categories
    unique_features = list(set(itertools.chain(*features_cats_evals)))
    
    num_unique_features = len(unique_features)
    
    features_map = {k:v for v, k in enumerate(unique_features)}
    features_cats_factorized = [[features_map[k] for k in l] for l in features_cats_evals]
    
    features_lengths = [
        len(features_cats_factorized[i]) 
        for i in series.cat.codes
    ]
    
    features_cats_rows = np.arange(series.shape[0]).repeat(features_lengths)
    
    features_cats_cols = np.array([
        v for c in 
        series.cat.codes
        for v in features_cats_factorized[c]
    ])
    
    total_length = len(features_cats_cols)
    
    dummies = scipy.sparse.coo_matrix(
        (np.ones(total_length, dtype=np.bool), (features_cats_rows, features_cats_cols)),
        shape=(series.shape[0], num_unique_features),
        dtype=np.bool
    )
    
    dummies = dummies.tocsr()
    
    m = dummies.sum(axis=0).A[0] > min_occurrences
    
    dummies = dummies[:, m]
    unique_features = [a for a, mm in zip(unique_features, m) if mm == 1]
    
    return dummies, unique_features
def get_similar(w, embeddings, features, max_features=10):
    
    i = features.index(w)
    v = embeddings[i]
    similarities = (embeddings @ v)
    
    ii = np.argsort(similarities)[::-1]
    
    similarities = similarities[ii[:max_features + 1]]
    
    similar_words = [features[j] for j in ii[:max_features + 1]]
        
    m = similarities > 1 - 1e-6
        
    assert w in np.array(similar_words)[m]
    
    similar_words.remove(w)
    
    return similar_words
%%time
subject_dummies, subjects = pd_categorical_to_dummies(
    news_data['subjects_tuples'],
    MIN_OCCURRENCES
)
subject_dummies.shape, len(subjects)
svd_reducer = sklearn.decomposition.TruncatedSVD(
    n_components=EMBEDDING_SIZE,
    algorithm='randomized',
    n_iter=5,
    random_state=None,
    tol=0.0
)
%%time
svd_reducer.fit(subject_dummies.T)
%%time
_subject_embeddings = svd_reducer.transform(subject_dummies.T)
assert np.abs(_subject_embeddings.sum(axis=1)).min() != 0
subject_embeddings = _subject_embeddings/np.linalg.norm(_subject_embeddings, axis=1)[:, np.newaxis]
# subject_embeddings[np.isnan(subject_embeddings)] = 0
assert np.isnan(subject_embeddings).sum() == 0
%%time
subject_tsne = sklearn.manifold.TSNE(n_components=2).fit_transform(subject_embeddings)
print(subject_tsne.shape)
fig, ax = plt.subplots(1, 1, figsize=(12, 6))

ax.scatter(subject_tsne[:, 0], subject_tsne[:, 1], s=1)

for i, txt in enumerate(subjects):
    if i % 20 == 0:
        ax.annotate(txt, (subject_tsne[i, 0], subject_tsne[i, 1]), fontsize=10)

plt.show()
" ".join(get_similar("FUND", subject_embeddings, subjects))
" ".join(get_similar("GOLF", subject_embeddings, subjects))
" ".join(get_similar("EPMICS", subject_embeddings, subjects))
" ".join(get_similar("TWAVE", subject_embeddings, subjects))
with open("subjects.json", "w") as f:
    json.dump(subjects, f)
np.save("subject_embeddings.npy", subject_embeddings)
if False:
    del subject_dummies, subjects, subject_embeddings
del subject_tsne, _subject_embeddings, svd_reducer
gc.collect()
%%time
audience_dummies, audiences = pd_categorical_to_dummies(
    news_data['audiences_tuples'], 
    MIN_OCCURRENCES
)
audience_dummies.shape, len(audiences)
svd_reducer = sklearn.decomposition.TruncatedSVD(
    n_components=EMBEDDING_SIZE,
    algorithm='randomized',
    n_iter=5,
    random_state=None,
    tol=0.0
)
%%time
svd_reducer.fit(audience_dummies.T)
%%time
_audience_embeddings = svd_reducer.transform(audience_dummies.T)
assert np.abs(_audience_embeddings.sum(axis=1)).min() != 0
audience_embeddings = _audience_embeddings/np.linalg.norm(_audience_embeddings, axis=1)[:, np.newaxis]
# audience_embeddings[np.isnan(audience_embeddings)] = 0
assert np.isnan(audience_embeddings).sum() == 0
audience_tsne = sklearn.manifold.TSNE(n_components=2).fit_transform(audience_embeddings)
print(audience_tsne.shape)
fig, ax = plt.subplots(1, 1, figsize=(12, 6))

ax.scatter(audience_tsne[:, 0], audience_tsne[:, 1], s=1)

for i, txt in enumerate(audiences):
    if i % 5 == 0:
        ax.annotate(txt, (audience_tsne[i, 0], audience_tsne[i, 1]))

plt.show()
" ".join(get_similar("OIL", audience_embeddings, audiences))
" ".join(get_similar("MTL", audience_embeddings, audiences))
" ".join(get_similar("NZP", audience_embeddings, audiences))
" ".join(get_similar("FN", audience_embeddings, audiences))
with open("audiences.json", "w") as f:
    json.dump(audiences, f)
np.save("audience_embeddings.npy", audience_embeddings)
if False:
    del audience_dummies, audiences
del audience_embeddings, audience_tsne, _audience_embeddings, svd_reducer
gc.collect()
audience_subject_map = {}
num = 5

global_subject_proportions = subject_dummies.sum(axis=0).A[0]/subject_dummies.shape[0]

for i in range(len(audiences)):
    m = audience_dummies[:, i].A[:, 0]
    a = audiences[i]
    
    c = subject_dummies[m].sum(axis=0).A[0]
    p = c/subject_dummies[m].shape[0]
    # s = np.abs(p - global_subject_proportions)
    s = np.clip(p - global_subject_proportions, 0, np.inf)
    
    ii = np.argsort(s)[::-1][:num]
    
    subs = np.array(subjects)[ii].tolist()
    cnts = c[ii]
    
    # print(a, subs)
    
    audience_subject_map[a] = subs
    
    #break
audience_subject_map['OIL']
audience_subject_map['MTL'] # Metal??
audience_subject_map['FN'] # Finland ??
audience_subject_map['NZP'] # New Zealand ??
# subject_embeddings = np.load("subject_embeddings.npy")
_audience_embeddings_using_sub = []

for i in range(len(audiences)):
    a = audiences[i]
    
    subs = audience_subject_map[a]
    
    ii = [i for i, s in enumerate(subjects) if s in subs]
    
    e = subject_embeddings[ii].mean(axis=0)
    
    _audience_embeddings_using_sub.append(e)
    
_audience_embeddings_using_sub = np.array(_audience_embeddings_using_sub)

print(_audience_embeddings_using_sub.shape)
audience_embeddings_using_sub = _audience_embeddings_using_sub/np.linalg.norm(_audience_embeddings_using_sub, axis=1)[:, np.newaxis]
audience_using_sub_tsne = sklearn.manifold.TSNE(n_components=2).fit_transform(
    audience_embeddings_using_sub
)
print(audience_using_sub_tsne.shape)
fig, ax = plt.subplots(1, 1, figsize=(12, 6))

ax.scatter(audience_using_sub_tsne[:, 0], audience_using_sub_tsne[:, 1], s=1)

for i, txt in enumerate(audiences):
    if i % 2 == 0:
        ax.annotate(txt, (audience_using_sub_tsne[i, 0], audience_using_sub_tsne[i, 1]))

plt.show()
" ".join(get_similar("OIL", audience_embeddings_using_sub, audiences))
" ".join(get_similar("MTL", audience_embeddings_using_sub, audiences))
" ".join(get_similar("FN", audience_embeddings_using_sub, audiences))
" ".join(get_similar("NZP", audience_embeddings_using_sub, audiences))
with open("audience_subject_map.json", "w") as f:
    json.dump(audience_subject_map, f)
    
np.save("audience_embeddings_using_sub.npy", audience_embeddings_using_sub)
import gensim
model_audience = gensim.models.Word2Vec(
    size=EMBEDDING_SIZE, #10,
    window=99999,
    sg=1,
    hs=0,
    min_count=MIN_OCCURRENCES,
    workers=4,
    compute_loss=True
)
model_audience.build_vocab(news_data['audiences_tuples'].values)
%%time
model_audience.train(
    sentences=news_data['audiences_tuples'],
    epochs=1,
    total_examples=news_data.shape[0],
    compute_loss=True,   
)

model_audience.get_latest_training_loss()
model_audience.wv.similar_by_word("OIL")
model_audience.wv.similar_by_word("MTL")
model_audience.wv.similar_by_word("NZP")
%%time
audience_word2vec_tsne = sklearn.manifold.TSNE(n_components=2).fit_transform(
    model_audience.wv.vectors_norm
)
print(audience_word2vec_tsne.shape)
fig, ax = plt.subplots(1, 1, figsize=(12, 6))

ax.scatter(audience_word2vec_tsne[:, 0], audience_word2vec_tsne[:, 1], s=1)

for i, txt in enumerate(model_audience.wv.index2word):
    if i % 2 == 0:
        ax.annotate(txt, (audience_word2vec_tsne[i, 0], audience_word2vec_tsne[i, 1]))

plt.show()
with open("audience_skipgram.json", "w") as f:
    json.dump(model_audience.wv.index2word, f)
np.save("audience_skipgram_embeddings.npy", model_audience.wv.vectors_norm)
model_subject = gensim.models.Word2Vec(
    size=EMBEDDING_SIZE, #10,
    window=99999,
    sg=1,
    hs=0,
    min_count=MIN_OCCURRENCES,
    workers=4,
    compute_loss=True
)
model_subject.build_vocab(news_data['subjects_tuples'].values)
%%time
model_subject.train(
    sentences=news_data['subjects_tuples'],
    epochs=1,
    total_examples=news_data.shape[0],
    compute_loss=True,   
)

model_subject.get_latest_training_loss()
model_subject.wv.similar_by_word("FUND")
model_subject.wv.similar_by_word("EPMICS")
# COMDIS    Communicable Diseases
# SL        Sierra Leone
%%time
subjects_word2vec_tsne = sklearn.manifold.TSNE(n_components=2).fit_transform(model_subject.wv.vectors_norm)
print(subjects_word2vec_tsne.shape)
fig, ax = plt.subplots(1, 1, figsize=(12, 6))

ax.scatter(subjects_word2vec_tsne[:, 0], subjects_word2vec_tsne[:, 1], s=1)

for i, txt in enumerate(model_subject.wv.index2word):
    if i % 20 == 0:
        ax.annotate(txt, (subjects_word2vec_tsne[i, 0], subjects_word2vec_tsne[i, 1]))

plt.show()
with open("subjects_skipgram.json", "w") as f:
    json.dump(model_subject.wv.index2word, f)
np.save("subject_skipgram_embeddings.npy", model_subject.wv.vectors_norm)