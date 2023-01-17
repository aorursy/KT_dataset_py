import pandas as pd
import numpy as np
import gc, copy
from gensim.models import Word2Vec # categorical feature to vectors
from random import shuffle
import hypertools as hyp
import warnings
warnings.filterwarnings('ignore')

%matplotlib inline
cat_cols = ['region', 'city', 'parent_category_name','category_name' 'user_type', 'user_id']
print('Loading data')
tr = pd.read_csv('../input/train.csv', index_col='item_id', parse_dates=['activation_date'], nrows=150340) # 1503424
te = pd.read_csv('../input/test.csv',  index_col='item_id', parse_dates=['activation_date'], nrows=50840) # 508438

y = tr.deal_probability
tr_index = tr.index
te_index = te.index

# tr.drop(['deal_probability'],axis=1,inplace=True)
daset = pd.concat([tr,te],axis=0)

print('Daset shape rows:{} cols:{}'.format(*daset.shape))
del tr
del te
gc.collect()
def apply_w2v(sentences, model, num_features):
    def _average_word_vectors(words, model, vocabulary, num_features):
        feature_vector = np.zeros((num_features,), dtype="float64")
        n_words = 0.
        for word in words:
            if word in vocabulary: 
                n_words = n_words + 1.
                feature_vector = np.add(feature_vector, model[word])

        if n_words:
            feature_vector = np.divide(feature_vector, n_words)
        return feature_vector
    
    vocab = set(model.wv.index2word)
    feats = [_average_word_vectors(s, model, vocab, num_features) for s in sentences]
    return np.array(feats)
def gen_cat2vec_sentences(data):
    X_w2v = copy.deepcopy(data)
    names = list(X_w2v.columns.values)
    for c in names:
        X_w2v[c] = X_w2v[c].fillna('unknow').astype('category')
        X_w2v[c].cat.categories = ["%s %s" % (c,g) for g in X_w2v[c].cat.categories]
    X_w2v = X_w2v.values.tolist()
    return X_w2v
print('Cat2Vec...')
n_cat2vec_feature  = len(cat_cols) # define the cat2vecs dimentions
n_cat2vec_window   = len(cat_cols) * 2 # define the w2v window size
def fit_cat2vec_model():
    X_w2v = gen_cat2vec_sentences(daset.loc[:,cat_cols].sample(frac=0.6))
    for i in X_w2v:
        shuffle(i)
    model = Word2Vec(X_w2v, size=n_cat2vec_feature, window=n_cat2vec_window)
    return model

print('Fit cat2vec model')
c2v_model = fit_cat2vec_model()
print('apply_w2v for cat2vec')
tr_c2v_matrix = apply_w2v(gen_cat2vec_sentences(daset.loc[tr_index,cat_cols]), c2v_model, n_cat2vec_feature)
te_c2v_matrix = apply_w2v(gen_cat2vec_sentences(daset.loc[te_index,cat_cols]), c2v_model, n_cat2vec_feature)
tr_c2v_matrix
te_c2v_matrix
hyp.plot(tr_c2v_matrix[:5000], '.', reduce='TSNE', hue=round(y[:5000]*10), ndims=2)