import pandas as pd
import numpy as np
train = pd.read_csv("../input/shopee-sentiment-analysis/train.csv")
test = pd.read_csv("../input/shopee-sentiment-analysis/test.csv")
train = train.sample(frac=1).reset_index()

print("train dimension: ", train.shape)
print("test dimension: ", test.shape)
train.head(5)
train['rating'] = train['rating']-1
import nltk
#nltk.download('popular')

import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer

# text cleaning & tokenization
def tokenize(text, stop_set = None, lemmatizer = None):
    
    # clean text
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = text.lower()
    
    text = re.sub(r'\b(?:(?:https?|ftp)://)?\w[\w-]*(?:\.[\w-]+)+\S*', ' ', text) # remove hyperlink,subs charact in the brackets
    text = re.sub("[\r\n]", ' ', text) # remove new line characters
    #text = re.sub(r'[^\w\s]','',text)
    text = text.strip() ## convert to lowercase split indv words
    
    #tokens = word_tokenize(text)
    # use TweetTokenizer instead of word_tokenize -> to prevent splitting at apostrophies
    tknzr = TweetTokenizer()
    tokens = tknzr.tokenize(text)
    
    # retain tokens with at least two words
    tokens = [token for token in tokens if re.match(r'.*[a-z]{2,}.*', token)]
    
    # remove stopwords - optional
    # removing stopwords lost important information
    if stop_set != None:
        tokens = [token for token in tokens if token not in stop_set]
    
    # lemmmatization - optional
    if lemmatizer != None:
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return tokens
# without lemmatization
train['tokens'] = train['review'].map(lambda x: tokenize(x))
test['tokens'] = test['review'].map(lambda x: tokenize(x))
test.head()
train.head()
def build_vocab(token_col):
    
    vocab = {}
    for tokens in token_col:
        for token in tokens:
            vocab[token] = vocab.get(token, 0) + 1

    return vocab

train_vocab = build_vocab(train['tokens'])
test_vocab = build_vocab(test['tokens'])
from gensim.models import KeyedVectors
news_path = '../input/googlenewsvectorsnegative300/GoogleNews-vectors-negative300.bin'
embeddings_index = KeyedVectors.load_word2vec_format(news_path, binary=True)
import operator
def check_coverage(vocab,embedding):    
    oov = {}
    k = 0
    i = 0
    for word in vocab:
        if word in embedding:
            k += vocab[word]
        else:
            oov[word] = vocab[word]
            i += vocab[word]
    print('Found embeddings for  {:.2%} of all text'.format(k / (k + i)))
    sorted_x = sorted(oov.items(), key=operator.itemgetter(1))[::-1]
    return sorted_x
not_found_vocab = check_coverage(train_vocab, embeddings_index)
not_found_vocab
to_remove = ['to','of','and', 'bgt', 'sdh', 'udg', 'shopee']

replace_dict = {'cepet':'fast', 'tq':'thanks', 
                'pesen': "request", 'brg': 'item', 
                'nyampe': 'reached', 'tpi':'but', 
                'alhamdulillah': 'gratitude',
                'nyesel': 'regret', "seller's": 'seller', 
               'well-packaged': 'good', 'baguss': 'good', "ship's":'ship', 'cuman':'only',
                'packingnya':'packaging', 'klo':'at', 'dateng':'comeon'
               }

def clean_token(tokens, remove_list, re_dict):
    tokens = [token for token in tokens if token not in remove_list]
    tokens = [re_dict[token] if token in re_dict else token for token in tokens]
    return tokens

train['clean_tokens'] = train['tokens'].map(lambda x: clean_token(x, to_remove, replace_dict))
test['clean_tokens'] = test['tokens'].map(lambda x: clean_token(x, to_remove, replace_dict))
train_vocab = build_vocab(train['clean_tokens'])
test_vocab = build_vocab(test['clean_tokens'])

not_found_vocab = check_coverage(train_vocab, embeddings_index)
not_found_vocab
def doc_mean(tokens, embedding):
    
    e_values = []
    e_values = [embedding[token] for token in tokens if token in embedding]
    
    if len(e_values) > 0:
        return np.mean(np.array(e_values), axis=0)
    else:
        return np.zeros(300)
      
X = np.vstack(train['clean_tokens'].apply(lambda x: doc_mean(x, embeddings_index)))
X_1 = np.vstack(test['clean_tokens'].apply(lambda x: doc_mean(x, embeddings_index)))
y = train['rating'].values
from sklearn import linear_model, tree, ensemble, metrics, model_selection, exceptions


def print_score(y_true, y_pred):
    print(' macro_accuracy : ', metrics.accuracy_score(y_true, y_pred))
    print('ave macro precision : ', metrics.precision_score(y_true, y_pred, average='macro', pos_label=1, sample_weight=None))
    print('   macro_recall : ', metrics.recall_score(y_true, y_pred, labels=None, pos_label=1, average='macro', sample_weight=None))
    print('       macro_F1 : ', metrics.f1_score(y_true, y_pred, labels=None, pos_label=1, average='macro', sample_weight=None))
    
    print(' micro_ave precision : ', metrics.precision_score(y_true, y_pred, average='micro', pos_label=1, sample_weight=None))
    print('   micro_recall : ', metrics.recall_score(y_true, y_pred, labels=None, pos_label=1, average='micro', sample_weight=None))
    print('       micro_F1 : ', metrics.f1_score(y_true, y_pred, labels=None, pos_label=1, average='micro', sample_weight=None))
    
    print(' weighted_precision : ', metrics.precision_score(y_true, y_pred, average='weighted', pos_label=1, sample_weight=None))
    print('   weighted_recall : ', metrics.recall_score(y_true, y_pred, labels=None, pos_label=1, average='weighted', sample_weight=None))
    print('       weighted_F1 : ', metrics.f1_score(y_true, y_pred, labels=None, pos_label=1, average='weighted', sample_weight=None))
    
# train-test split
X_train, X_val, y_train, y_val = model_selection.train_test_split(X, y, test_size = 0.8, random_state = 2020)

np.random.seed(2019)

# biased sampling
import xgboost as xgb

dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)

#from sklearn.datasets import dump_svmlight_file

#dump_svmlight_file(X_train, y_train, 'dtrain.svm', zero_based=True)
#dump_svmlight_file(X_test, y_test, 'dtest.svm', zero_based=True)
#dtrain_svm = xgb.DMatrix('dtrain.svm')
#dtest_svm = xgb.DMatrix('dtest.svm')
# param = {
#     'max_depth': 3,  # the maximum depth of each tree
#     'eta': 0.3,  # the training step for each iteration
#     #'silent': 1,  # logging mode - quiet
#     'objective': 'multi:softprob', # error evaluation for multiclass training
#     'num_class': 5,
#     'eval_metric': 'mlogloss'}  # the number of classes that exist in this datset

param = {'booster': "gblinear", 
        'objective': "reg:squarederror", 
        'lambda': 0.1, 
        'alpha': 0}
num_round = 15 # the number of training iterations

# bst = GridSearchCV()

bst = xgb.train(param, dtrain, num_round)
preds = bst.predict(dval)
#preds = np.array(preds)
# preds[:,3] = preds[:,3]+0.4
# preds[:,4] = preds[:,4]+0.4
# preds[:,0] = preds[:,0]+0.15
#best_preds = np.argsort(preds, axis=1)[:,preds.shape[1]-1::]
best_preds = [int(a+1+0.4) if a > 3 else int(a+1-0.4) for a in preds]
best_preds = np.clip(np.round(best_preds),1,5)
print_score(y_val, best_preds)
best_preds[:20]
y_val[:20]
print_score(y_val, best_preds)
# 1 round 0.37
# 10 rounds: 0.42
# 20 rounds: 0.43
# 30 rounds: 0.438
dtest = xgb.DMatrix(X_1)
y_pred_test = bst.predict(dtest)
y_pred_test = np.array(y_pred_test)
y_pred_test[:,3] = y_pred_test[:,3]+0.4
y_pred_test[:,4] = y_pred_test[:,4]+0.4
y_pred_test[:,0] = y_pred_test[:,0]+0.15
y_best_preds = np.argsort(y_pred_test, axis=1)[:,preds.shape[1]-1::]+1
submission = test[['review_id']]
result = pd.DataFrame(y_best_preds)
result.columns = ['rating']

submission = pd.concat([submission, result], axis=1)

submission.tail()
submission.to_csv("subs_3.csv", index=False)