import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import feature_extraction, linear_model, model_selection, preprocessing

import os

print(os.listdir("../input"))
train_df = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test_df = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")

train_df.head(2)
import re

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer

from nltk.tokenize import word_tokenize

from nltk.tokenize import TweetTokenizer



def tokenize(text, stop_set = None, lemmatizer = None):

    

    # clean text

    text = text.encode('ascii', 'ignore').decode('ascii')

    #text = text.lower()

    

    text = re.sub(r'\b(?:(?:https?|ftp)://)?\w[\w-]*(?:\.[\w-]+)+\S*', ' ', text) # remove hyperlink,subs charact in the brackets

    text = re.sub("[\r\n]", ' ', text) # remove new line characters

    #text = re.sub(r'[^\w\s]','',text)

    text = text.strip()

    

    #tokens = word_tokenize(text)

    # use TweetTokenizer instead of word_tokenize -> to prevent splitting at apostrophies

    tknzr = TweetTokenizer()

    tokens = tknzr.tokenize(text)

    

    # retain tokens with at least two words

    tokens = [token for token in tokens if re.match(r'.*[a-z]{1,}.*', token)]

    

    # remove stopwords - optional

    # removing stopwords lost important information

    if stop_set != None:

        tokens = [token for token in tokens if token not in stop_set]

    

    # lemmmatization - optional

    if lemmatizer != None:

        tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return tokens





train_df['tokens'] = train_df['text'].map(lambda x: tokenize(x))

test_df['tokens'] = test_df['text'].map(lambda x: tokenize(x))

train_df.head(1)

test_df.head(1)
from gensim.models import KeyedVectors



news_path = '../input/googlenewsvectorsnegative300/GoogleNews-vectors-negative300.bin'

embeddings_index = KeyedVectors.load_word2vec_format(news_path, binary=True)
# Google News Embeddings

to_remove = ['to','of','and', 'a']



replace_dict = {'favourite':'favorite', 'bitcoin':'Bitcoin', 'colour':'color', 'doesnt':'doesn\'t', 'centre':'center', 'Quorans':'Quora',

               'travelling':'traveling', 'counselling':'counseling', 'didnt':'didn\'t', 'btech':'BTech','isnt':'isn\'t',

               'Shouldn\'t':'shouldn\'t', 'programme':'program', 'realise':'realize', 'Wouldn\'t':'wouldn\'t', 'defence':'defense',

               'Aren\'t':'aren\'t', 'organisation':'organization', 'How\'s':'how\'s', 'e-commerce':'ecommerce', 'grey':'gray',

               'bitcoins':'Bitcoin', 'honours':'honors', 'learnt':'learned', 'licence':'license', 'mtech':'MTech', 'colours':'colors',

               'e-mail':'email', 't-shirt':'tshirt', 'Whatis':'What\'s', 'theatre':'theater', 'labour':'labor', 'Isnt':'Isn\'t',

               'behaviour':'behavior','aadhar':'Aadhar', 'Qoura':'Quora', 'aluminium':'aluminum'}



def clean_token(tokens, remove_list, re_dict, embedding):

    

    c_tokens = []

    for token in tokens:

        if token not in remove_list:

            token2 = token

            if token2 in embedding:

                c_tokens.append(token2)

            elif token2 in re_dict:

                token2 = re_dict[token2]

                c_tokens.append(token2)

            else:    

                # apostrophe

                if token2.endswith('\'s'):

                    token2 = token2[:-2]

                    

                if (token2.endswith('s')) & (token2[:-1] in embedding):

                    token2 = token2[:-1]

                    

                # break dash

                if "-" in token2:

                    token2 = token2.split('-')

                    c_tokens += token2

                else:

                    c_tokens.append(token2)

        



    return c_tokens



train_df['clean_tokens'] = train_df['tokens'].map(lambda x: clean_token(x, to_remove, replace_dict, embeddings_index))

test_df['clean_tokens'] = test_df['tokens'].map(lambda x: clean_token(x, to_remove, replace_dict, embeddings_index))

train_df.head(1)
test_df.head(1)
def doc_mean(tokens, embedding):

    

    e_values = []

    e_values = [embedding[token] for token in tokens if token in embedding]

    

    if len(e_values) > 0:

        return np.mean(np.array(e_values), axis=0)

    else:

        return np.zeros(300)

      

X = np.vstack(train_df['clean_tokens'].apply(lambda x: doc_mean(x, embeddings_index)))

X_test = np.vstack(test_df['clean_tokens'].apply(lambda x: doc_mean(x, embeddings_index)))



y = train_df['target'].values



indices = train_df.index
from sklearn import linear_model, tree, ensemble, metrics, model_selection, exceptions



def print_score(y_true, y_pred):

    print(' accuracy : ', metrics.accuracy_score(y_true, y_pred))

    print('precision : ', metrics.precision_score(y_true, y_pred))

    print('   recall : ', metrics.recall_score(y_true, y_pred))

    print('       F1 : ', metrics.f1_score(y_true, y_pred))



# train-test split

X_train, X_val, y_train, y_val, train_indices, test_indices = model_selection.train_test_split(X, y, indices, test_size = 0.2, random_state = 2019)
import lightgbm as lgb



lgb_c = lgb.LGBMClassifier(learning_rate = 0.02,n_estimators = 2000)



lgb_c.fit(X_train, y_train,

          eval_set = [(X_val, y_val)],

          early_stopping_rounds = 20,

          eval_metric = 'auc',

          verbose = 100)





y_val_pred = lgb_c.predict(X_val, num_iteration=lgb_c.best_iteration_)

print_score(y_val, y_val_pred)
y_test = lgb_c.predict(X_test, num_iteration=lgb_c.best_iteration_)
sample_submission = pd.concat([test_df['id'], pd.DataFrame(y_test)], axis = 1).rename(columns={0: "target"})

sample_submission.rename(columns={0: "target"})



#sample_submission = pd.DataFrame(sample_submission)

sample_submission.head()
sample_submission.to_csv("submission.csv", index=False)