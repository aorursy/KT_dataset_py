import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.preprocessing import text as keras_text, sequence as keras_seq

from keras.callbacks import EarlyStopping, ModelCheckpoint
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

train = train.sample(frac=1)



list_sentences_train = train["comment_text"].fillna("unknown").values

list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

y = train[list_classes].values

list_sentences_test = test["comment_text"].fillna("unknown").values



all_sentences = np.concatenate([list_sentences_train, list_sentences_test])
# We add all of the possible categories together as a proxy for level of toxicity

total_toxicity = np.sum(y,1)

print('Distribution of Total Toxicity Labels (important for validation)')

print(pd.value_counts(total_toxicity))
from sklearn.feature_extraction.text import CountVectorizer

import re, string

nums_chars = ''.join(['{}'.format(i) for i in range(10)])

re_prep = re.compile(f'([{string.ascii_letters}{nums_chars} ;:.,\t!?\-_\(\)\[\]])') # non-character things

re_prep = re.compile(r'[^\u263a-\U0001f645]') # just emojis

def prep_func(s): return re_prep.sub('', s)

print('Verify we are keeping the right things:',

      prep_func('Hello my name (µ-998)\t and I [😍] emojis ;-)'))

vec = CountVectorizer(preprocessor=prep_func, 

                      analyzer = 'char', 

                     binary = True)
%%time

vec.fit(all_sentences)

vocab_lookup = {idx: k for k,idx in vec.vocabulary_.items()}

print(len(vocab_lookup), 'unique characters found')
X_train = vec.transform(list_sentences_train)

X_test = vec.transform(list_sentences_test)
from sklearn.model_selection import train_test_split

X_t_train, X_t_test, y_train, y_test = train_test_split(X_train, 

                                                        total_toxicity, 

                                                        test_size = 0.2, 

                                                        stratify = total_toxicity,

                                                       random_state = 2017)

print('Training:', X_t_train.shape)

print('Testing:', X_t_test.shape)
from sklearn.ensemble import RandomForestRegressor

basic_rf = RandomForestRegressor()

basic_rf.fit(X_t_train, y_train)
%matplotlib inline

import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, roc_auc_score

y_pred = basic_rf.predict(X_t_test)

fig, ax1 = plt.subplots(1,1)

tpr, fpr, _ = roc_curve(y_test>0, y_pred)

ax1.plot(tpr, fpr, 'b.', label = 'ROC Curve')

ax1.plot(tpr, tpr, 'r-', label = 'Random Guessing')

ax1.set_ylabel('True Positive Rate')

ax1.set_xlabel('False Positive Rate')
show_characters = 100

for i in np.argsort(-1*basic_rf.feature_importances_)[:show_characters]:

    print(vocab_lookup[i], '\t%2.2f%%' % (100*basic_rf.feature_importances_[i]))
%%time

from sklearn.linear_model import LogisticRegressionCV

basic_logreg = LogisticRegressionCV()

basic_logreg.fit(X_train, total_toxicity>0)
show_characters = 100

for i in np.argsort(-1*np.abs(basic_logreg.coef_[0,:]))[:show_characters]:

    print(vocab_lookup[i], '\t%03.2f%%' % (100*basic_logreg.coef_[0,i]))