import pandas as pd

import numpy as np

import tensorflow as tf

import tokenizers

import transformers

from tqdm import tqdm

from sklearn import metrics

from sklearn.model_selection import KFold

import re
MAX_LEN = 64

TRAIN_BATCH_SIZE = 32

VALID_BATCH_SIZE = 8

LEARNING_RATE = 3e-5

EPOCHS = 3

TRAINING_FILE = "../input/nlp-getting-started/train.csv"

TEST_FILE = "../input/nlp-getting-started/test.csv"

ROBERTA_PATH = "../input/tf-roberta"

# TOKENIZER = tokenizers.ByteLevelBPETokenizer(

#         vocab_file=f"{ROBERTA_PATH}/vocab-roberta-base.json", 

#         merges_file=f"{ROBERTA_PATH}/merges-roberta-base.txt", 

#         lowercase=True,

#         add_prefix_space=True

#     )

roberta_tokenizer = transformers.RobertaTokenizer.from_pretrained('roberta-base', lower_case=True)
train = pd.read_csv(TRAINING_FILE).fillna('')

test = pd.read_csv(TEST_FILE).fillna('')

print("Training samples: {}".format(train.shape[0]))

print("Test samples: {}".format(test.shape[0]))

train.head(5)
# remove urls

url = "Great paper by Kalchbrenner https://arxiv.org/pdf/1404.2188.pdf?utm_medium=App.net&utm_source=PourOver"



def remove_urls(text):

    re_url = re.compile(r'https?://\S+|www\.\S+')

    return re_url.sub('', text).strip()



print(remove_urls(url))

train['text'] = train['text'].apply(lambda x : remove_urls(x))

test['text'] = test['text'].apply(lambda x : remove_urls(x))
html = """<div>

<h1>Hey</h1>

<p>Kaggle </p>

<a href="https://www.kaggle.com/c/nlp-getting-started">removed tags</a>

</div>"""

# remove html tags

def remove_html(text):

    re_html = re.compile(r'<.*?>')

    return re_html.sub('', text)



print(remove_html(html))

train['text'] = train['text'].apply(lambda x : remove_html(x)) 

test['text'] = test['text'].apply(lambda x : remove_html(x)) 
# remove emojis

def remove_emoji(text):

    emoji_pattern = re.compile("["

                           u"\U0001F600-\U0001F64F"  # emoticons

                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs

                           u"\U0001F680-\U0001F6FF"  # transport & map symbols

                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)

                           u"\U00002702-\U000027B0"

                           u"\U000024C2-\U0001F251"

                           "]+", flags=re.UNICODE)

    return emoji_pattern.sub(r'', text).strip()



print(remove_emoji("Difficult kernel 😔😔"))

train['text'] = train['text'].apply(lambda x: remove_emoji(x))

test['text'] = test['text'].apply(lambda x: remove_emoji(x))
# remove punctuations

punct = 'Cristiano. is #king .l'

import string

def remove_puncts(text):

    table = str.maketrans('','',string.punctuation)

    return text.translate(table).strip()



print(remove_puncts(punct))

train['text'] = train['text'].apply(lambda x: remove_puncts(x))

test['text'] = test['text'].apply(lambda x: remove_puncts(x))
num_classes = train.target.nunique()

num_classes
n_train = train.shape[0]

input_ids = np.ones((n_train, MAX_LEN), dtype='int32')

mask = np.zeros((n_train, MAX_LEN), dtype='int32')



# roberta tokenizer

for k in range(train.shape[0]):

    text = train.loc[k, 'text']

    output = roberta_tokenizer.encode_plus(text, max_length=MAX_LEN, pad_to_max_length=True)

    input_ids[k] = output['input_ids']

    mask[k] = output["attention_mask"]
n_test = test.shape[0]

input_ids_t = np.ones((n_test, MAX_LEN), dtype='int32')

mask_t = np.zeros((n_test, MAX_LEN), dtype='int32')



# roberta tokenizer

for k in range(test.shape[0]):

    text = test.loc[k, 'text']

    output = roberta_tokenizer.encode_plus(text, max_length=MAX_LEN, pad_to_max_length=True)

    input_ids_t[k] = output['input_ids']

    mask_t[k] = output["attention_mask"]
def create_dataset():

    xtrain = [input_ids, mask]

    xtest = [input_ids_t, mask_t]

    

    ytrain = tf.keras.utils.to_categorical(train['target'].values.reshape(-1, 1))

    return xtrain, ytrain, xtest
xtrain, ytrain, xtest = create_dataset()

print("X train : {0}".format(len(xtrain[0])))

print("Y train : {0}".format(len(ytrain)))

print("X test : {0}".format(len(xtest[0])))
def build_model():

    roberta = transformers.TFRobertaForSequenceClassification.from_pretrained('roberta-base')

    optim = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=2.0)

    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    roberta.compile(loss='binary_crossentropy', optimizer=optim, metrics=['accuracy'])

    return roberta
Folds = 5

kfold = KFold(n_splits=Folds)

predictions = list()



for i, (train_idx, test_idx) in enumerate(kfold.split(xtrain[0])):

    xtrain_fold = [xtrain[i][train_idx] for i in range(len(xtrain))]

    xvalid_fold = [xtrain[i][test_idx] for i in range(len(xtrain))]

    

    ytrain_fold = ytrain[train_idx]

    yvalid_fold = ytrain[test_idx]

    

    # class weights to deal with class imbalance

    positive = train.iloc[train_idx, :].target.value_counts()[0]

    negative = train.iloc[train_idx, :].target.value_counts()[1]

    pos_weight = positive / (positive + negative)

    neg_weight = negative / (positive + negative)



    class_weight = [{0:pos_weight, 1:neg_weight}, {0:neg_weight, 1:pos_weight}]

    

    tf.keras.backend.clear_session()

    

    roberta = build_model()

    roberta.fit(xtrain_fold, ytrain_fold, 

                batch_size=TRAIN_BATCH_SIZE, 

                epochs=EPOCHS, 

                class_weight=class_weight,

                validation_data=(xvalid_fold, yvalid_fold))

    val_preds = roberta.predict(xvalid_fold, batch_size=VALID_BATCH_SIZE, verbose=1)

    val_preds = np.argmax(val_preds, axis=1).flatten()

    print(metrics.accuracy_score(train.iloc[test_idx, :].target.values, val_preds))



    preds = roberta.predict(xtest, batch_size=TRAIN_BATCH_SIZE, verbose=1)

    predictions.append(preds)

sample_submission = pd.read_csv('../input/nlp-getting-started/sample_submission.csv')



predictions = np.average(predictions, axis=0)

predictions = np.argmax(predictions, axis=1).flatten()

sample_submission['target'] = predictions

sample_submission['target'].value_counts()

sample_submission.to_csv('submission.csv', index=False)