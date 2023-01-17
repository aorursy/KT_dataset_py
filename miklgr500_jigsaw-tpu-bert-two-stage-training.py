import os

import warnings



import numpy as np

import pandas as pd

import tensorflow as tf

from tensorflow.keras.layers import Dense, Input

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import ModelCheckpoint

from kaggle_datasets import KaggleDatasets

import transformers

import traitlets

import matplotlib.pyplot as plt

import seaborn as sns

from tqdm.notebook import tqdm

from tokenizers import BertWordPieceTokenizer

from sklearn.metrics import roc_auc_score



warnings.simplefilter("ignore")
class TextTransformation:

    def __call__(self, text: str, lang: str = None) -> tuple:

        raise NotImplementedError('Abstarct')   

        

class LowerCaseTransformation(TextTransformation):

    def __call__(self, text: str, lang: str = None) -> tuple:

        return text.lower(), lang

    

    

class URLTransformation(TextTransformation):

    def __call__(self, text: str, lang: str = None) -> tuple:

        for url in self.find_urls(text):

            if url in text:

                text.replace(url, ' external link ')

        return text.lower(), lang

    

    @staticmethod

    def find_urls(string): 

        # https://www.geeksforgeeks.org/python-check-url-string/

        urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', string) 

        return urls 

    

class PunctuationTransformation(TextTransformation):

    def __call__(self, text: str, lang: str = None) -> tuple:

        for p in '?!.,"#$%\'()*+-/:;<=>@[\\]^_`{|}~' + '“”’' +"/-'" + "&" + "¡¿":

            if '’' in text:

                text = text.replace('’', f' \' ')

                

            if '’' in text:

                text = text.replace('’', f' \' ')

              

            if '—' in text:

                text = text.replace('—', f' - ')

                

            if '−' in text:

                text = text.replace('−', f' - ')   

                

            if '–' in text:

                text = text.replace('–', f' - ')   

              

            if '“' in text:

                text = text.replace('“', f' " ')   

                

            if '«' in text:

                text = text.replace('«', f' " ')   

                

            if '»' in text:

                text = text.replace('»', f' " ')   

            

            if '”' in text:

                text = text.replace('”', f' " ') 

                

            if '`' in text:

                text = text.replace('`', f' \' ')              



            text = text.replace(p, f' {p} ')

                

        return text.strip(), lang

    

    

class NumericTransformation(TextTransformation):

    def __call__(self, text: str, lang: str = None) -> tuple:

        for i in range(10):

            text = text.replace(str(i), f' {str(i)} ')

        return text, lang

    

class WikiTransformation(TextTransformation):

    def __call__(self, text: str, lang: str = None) -> tuple:

        text = text.replace('wikiproject', ' wiki project ')

        for i in [' vikipedi ', ' wiki ', ' википедии ', " вики ", ' википедия ', ' viki ', ' wikipedien ', ' википедию ']:

            text = text.replace(i, ' wikipedia ')

        return text, lang

    

    

class MessageTransformation(TextTransformation):

    def __call__(self, text: str, lang: str = None) -> tuple:

        text = text.replace('wikiproject', ' wiki project ')

        for i in [' msg ', ' msj ', ' mesaj ']:

            text = text.replace(i, ' message ')

        return text, lang

    

    

class PixelTransformation(TextTransformation):

    def __call__(self, text: str, lang: str = None) -> tuple:

        for i in [' px ']:

            text = text.replace(i, ' pixel ')

        return text, lang

    

    

class SaleBotTransformation(TextTransformation):

    def __call__(self, text: str, lang: str = None) -> tuple:

        text = text.replace('salebot', ' sale bot ')

        return text, lang

    

    

class RuTransformation(TextTransformation):

    def __call__(self, text: str, lang: str = None) -> tuple:

        if lang is not None and lang == 'ru' and 'http' not in text and 'jpg' not in text and 'wikipedia' not in text:

            text = text.replace('t', 'т')

            text = text.replace('h', 'н')

            text = text.replace('b', 'в')

            text = text.replace('c', 'c')

            text = text.replace('k', 'к')

            text = text.replace('e', 'е')

            text = text.replace('a', 'а')

        return text, lang

    

class CombineTransformation(TextTransformation):

    def __init__(self, transformations: list, return_lang: bool = False):

        self._transformations = transformations

        self._return_lang = return_lang

        

    def __call__(self, text: str, lang: str = None) -> tuple:

        for transformation in self._transformations:

            text, lang = transformation(text, lang)

        if self._return_lang:

            return text, lang

        return text

    

    def append(self, transformation: TextTransformation):

        self._transformations.append(transformation)
def fast_encode(texts, tokenizer, chunk_size=256, maxlen=512):

    tokenizer.enable_truncation(max_length=maxlen)

    tokenizer.enable_padding(max_length=maxlen)

    all_ids = []

    

    for i in tqdm(range(0, len(texts), chunk_size)):

        text_chunk = texts[i:i+chunk_size].tolist()

        encs = tokenizer.encode_batch(text_chunk)

        all_ids.extend([enc.ids for enc in encs])

    

    return np.array(all_ids)
def build_model(transformer, loss='binary_crossentropy', max_len=512):

    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")

    sequence_output = transformer(input_word_ids)[0]

    cls_token = sequence_output[:, 0, :]

    x = tf.keras.layers.Dropout(0.35)(cls_token)

    out = Dense(1, activation='sigmoid')(x)

    

    model = Model(inputs=input_word_ids, outputs=out)

    model.compile(Adam(lr=3e-5), loss=loss, metrics=[tf.keras.metrics.AUC()])

    

    return model
# https://stackoverflow.com/questions/8897593/how-to-compute-the-similarity-between-two-text-documents

import nltk, string

from sklearn.feature_extraction.text import TfidfVectorizer



nltk.download('punkt') # if necessary...





stemmer = nltk.stem.porter.PorterStemmer()

remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)



def stem_tokens(tokens):

    return [stemmer.stem(item) for item in tokens]



'''remove punctuation, lowercase, stem'''

def normalize(text):

    return stem_tokens(nltk.word_tokenize(text.lower().translate(remove_punctuation_map)))



vectorizer = TfidfVectorizer(tokenizer=normalize, stop_words='english')



def cosine_sim(text1, text2):

    tfidf = vectorizer.fit_transform([text1, text2])

    return ((tfidf * tfidf.T).A)[0,1]
AUTO = tf.data.experimental.AUTOTUNE



# Create strategy from tpu

tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

tf.config.experimental_connect_to_cluster(tpu)

tf.tpu.experimental.initialize_tpu_system(tpu)

strategy = tf.distribute.experimental.TPUStrategy(tpu)



# Data access

#GCS_DS_PATH = KaggleDatasets().get_gcs_path('kaggle/input/') 
# First load the real tokenizer

tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-cased')



# Save the loaded tokenizer locally

save_path = '/kaggle/working/distilbert_base_uncased/'

if not os.path.exists(save_path):

    os.makedirs(save_path)

tokenizer.save_pretrained(save_path)



# Reload it with the huggingface tokenizers library

fast_tokenizer = BertWordPieceTokenizer('distilbert_base_uncased/vocab.txt', lowercase=False)

fast_tokenizer
transformer = CombineTransformation(

    [

        LowerCaseTransformation(),

        PunctuationTransformation(),

        NumericTransformation(),

        PixelTransformation(),

        MessageTransformation(),

        WikiTransformation(),

        SaleBotTransformation()

    ]

)
train1 = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv")

train1['comment_text'] = train1.apply(lambda x: transformer(x.comment_text), axis=1)

#train2 = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-unintended-bias-train.csv")



valid1 = pd.read_csv('/kaggle/input/val-en-df/validation_en.csv')

valid1['comment_text'] = valid1.apply(lambda x: transformer(x.comment_text_en), axis=1)

valid2 = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-test-translated/jigsaw_miltilingual_valid_translated.csv')

valid2['comment_text'] = valid2.apply(lambda x: transformer(x.translated), axis=1)

test1 = pd.read_csv('/kaggle/input/test-en-df/test_en.csv')

test1['comment_text'] = test1.apply(lambda x: transformer(x.content_en), axis=1)

test2 = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-test-translated/jigsaw_miltilingual_test_translated.csv')

test2['comment_text'] = test2.apply(lambda x: transformer(x.translated), axis=1)

sub = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/sample_submission.csv')
test1.tail(15)
plt.figure(figsize=(12, 8))



sns.distplot(train1.comment_text.str.len(), label='train')

sns.distplot(test1.content_en.str.len(), label='test1')

sns.distplot(test2.translated.str.len(), label='test2')

plt.legend();
plt.figure(figsize=(12, 8))



sns.distplot(train1.comment_text.str.len(), label='train')

sns.distplot(test1.content_en.str.len(), label='test1')

sns.distplot(test2.translated.str.len(), label='test2')

plt.xlim([0, 512])

plt.legend();
x_train = fast_encode(train1.comment_text.astype(str), fast_tokenizer, maxlen=512)

x_valid1 = fast_encode(valid1.comment_text_en.astype(str), fast_tokenizer, maxlen=512)

x_valid2 = fast_encode(valid2.translated.astype(str), fast_tokenizer, maxlen=512)

x_valid = np.concatenate([x_valid1, x_valid2], axis=0)

x_test1 = fast_encode(test1.content_en.astype(str), fast_tokenizer, maxlen=512)

x_test2 = fast_encode(test2.translated.astype(str), fast_tokenizer, maxlen=512)



y_train = train1.toxic.values

y_valid1 = valid1.toxic.values

y_valid2 = valid2.toxic.values

y_valid = np.concatenate([y_valid1, y_valid2], axis=0)
train_dataset = (

    tf.data.Dataset

    .from_tensor_slices((x_train, y_train))

    .repeat()

    .shuffle(2048)

    .batch(64)

    .prefetch(AUTO)

)



valid_dataset = (

    tf.data.Dataset

    .from_tensor_slices((x_valid, y_valid))

    .batch(64)

    .cache()

    .prefetch(AUTO)

)



test_dataset = [(

    tf.data.Dataset

    .from_tensor_slices(x_test1)

    .batch(64)

),

    (

    tf.data.Dataset

    .from_tensor_slices(x_test2)

    .batch(64)

)]
from tensorflow.keras import backend as K



def focal_loss(gamma=2., alpha=.2):

    def focal_loss_fixed(y_true, y_pred):

        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))

        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

    return focal_loss_fixed
%%time

with strategy.scope():

    transformer_layer = transformers.TFBertModel.from_pretrained('bert-base-cased')

    model = build_model(transformer_layer, loss=focal_loss(gamma=1.5), max_len=512)

model.summary()
from tensorflow.keras.callbacks import Callback 



class RocAucCallback(Callback):

    def __init__(self, test_data, score_thr):

        self.test_data = test_data

        self.score_thr = score_thr

        self.test_pred = []

        

    def on_epoch_end(self, epoch, logs=None):

        if logs['val_auc'] > self.score_thr:

            print('\nRun TTA...')

            for td in self.test_data:

                self.test_pred.append(self.model.predict(td))
def build_lrfn(lr_start=0.000001, lr_max=0.000004, 

               lr_min=0.0000001, lr_rampup_epochs=7, 

               lr_sustain_epochs=0, lr_exp_decay=.87):

    lr_max = lr_max * strategy.num_replicas_in_sync



    def lrfn(epoch):

        if epoch < lr_rampup_epochs:

            lr = (lr_max - lr_start) / lr_rampup_epochs * epoch + lr_start

        elif epoch < lr_rampup_epochs + lr_sustain_epochs:

            lr = lr_max

        else:

            lr = (lr_max - lr_min) * lr_exp_decay**(epoch - lr_rampup_epochs - lr_sustain_epochs) + lr_min

        return lr

    

    return lrfn
import matplotlib.pyplot as plt



plt.figure(figsize=(10, 7))



_lrfn = build_lrfn()

plt.plot([i for i in range(35)], [_lrfn(i) for i in range(35)]);
roc_auc = RocAucCallback(test_dataset, 0.935)

lrfn = build_lrfn()

lr_schedule = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=1)



train_history = model.fit(

    train_dataset,

    steps_per_epoch=150,

    validation_data=valid_dataset,

    callbacks=[lr_schedule, roc_auc],

    epochs=35

)
import matplotlib.pyplot as plt



plt.figure(figsize=(10, 7))



lrfn = build_lrfn(lr_start=0.000001, lr_max=0.0000005, 

               lr_min=0.0000001, lr_rampup_epochs=5, 

               lr_sustain_epochs=0, lr_exp_decay=.87)

plt.plot([i for i in range(25)], [lrfn(i) for i in range(25)]);
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split



x_valid_train1, x_valid_valid1, y_valid_train1, y_valid_valid1 = train_test_split(x_valid1, y_valid1, test_size=0.15, shuffle=True, random_state=123, stratify=y_valid1)



x_valid_train2, x_valid_valid2, y_valid_train2, y_valid_valid2 = train_test_split(x_valid2, y_valid2, test_size=0.15, shuffle=True, random_state=123, stratify=y_valid2)



x_valid_train = np.concatenate([x_valid_train1, x_valid_train2], axis=0)

y_valid_train = np.concatenate([y_valid_train1, y_valid_train2], axis=0)



x_valid_valid = np.concatenate([x_valid_valid1, x_valid_valid2], axis=0)

y_valid_valid = np.concatenate([y_valid_valid1, y_valid_valid2], axis=0)



valid_train_dataset = (

tf.data.Dataset

.from_tensor_slices((x_valid_train, y_valid_train))

.shuffle(2048)

.batch(64)

.cache()

.prefetch(AUTO)

)



valid_valid_dataset = (

tf.data.Dataset

.from_tensor_slices((x_valid_valid, y_valid_valid))

.batch(64)

.cache()

.prefetch(AUTO)

)
lr_schedule = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=1)



train_history = model.fit(

valid_train_dataset,

steps_per_epoch=75,

validation_data=valid_valid_dataset,

callbacks=[lr_schedule, roc_auc],

epochs=25

)
sub['toxic'] = np.mean(roc_auc.test_pred, axis=0)

sub.to_csv('submission.csv', index=False)