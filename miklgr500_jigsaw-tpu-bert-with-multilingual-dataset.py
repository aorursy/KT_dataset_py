import os

import gc

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



np.random.seed(100)
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

    emb = transformer(input_word_ids)[0]

    _avg = tf.keras.layers.GlobalAveragePooling1D()(emb)

    _max = tf.keras.layers.GlobalMaxPooling1D()(emb)

    x = tf.keras.layers.Concatenate()([_avg, _max])

    x = tf.keras.layers.Dropout(0.15)(x)

    out = Dense(1, activation='sigmoid')(x)

    

    model = Model(inputs=input_word_ids, outputs=out)

    model.compile(Adam(lr=3e-5), loss=loss, metrics=[tf.keras.metrics.AUC()])

    

    return model
AUTO = tf.data.experimental.AUTOTUNE



# Create strategy from tpu

tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

tf.config.experimental_connect_to_cluster(tpu)

tf.tpu.experimental.initialize_tpu_system(tpu)

strategy = tf.distribute.experimental.TPUStrategy(tpu)



# Data access

#GCS_DS_PATH = KaggleDatasets().get_gcs_path('kaggle/input/') 
# First load the real tokenizer

tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
valid = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/validation.csv')

test = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/test.csv')

sub = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/sample_submission.csv')
lang_densety = test.groupby('lang').count()['id']/ len(test)

target_densety = valid.groupby('toxic').count()['id']/ len(valid)
lang_densety
target_densety
N_SAMPLES = 500000



train_dfs = []

for lang in ['es', 'it', 'pt', 'tr', 'ru', 'fr']:

    _df = pd.read_csv(f"/kaggle/input/jigsaw-train-multilingual-coments-google-api/jigsaw-toxic-comment-train-google-{lang}-cleaned.csv")

    

    _df0 = _df.loc[_df.toxic == 0, :]

    _df1 = _df.loc[_df.toxic == 1, :]

    

    n_samples = int(N_SAMPLES * lang_densety[lang])

    

    n_samples_0 = int(n_samples * target_densety[0])

    n_samples_1 = int(n_samples * target_densety[1])

    

    _df0 = _df0.sample(n_samples_0)

    _df1 = _df1.sample(n_samples_1)

    

    train_dfs.append(pd.concat([_df0, _df1], ignore_index=True).sample(n_samples_0 + n_samples_1))

    

train = pd.concat(train_dfs, ignore_index=True)

train = train.sample(min(N_SAMPLES, len(train)))

train.toxic = train.toxic.round().astype(int)
train.groupby('toxic').count()['id']/ len(train)
class TextTransformation:

    def __call__(self, text: str, lang: str = None) -> tuple:

        raise NotImplementedError('Abstarct')

        

class LowerCaseTransformation(TextTransformation):

    def __call__(self, text: str, lang: str = None) -> tuple:

        return text.lower(), lang

    

class PunctuationTransformation(TextTransformation):

    def __call__(self, text: str, lang: str = None) -> tuple:

        for p in '?!.,"#$%\'()*+-/:;<=>@[\\]^_`{|}~' + '“”’' +"/-'" + "&" + "¡¿":

            if '’' in text:

                text = text.replace('’', f' \' ')

                

            if '’' in text:

                text = text.replace('’', f' \' ')

              

            if '—' in text:

                text = text.replace('—', f' - ')

                

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

    

class ETransformation(TextTransformation):

    def __call__(self, text: str, lang: str = None) -> tuple:

        for i in ['\u00E8', '\u00E9', '\u00EA', '\u00EB', '\u0450', '\u0451']:

            text = text.replace(i, 'e')

        return text, lang

    

class ATransformation(TextTransformation):

    def __call__(self, text: str, lang: str = None) -> tuple:

        for i in ['à', 'á', '\u00E2', '\u00E3', '\u00E4', '\u00E5']:

            text = text.replace(i, 'a')

        return text, lang

    

class OTransformation(TextTransformation):

    def __call__(self, text: str, lang: str = None) -> tuple:

        for i in ['ó', 'ò', 'ö','õ', 'ô']:

            text = text.replace(i, 'o')

        return text, lang

    



class CTransformation(TextTransformation):

    def __call__(self, text: str, lang: str = None) -> tuple:

        for i in ['ç']:

            text = text.replace(i, 'c')

        return text, lang

    

class ITransformation(TextTransformation):

    def __call__(self, text: str, lang: str = None) -> tuple:

        for i in ['í', 'ı', 'ì']:

            text = text.replace(i, 'i')

        return text, lang

    

class STransformation(TextTransformation):

    def __call__(self, text: str, lang: str = None) -> tuple:

        for i in ['ş']:

            text = text.replace(i, 's')

        return text, lang

    

    

class NTransformation(TextTransformation):

    def __call__(self, text: str, lang: str = None) -> tuple:

        for i in ['ñ', 'n']:

            text = text.replace(i, 'n')

        return text, lang

    

    

class UTransformation(TextTransformation):

    def __call__(self, text: str, lang: str = None) -> tuple:

        for i in ['ù', 'ü', 'û', 'ú']:

            text = text.replace(i, 'u')

        return text, lang

    

class GTransformation(TextTransformation):

    def __call__(self, text: str, lang: str = None) -> tuple:

        for i in ['ğ']:

            text = text.replace(i, 'g')

        return text, lang



class RTransformation(TextTransformation):

    def __call__(self, text: str, lang: str = None) -> tuple:

        for i in ['r']:

            text = text.replace(i, 'r')

        return text, lang

    

class WikiTransformation(TextTransformation):

    def __call__(self, text: str, lang: str = None) -> tuple:

        text = text.replace('wikiproject', ' wiki project ')

        for i in [' vikipedi ', ' wiki ', ' википедии ', " вики ", ' википедия ', ' viki ', ' wikipedien ', ' википедию ']:

            text = text.replace(i, ' wikipedia ')

        return text, lang

    

class PixelTransformation(TextTransformation):

    def __call__(self, text: str, lang: str = None) -> tuple:

        for i in [' px ']:

            text = text.replace(i, ' pixel ')

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

    def __init__(self, transformations: list):

        self._transformations = transformations

        

    def __call__(self, text: str, lang: str = None) -> tuple:

        for transformation in self._transformations:

            text, lang = transformation(text, lang)

        return text, lang

    

    def append(self, transformation: TextTransformation):

        self._transformations.append(transformation)

        

        

transformer = CombineTransformation(

    [

        LowerCaseTransformation(),

        PunctuationTransformation(),

        NumericTransformation(),

        ETransformation(),

        ATransformation(),

        OTransformation(),

        CTransformation(),

        ITransformation(),

        STransformation(),

        UTransformation(),

        GTransformation(),

        NTransformation(),

        WikiTransformation(),

        PixelTransformation(),

        RuTransformation()

    ]

)
train['comment_text'] = [v[0] for v in train.apply(lambda x: transformer(x.comment_text), axis=1).values]

valid['comment_text'] = [v[0] for v in valid.apply(lambda x: transformer(x.comment_text, x.lang), axis=1).values]

test['content'] = [v[0] for v in test.apply(lambda x: transformer(x.content, x.lang), axis=1).values]
sentences = train["comment_text"].apply(lambda x: x.split()).values.tolist() + valid["comment_text"].apply(lambda x: x.split()).values.tolist() + test['content'].apply(lambda x: x.split()).values.tolist()
import operator 

import seaborn as sns

import matplotlib.pyplot as plt



def unknown_plot(data):

    fig, axes = plt.subplots(ncols=1, figsize=(10, 20))

    plt.tight_layout()

    

    sns.barplot(y=list(data.keys()), x=list(data.values()), ax=axes, color='green')



    axes.spines['right'].set_visible(False)

    axes.set_xlabel('')

    axes.set_ylabel('')

    axes.tick_params(axis='x', labelsize=13)

    axes.tick_params(axis='y', labelsize=13)



    axes.set_title(f'Most unknown tokens', fontsize=15)



    plt.show()

    

def check_coverage(vocab,embeddings_index):

    a = {}

    oov = {}

    k = 0

    i = 0

    for word in tqdm(vocab):

        try:

            a[word] = embeddings_index[word]

            k += vocab[word]

        except:



            oov[word] = vocab[word]

            i += vocab[word]

            pass



    print('Found embeddings for {:.2%} of vocab'.format(len(a) / len(vocab)))

    print('Found embeddings for  {:.2%} of all text'.format(k / (k + i)))

    sorted_x = sorted(oov.items(), key=operator.itemgetter(1))[::-1]



    return sorted_x





def build_vocab(sentences, verbose =  True):

    """

    :param sentences: list of list of words

    :return: dictionary of words and their count

    """

    vocab = {}

    for sentence in tqdm(sentences, disable = (not verbose)):

        for word in sentence:

            try:

                vocab[word] += 1

            except KeyError:

                vocab[word] = 1

    return vocab





vocab = build_vocab(sentences)

unknown_vocab = check_coverage(vocab, tokenizer.get_vocab())

len(unknown_vocab)
unknown_plot({k:v for i, (k,v) in enumerate(unknown_vocab) if i < 100})
del sentences
# Save the loaded tokenizer locally

save_path = '/kaggle/working/distilbert_base_uncased/'

if not os.path.exists(save_path):

    os.makedirs(save_path)

tokenizer.save_pretrained(save_path)



# Reload it with the huggingface tokenizers library

fast_tokenizer = BertWordPieceTokenizer('distilbert_base_uncased/vocab.txt', lowercase=True)

fast_tokenizer
x_train = fast_encode(train.comment_text.astype(str), fast_tokenizer, maxlen=512)

x_valid = fast_encode(valid.comment_text.astype(str), fast_tokenizer, maxlen=512)

x_test = fast_encode(test.content.astype(str), fast_tokenizer, maxlen=512)

y_train = train.toxic.values

y_valid = valid.toxic.values



del train_dfs, train, valid, test

gc.collect()
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



test_dataset = (

    tf.data.Dataset

    .from_tensor_slices(x_test)

    .batch(64)

)

from tensorflow.keras import backend as K



def focal_loss(gamma=2., alpha=.15):

    def focal_loss_fixed(y_true, y_pred):

        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))

        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

    return focal_loss_fixed
%%time

with strategy.scope():

    transformer_layer = transformers.TFBertModel.from_pretrained('bert-base-multilingual-uncased')

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

            self.test_pred.append(self.model.predict(self.test_data))
def build_lrfn(lr_start=0.000001, lr_max=0.000004, 

               lr_min=0.0000001, lr_rampup_epochs=5, 

               lr_sustain_epochs=3, lr_exp_decay=.87):

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
lrfn = build_lrfn()

lr_schedule = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=1)

er = tf.keras.callbacks.EarlyStopping(monitor='val_auc', patience=20, restore_best_weights=True, mode='max')



train_history = model.fit(

    train_dataset,

    steps_per_epoch=200,

    validation_data=valid_dataset,

    callbacks=[lr_schedule, er],

    epochs=35

)
bert_weights = model.get_weights()
import matplotlib.pyplot as plt



plt.figure(figsize=(10, 7))



lrfn = build_lrfn(lr_start=0.000001, lr_max=0.000001, 

               lr_min=0.0000001, lr_rampup_epochs=2, 

               lr_sustain_epochs=1, lr_exp_decay=.65)

plt.plot([i for i in range(15)], [lrfn(i) for i in range(15)]);
from sklearn.model_selection import train_test_split, StratifiedKFold
skf = StratifiedKFold(n_splits=5)



test_preds = np.zeros((len(x_test),))



for tr_idx, vl_idx in skf.split(x_valid, y_valid):

    lr_schedule = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=1)

    er = tf.keras.callbacks.EarlyStopping(monitor='val_auc', patience=20, restore_best_weights=True, mode='max')

    x_tr, x_val, y_tr, y_val = x_valid[tr_idx], x_valid[vl_idx], y_valid[tr_idx], y_valid[vl_idx]



    train_dataset = (

        tf.data.Dataset

        .from_tensor_slices((x_tr, y_tr))

        .repeat()

        .shuffle(2048)

        .batch(32)

        .prefetch(AUTO)

    )



    valid_dataset = (

        tf.data.Dataset

        .from_tensor_slices((x_val, y_val))

        .batch(32)

        .cache()

        .prefetch(AUTO)

    )

    

    model.set_weights(bert_weights)

    train_history = model.fit(

        train_dataset,

        steps_per_epoch=40,

        validation_data=valid_dataset,

        callbacks=[lr_schedule, er],

        epochs=15)

    test_preds += model.predict(test_dataset)[:, 0] / 5
sub['toxic'] = test_preds

sub.to_csv('submission.csv', index=False)