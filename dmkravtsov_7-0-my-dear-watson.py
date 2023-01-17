# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
## based on https://www.kaggle.com/mattbast/training-transformers-with-tensorflow-and-tpus
!pip install git+https://github.com/ssut/py-googletrans.git
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import tensorflow as tf

from transformers import AutoTokenizer, TFAutoModel

from tqdm.notebook import tqdm

from sklearn.utils import shuffle

from googletrans import Translator

from dask import bag, diagnostics

from tqdm import tqdm, tqdm_gui

tqdm.pandas(ncols=75) 

from bs4 import BeautifulSoup

import re

import warnings

warnings.filterwarnings('ignore')



from nltk.corpus import stopwords

", ".join(stopwords.words('english'))

STOPWORDS = set(stopwords.words('english'))



import string



from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
## setting up TPUs strategy/don't forget to turn on Accelerator to TPU



try:

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  

    print('Running on TPU ', tpu.master())

except ValueError:

    tpu = None



if tpu:

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

else:

    strategy = tf.distribute.get_strategy() 



print("REPLICAS: ", strategy.num_replicas_in_sync)
MODEL_NAME = 'jplu/tf-xlm-roberta-large'

EPOCHS = 10

MAX_LEN = 80

RATE = 1e-5

BATCH_SIZE = 64 * strategy.num_replicas_in_sync
## loading dataframes

train = pd.read_csv('/kaggle/input/contradictory-my-dear-watson/train.csv')

test = pd.read_csv('/kaggle/input/contradictory-my-dear-watson/test.csv')

submission = pd.read_csv('/kaggle/input/contradictory-my-dear-watson/sample_submission.csv')
## selecting actual features in df train

train = train[['premise', 'hypothesis', 'label', 'lang_abv']]

train.head(10)
def trans_parallel(df, dest):

    premise_bag = bag.from_sequence(df.premise.tolist()).map(translate, dest)

    hypo_bag =  bag.from_sequence(df.hypothesis.tolist()).map(translate, dest)

    with diagnostics.ProgressBar():

        premises = premise_bag.compute()

        hypos = hypo_bag.compute()

    df[['premise', 'hypothesis']] = list(zip(premises, hypos))

    return df
def translate(text, dest):

    translator = Translator()

    return translator.translate(text, dest='en').text
# eng = train.loc[train.lang_abv == "en"].copy().pipe(trans_parallel, dest=None)

eng = train.loc[train.lang_abv == "en"].copy()

non_eng =  train.loc[train.lang_abv != "en"].copy().pipe(trans_parallel, dest='en')

train = eng.append(non_eng)
# eng = test.loc[test.lang_abv == "en"].copy().pipe(trans_parallel, dest=None)

eng = test.loc[test.lang_abv == "en"].copy()

non_eng =  test.loc[test.lang_abv != "en"].copy().pipe(trans_parallel, dest='en')

test = eng.append(non_eng)
def remove_space(text):

    return " ".join(text.split())



def remove_punctuation(text):

    return re.sub("[!@#$+%*:()'-]", ' ', text)



def remove_html(text):

    soup = BeautifulSoup(text, 'lxml')

    return soup.get_text()



def remove_url(text):

    return re.sub(r"http\S+", "", text)





def remove_stopwords(text):

    """custom function to remove the stopwords"""

    return " ".join([word for word in str(text).split() if word not in STOPWORDS])



def lemmatize_words(text):

    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])





def clean_text(text):

    text = remove_space(text)

    text = remove_html(text)

    text = remove_url(text)

    text = remove_punctuation(text)

    text = remove_stopwords(text)

    text = lemmatize_words(text)

    text = text.lower()

    return text



train['premise'] = train.premise.progress_apply(lambda text : clean_text(text))

train['hypothesis'] = train.hypothesis.apply(lambda text : clean_text(text))



test['premise'] = test.premise.progress_apply(lambda text : clean_text(text))

test['hypothesis'] = test.hypothesis.apply(lambda text : clean_text(text))
train.tail(60)
train.shape
test.tail(60)
# ## adding additional datasets for more samples

# # first dataset

# mnli = nlp.load_dataset(path='glue', name='mnli')
# # re-shape nlp dataset into pandas

# index = []

# premise = []

# hypothesis = []

# label = []



# for example in mnli['train']:

#     premise.append(example['premise'])

#     hypothesis.append(example['hypothesis'])

#     label.append(example['label'])

# for example in mnli['validation_matched']:

#     premise.append(example['premise'])

#     hypothesis.append(example['hypothesis'])

#     label.append(example['label'])

# for example in mnli['validation_mismatched']:

#     premise.append(example['premise'])

#     hypothesis.append(example['hypothesis'])

#     label.append(example['label'])

# for example in mnli['test_matched']:

#     premise.append(example['premise'])

#     hypothesis.append(example['hypothesis'])

#     label.append(example['label'])

# # pandas dataset into pands dataframe

# mnli = pd.DataFrame(data={

#     'premise': premise,

#     'hypothesis': hypothesis,

#     'label': label

# })

# # we got:

# mnli.head()
# ## second dataset

# anli = nlp.load_dataset(path='anli')
# # re-shape 2nd nlp dataset into pandas

# index = []

# premise = []

# hypothesis = []

# label = []



# for example in anli['train_r1']:

#     premise.append(example['premise'])

#     hypothesis.append(example['hypothesis'])

#     label.append(example['label'])

    

# for example in anli['train_r2']:

#     premise.append(example['premise'])

#     hypothesis.append(example['hypothesis'])

#     label.append(example['label'])

    

# for example in anli['train_r3']:

#     premise.append(example['premise'])

#     hypothesis.append(example['hypothesis'])

#     label.append(example['label'])
# # pandas 2 nd dataset into pands dataframe

# anli = pd.DataFrame(data={

#     'premise': premise,

#     'hypothesis': hypothesis,

#     'label': label

# })
# # re-shape nlp dataset into pandas

# index = []

# premise = []

# hypothesis = []

# label = []



# for example in snli['test']:

#     premise.append(example['premise'])

#     hypothesis.append(example['hypothesis'])

#     label.append(example['label'])

# for example in snli['train']:

#     premise.append(example['premise'])

#     hypothesis.append(example['hypothesis'])

#     label.append(example['label'])

# for example in snli['validation']:

#     premise.append(example['premise'])

#     hypothesis.append(example['hypothesis'])

#     label.append(example['label'])
# # pandas dataset into pands dataframe

# snli = pd.DataFrame(data={

#     'premise': premise,

#     'hypothesis': hypothesis,

#     'label': label

# })
train = pd.concat([train])

# train = pd.concat([train, mnli, anli, snli])

# train = shuffle(train)
train.info()
train.head()
## let's make numbers from text



tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
train_text = train[['premise', 'hypothesis']].values.tolist()

test_text = test[['premise', 'hypothesis']].values.tolist()

## splitting sentences into array of numbers

train_encoded = tokenizer.batch_encode_plus(

    train_text,

    pad_to_max_length=True,

    max_length=MAX_LEN

)

test_encoded = tokenizer.batch_encode_plus(

    test_text,

    pad_to_max_length=True,

    max_length=MAX_LEN

)

# splitting dataset into test and train datasets

x_train, x_valid, y_train, y_valid = train_test_split(

    train_encoded['input_ids'], 

    train.label.values, 

    test_size=0.2, 

    random_state=2020

)

x_test = test_encoded['input_ids']
## data pipeline building

auto = tf.data.experimental.AUTOTUNE



train_dataset = (

    tf.data.Dataset

    .from_tensor_slices((x_train, y_train))

    .repeat()

    .shuffle(2048)

    .batch(BATCH_SIZE)

    .prefetch(auto)

)
valid_dataset = (

    tf.data.Dataset

    .from_tensor_slices((x_valid, y_valid))

    .batch(BATCH_SIZE)

    .cache()

    .prefetch(auto)

)
test_dataset = (

    tf.data.Dataset

    .from_tensor_slices(x_test)

    .batch(BATCH_SIZE)

)
with strategy.scope():

    backbone = TFAutoModel.from_pretrained(MODEL_NAME)
with strategy.scope():

    x_input = tf.keras.Input(shape=(MAX_LEN,), dtype=tf.int32, name="input_ids")



    x = backbone(x_input)[0]



    x = x[:, 0, :]



    x = tf.keras.layers.Dense(3, activation='softmax')(x)



    model = tf.keras.models.Model(inputs=x_input, outputs=x)

    





model.summary()
model.compile(

    tf.keras.optimizers.Adam(lr=RATE), 

    loss='sparse_categorical_crossentropy', 

    metrics=['accuracy']

)
steps = len(x_train) // BATCH_SIZE



history = model.fit(

    train_dataset,

    validation_data=valid_dataset,

    epochs=10,

    steps_per_epoch=steps,

)
test_preds = model.predict(test_dataset, verbose=1)

submission['prediction'] = test_preds.argmax(axis=1)

submission.to_csv('submission.csv', index=False)

submission.head()
## some outputs:

# accuracy based only on original dataset 0.724183/0.7459 (takes abt 10 mins)/10 epochs

# translator + cleaner 0.9784 / 0.7876

# accuracy based on original dataset+mnli+anli (570000 samples) 0.8633 (takes abt 2 hours)