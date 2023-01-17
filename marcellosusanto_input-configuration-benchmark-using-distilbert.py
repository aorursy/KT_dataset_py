# Setting package umum 

import pandas as pd

import pandas_profiling as pp

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

import seaborn as sns

from tqdm import tqdm_notebook as tqdm

%matplotlib inline



from matplotlib.pylab import rcParams

# For every plotting cell use this

# grid = gridspec.GridSpec(n_row,n_col)

# ax = plt.subplot(grid[i])

# fig, axes = plt.subplots()

rcParams['figure.figsize'] = [10,5]

plt.style.use('fivethirtyeight') 

sns.set_style('whitegrid')



import warnings

warnings.filterwarnings('ignore')

from tqdm import tqdm



pd.set_option('display.max_rows', 50)

pd.set_option('display.max_columns', 150)

pd.options.display.float_format = '{:.4f}'.format
### Install packages

from IPython.display import clear_output



!pip install googletrans

!pip install p_tqdm

!pip install transformers==3.0.2

clear_output()
# Load dataset

df_train = pd.read_csv('../input/contradictory-my-dear-watson/train.csv')

df_test = pd.read_csv('../input/contradictory-my-dear-watson/test.csv')
### Overview dataset

df_train.head(11)
### Translate text into english

import multiprocessing

from multiprocessing import Pool

from p_tqdm import p_map

from googletrans import Translator

translator = Translator()



### Function for paralallel translation

def translate_to_en(text) :

    global translator

    translated_text = translator.translate(text, dest = 'en' ).text

    

    return translated_text



def pool_translated_en(list_text) :

    threads = 8*multiprocessing.cpu_count()

    

    with Pool(threads) as executor:

        result = list(tqdm(executor.imap(translate_to_en, list_text), total=len(list_text)))

        

    return result



def dataset_translated_en(df) :

    

    ### Initialize translated dataset

    df_trans = df.copy()

    

    ### Get non-english text

    list_idx = list(df_trans[df_trans['lang_abv']!='en'].index)

    list_prem = list(df_trans.loc[list_idx]['premise'])

    list_hyp = list(df_trans.loc[list_idx]['hypothesis'])

    

    ### Translate premise

    print('\nPremise Translation - En')

    df_trans.loc[list_idx,'premise'] = pool_translated_en(list_prem)

    

    ### Translate hypotheses

    print('\nPremise Hypotheses - En')

    df_trans.loc[list_idx,'hypothesis'] = pool_translated_en(list_hyp)



    ### Change languange value

    df_trans['lang_abv'] = 'en'

    df_trans['language'] = 'English'

    

    return df_trans

    

### Translate!

df_train_en = dataset_translated_en(df_train)

df_test_en = dataset_translated_en(df_test)
### Compare translation result

import random

list_non_en_idx = list(df_train[df_train['lang_abv']!='en'].index)



for i in range(5) :

    idx = random.randint(0,len(list_non_en_idx))

    print('\nORIGINAL TEXT :',df_train['premise'][list_non_en_idx[idx]])

    print('TRANSLATED TEXT :',df_train_en['premise'][list_non_en_idx[idx]])
### Proportion for each languange in train dataset

print('Proportion for each languange in train dataset')

df_train['language'].value_counts() / len(df_train) * 100
### Remove unnecessary whitespaces, lower text and remove punctuation

import string



def remove_punctuation(text) :

    no_punct = ''.join([c for c in text if c not in string.punctuation])

    

    return no_punct



def quick_clean_data(dataset, var) :

    df = dataset.copy()

    

    # Lowercase

    df[var] = df[var].str.lower()

    

    # Strip whitespaces

    df[var] = df[var].str.strip()

    

    # Remove punctuation

    df[var] = df.apply(lambda x : remove_punctuation(x[var]), axis=1)

    

    # Remove double whitespaces

    df[var] = df.apply(lambda x : " ".join(x[var].split()), axis=1)

    

    return df



list_var = ['premise','hypothesis']

for var in list_var :

    df_train = quick_clean_data(df_train, var)

    df_test = quick_clean_data(df_test, var)

    df_train_en = quick_clean_data(df_train_en, var)

    df_test_en = quick_clean_data(df_test_en, var)
### Make another dataset with stopwords removed

from nltk.corpus import stopwords

from tqdm._tqdm_notebook import tqdm_notebook

tqdm_notebook.pandas()



def remove_stop_words(text) :

    

    # List of stop words

    en_stop_words = stopwords.words('english')

    

    # Remove stop words 

    text = ' '.join([c for c in text.split() if c not in en_stop_words])    

    

    return text



### Initialize dataset

df_train_no_stop = df_train_en.copy()

df_test_no_stop = df_test_en.copy()

list_var = ['premise','hypothesis']



for var in list_var :

    df_train_no_stop[var] = df_train_no_stop.progress_apply(lambda x : remove_stop_words(x[var]), axis=1)

    df_test_no_stop[var] = df_test_no_stop.progress_apply(lambda x : remove_stop_words(x[var]), axis=1)
### Compare stopwords result

import random

list_idx = list(df_train_no_stop.index)



for i in range(5) :

    idx = random.randint(0,len(list_idx))

    print('\nORIGINAL TEXT :',df_train_en['premise'][list_idx[idx]])

    print('NO STOPWORDS TEXT :',df_train_no_stop['premise'][list_idx[idx]])
### Distribution of word count in original dataset

rcParams['figure.figsize'] = [15,5]

plt.style.use('fivethirtyeight') 

sns.set_style('whitegrid')

grid = gridspec.GridSpec(1,2)



### Setting up data for plot

df_plot = df_train.copy()

df_plot['premise_word_count'] = df_plot.apply(lambda x : len(x['premise'].split()), axis=1)

df_plot['hypothesis_word_count'] = df_plot.apply(lambda x : len(x['hypothesis'].split()), axis=1)



### Plotting

list_var = ['premise_word_count', 'hypothesis_word_count']

list_color = ['#db3236','#4885ed']

for i,var in enumerate(list_var) :

    ax = plt.subplot(grid[i])

    sns.distplot(df_plot[var], kde=False, ax=ax, color=list_color[i])



plt.suptitle('Distribution of word count on ORIGINAL dataset') ;

plt.tight_layout() ;

plt.subplots_adjust(top=0.9) ;
### Distribution of word count in all english dataset

rcParams['figure.figsize'] = [15,5]

plt.style.use('fivethirtyeight') 

sns.set_style('whitegrid')

grid = gridspec.GridSpec(1,2)



### Setting up data for plot

df_plot = df_train_en.copy()

df_plot['premise_word_count'] = df_plot.apply(lambda x : len(x['premise'].split()), axis=1)

df_plot['hypothesis_word_count'] = df_plot.apply(lambda x : len(x['hypothesis'].split()), axis=1)



### Plotting

list_var = ['premise_word_count', 'hypothesis_word_count']

list_color = ['#db3236','#4885ed']

for i,var in enumerate(list_var) :

    ax = plt.subplot(grid[i])

    sns.distplot(df_plot[var], kde=False, ax=ax, color=list_color[i])



plt.suptitle('Distribution of word count on ALL ENGLISH dataset') ;

plt.tight_layout() ;

plt.subplots_adjust(top=0.9) ;
### Distribution of word count in no stopwords dataset

rcParams['figure.figsize'] = [15,5]

plt.style.use('fivethirtyeight') 

sns.set_style('whitegrid')

grid = gridspec.GridSpec(1,2)



### Setting up data for plot

df_plot = df_train_no_stop.copy()

df_plot['premise_word_count'] = df_plot.apply(lambda x : len(x['premise'].split()), axis=1)

df_plot['hypothesis_word_count'] = df_plot.apply(lambda x : len(x['hypothesis'].split()), axis=1)



### Plotting

list_var = ['premise_word_count', 'hypothesis_word_count']

list_color = ['#db3236','#4885ed']

for i,var in enumerate(list_var) :

    ax = plt.subplot(grid[i])

    sns.distplot(df_plot[var], kde=False, ax=ax, color=list_color[i])



plt.suptitle('Distribution of word count on NO STOPWORDS dataset') ;

plt.tight_layout() ;

plt.subplots_adjust(top=0.9) ;
### Proportion of target class

rcParams['figure.figsize'] = [8,5]

plt.style.use('fivethirtyeight') 

sns.set_style('whitegrid')



### Function to plot donut chart

def make_donut_chart(sizes, labels, colors=None, explode=None) :

  

    # Make pie chart

    plt.pie(sizes, colors = colors, labels=labels, autopct='%1.1f%%', startangle=90, pctdistance=0.85, explode = explode)



    # Make inner circle

    centre_circle = plt.Circle((0,0),0.70,fc='white')

    fig = plt.gcf()

    fig.gca().add_artist(centre_circle)



    plt.axis('equal')  

    plt.tight_layout()

    

# Plot preparation

sizes = df_train['label'].value_counts() / len(df_train) * 100

labels = ['entailment','contradiction','neutral']

colors = ['#4285F4','#EA4335','#6a737b']

explode_donut = [0.05, 0.05, 0.05]



# Plot

make_donut_chart(sizes, labels, colors, explode_donut)

plt.title('Percentage of label class', fontsize=18, fontname='Monospace', fontweight="bold") ;
### Make dummy label for stratified sampling on original dataset

LANGUAGE_MAP = {

            "English"   : 0,

            "Chinese"   : 1,

            "Arabic"    : 2,

            "French"    : 3,

            "Swahili"   : 4,

            "Urdu"      : 5,

            "Vietnamese": 6,

            "Russian"   : 7,

            "Hindi"     : 8,

            "Greek"     : 9,

            "Thai"      : 10,

            "Spanish"   : 11,

            "German"    : 12,

            "Turkish"   : 13,

            "Bulgarian" : 14

        }



df_train['language'] = df_train['language'].map(LANGUAGE_MAP)

df_train['language_label'] = df_train['language'].astype(str) + "_" + df_train['label'].astype(str)
### Initialize accelerator

import tensorflow as tf



def initialize_accelerator(ACCELERATOR) :



    # checking TPU first

    if ACCELERATOR == "TPU":

        print("Connecting to TPU")

        try:

            tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

            print(f"Running on TPU {tpu.master()}")

        except ValueError:

            print("Could not connect to TPU")

            tpu = None



        if tpu:

            try:

                print("Initializing TPU")

                tf.config.experimental_connect_to_cluster(tpu)

                tf.tpu.experimental.initialize_tpu_system(tpu)

                strategy = tf.distribute.experimental.TPUStrategy(tpu)

                print("TPU initialized")

            except :

                print("Failed to initialize TPU")

                strategy = tf.distribute.get_strategy()

        else:

            print("Unable to initialize TPU")

            ACCELERATOR = "GPU"



    # default for CPU and GPU

    if ACCELERATOR != "TPU":

        print("Using default strategy for CPU and single GPU")

        strategy = tf.distribute.get_strategy()



    # checking GPUs

    if ACCELERATOR == "GPU":

        print(f"GPUs Available: {len(tf.config.experimental.list_physical_devices('GPU'))}")



    # defining replicas

    AUTO = tf.data.experimental.AUTOTUNE

    REPLICAS = strategy.num_replicas_in_sync

    print(f"REPLICAS: {REPLICAS}")

    

    return strategy, AUTO, REPLICAS



STRATEGY, AUTO, REPLICAS =  initialize_accelerator('TPU')
### Function to do experiment

from sklearn.model_selection import StratifiedKFold

import tensorflow.keras.backend as K

import gc



def run_experiments(df, var_stratified, encode_text) :



    # Stratified K-fold

    skf = StratifiedKFold(n_splits = CV_SPLIT, shuffle = True, random_state = SEED)



    # Initializing predictions

    acc_oof = []



    # Iterating over folds

    for (fold, (train_index, valid_index)) in enumerate(skf.split(df, df[var_stratified])):

        

        # Initialize Accelerator

        STRATEGY, AUTO, REPLICAS =  initialize_accelerator('TPU')

        

        # Building model

        K.clear_session()

        with STRATEGY.scope():

            model = build_model(MODEL_NAME, MAX_LENGTH, METRICS)

            if fold == 0:

                print(model.summary())



        print("\n")

        print("#" * 19)

        print(f"##### Fold: {fold + 1} #####")

        print("#" * 19)



        # Splitting data into training and validation

        X_train = df.iloc[train_index].sample(frac=1)

        X_valid = df.iloc[valid_index]

        

        from tensorflow.keras.utils import to_categorical

        y_train = to_categorical(X_train['label'].values)

        y_valid = to_categorical(X_valid['label'].values)



        print("\nTokenizing")



        # Encoding text data using tokenizer

        X_train_encoded = encode_text(texts = X_train, tokenizer = TOKENIZER, maxlen = MAX_LENGTH, padding = PADDING)

        X_valid_encoded = encode_text(texts = X_valid, tokenizer = TOKENIZER, maxlen = MAX_LENGTH, padding = PADDING)

        

        # Creating TF Dataset

        ds_train = (

                            tf.data.Dataset

                            .from_tensor_slices((X_train_encoded, y_train))

                            .repeat()

                            .shuffle(SEED)

                            .batch(BATCH_SIZE)

                            .cache()

                            .prefetch(AUTO)

                            )



        ds_valid = (

                            tf.data.Dataset

                            .from_tensor_slices((X_valid_encoded, y_valid))

                            .batch(BATCH_SIZE)

                            .cache()

                            .prefetch(AUTO)

                            )

        

        n_train = X_train.shape[0]



        # Saving model at best accuracy epoch

        sv = tf.keras.callbacks.ModelCheckpoint(

            "model.h5",

            monitor = 'val_'+METRICS[0],

            verbose = 0,

            save_best_only = True,

            save_weights_only = True,

            mode = "max",

            save_freq = "epoch"

        )



        print("\nTraining")



        # Training model

        history = model.fit(

            ds_train,

            epochs = EPOCHS,

            callbacks = [sv],

            steps_per_epoch = n_train // BATCH_SIZE,

            validation_data = ds_valid,

            verbose = VERBOSE

        )

        

        

        # Validation

        model.load_weights("model.h5")

        

        from sklearn.metrics import accuracy_score

        pred = model.predict(ds_valid)

        acc = accuracy_score(X_valid['label'].values, np.argmax(pred, axis=1))

        acc_oof.append(acc)



        print(f"\nFold {fold + 1} Accuracy: {round(acc, 4)}\n")



        g = gc.collect()



    # overall CV score and standard deviation

    print(f"\nCV Mean Accuracy: {round(np.mean(acc_oof), 4)}")

    print(f"CV StdDev Accuracy: {round(np.std(acc_oof), 4)}\n")
### Function to build model

import tensorflow as tf

from tensorflow.keras.layers import Dense, Input, Dropout

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import ModelCheckpoint

from transformers import TFAutoModel



def build_model(model_name, max_len, metrics):



    # Defining encoded inputs

    input_ids = Input(shape = (max_len,), dtype = tf.int32, name = "input_ids")

    

    # Defining transformer model embeddings

    transformer_model = TFAutoModel.from_pretrained(model_name)

    transformer_embeddings = transformer_model(input_ids)[0]

    transformer_token = transformer_embeddings[:, 0, :]

    

    # Defining output layer

    output_values = Dense(3, activation = "softmax")(transformer_token)



    # defining model

    model = Model(inputs = input_ids, outputs = output_values)



    model.compile(optimizer = Adam(learning_rate = 1e-5), 

                  loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.2), 

                  metrics = metrics)



    return model
### Make the long sentence to be predictor

df_train['predictor'] = " [CLS] " + df_train['premise'] + " [SEP] " + df_train['hypothesis']

df_test['predictor'] = " [CLS] " + df_test['premise'] + " [SEP] " + df_test['hypothesis']
### Experiment configuration

# Note that all this parameter are being used in the run_experiments function

# Think of this as a global parameter for the function (cause I'm lazy to code it as function parameter)

MAX_LENGTH = 75

MODEL_NAME = "distilbert-base-multilingual-cased"

PADDING = True

BATCH_SIZE = 16 * REPLICAS

EPOCHS = 10

CV_SPLIT = 5

SEED = 2020

VERBOSE = 1

METRICS = ["categorical_accuracy"]
### Load tokenizer

import transformers

from transformers import AutoTokenizer



TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
### Function to encode predictor

def original_encode(texts, tokenizer, maxlen, padding):

    

    enc_di = tokenizer.batch_encode_plus(

             texts['predictor'], 

             return_attention_masks=False, 

             return_token_type_ids=False,

             pad_to_max_length=padding,

             max_length=maxlen)

    

    return enc_di["input_ids"]
### Start experiment

import time

start = time.time()



run_experiments(df_train, 'language_label', original_encode)



end = time.time()

print('Time used :',(end-start)/60)
### Make the long sentence to be predictor

df_train_en['predictor'] = " [CLS] " + df_train_en['premise'] + " [SEP] " + df_train_en['hypothesis']

df_test_en['predictor'] = " [CLS] " + df_test_en['premise'] + " [SEP] " + df_test_en['hypothesis']
### Experiment configuration

# Note that all this parameter are being used in the run_experiments function

# Think of this as a global parameter for the function (cause I'm lazy to code it as function parameter)

MAX_LENGTH = 75

MODEL_NAME = "distilbert-base-multilingual-cased"

PADDING = True

BATCH_SIZE = 16 * REPLICAS

EPOCHS = 10

CV_SPLIT = 5

SEED = 2020

VERBOSE = 1

METRICS = ["categorical_accuracy"]
### Load tokenizer

import transformers

from transformers import AutoTokenizer



TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
### Function to encode predictor

def original_encode(texts, tokenizer, maxlen, padding):

    

    enc_di = tokenizer.batch_encode_plus(

             texts['predictor'], 

             return_attention_masks=False, 

             return_token_type_ids=False,

             pad_to_max_length=padding,

             max_length=maxlen)

    

    return enc_di["input_ids"]
### Start experiment

import time

start = time.time()



run_experiments(df_train_en, 'label', original_encode)



end = time.time()

print('Time used :',(end-start)/60)
### Make the long sentence to be predictor

df_train_no_stop['predictor'] = " [CLS] " + df_train_no_stop['premise'] + " [SEP] " + df_train_no_stop['hypothesis']

df_test_no_stop['predictor'] = " [CLS] " + df_test_no_stop['premise'] + " [SEP] " + df_test_no_stop['hypothesis']
### Experiment configuration

# Note that all this parameter are being used in the run_experiments function

# Think of this as a global parameter for the function (cause I'm lazy to code it as function parameter)

MAX_LENGTH = 50

MODEL_NAME = "distilbert-base-multilingual-cased"

PADDING = True

BATCH_SIZE = 16 * REPLICAS

EPOCHS = 10

CV_SPLIT = 5

SEED = 2020

VERBOSE = 1

METRICS = ["categorical_accuracy"]
### Function to encode predictor

def original_encode(texts, tokenizer, maxlen, padding):

    

    enc_di = tokenizer.batch_encode_plus(

             texts['predictor'], 

             return_attention_masks=False, 

             return_token_type_ids=False,

             pad_to_max_length=padding,

             max_length=maxlen)

    

    return enc_di["input_ids"]
### Start experiment

import time

start = time.time()



run_experiments(df_train_no_stop, 'label', original_encode)



end = time.time()

print('Time used :',(end-start)/60)
### Function to build model

import tensorflow as tf

from tensorflow.keras.layers import Dense, Input, Dropout, Concatenate

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import ModelCheckpoint

from transformers import TFAutoModel



def build_model_multimodal(model_name, max_len, metrics):



    # Defining encoded inputs

    input_ids_prem = Input(shape = (MAX_LENGTH_PREM,), dtype = tf.int32, name = "input_ids_prem")

    input_ids_hyp = Input(shape = (MAX_LENGTH_HYP,), dtype = tf.int32, name = "input_ids_hyp")

    

    # Defining transformer model embeddings

    transformer_model_prem = TFAutoModel.from_pretrained(model_name)

    transformer_embeddings_prem = transformer_model_prem(input_ids_prem)[0]

    transformer_token_prem = transformer_embeddings_prem[:, 0, :]

    

    transformer_model_hyp = TFAutoModel.from_pretrained(model_name)

    transformer_embeddings_hyp = transformer_model_hyp(input_ids_hyp)[0]

    transformer_token_hyp = transformer_embeddings_hyp[:, 0, :]

    

    # Concat 2 token

    transformer_concat = Concatenate()([transformer_token_prem, transformer_token_hyp])

    

    # Defining output layer

    output_values = Dense(3, activation = "softmax")(transformer_concat)



    # defining model

    model = Model(inputs = [input_ids_prem, input_ids_hyp], outputs = output_values)



    model.compile(optimizer = Adam(learning_rate = 1e-5), 

                  loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.2 ), 

                  metrics = metrics)



    return model
### Function to encode predictor

def original_encode_multimodal(texts, tokenizer, maxlen, padding):

    

    enc_di = tokenizer.batch_encode_plus(

             texts, 

             return_attention_masks=False, 

             return_token_type_ids=False,

             pad_to_max_length=padding,

             max_length=maxlen)

    

    return enc_di["input_ids"]
### Function to do experiment multimodal

from sklearn.model_selection import StratifiedKFold

import tensorflow.keras.backend as K

import gc



def run_experiments_multimodal(df, var_stratified, encode_text) :



    # Stratified K-fold

    skf = StratifiedKFold(n_splits = CV_SPLIT, shuffle = True, random_state = SEED)



    # Initializing predictions

    acc_oof = []



    # Iterating over folds

    for (fold, (train_index, valid_index)) in enumerate(skf.split(df, df[var_stratified])):

        

        # Initialize Accelerator

        STRATEGY, AUTO, REPLICAS =  initialize_accelerator('TPU')

        

        # Building model

        K.clear_session()

        with STRATEGY.scope():

            model = build_model_multimodal(MODEL_NAME, MAX_LENGTH_PREM, METRICS)

            if fold == 0:

                print(model.summary())



        print("\n")

        print("#" * 19)

        print(f"##### Fold: {fold + 1} #####")

        print("#" * 19)



        # Splitting data into training and validation

        X_train = df.iloc[train_index]

        X_valid = df.iloc[valid_index]



        from tensorflow.keras.utils import to_categorical

        y_train = to_categorical(X_train['label'].values)

        y_valid = to_categorical(X_valid['label'].values)



        print("\nTokenizing")



        # Encoding text data using tokenizer

        X_train_encoded_prem = encode_text(texts = X_train['premise'], tokenizer = TOKENIZER, maxlen = MAX_LENGTH_PREM, padding = PADDING)

        X_train_encoded_hyp = encode_text(texts = X_train['hypothesis'], tokenizer = TOKENIZER, maxlen = MAX_LENGTH_HYP, padding = PADDING)

        X_valid_encoded_prem = encode_text(texts = X_valid['premise'], tokenizer = TOKENIZER, maxlen = MAX_LENGTH_PREM, padding = PADDING)

        X_valid_encoded_hyp = encode_text(texts = X_valid['hypothesis'], tokenizer = TOKENIZER, maxlen = MAX_LENGTH_HYP, padding = PADDING)

        

        # Creating TF Dataset

        ds_train = (

                            tf.data.Dataset

                            .from_tensor_slices(({"input_ids_prem": X_train_encoded_prem, "input_ids_hyp": X_train_encoded_hyp}, y_train))

                            .repeat()

                            .shuffle(SEED)

                            .batch(BATCH_SIZE)

                            .cache()

                            .prefetch(AUTO)

                            )



        ds_valid = (

                            tf.data.Dataset

                            .from_tensor_slices(({"input_ids_prem": X_valid_encoded_prem, "input_ids_hyp": X_valid_encoded_hyp}, y_valid))

                            .batch(BATCH_SIZE)

                            .cache()

                            .prefetch(AUTO)

                            )

        

        n_train = X_train.shape[0]



        # Saving model at best accuracy epoch

        sv = tf.keras.callbacks.ModelCheckpoint(

            "model.h5",

            monitor = 'val_'+METRICS[0],

            verbose = 0,

            save_best_only = True,

            save_weights_only = True,

            mode = "max",

            save_freq = "epoch"

        )



        print("\nTraining")



        # Training model

        history = model.fit(

            ds_train,

            epochs = EPOCHS,

            callbacks = [sv],

            steps_per_epoch = n_train // BATCH_SIZE,

            validation_data = ds_valid,

            verbose = VERBOSE

        )

        

        # Validation

        model.load_weights("model.h5")

        

        from sklearn.metrics import accuracy_score

        pred = model.predict(ds_valid)

        acc = accuracy_score(X_valid['label'].values, np.argmax(pred, axis=1))

        acc_oof.append(acc)



        print(f"\nFold {fold + 1} Accuracy: {round(acc, 4)}\n")





        g = gc.collect()



    # overall CV score and standard deviation

    print(f"\nCV Mean Accuracy: {round(np.mean(acc_oof), 4)}")

    print(f"CV StdDev Accuracy: {round(np.std(acc_oof), 4)}\n")
### Experiment configuration

# Note that all this parameter are being used in the run_experiments function

# Think of this as a global parameter for the function (cause I'm lazy to code it as function parameter)

MAX_LENGTH_PREM = 50

MAX_LENGTH_HYP = 25



# You can use different pre trained model for premise and hypothesis

# In this notebook I use XLM-RoBERTa for both

MODEL_NAME = "distilbert-base-multilingual-cased"

PADDING = True

BATCH_SIZE = 16 * REPLICAS

EPOCHS = 10

CV_SPLIT = 5

SEED = 2020

VERBOSE = 1

METRICS = ["categorical_accuracy"]
### Load tokenizer

import transformers

from transformers import AutoTokenizer



TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
### Start experiment

import time

start = time.time()



run_experiments_multimodal(df_train_en, 'label', original_encode_multimodal)



end = time.time()

print('Time used :',(end-start)/60)
### Experiment configuration

# Note that all this parameter are being used in the run_experiments function

# Think of this as a global parameter for the function (cause I'm lazy to code it as function parameter)

MAX_LENGTH = 75

MODEL_NAME = "jplu/tf-xlm-roberta-large"

PADDING = True

BATCH_SIZE = 16 * REPLICAS

EPOCHS = 10

CV_SPLIT = 5

SEED = 2020

VERBOSE = 1

METRICS = ["categorical_accuracy"]
### Load tokenizer

import transformers

from transformers import AutoTokenizer



TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
### Encode the dataset

from tensorflow.keras.utils import to_categorical

y_train = to_categorical(df_train_en['label'].values)



# Encoding text data using tokenizer

X_train_encoded = original_encode(texts = df_train_en, tokenizer = TOKENIZER, maxlen = MAX_LENGTH, padding = PADDING)

X_test_encoded = original_encode(texts = df_test_en, tokenizer = TOKENIZER, maxlen = MAX_LENGTH, padding = PADDING)
### Make TF dataset

ds_train = (

                    tf.data.Dataset

                    .from_tensor_slices((X_train_encoded, y_train))

                    .repeat()

                    .shuffle(SEED)

                    .batch(BATCH_SIZE)

                    .cache()

                    .prefetch(AUTO)

                    )



ds_test = (

                    tf.data.Dataset

                    .from_tensor_slices((X_test_encoded))

                    .batch(BATCH_SIZE)

                    .prefetch(AUTO)

                    )
# Building model

K.clear_session()

with STRATEGY.scope():

    model = build_model(MODEL_NAME, MAX_LENGTH, METRICS)
### Train model

n_train = df_train_en.shape[0]



history = model.fit(

    ds_train,

    epochs = EPOCHS,

    steps_per_epoch = n_train // BATCH_SIZE,

    verbose = VERBOSE

)
### Make prediction

pred = model.predict(ds_test)
### Make submission

sub = pd.read_csv('../input/contradictory-my-dear-watson/sample_submission.csv')

sub['prediction'] = np.argmax(pred, axis=1)



sub.to_csv('submission.csv', index=False)