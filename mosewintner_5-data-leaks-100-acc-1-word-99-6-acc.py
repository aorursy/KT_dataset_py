import numpy as np

import pandas as pd

import os

import tensorflow as tf



import gc

import random

import transformers

import warnings



import tensorflow.keras.backend as K



from pathlib import Path

from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, fbeta_score

from sklearn.model_selection import KFold, train_test_split

from tensorflow.keras import Model

from tensorflow.keras.layers import Input, Dense

from tensorflow.keras.losses import BinaryCrossentropy

from tensorflow.keras.optimizers import Adam

from transformers import AutoTokenizer, TFAutoModel



# print(f"TensorFlow version: {tf.__version__}")

# print(f"Transformers version: {transformers.__version__}")



warnings.filterwarnings("ignore")



from matplotlib import pyplot as plt

import seaborn as sns
def print_cm(cm, labels, counts=True, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):

    # https://gist.github.com/zachguo/10296432

    """pretty print for confusion matrixes"""

    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length

    empty_cell = " " * columnwidth

    

    fst_empty_cell = (columnwidth-3)//2 * " " + "T\P" + (columnwidth-3)//2 * " "

    

    if len(fst_empty_cell) < len(empty_cell):

        fst_empty_cell = " " * (len(empty_cell) - len(fst_empty_cell)) + fst_empty_cell

    # Print header

    print("    " + fst_empty_cell, end=" ")

    

    for label in labels:

        print("%{0}s".format(columnwidth) % label, end=" ")

        

    print()

    # Print rows

    for i, label1 in enumerate(labels):

        print("    %{0}s".format(columnwidth) % label1, end=" ")

        for j in range(len(labels)):

            if counts:

                cell = "%{0}.1d".format(columnwidth) % cm[i, j]

            else:

                cell = "%{0}.1f".format(columnwidth) % cm[i, j]

            if hide_zeroes:

                cell = cell if float(cm[i, j]) != 0 else empty_cell

            if hide_diagonal:

                cell = cell if i != j else empty_cell

            if hide_threshold:

                cell = cell if cm[i, j] > hide_threshold else empty_cell

            print(cell, end=" ")

        print()
fake = pd.read_csv('/kaggle/input/fake-and-real-news-dataset/Fake.csv')

true = pd.read_csv('/kaggle/input/fake-and-real-news-dataset/True.csv')

true = true.assign(fake = 0)

fake = fake.assign(fake = 1)

full = pd.concat([true,fake],axis=0)
true.head()
fake.head()
true.text.str.contains("\(Reuters\)").mean(), fake.text.str.contains("\(Reuters\)").mean()
non_reuters_tag = ~full.text.str.contains("\(Reuters\)")

labels = ['True','Fake']

print_cm(confusion_matrix(full['fake'],non_reuters_tag),labels=labels,counts=True)
accuracy_score(full['fake'],non_reuters_tag)
def drop_prefix(text,prefix='(Reuters)',n=5):

    ts = str.split(text,' ')

    if prefix in ts[:n]:

        return str.split(text,prefix)[-1]

    else:

        return text
full = full.assign(text = full.text.apply(lambda x: drop_prefix(x,'(Reuters)')))

full = full.assign(text = full.text.str.strip(' -'))
true.subject.value_counts()
fake.subject.value_counts()
true.date.value_counts().sort_index()
fake.date.value_counts().sort_index().tail(15)
tct = pd.to_datetime(true.date,errors='coerce').value_counts().sort_index()

fct = pd.to_datetime(fake.date,errors='coerce').value_counts().sort_index()



plt.plot(tct.index,tct.values,label='True',alpha=0.4)

plt.plot(fct.index,fct.values,'red',label='Fake',alpha=0.4)

plt.xticks(rotation=45)

plt.ylabel("Document count")

plt.legend(loc='upper left')

plt.title("Documents published per day")
full.shape[0] - full.drop_duplicates().shape[0]
true = true.drop(columns=['date','subject'])

fake = fake.drop(columns=['date','subject'])

full = full.drop(columns=['date','subject'])
full.shape[0], full.shape[0] - full.drop_duplicates().shape[0], 44898/5
(full.shape[0] - full.drop_duplicates().shape[0])/full.drop_duplicates().shape[0]
def first_n_words_of_text(df,n=2):

    word_list_series = df.text.str.strip(' -').str.split(' ').apply(lambda x: x[:n]).astype(str)

    return word_list_series
first_2_words = first_n_words_of_text(full)

first_2_words.value_counts().head(20)
def prefix_classifier_performance(df,prefix):

    # Return the fraction of documents in each class which begin with prefix

    n = len(prefix)

    return df.groupby('fake')['text'].agg(lambda x: x.apply(lambda y: y[:n] == prefix).mean())
prefix_classifier_performance(full, 'Donald Trump')
prefix_classifier_performance(full, '21st Century Wire')
full = full.assign(text = full.text.apply(lambda x: drop_prefix(x,'21st Century Wire')))
full = full.assign(ncw = full.title.str.split(' ').apply(lambda x: np.sum([y==str.upper(y) for y in x])))



plt.hist(full.loc[full.fake.astype(bool),'ncw'],color='red',label='fake',alpha=0.4)

plt.hist(full.loc[~full.fake.astype(bool),'ncw'],color='blue',label='true',alpha=0.4)

plt.xlabel("Number of fully capitalized words in title")

_=plt.legend()
# Average class label by number of capitalized words in title.

full.groupby('ncw')['fake'].mean()
yhat_capital_classifier = full.ncw.ge(2)

(yhat_capital_classifier == full['fake']).mean()
## defining configuration



class Configuration():

    """

    All configuration for running an experiment

    """

    def __init__(

        self,

        model_name,

        train,

        test,

        max_length = 64,

        padding = True,

        batch_size = 128,

        epochs = 5,

        learning_rate = 1e-5,

        metrics = ["binary_accuracy"],

        verbose = 1,

        train_splits = 4,

        accelerator = "TPU",

        target_col = 'fake',

        seed = 13

    ):

        # seed and accelerator

        self.SEED = seed

        self.ACCELERATOR = accelerator



        self.TRAIN = train

        self.TEST = test

        self.TARGET_COL = target_col

        

        # splits

        self.TRAIN_SPLITS = train_splits

        

        # model configuration

        self.MODEL_NAME = model_name

        self.TOKENIZER = AutoTokenizer.from_pretrained(self.MODEL_NAME)



        # model hyperparameters

        self.MAX_LENGTH = max_length

        self.PAD_TO_MAX_LENGTH = padding

        self.BATCH_SIZE = batch_size

        self.EPOCHS = epochs

        self.LEARNING_RATE = learning_rate

        self.METRICS = metrics

        self.VERBOSE = verbose

        

        # initializing accelerator

        self.initialize_accelerator()



    def initialize_accelerator(self):

        """

        Initializing accelerator

        """

        # checking TPU first

        if self.ACCELERATOR == "TPU":

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

                    self.strategy = tf.distribute.experimental.TPUStrategy(tpu)

                    self.tpu = tpu

                    print("TPU initialized")

                except _:

                    print("Failed to initialize TPU")

            else:

                print("Unable to initialize TPU")

                self.ACCELERATOR = "GPU"



        # default for CPU and GPU

        if self.ACCELERATOR != "TPU":

            print("Using default strategy for CPU and single GPU")

            self.strategy = tf.distribute.get_strategy()



        # checking GPUs

        if self.ACCELERATOR == "GPU":

            print(f"GPUs Available: {len(tf.config.experimental.list_physical_devices('GPU'))}")



        # defining replicas

        self.AUTO = tf.data.experimental.AUTOTUNE

        self.REPLICAS = self.strategy.num_replicas_in_sync

        print(f"REPLICAS: {self.REPLICAS}")

        

        

        

def encode_text(df, tokenizer, max_len, padding):

    """

    Preprocessing textual data into encoded tokens.

    """

    text = df[["title", "text"]].values.tolist()



    # encoding text using tokenizer of the model

    text_encoded = tokenizer.batch_encode_plus(

        text,

        pad_to_max_length = padding,

        max_length = max_len

    )



    return text_encoded





def get_tf_dataset(X, y, auto, labelled = True, repeat = False, shuffle = False, batch_size = 128):

    """

    Creating tf.data.Dataset for TPU.

    """

    if labelled:

        ds = (tf.data.Dataset.from_tensor_slices((X["input_ids"], y)))

    else:

        ds = (tf.data.Dataset.from_tensor_slices(X["input_ids"]))



    if repeat:

        ds = ds.repeat()



    if shuffle:

        ds = ds.shuffle(2048)



    ds = ds.batch(batch_size)

    ds = ds.prefetch(auto)



    return ds







## building model

def build_model(model_name, max_len, learning_rate, metrics):

    """

    Building the Deep Learning architecture

    """

    # defining encoded inputs

    input_ids = Input(shape = (max_len,), dtype = tf.int32, name = "input_ids")

    

    # defining transformer model embeddings

    transformer_model = TFAutoModel.from_pretrained(model_name)

    transformer_embeddings = transformer_model(input_ids)[0]



    # defining output layer

    output_values = Dense(1, activation = "sigmoid")(transformer_embeddings[:, 0, :])



    # defining model

    model = Model(inputs = input_ids, outputs = output_values)

    opt = Adam(learning_rate = learning_rate)

    loss = tf.keras.losses.BinaryCrossentropy(from_logits = True)

    metrics = metrics



    model.compile(optimizer = opt, loss = loss, metrics = metrics)



    return model



def run_model(config):

    """

    Running the model

    """

    ## reading data

    df_train = config.TRAIN

    df_test = config.TEST

    

    # stratified K-fold on language and label

    skf = KFold(n_splits = config.TRAIN_SPLITS, shuffle = True, random_state = config.SEED)



    # initializing predictions

    preds_oof = np.zeros((df_train.shape[0], 1))

    preds_test = np.zeros((df_test.shape[0], 1))

    acc_oof = []



    # iterating over folds

    for (fold, (train_index, valid_index)) in enumerate(skf.split(df_train)):

        # initializing TPU

        if config.ACCELERATOR == "TPU":

            if config.tpu:

                config.initialize_accelerator()



        # building model

        K.clear_session()

        with config.strategy.scope():

            model = build_model(config.MODEL_NAME, config.MAX_LENGTH, config.LEARNING_RATE, config.METRICS)

            if fold == 0:

                print(model.summary())



        print("\n")

        print("#" * 19)

        print(f"##### Fold: {fold + 1} #####")

        print("#" * 19)



        # splitting data into training and validation

        X_train = df_train.iloc[train_index]

        X_valid = df_train.iloc[valid_index]



        y_train = X_train[config.TARGET_COL].values

        y_valid = X_valid[config.TARGET_COL].values

        

        print("\nTokenizing")



        # encoding text data using tokenizer

        X_train_encoded = encode_text(df = X_train, tokenizer = config.TOKENIZER, max_len = config.MAX_LENGTH, padding = config.PAD_TO_MAX_LENGTH)

        X_valid_encoded = encode_text(df = X_valid, tokenizer = config.TOKENIZER, max_len = config.MAX_LENGTH, padding = config.PAD_TO_MAX_LENGTH)



        # creating TF Dataset

        ds_train = get_tf_dataset(X_train_encoded, y_train, config.AUTO, repeat = True, shuffle = True, batch_size = config.BATCH_SIZE * config.REPLICAS)

        ds_valid = get_tf_dataset(X_valid_encoded, y_valid, config.AUTO, batch_size = config.BATCH_SIZE * config.REPLICAS * 4)



        n_train = X_train.shape[0]



        if fold == 0:

            X_test_encoded = encode_text(df = df_test, tokenizer = config.TOKENIZER, max_len = config.MAX_LENGTH, padding = config.PAD_TO_MAX_LENGTH)



        # saving model at best accuracy epoch

        sv = tf.keras.callbacks.ModelCheckpoint(

            "model.h5",

            monitor = "binary_accuracy",

            verbose = 0,

            save_best_only = True,

            save_weights_only = True,

            mode = "max",

            save_freq = "epoch"

        )



        print("\nTraining")



        # training model

        model_history = model.fit(

            ds_train,

            epochs = config.EPOCHS,

            callbacks = [sv],

            steps_per_epoch = n_train / config.BATCH_SIZE // config.REPLICAS,

            validation_data = ds_valid,

            verbose = config.VERBOSE

        )



        print("\nValidating")



        # scoring validation data

        model.load_weights("model.h5")

        ds_valid = get_tf_dataset(X_valid_encoded, -1, config.AUTO, labelled = False, batch_size = config.BATCH_SIZE * config.REPLICAS * 4)



        preds_valid = model.predict(ds_valid, verbose = config.VERBOSE)

        acc = accuracy_score(y_valid, np.argmax(preds_valid, axis = 1))



        preds_oof[valid_index] = preds_valid

        acc_oof.append(acc)



        print("\nInferencing")



        # scoring test data

        ds_test = get_tf_dataset(X_test_encoded, -1, config.AUTO, labelled = False, batch_size = config.BATCH_SIZE * config.REPLICAS * 4)

        preds_test += model.predict(ds_test, verbose = config.VERBOSE) / config.TRAIN_SPLITS



        print(f"\nFold {fold + 1} Accuracy: {round(acc, 4)}\n")



        g = gc.collect()



    # overall CV score and standard deviation

    print(f"\nCV Mean Accuracy: {round(np.mean(acc_oof), 4)}")

    print(f"CV StdDev Accuracy: {round(np.std(acc_oof), 4)}\n")



    return preds_oof, preds_test
TEST_FRAC = 0.2



train, test = train_test_split(full,random_state=32094,test_size=TEST_FRAC)
# # Model: Bert Base Cased

# config_1 = Configuration("bert-base-cased", train=train, test=test, max_length = 32, batch_size = 8, epochs = 5, train_splits = 4)

# preds_train_1, preds_test_1 = run_model(config_1)



# # Model: Bert Base Uncased

# config_2 = Configuration("bert-base-uncased", max_length = 32, batch_size = 32, epochs = 2, train_splits = 4)

# preds_train_2, preds_test_2 = run_model(config_2)



# Model: Bert Large Cased

#config_3 = Configuration("bert-large-cased", max_length = 32, batch_size = 32, epochs = 2, train_splits = 4)

#preds_train_3, preds_test_3 = run_model(config_3)



# Model: Bert Large Uncased

#config_4 = Configuration("bert-large-uncased", max_length = 32, batch_size = 32, epochs = 2, train_splits = 4)

#preds_train_4, preds_test_4 = run_model(config_4)



# Model: Bert Multilingual Base Cased

#config_5 = Configuration("bert-base-multilingual-cased", translation = False, max_length = 32, batch_size = 32, epochs = 2, train_splits = 4)

#preds_train_5, preds_test_5 = run_model(config_5)



# Model: Distilbert Base Cased

#config_6 = Configuration("distilbert-base-cased", max_length = 32, batch_size = 32, epochs = 2, train_splits = 4)

#preds_train_6, preds_test_6 = run_model(config_6)



# Model: Distilbert Base Uncased

#config_7 = Configuration("distilbert-base-uncased", max_length = 32, batch_size = 32, epochs = 2, train_splits = 4)

#preds_train_7, preds_test_7 = run_model(config_7)



# Model: Distilbert Multilingual Base Cased

#config_8 = Configuration("distilbert-base-multilingual-cased", translation = False, max_length = 32, batch_size = 32, epochs = 2, train_splits = 4)

#preds_train_8, preds_test_8 = run_model(config_8)



# Model: Roberta Base

#config_9 = Configuration("roberta-base", max_length = 32, batch_size = 32, epochs = 2, train_splits = 4)

#preds_train_9, preds_test_9 = run_model(config_9)



# Model: Roberta Large

#config_10 = Configuration("roberta-large", max_length = 32, batch_size = 32, epochs = 2, train_splits = 4)

#preds_train_10, preds_test_10 = run_model(config_10)



# Model: XLM Roberta Base

#config_11 = Configuration("jplu/tf-xlm-roberta-base", max_length = 32, batch_size = 32, epochs = 2, train_splits = 4)

#preds_train_11, preds_test_11 = run_model(config_11)



# Model: XLM Roberta Large

#config_12 = Configuration("jplu/tf-xlm-roberta-large", translation = False, max_length = 32, batch_size = 32, epochs = 2, train_splits = 4)

#preds_train_12, preds_test_12 = run_model(config_12)