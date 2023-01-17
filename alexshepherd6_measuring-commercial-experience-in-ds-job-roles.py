import numpy as np

import tensorflow as tf

from tensorflow import keras as ks

from tensorflow.estimator import LinearRegressor

from sklearn import datasets

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error, r2_score

print(tf.__version__)

import pandas as pd

import seaborn as sns

import os



%matplotlib inline

import matplotlib.pyplot as plt



plt.rcParams['figure.figsize'] = (7, 6)

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']



from tensorflow import feature_column

from tensorflow.keras import layers

from tensorflow.keras import regularizers



from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

import re

import string



import os



os.chdir("/kaggle/input/linkedin-uk-entrylevel-data-scientist-roles-2020/DATA/")
# Agreed labels

df = pd.read_csv("sentence_labels_v3.csv", usecols = ["Text", "none", "bt", "tt"])

df = df.drop_duplicates("Text")

df.head()

print("Number of sentences: {}".format(len(df)))
punctuation = list(string.punctuation)

punctuation.extend(["\xa0", "\n"])

punctuation.extend(stopwords.words('english'))

        



def clean_text(txt):

        """

        Helper method to clean text of punctuation, contractions and adds sentence beginning markers at beginning of sentences.

        Also removes hashtags and mentions.



        :param txt: String raw text



        :return c_txt: Nested list of sentences containing list of tokens which form the cleaned the sentences.

        """



        #print("Cleaning text...")

        # Step 1 tokenize by sentence

        c_txt = txt.lower()



        # Step 2 remove punctuation

        for punc in punctuation:

            try:

                #code.interact(local = locals(), banner = "@, # removal")

                if punc == "@":

                    c_txt = re.sub(r"{0}\w+".format(punc), " ", c_txt)

                if punc in c_txt:

                    if punc in stopwords.words('english'):

                        c_txt = re.sub(r" {0} ".format(punc), " ", c_txt)

                    else:

                        try:

                            if punc != ".":

                                c_txt = re.sub(r"{0}".format(punc), " ", c_txt)

                            else:

                                c_txt = re.sub(r"\{0}".format(punc), " ", c_txt)

                        except:

                            c_txt = re.sub(r"\{0}".format(punc), "", c_txt)

                else:

                    pass

            except Exception as e:

                pass

            

        # Remove non-ASCII characters

        if c_txt.isascii():

            pass

        else:

            txt = c_txt

            c_txt = ""

            for char in txt:

                if char.isascii():

                    c_txt = c_txt + char

                else:

                    pass

                

        # Replace digits with DIGIT flag

        c_txt = re.sub(r"\d+".format(punc), " DIGIT ", c_txt)

        

                

        return c_txt
df.loc[:,"Clean Text"] = df["Text"].apply(clean_text)
# class distribution

df.loc[:, ["none", "bt", "tt"]].sum()



# converting one hot encoding to indices to represent the classes.

# [none, bt, tt] == [0, 1, 2]

df.loc[:, "Class"] = df.loc[:,"bt"]

df.loc[:, "Class"] += 2 * df.loc[:,"tt"] 



sns.countplot(df["Class"])

plt.title("""Class distribution over LinkedIn UK Entry-level

Data Scientist Job Listing dataset \n (n = {})""".format(len(df)))

plt.xticks(np.arange(3), ["none", "bt", "tt"])

plt.ylabel("Frequency")

plt.show()
"""

The unknown tokens were identified by the following:



- Tokens which have a word count of 1 in the sentences dataset.

- Tokens which appear lower than 25th percentile = 3 of those word frequency over overall job listing dataset.



"""

unk_words = pd.read_csv("UNK_flag.csv", index_col = 0)

# Applied 50th percentile as UNK flag

unk_words = set(list(unk_words["word"].values))
# vocab size



vocab = []



# converting unknown flagged words with unknown flag UNK

for i in df["Clean Text"].values:

    for w in unk_words:

        i = re.sub(" {} ".format(w), " UNK ", i)

    vocab.extend(word_tokenize(i))

    vocab = list(set(vocab))

    

vocab = sorted(vocab)



print("""Vocabulary: {} 

Unknown Words: {}""".format(len(unk_words), len(vocab)))
def encode_idxs(txt):

    """

    Converts sentence of words into a sequence of indices as

    per the vocab list.

    

    :param txt: (String) input sentence

    

    :return idxs: (List of Integers) sequence of indices

    """

    idxs = []

    for word in word_tokenize(txt):

        if word in unk_words:

            idxs.append(vocab.index("UNK"))

        else:

            idxs.append(vocab.index(word))

        

    return idxs
seq = [encode_idxs(txt) for txt in df["Clean Text"]] 

X = tf.keras.preprocessing.sequence.pad_sequences(seq)

y = tf.keras.utils.to_categorical(df["Class"], df["Class"].unique().size)



data = np.hstack((X, y))



print(data.shape)
train, test = train_test_split(data, test_size=0.2)

train, val = train_test_split(train, test_size=0.2)

print(len(train), 'train examples')

print(len(val), 'validation examples')

print(len(test), 'test examples')



batch_size = 10



def prepare_dataset(d, classes):

    

    X = d[:, :-classes]

    y = d[:, -classes:]

    

    return X, y



def prepare_all(datasets, classes):

    """

    Prepare all datasets for multi-class classification.

    """

    dfs = ()

    for d in datasets:

        dfs = dfs + prepare_dataset(d, classes)

        

    return dfs
all_data = prepare_all([train, val, test], classes = 3)



X_train, y_train, X_val, y_val, X_test, y_test = all_data
print("""Examples:



Training Set : {}

Validation Set : {}

Test Set : {}""".format(X_train.shape[0], X_val.shape[0], X_test.shape[0]))
# Hyperparameters



embed_dim = 64

lstm_output = 15

output_dim = df["Class"].unique().size
baseline = tf.keras.Sequential([

    layers.Embedding(input_dim = len(vocab), output_dim = embed_dim),

    layers.LSTM(lstm_output),

    layers.Dense(output_dim, activation = "softmax")

], name = "baseline")



METRICS = [ 

      tf.keras.metrics.Precision(name='precision'),

      tf.keras.metrics.Recall(name='recall'),

      tf.keras.metrics.AUC(name='auc'),

]



baseline.compile(optimizer='adam',

              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),

              metrics=METRICS)





baseline.summary()
bs_history = baseline.fit(X_train, y_train, 

                       epochs=20,

                       validation_data=(X_val, y_val)) 
def plot_loss(history, label, n):

    # Use a log scale to show the wide range of values.

    plt.semilogy(history.epoch,

                 history.history['loss'],

                 color=colors[n],

                 label='Train '+label)

    plt.semilogy(history.epoch,

                 history.history['val_loss'],

                 color=colors[n],

                 label='Val '+label,

                 linestyle="--")

    plt.xlabel('Epoch')

    plt.ylabel('Loss')



    plt.legend()
baseline.evaluate(X_test, y_test)
def cm(model, classes = 3, datasets = None):



    f, (ax1, ax2) = plt.subplots(1,2, sharex = True, figsize = (15, 7))

    

    if datasets is None:

        datasets = [(X_val, y_val), (X_test, y_test)]

    

    for a, ds, t in zip([ax1, ax2], datasets, ["validation", "test"]):

        prediction = np.argmax(model.predict(ds[0]), axis = 1)

        tst = np.vstack((np.argmax(ds[1], axis = 1), prediction)).T



        cm = np.zeros((classes,classes))



        for i in tst:

            cm[i[0], i[1]] += 1



        sns.heatmap(cm/ np.atleast_2d(np.sum(cm, axis = 1)).T, annot = True, ax = a)

        a.set_title("Confusion Matrix over {} set".format(t))

        a.set_ylabel("Actual")

        a.set_xlabel("Predicted")
cm(baseline)

def balance_classes(data):

    d = pd.DataFrame(data["Class"].value_counts())

    max_class = d.idxmax(axis = 0).values[0]

    max_class_count = d.loc[max_class][0]



    new_data = data[data["Class"] == max_class]



    for c in list(set(data["Class"].unique()) - set([max_class])):

        c_idxs = data[data["Class"] == c].index.values

        c_idxs = np.random.choice(c_idxs, max_class_count)

        new_data = pd.concat([new_data, df.loc[c_idxs,:]], ignore_index = True)

        

    return new_data
new_data = balance_classes(df)



seq = [encode_idxs(txt) for txt in new_data["Clean Text"]] 

X = tf.keras.preprocessing.sequence.pad_sequences(seq)

y = tf.keras.utils.to_categorical(new_data["Class"], new_data["Class"].unique().size)



data = np.hstack((X, y))

print(data.shape)



train, test = train_test_split(data, test_size=0.2)

train, val = train_test_split(train, test_size=0.2)

print(len(train), 'train examples')

print(len(val), 'validation examples')

print(len(test), 'test examples')



all_data = prepare_all([train, val, test], 3)



X_train, y_train, X_val, y_val, X_test, y_test = all_data
# Look at distribution of UNK tag in data by class



UNK_count = np.where(X == vocab.index("UNK"), 1, 0)

UNK_count = UNK_count.sum(axis = 1)



UNK_count = pd.DataFrame(np.vstack((UNK_count, new_data["Class"].values))).T

UNK_count.columns = ["UNK", "Class"]



plt.figure(figsize = (5,6))

plt.title("Distribution of UNK tags by class")

sns.boxplot(x = "Class", y = "UNK", data = UNK_count)

plt.show()
base_balance = tf.keras.Sequential([

    layers.Embedding(input_dim = len(vocab), output_dim = embed_dim),

    layers.LSTM(lstm_output),

    layers.Dense(output_dim, activation = "softmax")

], name = "baseline-balanced")



base_balance.compile(optimizer='adam',

              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),

              metrics=METRICS)





base_balance.summary()
bs_balanced_history = base_balance.fit(X_train, y_train, 

                       epochs=20,

                       validation_data=(X_val, y_val))
cm(base_balance)
inc_embed = tf.keras.Sequential([

    layers.Embedding(input_dim = len(vocab), output_dim = embed_dim * 3),

    layers.LSTM(lstm_output),

    layers.Dense(output_dim, activation = "softmax")

], name = "increase-embedding")



inc_embed.compile(optimizer='adam',

              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),

              metrics=METRICS)





inc_embed.summary()
inc_embed_history = inc_embed.fit(X_train, y_train, 

                       epochs=100,

                       validation_data=(X_val, y_val))
cm(inc_embed)
inc_layers = tf.keras.Sequential([

    layers.Embedding(input_dim = len(vocab), output_dim = embed_dim),

    layers.LSTM(lstm_output, return_sequences=True),

    layers.LSTM(lstm_output, return_sequences=True),

    layers.LSTM(lstm_output),

    layers.Dense(output_dim, activation = "softmax")

], name = "increase-layers")



inc_layers.compile(optimizer='adam',

              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),

              metrics=METRICS)





inc_layers.summary()
inc_layers_history = inc_layers.fit(X_train, y_train, 

                       epochs=20,

                       validation_data=(X_val, y_val))
cm(inc_layers)
bi_base = tf.keras.Sequential([

    layers.Embedding(input_dim = len(vocab), output_dim = embed_dim),

    layers.Bidirectional(layers.LSTM(lstm_output)),

    layers.Dense(output_dim, activation = "softmax")

], name = "bidirection-baseline")



bi_base.compile(optimizer='adam',

              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),

              metrics=METRICS)



earlystop_callback = tf.keras.callbacks.EarlyStopping(

  monitor='val_loss', min_delta=0.001, mode = "min",

  patience=20, verbose=1)



bi_base.summary()
bi_base_history = bi_base.fit(X_train, y_train, 

                              # worse performance (recall, precision) beyond 21 epochs

                       epochs=22,

                       validation_data=(X_val, y_val),

                              callbacks=[earlystop_callback])
cm(bi_base)
bi_inc_embed = tf.keras.Sequential([

    layers.Embedding(input_dim = len(vocab), output_dim = embed_dim * 3),

    layers.Bidirectional(layers.LSTM(lstm_output)),

    layers.Dense(output_dim, activation = "softmax")

], name = "bidirection-inc-embed")



bi_inc_embed.compile(optimizer='adam',

              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),

              metrics=METRICS)



bi_inc_embed.summary()
bi_inc_embed_history = bi_inc_embed.fit(X_train, y_train, 

                                        # 23 epochs was chosen as the optimal number of epochs

                       epochs=23,

                       validation_data=(X_val, y_val))
cm(bi_inc_embed)
bi_inc_layers = tf.keras.Sequential([

    layers.Embedding(input_dim = len(vocab), output_dim = embed_dim),

    layers.Bidirectional(layers.LSTM(lstm_output, return_sequences = True)),

    layers.Bidirectional(layers.LSTM(lstm_output)),

    layers.Dense(output_dim, activation = "softmax")

], name = "bidirection-inc-layers")



bi_inc_layers.compile(optimizer='adam',

              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),

              metrics=METRICS)





bi_inc_layers.summary()
bi_inc_layers_history = bi_inc_layers.fit(X_train, y_train, 

                       epochs=20,

                       validation_data=(X_val, y_val))
cm(bi_inc_layers)
plot_loss(bs_history, "Baseline", 0)

plot_loss(inc_embed_history, "Inc Embed", 1)

plot_loss(inc_layers_history, "Inc Layers", 2)

plot_loss(bi_base_history, "Bi Baseline", 3)

plot_loss(bi_inc_embed_history, "Bi Inc Embed", 4)

plot_loss(bs_balanced_history, "Baseline Balanced", 1)
df.loc[:, "bt/tt"] = (df["Class"] > 0).astype(int)



none_vs_other = df.loc[:,["Clean Text", "bt/tt"]]

bt_vs_tt = df[df["none"] == 0].loc[:,["Clean Text", "tt"]]
none_vs_other = balance_classes(df)



seq = [encode_idxs(txt) for txt in none_vs_other["Clean Text"]] 

X = tf.keras.preprocessing.sequence.pad_sequences(seq)

y = tf.keras.utils.to_categorical(none_vs_other["bt/tt"], none_vs_other["bt/tt"].unique().size)



data = np.hstack((X, y))

print(data.shape)



train, test = train_test_split(data, test_size=0.2)

train, val = train_test_split(train, test_size=0.2)

print(len(train), 'train examples')

print(len(val), 'validation examples')

print(len(test), 'test examples')



all_data = prepare_all([train, val, test], 2)



nvo_X_train, nvo_y_train, nvo_X_val, nvo_y_val, nvo_X_test, nvo_y_test = all_data



print(nvo_X_train.shape)

print(nvo_y_train.shape)
bt_vs_tt = balance_classes(df)



seq = [encode_idxs(txt) for txt in bt_vs_tt["Clean Text"]] 

X = tf.keras.preprocessing.sequence.pad_sequences(seq)

y = tf.keras.utils.to_categorical(bt_vs_tt["tt"], bt_vs_tt["tt"].unique().size)



data = np.hstack((X, y))

print(data.shape)



train, test = train_test_split(data, test_size=0.2)

train, val = train_test_split(train, test_size=0.2)

print(len(train), 'train examples')

print(len(val), 'validation examples')

print(len(test), 'test examples')



all_data = prepare_all([train, val, test], 2)



bvt_X_train, bvt_y_train, bvt_X_val, bvt_y_val, bvt_X_test, bvt_y_test = all_data
bi_nvo = tf.keras.Sequential([

    layers.Embedding(input_dim = len(vocab), output_dim = embed_dim),

    layers.Bidirectional(layers.LSTM(lstm_output)),

    layers.Dense(2, activation = "softmax")

], name = "bidirectional-none-vs-other")



bi_nvo.compile(optimizer='adam',

              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),

              metrics=METRICS)





bi_nvo.summary()
bi_nvo_embed_history = bi_nvo.fit(nvo_X_train, nvo_y_train, 

                       epochs=20,

                       validation_data=(nvo_X_val, nvo_y_val), 

                       validation_steps=30)
cm(bi_nvo, 2, [(nvo_X_test, nvo_y_test),(nvo_X_val, nvo_y_val)])
bi_bvt = tf.keras.Sequential([

    layers.Embedding(input_dim = len(vocab), output_dim = embed_dim),

    layers.Bidirectional(layers.LSTM(lstm_output)),

    layers.Dense(2, activation = "softmax")

], name = "bidirectional-bt-vs-tt")



bi_bvt.compile(optimizer='adam',

              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),

              metrics=METRICS)





bi_bvt.summary()
bi_bvt_embed_history = bi_bvt.fit(bvt_X_train, bvt_y_train, 

                       epochs=20,

                       validation_data=(bvt_X_val, bvt_y_val), 

                       validation_steps=30)
cm(bi_bvt, 2, [(bvt_X_test, bvt_y_test),(bvt_X_val, bvt_y_val)])
bi_nvo_inc_embed = tf.keras.Sequential([

    layers.Embedding(input_dim = len(vocab), output_dim = embed_dim * 2),

    layers.Bidirectional(layers.LSTM(lstm_output)),

    layers.Dense(2, activation = "softmax")

], name = "bidirectional-none-vs-other-inc-embed")



bi_nvo_inc_embed.compile(optimizer='adam',

              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),

              metrics=METRICS)





bi_nvo_inc_embed.summary()
bi_nvo_inc_embed_history = bi_nvo.fit(nvo_X_train, nvo_y_train, 

                       epochs=20,

                       validation_data=(nvo_X_val, nvo_y_val), 

                       validation_steps=30)
cm(bi_nvo_inc_embed, 2, [(nvo_X_test, nvo_y_test),(nvo_X_val, nvo_y_val)])
bi_bvt_inc_embed = tf.keras.Sequential([

    layers.Embedding(input_dim = len(vocab), output_dim = embed_dim * 2),

    layers.Bidirectional(layers.LSTM(lstm_output)),

    layers.Dense(2, activation = "softmax")

], name = "bidirectional-bt-vs-tt-inc-embed")



bi_bvt_inc_embed.compile(optimizer='adam',

              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),

              metrics=METRICS)





bi_bvt_inc_embed.summary()
bi_bvt_inc_embed_history = bi_bvt_inc_embed.fit(bvt_X_train, bvt_y_train, 

                       epochs=20,

                       validation_data=(bvt_X_val, bvt_y_val), 

                       validation_steps=30)
cm(bi_bvt_inc_embed, 2, [(bvt_X_test, bvt_y_test),(bvt_X_val, bvt_y_val)])