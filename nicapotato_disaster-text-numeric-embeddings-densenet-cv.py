from keras import backend as K

from keras.models import Model

from keras.layers import Input, Flatten, Dense, Embedding, SpatialDropout1D, concatenate, Dropout, BatchNormalization, Activation

from keras.layers import LSTM, Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D

from keras.preprocessing import text, sequence

from keras import optimizers

from gensim.models import KeyedVectors

from nltk.sentiment.vader import SentimentIntensityAnalyzer



from sklearn.model_selection import KFold

from tensorflow.keras import callbacks

from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.preprocessing import StandardScaler

from scipy.sparse import hstack, csr_matrix



import time

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import pprint

import re

import nltk



import seaborn as sns

sns.set_style("whitegrid")

notebookstart = time.time()

pd.options.display.max_colwidth = 500



import keras

print("Keras Version: ",keras.__version__)

import tensorflow

print("Tensorflow Version: ", tensorflow.__version__)



EMBEDDING_FILES = [

    '../input/gensim-embeddings-dataset/crawl-300d-2M.gensim',

    '../input/gensim-embeddings-dataset/glove.840B.300d.gensim'

]



seed = 25



N_ROWS = None

BATCH_SIZE = 64

EPOCHS = 100

N_CLASSES = 1

f1_strategy = 'macro'



MAX_LEN = 34

AUX_COLUMNS = ['target']

TEXT_COLUMN = 'text'

TARGET_COLUMN = 'target'

CHARS_TO_REMOVE = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n“”’\'∞θ÷α•à−β∅³π‘₹´°£€\×™√²—'
def build_matrix(word_index, path):

    embedding_index = KeyedVectors.load(path, mmap='r')

    embedding_matrix = np.zeros((len(word_index) + 1, 300))

    for word, i in word_index.items():

        for candidate in [word, word.lower()]:

            if candidate in embedding_index:

                embedding_matrix[i] = embedding_index[candidate]

                break

    return embedding_matrix



def text_processing(df):

    df['keyword'] = df['keyword'].str.replace("%20", " ")

    df['hashtags'] = df['text'].apply(lambda x: " ".join(re.findall(r"#(\w+)", x)))

    df['hash_loc_key'] = df[['hashtags', 'location','keyword']].astype(str).apply(lambda x: " ".join(x), axis=1)

    df['hash_loc_key'] = df["hash_loc_key"].astype(str).str.lower().str.strip().fillna('nan')

    

    textfeats = ['hash_loc_key', 'text']

    for cols in textfeats:

        df[cols + '_num_words'] = df[cols].apply(lambda comment: len(comment.split())) # Count number of Words

        df[cols + '_num_unique_words'] = df[cols].apply(lambda comment: len(set(w for w in comment.split())))

        df[cols + '_words_vs_unique'] = df[cols+'_num_unique_words'] / df[cols+'_num_words'] * 100 # Count Unique Words

        if cols == "text":

            df[cols+"_vader_Compound"]= df[cols].apply(lambda x:SIA.polarity_scores(x)['compound'])



    return df
print("Read Data")

train_df = pd.read_csv('../input/nlp-getting-started/train.csv', nrows = N_ROWS)

test_df = pd.read_csv('../input/nlp-getting-started/test.csv', nrows = N_ROWS)



X = train_df[TEXT_COLUMN].astype(str)

y = train_df[TARGET_COLUMN].values

test = test_df[TEXT_COLUMN].astype(str)



print("Train Shape: {} Rows".format(X.shape[0]))

print("Test Shape: {} Rows".format(test.shape[0]))

print('Dependent Variable Factor Ratio: ',train_df[TARGET_COLUMN].value_counts(normalize=True).to_dict())



tokenizer = text.Tokenizer(filters=CHARS_TO_REMOVE, lower=False)

tokenizer.fit_on_texts(list(X) + list(test))



X = tokenizer.texts_to_sequences(X)

test = tokenizer.texts_to_sequences(test)



length_info = [len(x) for x in X]

print("Train Sequence Length - Mean {:.1f} +/- {:.1f}, Max {:.1f}, Min {:.1f}".format(

    np.mean(length_info), np.std(length_info), np.max(length_info), np.min(length_info)))



X = sequence.pad_sequences(X, maxlen=MAX_LEN)

test = sequence.pad_sequences(test, maxlen=MAX_LEN)



embedding_matrix = np.concatenate(

    [build_matrix(tokenizer.word_index, f) for f in EMBEDDING_FILES], axis=-1)



print("Embeddings Matrix Shape:", embedding_matrix.shape)



checkpoint_predictions = []

weights = []
# Text Processing

SIA = SentimentIntensityAnalyzer()

train_df = text_processing(train_df)

test_df = text_processing(test_df)



# TF-IDF

count_vectorizer = TfidfVectorizer(

    analyzer="word",

    tokenizer=nltk.word_tokenize,

    preprocessor=None,

    stop_words='english',

    ngram_range=(1, 1),

    max_features=None)    



hash_loc_tfidf = count_vectorizer.fit(train_df['hash_loc_key'])

tfvocab = hash_loc_tfidf.get_feature_names()

print("Number of TF-IDF Features: {}".format(len(tfvocab)))



train_tfidf = count_vectorizer.transform(train_df['hash_loc_key'])

test_tfidf = count_vectorizer.transform(test_df['hash_loc_key'])
# Sparse Stack Numerical and TFIDF

dense_vars = [

    'hash_loc_key_num_words',

    'hash_loc_key_num_unique_words',

    'hash_loc_key_words_vs_unique',

    'text_num_words',

    'text_num_unique_words',

    'text_words_vs_unique',

    'text_vader_Compound']



# Normalisation - Standard Scaler

for d_i in dense_vars:

    scaler = StandardScaler()

    scaler.fit(train_df.loc[:,d_i].values.reshape(-1, 1))

    train_df.loc[:,d_i] = scaler.transform(train_df.loc[:,d_i].values.reshape(-1, 1))

    test_df.loc[:,d_i] = scaler.transform(test_df.loc[:,d_i].values.reshape(-1, 1))

    

# Sparse Stack

train_num = hstack([csr_matrix(train_df.loc[:,dense_vars].values),train_tfidf]).tocsr()

test_num = hstack([csr_matrix(test_df.loc[:,dense_vars].values),test_tfidf]).tocsr()

num_cols = train_df[dense_vars].columns.tolist() + tfvocab
def build_model(embedding_matrix, n_classes):

    words_inputs = Input(shape=(None,))

    numeric_inputs = Input(shape=(len(num_cols),))

    

    # Dense Inputs

    numeric_x = Dense(512, activation='relu')(numeric_inputs)

    numeric_x = Dropout(.4)(numeric_x)

    numeric_x = Dense(64, activation='relu')(numeric_x)

    

    # Embeddings Inputs

    words_x = Embedding(*embedding_matrix.shape, weights=[embedding_matrix],

                  trainable=False, input_length=MAX_LEN)(words_inputs)

    words_x = Flatten()(words_x)

    

    # Concat

    concat_x = concatenate([words_x, numeric_x])

    concat_x = Dropout(.4)(concat_x)

    output = Dense(n_classes, activation='sigmoid')(concat_x)

    model = Model(inputs=[words_inputs,numeric_inputs], outputs=output)

    opt = optimizers.Adam(learning_rate=0.00004, beta_1=0.9, beta_2=0.999, amsgrad=True)

    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['acc'])

    

    return model



model = build_model(embedding_matrix, N_CLASSES)

model.summary()
oof_preds = np.zeros(X.shape[0])

test_preds = np.zeros(test.shape[0])



n_splits = 6

folds = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

plot_metrics = ['loss','acc']



fold_hist = {}

for i, (trn_idx, val_idx) in enumerate(folds.split(X)):

    modelstart = time.time()

    model = build_model(embedding_matrix, N_CLASSES)

    

    es = callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=4, verbose=1,

                                 mode='min', baseline=None, restore_best_weights=True)

    rlr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-7,

                                      mode='min', verbose=1)

    

    history = model.fit(

            [X[trn_idx], train_num[trn_idx]],

            y[trn_idx],

            validation_data=([X[val_idx], train_num[val_idx]], y[val_idx]),

            batch_size=BATCH_SIZE,

            epochs=EPOCHS,

            verbose=0,

            callbacks=[es, rlr]

        )



    best_index = np.argmin(history.history['val_loss'])

    fold_hist[i] = history

    

    oof_preds[val_idx] = model.predict([X[val_idx], train_num[val_idx]]).ravel()

    test_preds += model.predict([test, test_num]).ravel()

    f1_sc = f1_score(y[val_idx], (oof_preds[val_idx] > 0.5).astype(int), average=f1_strategy)

    print("\nFOLD {} COMPLETE in {:.1f} Minutes - Avg F1 {:.5f} - Best Epoch {}".format(i, (time.time() - modelstart)/60, f1_sc, best_index + 1))

    best_metrics = {metric: scores[best_index] for metric, scores in history.history.items()}

    pprint.pprint(best_metrics)

    

    f, ax = plt.subplots(1,len(plot_metrics),figsize = [12,4])

    for p_i,metric in enumerate(plot_metrics):

        ax[p_i].plot(history.history[metric], label='Train ' + metric)

        ax[p_i].plot(history.history['val_' + metric], label='Val ' + metric)

        ax[p_i].set_title("{} Fold Loss Curve - {}\nBest Epoch {}".format(i, metric, best_index))

        ax[p_i].legend()

        ax[p_i].axvline(x=best_index, c='black')

    plt.show()
# OOF F1 Cutoff

save_f1_opt = []

for cutoff in np.arange(.38,.62, .01):

    save_f1_opt.append([cutoff, f1_score(y, (oof_preds > cutoff).astype(int), average=f1_strategy)])

f1_pd = pd.DataFrame(save_f1_opt, columns = ['cutoff', 'f1_score'])



best_cutoff = f1_pd.loc[f1_pd['f1_score'].idxmax(),'cutoff']

print("F1 Score: {:.4f}, Optimised Cufoff: {:.2f}".format(f1_pd.loc[f1_pd['f1_score'].idxmax(),'f1_score'], best_cutoff))



f,ax = plt.subplots(1,2,figsize = [10,4])



ax[0].plot(f1_pd['cutoff'], f1_pd['f1_score'], c = 'red')

ax[0].set_ylabel("F1 Score")

ax[0].set_xlabel("Cutoff")

ax[0].axvline(x=best_cutoff, c='black')

ax[0].set_title("F1 Score and Cutoff on OOF")





train_df['oof_preds'] = oof_preds

train_df['error'] = train_df['target'] - train_df['oof_preds']



sns.distplot(train_df['error'], ax = ax[1])

ax[1].set_title("Classification Errors: Target - Pred Probability")

ax[1].axvline(x=.5, c='black')

ax[1].axvline(x=-.5, c='black')

plt.tight_layout(pad=1)

plt.show()
print("OOF Classification Report for Optimised Threshold: {:.3f}".format(best_cutoff))

print(classification_report(y, (oof_preds > best_cutoff).astype(int), digits = 4))

print(f1_score(y, (oof_preds > cutoff).astype(int), average=f1_strategy))



print("\nOOF Non-Optimised Cutoff (.5)")

print(classification_report(y, (oof_preds > .5).astype(int), digits = 4))

print(f1_score(y, (oof_preds > .5).astype(int), average=f1_strategy))



cnf_matrix = confusion_matrix(y, (oof_preds > .5).astype(int))

print("OOF Confusion Matrix")

print(cnf_matrix)

print("OOF Normalised Confusion Matrix")

print((cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]).round(3))
show_cols = [

    'id',

    'keyword',

    'location',

    'text',

    'target',

    'oof_preds',

    'error']



print("Look at False Negative")

display(train_df[show_cols].sort_values(by = 'error', ascending=False).iloc[:20])



print("Look at False Positives")

display(train_df[show_cols].sort_values(by = 'error', ascending=True).iloc[:20])
print(test_preds[:5])

print(test_preds[:5] / n_splits)
submission = pd.DataFrame.from_dict({

    'id': test_df.id,

    TARGET_COLUMN: ((test_preds / n_splits) > best_cutoff).astype(int)

})

submission.to_csv('submission_optimised_cutoff.csv', index=False)

print(submission[TARGET_COLUMN].value_counts(normalize = True).to_dict())

submission.head()
submission = pd.DataFrame.from_dict({

    'id': test_df.id,

    TARGET_COLUMN: ((test_preds / n_splits) > .5).astype(int)

})

submission.to_csv('submission_fixed_cutoff.csv', index=False)

print(submission[TARGET_COLUMN].value_counts(normalize = True).to_dict())

submission.head()
oof_pd = pd.DataFrame(oof_preds, columns = ['dense_oof'])

oof_pd.to_csv("oof_dense_nn.csv")
print("Notebook Runtime: %0.2f Minutes"%((time.time() - notebookstart)/60))