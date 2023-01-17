# We will use the official tokenization script created by the Google team

!wget --quiet https://raw.githubusercontent.com/tensorflow/models/master/official/nlp/bert/tokenization.py
import tensorflow as tf

from tensorflow.keras.layers import Dense, Input, Dropout, LSTM, Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D, concatenate, SpatialDropout1D

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import ModelCheckpoint

import tensorflow_hub as hub

from tensorflow.keras import callbacks



from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.preprocessing import StandardScaler

from nltk.sentiment.vader import SentimentIntensityAnalyzer

from nltk import word_tokenize

from scipy.sparse import hstack, csr_matrix

from sklearn.model_selection import KFold

from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

from sklearn.metrics import classification_report, confusion_matrix

import pprint



import tokenization



import re

import gc

import time



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



sns.set_style("whitegrid")

notebookstart = time.time()

pd.options.display.max_colwidth = 500



print("Tensorflow Version: ", tf.__version__)
MAX_LEN = 36

BATCH_SIZE = 36

EPOCHS = 5

SEED = 42

NROWS = None

f1_strategy = 'macro'

TARGET_COLUMN = 'target'
def bert_encode(texts, tokenizer, max_len=512):

    all_tokens = []

    all_masks = []

    all_segments = []

    

    for text in texts:

        text = tokenizer.tokenize(text)

            

        text = text[:max_len-2]

        input_sequence = ["[CLS]"] + text + ["[SEP]"]

        pad_len = max_len - len(input_sequence)

        

        tokens = tokenizer.convert_tokens_to_ids(input_sequence)

        tokens += [0] * pad_len

        pad_masks = [1] * len(input_sequence) + [0] * pad_len

        segment_ids = [0] * max_len

        

        all_tokens.append(tokens)

        all_masks.append(pad_masks)

        all_segments.append(segment_ids)

    

    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)



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



def build_model(bert_layer, max_len=512, dropout=.2):

    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")

    input_mask = Input(shape=(max_len,), dtype=tf.int32, name="input_mask")

    segment_ids = Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")

    numeric_inputs = Input(shape=(len(num_cols),), dtype=tf.float32, name="numeric_inputs")

    

    # Bert Layer

    pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])

    

    # Sequence Output

    sequence_output = SpatialDropout1D(dropout)(sequence_output)

    sequence_output = Bidirectional(LSTM(128, return_sequences=True))(sequence_output)

    sequence_output = GlobalAveragePooling1D()(sequence_output)

    

    # Pooled Output

    pooled_output = Dense(36, activation='relu')(pooled_output)

    

    # Dense Inputs

    numeric_x = Dense(512, activation='relu')(numeric_inputs)

    numeric_x = Dropout(dropout)(numeric_x)

    numeric_x = Dense(64, activation='relu')(numeric_x)

    

    # Concatenate

    cat = concatenate([

        pooled_output,

        sequence_output,

        numeric_x

    ])

    cat = Dropout(dropout)(cat)

    

    # Output Layer

    out = Dense(1, activation='sigmoid')(cat)

    

    model = Model(inputs=[input_word_ids, input_mask, segment_ids, numeric_inputs], outputs=out)

    model.compile(Adam(lr=1e-6), loss='binary_crossentropy', metrics=['acc'])

    

    return model
%%time

module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/1"

bert_layer = hub.KerasLayer(module_url, trainable=True)
train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv", nrows=NROWS)

test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv", nrows=NROWS)

testdex = test.id

submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv", nrows=NROWS)



print("Train Shape: {} Rows, {} Columns".format(*train.shape))

print("Test Shape: {} Rows, {} Columns".format(*test.shape))



length_info = [len(x) for x in np.concatenate([train.text.values, test.text.values])]

print("Train Sequence Length - Mean {:.1f} +/- {:.1f}, Max {:.1f}, Min {:.1f}".format(

    np.mean(length_info), np.std(length_info), np.max(length_info), np.min(length_info)))
# Text Processing

SIA = SentimentIntensityAnalyzer()

train_df = text_processing(train)

test_df = text_processing(test)



# TF-IDF

count_vectorizer = TfidfVectorizer(

    analyzer="word",

    tokenizer=word_tokenize,

    preprocessor=None,

    stop_words='english',

    sublinear_tf=True,

    ngram_range=(1, 1),

    max_features=500)    



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

train_num = hstack([csr_matrix(train_df.loc[:,dense_vars].values),train_tfidf]).toarray()

test_num = hstack([csr_matrix(test_df.loc[:,dense_vars].values),test_tfidf]).toarray()

num_cols = train_df[dense_vars].columns.tolist() + tfvocab
# Bert Pre-Processing

vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()

do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()

tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)



train_input_word_ids, train_input_mask, train_segment_ids, train_numeric_inputs = *bert_encode(train.text.values, tokenizer, max_len=MAX_LEN), train_num

test_input = (*bert_encode(test.text.values, tokenizer, max_len=MAX_LEN), test_num)

train_labels = train.target.values



del test, train_num, test_num, train_df, test_df

_ = gc.collect()
model = build_model(bert_layer, max_len=MAX_LEN)

model.summary()
oof_preds = np.zeros(train_input_word_ids.shape[0])

test_preds = np.zeros(test_input[0].shape[0])



n_splits = 3

folds = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)

plot_metrics = ['loss','acc']



fold_hist = {}

for i, (trn_idx, val_idx) in enumerate(folds.split(train_input_word_ids)):

    modelstart = time.time()

    model = build_model(bert_layer, max_len=MAX_LEN)

    

    es = callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=4, verbose=1,

                                 mode='min', baseline=None, restore_best_weights=True)

    

    

    history = model.fit(

        x=[train_input_word_ids[trn_idx],

            train_input_mask[trn_idx],

            train_segment_ids[trn_idx],

            train_numeric_inputs[trn_idx]],

        y=train_labels[trn_idx],

        validation_data=(

            [train_input_word_ids[val_idx],

            train_input_mask[val_idx],

            train_segment_ids[val_idx],

            train_numeric_inputs[val_idx]],

            train_labels[val_idx]),

        epochs=EPOCHS,

        batch_size=18,

        callbacks=[es]

    )



    best_index = np.argmin(history.history['val_loss'])

    fold_hist[i] = history

    

    oof_preds[val_idx] = model.predict(

        [train_input_word_ids[val_idx],

        train_input_mask[val_idx],

        train_segment_ids[val_idx],

        train_numeric_inputs[val_idx]]).ravel()

    test_preds += model.predict(test_input).ravel()

    f1_sc = f1_score(train_labels[val_idx], (oof_preds[val_idx] > 0.5).astype(int), average=f1_strategy)

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

    save_f1_opt.append([cutoff, f1_score(train_labels, (oof_preds > cutoff).astype(int), average=f1_strategy)])

f1_pd = pd.DataFrame(save_f1_opt, columns = ['cutoff', 'f1_score'])



best_cutoff = f1_pd.loc[f1_pd['f1_score'].idxmax(),'cutoff']

print("F1 Score: {:.4f}, Optimised Cufoff: {:.2f}".format(f1_pd.loc[f1_pd['f1_score'].idxmax(),'f1_score'], best_cutoff))



f,ax = plt.subplots(1,2,figsize = [10,4])



ax[0].plot(f1_pd['cutoff'], f1_pd['f1_score'], c = 'red')

ax[0].set_ylabel("F1 Score")

ax[0].set_xlabel("Cutoff")

ax[0].axvline(x=best_cutoff, c='black')

ax[0].set_title("F1 Score and Cutoff on OOF")





train['oof_preds'] = oof_preds

train['error'] = train['target'] - train['oof_preds']



sns.distplot(train['error'], ax = ax[1])

ax[1].set_title("Classification Errors: Target - Pred Probability")

ax[1].axvline(x=.5, c='black')

ax[1].axvline(x=-.5, c='black')

plt.tight_layout(pad=1)

plt.show()
print("OOF Classification Report for Optimised Threshold: {:.3f}".format(best_cutoff))

print(classification_report(train_labels, (oof_preds > best_cutoff).astype(int), digits = 4))

print(f1_score(train_labels, (oof_preds > cutoff).astype(int), average=f1_strategy))



print("\nOOF Non-Optimised Cutoff (.5)")

print(classification_report(train_labels, (oof_preds > .5).astype(int), digits = 4))

print(f1_score(train_labels, (oof_preds > .5).astype(int), average=f1_strategy))



cnf_matrix = confusion_matrix(train_labels, (oof_preds > .5).astype(int))

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

display(train[show_cols].sort_values(by = 'error', ascending=False).iloc[:20])



print("Look at False Positives")

display(train[show_cols].sort_values(by = 'error', ascending=True).iloc[:20])
submission = pd.DataFrame.from_dict({

    'id': testdex,

    TARGET_COLUMN: ((test_preds / n_splits) > .5).astype(int)

})

submission.to_csv('submission_fixed_cutoff.csv', index=False)

print(submission[TARGET_COLUMN].value_counts(normalize = True).to_dict())

submission.head()
submission = pd.DataFrame.from_dict({

    'id': testdex,

    TARGET_COLUMN: ((test_preds / n_splits) > best_cutoff).astype(int)

})

submission.to_csv('submission_optimised_cutoff.csv', index=False)

print(submission[TARGET_COLUMN].value_counts(normalize = True).to_dict())

submission.head()
oof_pd = pd.DataFrame(oof_preds, columns = ['dense_oof'])

oof_pd.to_csv("oof_dense_bert.csv")
print("Notebook Runtime: %0.2f Minutes"%((time.time() - notebookstart)/60))