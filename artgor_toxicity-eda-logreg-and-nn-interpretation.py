import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



from nltk.tokenize import TweetTokenizer

import datetime

import lightgbm as lgb

from scipy import stats

from scipy.sparse import hstack, csr_matrix

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn import metrics

from wordcloud import WordCloud

from collections import Counter

from nltk.corpus import stopwords

from nltk.util import ngrams

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.svm import LinearSVC

from sklearn.multiclass import OneVsRestClassifier

pd.set_option('max_colwidth',400)

pd.set_option('max_columns', 50)

import json

import altair as alt

from  altair.vega import v3

from IPython.display import HTML

import gc

import os

print(os.listdir("../input"))

import lime

import eli5

from eli5.lime import TextExplainer

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Conv1D, GRU, CuDNNGRU, CuDNNLSTM, BatchNormalization

from keras.layers import Bidirectional, GlobalMaxPool1D, MaxPooling1D, Add, Flatten

from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D

from keras.models import Model, load_model

from keras import initializers, regularizers, constraints, optimizers, layers, callbacks

from keras import backend as K

from keras.engine import InputSpec, Layer

from keras.optimizers import Adam



from keras.callbacks import ModelCheckpoint, TensorBoard, Callback, EarlyStopping
# Preparing altair. I use code from this great kernel: https://www.kaggle.com/notslush/altair-visualization-2018-stackoverflow-survey



vega_url = 'https://cdn.jsdelivr.net/npm/vega@' + v3.SCHEMA_VERSION

vega_lib_url = 'https://cdn.jsdelivr.net/npm/vega-lib'

vega_lite_url = 'https://cdn.jsdelivr.net/npm/vega-lite@' + alt.SCHEMA_VERSION

vega_embed_url = 'https://cdn.jsdelivr.net/npm/vega-embed@3'

noext = "?noext"



paths = {

    'vega': vega_url + noext,

    'vega-lib': vega_lib_url + noext,

    'vega-lite': vega_lite_url + noext,

    'vega-embed': vega_embed_url + noext

}



workaround = """

requirejs.config({{

    baseUrl: 'https://cdn.jsdelivr.net/npm/',

    paths: {}

}});

"""



#------------------------------------------------ Defs for future rendering

def add_autoincrement(render_func):

    # Keep track of unique <div/> IDs

    cache = {}

    def wrapped(chart, id="vega-chart", autoincrement=True):

        if autoincrement:

            if id in cache:

                counter = 1 + cache[id]

                cache[id] = counter

            else:

                cache[id] = 0

            actual_id = id if cache[id] == 0 else id + '-' + str(cache[id])

        else:

            if id not in cache:

                cache[id] = 0

            actual_id = id

        return render_func(chart, id=actual_id)

    # Cache will stay outside and 

    return wrapped

            

@add_autoincrement

def render(chart, id="vega-chart"):

    chart_str = """

    <div id="{id}"></div><script>

    require(["vega-embed"], function(vg_embed) {{

        const spec = {chart};     

        vg_embed("#{id}", spec, {{defaultStyle: true}}).catch(console.warn);

        console.log("anything?");

    }});

    console.log("really...anything?");

    </script>

    """

    return HTML(

        chart_str.format(

            id=id,

            chart=json.dumps(chart) if isinstance(chart, dict) else chart.to_json(indent=None)

        )

    )



HTML("".join((

    "<script>",

    workaround.format(json.dumps(paths)),

    "</script>",

)))
train = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')

test = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')

sub = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv')
train.head()
train.shape, test.shape, (train['target'] > 0).sum() / train.shape[0], (train['target'] >= 0.5).sum() / train.shape[0]
train['comment_text'].value_counts().head(20)
train.loc[train['comment_text'] == 'Well said.', 'target'].unique()
print('Rate of unique comments:', train['comment_text'].nunique() / train['comment_text'].shape[0])
train_comments = set(train['comment_text'].values)

test_comments = set(test['comment_text'].values)

len(train_comments.intersection(test_comments)), len(test.loc[test['comment_text'].isin(list(train_comments.intersection(test_comments)))])
hist_df = pd.cut(train['target'], 20).value_counts().sort_index().reset_index().rename(columns={'index': 'bins'})

hist_df['bins'] = hist_df['bins'].astype(str)

render(alt.Chart(hist_df).mark_bar().encode(

    x=alt.X("bins:O", axis=alt.Axis(title='Target bins')),

    y=alt.Y('target:Q', axis=alt.Axis(title='Count')),

    tooltip=['target', 'bins']

).properties(title="Counts of target bins", width=400).interactive())
train['target'].value_counts().head(20)
train['created_date'] = pd.to_datetime(train['created_date']).values.astype('datetime64[M]')

counts = train.groupby(['created_date'])['target'].mean().sort_index().reset_index()

means = train.groupby(['created_date'])['target'].count().sort_index().reset_index()

c = alt.Chart(counts).mark_line().encode(

    x=alt.X("created_date:T", axis=alt.Axis(title='Date')),

    y=alt.Y('target:Q', axis=alt.Axis(title='Rate')),

    tooltip=[alt.Tooltip('created_date:T', timeUnit='yearmonth'), alt.Tooltip('target:Q')]

).properties(title="Counts and toxicity rate of comments", width=800).interactive()

r = alt.Chart(means).mark_line(color='green').encode(

    x=alt.X("created_date:T", axis=alt.Axis(title='Date')),

    y=alt.Y('target:Q', axis=alt.Axis(title='Counts')),

    tooltip=[alt.Tooltip('created_date:T', timeUnit='yearmonth'), alt.Tooltip('target:Q')],

).properties().interactive()

render(alt.layer(

    c,

    r

).resolve_scale(

    y='independent'

))
plot_dict = {}

for col in ['severe_toxicity', 'obscene', 'threat', 'insult', 'identity_attack', 'sexual_explicit']:

    df_ = train.loc[train[col] > 0]

    hist_df = pd.cut(df_[col], 20).value_counts().sort_index().reset_index().rename(columns={'index': 'bins'})

    hist_df['bins'] = hist_df['bins'].astype(str)

    plot_dict[col] = alt.Chart(hist_df).mark_bar().encode(

        x=alt.X("bins:O", axis=alt.Axis(title='Target bins')),

        y=alt.Y(f'{col}:Q', axis=alt.Axis(title='Count')),

        tooltip=[col, 'bins']

    ).properties(title=f"Counts of {col} bins", width=300, height=200).interactive()

    

render((plot_dict['severe_toxicity'] | plot_dict['obscene']) & (plot_dict['threat'] | plot_dict['insult']) & (plot_dict['identity_attack'] | plot_dict['sexual_explicit']))
hist_df = pd.cut(train['comment_text'].apply(lambda x: len(x)), 10).value_counts().sort_index().reset_index().rename(columns={'index': 'bins'})

hist_df['bins'] = hist_df['bins'].astype(str)

render(alt.Chart(hist_df).mark_bar().encode(

    x=alt.X("bins:O", axis=alt.Axis(title='Target bins'), sort=list(hist_df['bins'].values)),

    y=alt.Y('comment_text:Q', axis=alt.Axis(title='Count')),

    tooltip=['comment_text', 'bins']

).properties(title="Counts of target bins of text length", width=400).interactive())
text_length = train['comment_text'].apply(lambda x: len(x)).value_counts(normalize=True).sort_index().cumsum().reset_index().rename(columns={'index': 'Text length'})
render(alt.Chart(text_length).mark_line().encode(

    x=alt.X("Text length:Q", axis=alt.Axis(title='Text length')),

    y=alt.Y('comment_text:Q', axis=alt.Axis(title='Cummulative rate')),

    tooltip=['Text length', 'comment_text']

).properties(title="Cummulative text length", width=400).interactive())
hist_df = pd.cut(train['comment_text'].apply(lambda x: len(x.split())), 10).value_counts().sort_index().reset_index().rename(columns={'index': 'bins'})

hist_df['bins'] = hist_df['bins'].astype(str)

render(alt.Chart(hist_df).mark_bar().encode(

    x=alt.X("bins:O", axis=alt.Axis(title='Target bins'), sort=list(hist_df['bins'].values)),

    y=alt.Y('comment_text:Q', axis=alt.Axis(title='Count')),

    tooltip=['comment_text', 'bins']

).properties(title="Counts of target bins of word count", width=400).interactive())
word_count = train['comment_text'].apply(lambda x: len(x.split())).value_counts(normalize=True).sort_index().cumsum().reset_index().rename(columns={'index': 'Word count'})

render(alt.Chart(word_count).mark_line().encode(

    x=alt.X("Word count:Q", axis=alt.Axis(title='Text length')),

    y=alt.Y('comment_text:Q', axis=alt.Axis(title='Cummulative rate')),

    tooltip=['Word count:Q', 'comment_text']

).properties(title="Cummulative word cound", width=400).interactive())
# I'll load processed texts from my kernel

train = pd.read_csv('../input/jigsaw-public-files/train.csv')

test = pd.read_csv('../input/jigsaw-public-files/test.csv')

train['comment_text'] = train['comment_text'].fillna('')

test['comment_text'] = test['comment_text'].fillna('')
identity_columns = ['male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish', 'muslim', 'black', 'white', 'psychiatric_or_mental_illness']

for col in identity_columns + ['target']:

    train[col] = np.where(train[col] >= 0.5, True, False)
train_df, valid_df = train_test_split(train, test_size=0.1, stratify=train['target'])

y_train = train_df['target']

y_valid = valid_df['target']
%%time

tokenizer = TweetTokenizer()



vectorizer = TfidfVectorizer(ngram_range=(1, 2), tokenizer=tokenizer.tokenize, max_features=30000)

vectorizer.fit(train['comment_text'].values)

train_vectorized = vectorizer.transform(train_df['comment_text'].values)

valid_vectorized = vectorizer.transform(valid_df['comment_text'].values)
%%time

logreg = LogisticRegression()

logreg.fit(train_vectorized, y_train)

oof_name = 'predicted_target'

valid_df[oof_name] = logreg.predict_proba(valid_vectorized)[:, 1]
SUBGROUP_AUC = 'subgroup_auc'

BPSN_AUC = 'bpsn_auc'  # stands for background positive, subgroup negative

BNSP_AUC = 'bnsp_auc'  # stands for background negative, subgroup positive



def compute_auc(y_true, y_pred):

    try:

        return metrics.roc_auc_score(y_true, y_pred)

    except ValueError:

        return np.nan



def compute_subgroup_auc(df, subgroup, label, oof_name):

    subgroup_examples = df[df[subgroup]]

    return compute_auc(subgroup_examples[label], subgroup_examples[oof_name])



def compute_bpsn_auc(df, subgroup, label, oof_name):

    """Computes the AUC of the within-subgroup negative examples and the background positive examples."""

    subgroup_negative_examples = df[df[subgroup] & ~df[label]]

    non_subgroup_positive_examples = df[~df[subgroup] & df[label]]

    examples = subgroup_negative_examples.append(non_subgroup_positive_examples)

    return compute_auc(examples[label], examples[oof_name])



def compute_bnsp_auc(df, subgroup, label, oof_name):

    """Computes the AUC of the within-subgroup positive examples and the background negative examples."""

    subgroup_positive_examples = df[df[subgroup] & df[label]]

    non_subgroup_negative_examples = df[~df[subgroup] & ~df[label]]

    examples = subgroup_positive_examples.append(non_subgroup_negative_examples)

    return compute_auc(examples[label], examples[oof_name])



def compute_bias_metrics_for_model(dataset,

                                   subgroups,

                                   model,

                                   label_col,

                                   include_asegs=False):

    """Computes per-subgroup metrics for all subgroups and one model."""

    records = []

    for subgroup in subgroups:

        record = {

            'subgroup': subgroup,

            'subgroup_size': len(dataset[dataset[subgroup]])

        }

        record[SUBGROUP_AUC] = compute_subgroup_auc(dataset, subgroup, label_col, model)

        record[BPSN_AUC] = compute_bpsn_auc(dataset, subgroup, label_col, model)

        record[BNSP_AUC] = compute_bnsp_auc(dataset, subgroup, label_col, model)

        records.append(record)

    return pd.DataFrame(records).sort_values('subgroup_auc', ascending=True)

oof_name = 'predicted_target'

bias_metrics_df = compute_bias_metrics_for_model(valid_df, identity_columns, oof_name, 'target')

bias_metrics_df
def calculate_overall_auc(df, oof_name):

    true_labels = df['target']

    predicted_labels = df[oof_name]

    return metrics.roc_auc_score(true_labels, predicted_labels)



def power_mean(series, p):

    total = sum(np.power(series, p))

    return np.power(total / len(series), 1 / p)



def get_final_metric(bias_df, overall_auc, POWER=-5, OVERALL_MODEL_WEIGHT=0.25):

    bias_score = np.average([

        power_mean(bias_df[SUBGROUP_AUC], POWER),

        power_mean(bias_df[BPSN_AUC], POWER),

        power_mean(bias_df[BNSP_AUC], POWER)

    ])

    return (OVERALL_MODEL_WEIGHT * overall_auc) + ((1 - OVERALL_MODEL_WEIGHT) * bias_score)

    

get_final_metric(bias_metrics_df, calculate_overall_auc(valid_df, oof_name))
import eli5

from eli5.lime import TextExplainer



te = TextExplainer(random_state=42)

def model_predict(x):

    return logreg.predict_proba(vectorizer.transform(x))

te.fit(valid_df['comment_text'].values[2:3][0], model_predict)

te.show_prediction()
te.fit(valid_df['comment_text'].values[12:13][0], model_predict)

te.show_prediction()
test_vectorized = vectorizer.transform(test['comment_text'].values)

sub['prediction'] = logreg.predict_proba(test_vectorized)[:, 1]

sub.to_csv('submission.csv', index=False)

del logreg, vectorizer, test_vectorized, train_vectorized, valid_vectorized
def build_model(X_train, y_train, X_valid, y_valid, max_len, max_features, embedding_matrix, lr=0.0, lr_d=0.0, spatial_dr=0.0,  dense_units=128, dr=0.1):

    file_path = "best_model.hdf5"

    check_point = ModelCheckpoint(file_path, monitor = "val_loss", verbose = 1,

                                  save_best_only = True, mode = "min")

    early_stop = EarlyStopping(monitor = "val_loss", mode = "min", patience = 3)

    

    inp = Input(shape = (max_len,))

    x = Embedding(max_features, embed_size, weights = [embedding_matrix], trainable = False)(inp)

    x1 = SpatialDropout1D(spatial_dr)(x)

    # from benchmark kernel

    x = Conv1D(128, 2, activation='relu', padding='same')(x1)

    x = MaxPooling1D(5, padding='same')(x)

    x = Conv1D(128, 3, activation='relu', padding='same')(x)

    x = MaxPooling1D(5, padding='same')(x)

    x = Flatten()(x)

    

    x = Dropout(dr)(Dense(dense_units, activation='relu') (x))

    x = Dense(2, activation = "softmax")(x)

    

    model = Model(inputs = inp, outputs = x)

    model.compile(loss = "binary_crossentropy", optimizer = Adam(lr = lr, decay = lr_d), metrics = ["accuracy"])

    history = model.fit(X_train, y_train, batch_size = 128, epochs = 3, validation_data=(X_valid, y_valid), 

                        verbose = 0, callbacks = [check_point, early_stop])

    model = load_model(file_path)

    return model
full_text = list(train['comment_text'].values) + list(test['comment_text'].values)

embedding_path = "../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec"

embed_size = 300

oof_name = 'oof_name'



def calculate_score(num_words, max_len, full_text, train_df, valid_df, embedding_path, embed_size, identity_columns, oof_name):

    tk = Tokenizer(lower = True, filters='', num_words=num_words)

    tk.fit_on_texts(full_text)

    

    def get_coefs(word,*arr):

        return word, np.asarray(arr, dtype='float32')



    embedding_index = dict(get_coefs(*o.strip().split(" ")) for o in open(embedding_path))

    embedding_matrix = np.zeros((num_words + 1, embed_size))

    for word, i in tk.word_index.items():

        if i >= num_words: continue

        embedding_vector = embedding_index.get(word)

        if embedding_vector is not None:

            embedding_matrix[i] = embedding_vector

    del embedding_index

            

    train_tokenized = tk.texts_to_sequences(train_df['comment_text'])

    valid_tokenized = tk.texts_to_sequences(valid_df['comment_text'])



    X_train = pad_sequences(train_tokenized, maxlen = max_len)

    X_valid = pad_sequences(valid_tokenized, maxlen = max_len)

    

    model = build_model(X_train=X_train, y_train=y_train, X_valid=X_valid, y_valid=y_valid, max_len=max_len, max_features=embedding_matrix.shape[0], embedding_matrix=embedding_matrix,

                        lr = 1e-3, lr_d = 0, spatial_dr = 0.0, dr=0.1)

    

    valid_df[oof_name] = model.predict(X_valid)

    bias_metrics_df = compute_bias_metrics_for_model(valid_df, identity_columns, oof_name, 'target')

    score = get_final_metric(bias_metrics_df, calculate_overall_auc(valid_df, oof_name))

    del embedding_matrix, tk

    gc.collect()

    

    return score
# scores = []

# for n_words in [50000, 100000]:

#     for seq_len in [150, 300]:

#         loc_score = calculate_score(n_words, seq_len, full_text, train_df, valid_df, embedding_path, embed_size, identity_columns, oof_name)

#         scores.append((n_words, seq_len, loc_score))
num_words = 150000

max_len = 220

tk = Tokenizer(lower = True, filters='', num_words=num_words)

tk.fit_on_texts(full_text)



def get_coefs(word,*arr):

    return word, np.asarray(arr, dtype='float32')



embedding_index = dict(get_coefs(*o.strip().split(" ")) for o in open(embedding_path))

embedding_matrix = np.zeros((num_words + 1, embed_size))

for word, i in tk.word_index.items():

    if i >= num_words: continue

    embedding_vector = embedding_index.get(word)

    if embedding_vector is not None:

        embedding_matrix[i] = embedding_vector

del embedding_index



train_tokenized = tk.texts_to_sequences(train_df['comment_text'])

valid_tokenized = tk.texts_to_sequences(valid_df['comment_text'])



X_train = pad_sequences(train_tokenized, maxlen = max_len)

X_valid = pad_sequences(valid_tokenized, maxlen = max_len)



model = build_model(X_train=X_train, y_train=pd.get_dummies(y_train), X_valid=X_valid, y_valid=pd.get_dummies(y_valid), max_len=max_len, max_features=embedding_matrix.shape[0],

                    embedding_matrix=embedding_matrix,

                    lr = 1e-3, lr_d = 0, spatial_dr = 0.0, dr=0.1)
te = TextExplainer(random_state=42)

def dl_predict(x):

    return model.predict(pad_sequences(tk.texts_to_sequences(np.array(x)), maxlen = max_len))

te.fit(valid_df['comment_text'].values[3:4][0], dl_predict)

te.show_prediction(target_names=[0, 1])