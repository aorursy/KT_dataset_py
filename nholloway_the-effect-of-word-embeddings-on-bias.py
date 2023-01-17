import numpy as np

import pandas as pd

from sklearn.metrics import roc_auc_score as auc

from sklearn.model_selection import train_test_split



import plotly

import colorlover as cl

import plotly.offline as py

import plotly.graph_objs as go



from keras.models import Model

from keras.layers import Input, Dense, Embedding, SpatialDropout1D, Dropout, add, concatenate

from keras.layers import CuDNNLSTM, Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D

from keras.preprocessing import text, sequence

from keras.callbacks import LearningRateScheduler

plotly.tools.set_credentials_file(username='nholloway', api_key='Ef8vuHMUdvaIpvtC2lux')

py.init_notebook_mode(connected=True)
FASTTEXT_PATH = '../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec'

GLOVE_PATH = '../input/glove840b300dtxt/glove.840B.300d.txt'

NUMBERBATCH_PATH = '../input/conceptnet-numberbatch-vectors/numberbatch-en-17.06.txt/numberbatch-en-17.06.txt'

NUM_MODELS = 2

BATCH_SIZE = 512

LSTM_UNITS = 128

DENSE_HIDDEN_UNITS = 4 * LSTM_UNITS

EPOCHS = 4

MAX_LEN = 220
def get_coefs(word, *arr):

    return word, np.asarray(arr, dtype='float32')



def load_embeddings(path):

    with open(path) as f:

        return dict(get_coefs(*line.strip().split(' ')) for line in f)



def build_matrix(word_index, path):

    embedding_index = load_embeddings(path)

    embedding_matrix = np.zeros((len(word_index) + 1, 300))

    for word, i in word_index.items():

        try:

            embedding_matrix[i] = embedding_index[word]

        except KeyError:

            pass

    return embedding_matrix
def build_model(embedding_matrix, num_aux_targets):

    words = Input(shape=(MAX_LEN,))

    x = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(words)

    x = SpatialDropout1D(0.3)(x)

    x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x)

    x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x)



    hidden = concatenate([

        GlobalMaxPooling1D()(x),

        GlobalAveragePooling1D()(x),

    ])

    hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])

    hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])

    result = Dense(1, activation='sigmoid')(hidden)

    aux_result = Dense(num_aux_targets, activation='sigmoid')(hidden)

    

    model = Model(inputs=words, outputs=[result, aux_result])

    model.compile(loss='binary_crossentropy', optimizer='adam')



    return model
def preprocess(text):

    s_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have" }

    specials = ["’", "‘", "´", "`"]

    p_mapping = {"_":" ", "`":" "}    

    punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'

    

    for s in specials:

        text = text.replace(s, "'")

    text = ' '.join([s_mapping[t] if t in s_mapping else t for t in text.split(" ")])

    for p in p_mapping:

        text = text.replace(p, p_mapping[p])    

    for p in punct:

        text = text.replace(p, f' {p} ')     

    return text
train = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')

test_columns = ['target', 'black', 'white', 'male', 'female', 'homosexual_gay_or_lesbian',

                'christian', 'jewish', 'muslim', 'psychiatric_or_mental_illness', 

                'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat'] 



x_train, x_val, y_train, y_val = train_test_split(train['comment_text'], train[test_columns], test_size=.2, random_state=42)



x_train = x_train.apply(lambda x: preprocess(x.lower()))

y_train['target'] = np.where(y_train['target'] >= 0.5, 1, 0)

x_val = x_val.apply(lambda x: preprocess(x.lower()))

y_val['target'] = np.where(y_val['target'] >= 0.5, 1, 0)

y_aux_train = y_train[['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']].copy()

y_train = y_train['target']



tokenizer = text.Tokenizer()

tokenizer.fit_on_texts(list(x_train) + list(x_val))

 

x_train = tokenizer.texts_to_sequences(x_train)

x_val = tokenizer.texts_to_sequences(x_val)



x_train = sequence.pad_sequences(x_train, maxlen=MAX_LEN)

x_val = sequence.pad_sequences(x_val, maxlen=MAX_LEN)
def train_model(embedding_matrix, model_name):

    checkpoint_predictions = []

    weights = []

    

    for model_idx in range(NUM_MODELS):

        model = build_model(embedding_matrix, y_aux_train.shape[-1])

        for global_epoch in range(EPOCHS):

            model.fit(

                x_train,

                [y_train, y_aux_train],

                batch_size=BATCH_SIZE,

                epochs=1,

                verbose=2,

                callbacks=[

                    LearningRateScheduler(lambda epoch: 1e-3 * (0.6 ** global_epoch))

                ])

            checkpoint_predictions.append(model.predict(x_val, batch_size=2048)[0].flatten())

            weights.append(2 ** global_epoch)

            model.save(f'data/toxicity/{model_name}_model.h5')



    predictions = np.average(checkpoint_predictions, weights=weights, axis=0)

    return predictions
IDENTITY_COLUMNS = ['black', 'white', 'male', 'female', 'homosexual_gay_or_lesbian',

                   'christian', 'jewish', 'muslim', 'psychiatric_or_mental_illness'] 

    

def compute_bpsn_auc(df, subgroup, model, label):

    subgroup_positive_examples = df.loc[(df[subgroup] == 1) & (df[label] == 1)]

    non_subgroup_negative_examples = df.loc[df[subgroup] != 1 & (df[label] == 0)]

    examples = subgroup_positive_examples.append(non_subgroup_negative_examples)

    return auc(examples[label], examples[model])  

    

def compute_bnsp_auc(df, subgroup, model, label):

    subgroup_negative_examples = df.loc[(df[subgroup] == 1) & (df[label] == 0)]

    non_subgroup_positive_examples = df.loc[(df[subgroup] != 1) & (df[label] == 1)]

    examples = subgroup_negative_examples.append(non_subgroup_positive_examples)

    return auc(examples[label], examples[model])



def power_mean(series, p):

    total = sum(np.power(series, p))

    return np.power(total/len(series), 1/p)



def compute_final_bias(bias_df, overall_auc, power=-5, overall_model_weight=0.25):

    bias_score = np.average([

        power_mean(bias_df['subgroup_auc'], power),

        power_mean(bias_df['bpsn_auc'], power),

        power_mean(bias_df['bnsp_auc'], power)

    ])

    return (overall_model_weight * overall_auc) + ((1 - overall_model_weight)* bias_score)

    

def compute_subgroup_bias_metrics(df, subgroups, model, label):

    records = []

    for subgroup in subgroups:

        subgroup_df = df.loc[df[subgroup] == 1]

        record = {

            'subgroup': subgroup, 

            'subgroup_size': len(subgroup_df)

        }

        record['subgroup_auc'] = auc(subgroup_df['target'], subgroup_df[model])

        record['bpsn_auc'] = compute_bpsn_auc(df, subgroup, model, label)

        record['bnsp_auc'] = compute_bnsp_auc(df, subgroup, model, label)

        records.append(record)

    return pd.DataFrame(records).sort_values('subgroup_auc', 

                                                 ascending=True)
def build_control_model(embedding_size, max_features, num_aux_targets):

    words = Input(shape=(MAX_LEN,))

    x = Embedding(max_features, embedding_size)(words)

    x = SpatialDropout1D(0.3)(x)

    x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x)

    x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x)



    hidden = concatenate([

        GlobalMaxPooling1D()(x),

        GlobalAveragePooling1D()(x),

    ])

    hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])

    hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])

    result = Dense(1, activation='sigmoid')(hidden)

    aux_result = Dense(num_aux_targets, activation='sigmoid')(hidden)

    

    model = Model(inputs=words, outputs=[result, aux_result])

    model.compile(loss='binary_crossentropy', optimizer='adam')



    return model



def train_control_model(embedding_size, max_features, model_name):

    checkpoint_predictions = []

    weights = []

    

    for model_idx in range(NUM_MODELS):

        model = build_control_model(embedding_size, max_features, y_aux_train.shape[-1])

        for global_epoch in range(EPOCHS):

            model.fit(

                x_train,

                [y_train, y_aux_train],

                batch_size=BATCH_SIZE,

                epochs=1,

                verbose=2,

                callbacks=[

                    LearningRateScheduler(lambda epoch: 1e-3 * (0.6 ** global_epoch))

                ])

            checkpoint_predictions.append(model.predict(x_val, batch_size=2048)[0].flatten())

            weights.append(2 ** global_epoch)

            model.save(f'data/toxicity/{model_name}_model.h5')



    predictions = np.average(checkpoint_predictions, weights=weights, axis=0)

    return predictions
max_features = len(tokenizer.word_index.keys())

emb_size = 300

# This model took 3hr 47min to train compared to 

# ~3hr 3min for most the pretrained embedding models

# %time control_preds = train_control_model(emb_size, max_features, 'no pretrained embeddings')

results_df = pd.read_pickle('../input/embedding-bias-benchmark/final_results.pkl')
# results_df = pd.concat([pd.DataFrame(control_preds, columns=['w/o pretrained']).reset_index(drop=True), results_df], axis=1)

ctrl_bias_metrics = compute_subgroup_bias_metrics(results_df, IDENTITY_COLUMNS, 'w/o pretrained', 'target')

overall_auc = auc(results_df['target'], results_df['w/o pretrained'])

ctrl_final_bias = compute_final_bias(ctrl_bias_metrics, overall_auc)

display(ctrl_bias_metrics)

print(f'Final Metric: {ctrl_final_bias}')
fasttext_matrix = build_matrix(tokenizer.word_index, FASTTEXT_PATH)

# %time fast_preds = train_model(fasttext_matrix, 'fasttext')
# results_df = pd.concat([pd.DataFrame(fast_preds, columns=['fasttext']).reset_index(drop=True), y_val.reset_index(drop=True)], axis=1).fillna(0)

ft_bias_metrics = compute_subgroup_bias_metrics(results_df, IDENTITY_COLUMNS, 'fasttext', 'target')

overall_auc = auc(results_df['target'], results_df['fasttext'])

ft_final_bias = compute_final_bias(ft_bias_metrics, overall_auc)

display(ft_bias_metrics)

print(f'Final Metric: {ft_final_bias}')
glove_matrix = build_matrix(tokenizer.word_index, GLOVE_PATH)

# %time glove_preds = train_model(glove_matrix, 'glove')
# results_df = pd.concat([pd.DataFrame(glove_preds, columns=['glove']).reset_index(drop=True), results_df], axis=1)

gl_bias_metrics = compute_subgroup_bias_metrics(results_df, IDENTITY_COLUMNS, 'glove', 'target')

overall_auc = auc(results_df['target'], results_df['glove'])

gl_final_bias = compute_final_bias(gl_bias_metrics, overall_auc)

display(gl_bias_metrics)

print(f'Final Metric: {gl_final_bias}')
numb_matrix = build_matrix(tokenizer.word_index, NUMBERBATCH_PATH)

# %time numb_preds = train_model(numb_matrix, 'numberbatch')
# results_df = pd.concat([pd.DataFrame(numb_preds, columns=['numberbatch']).reset_index(drop=True), results_df], axis=1)

nb_bias_metrics = compute_subgroup_bias_metrics(results_df, IDENTITY_COLUMNS, 'numberbatch', 'target')

overall_auc = auc(results_df['target'], results_df['numberbatch'])

nb_final_bias = compute_final_bias(nb_bias_metrics, overall_auc)

display(nb_bias_metrics)

print(f'Final Metric: {nb_final_bias}')
! pip install bert_embedding

from bert_embedding import BertEmbedding



def get_bert_embed_matrix():

    %%time

    # Total CPU time (my machine): 1d 4h 7min

    vocab = list(tokenizer.word_index.keys())

    embedding_results = bert_embedding(vocab)

    bert_embeddings = {}

    for emb in embedding_results:

        try: 

            bert_embeddings[emb[0][0]] = emb[1][0]

        except:

            pass

    with open('../input/bert.768.pkl', 'wb') as f:

        pickle.dump(bert_embeddings, f)

def scale(x):

    mini = 272

    maxi = 6047

    scale_rng = [10, 55]

    return (scale_rng[1] - scale_rng[0])*((x-mini)/(maxi-mini))+scale_rng[0]

    

trace0 = go.Scatter(

{

        'x': ctrl_bias_metrics['subgroup_auc'], 

        'y': ctrl_bias_metrics['subgroup_size'],

        'legendgroup': 'w/o pretrained',

        'name': 'w/o pretrained', 

        'mode': 'markers', 

        'marker': {

            'color': cl.scales['9']['div']['Spectral'][0],

            'size': [scale(x) for x in ctrl_bias_metrics['subgroup_size']]

        },

        'text': ctrl_bias_metrics['subgroup']

    })



trace1 = go.Scatter(

{

        'x': ft_bias_metrics['subgroup_auc'], 

        'y': ft_bias_metrics['subgroup_size'],

        'legendgroup': 'fasttext',

        'name': 'fasttext', 

        'mode': 'markers', 

        'marker': {

            'color': cl.scales['9']['div']['Spectral'][2],

            'size': [scale(x) for x in ft_bias_metrics['subgroup_size']]

        },

        'text': ft_bias_metrics['subgroup']

    })



trace2 = go.Scatter(

{

        'x': gl_bias_metrics['subgroup_auc'], 

        'y': gl_bias_metrics['subgroup_size'],

        'legendgroup': 'glove',

        'name': 'glove', 

        'mode': 'markers', 

        'marker': {

            'color': cl.scales['9']['div']['Spectral'][4],

            'size': [scale(x) for x in gl_bias_metrics['subgroup_size']]

        },

        'text': gl_bias_metrics['subgroup']

    })



trace3 = go.Scatter(

{

        'x': nb_bias_metrics['subgroup_auc'], 

        'y': nb_bias_metrics['subgroup_size'],

        'legendgroup': 'numberbatch',

        'name': 'numberbatch', 

        'mode': 'markers', 

        'marker': {

            'color': cl.scales['9']['div']['Spectral'][6],

            'size': [scale(x) for x in nb_bias_metrics['subgroup_size']]

        },

        'text': nb_bias_metrics['subgroup']

    })



layout = go.Layout(

    title= 'Subgroup Size vs Subgroup AUC',

    hovermode = 'closest',

    xaxis = dict(

        title='Subgroup AUC'

    ),

    yaxis = dict(

        title='Subgroup Size'

    ),

    showlegend = True

)



fig = go.Figure(data=[trace0, trace1, trace2, trace3], layout=layout)

py.iplot(fig)
def scale(x):

    mini = 272

    maxi = 6047

    scale_rng = [10, 55]

    return (scale_rng[1] - scale_rng[0])*((x-mini)/(maxi-mini))+scale_rng[0]

    

trace0 = go.Scatter(

{

        'x': ctrl_bias_metrics['bnsp_auc'], 

        'y': ctrl_bias_metrics['bpsn_auc'],

        'legendgroup': 'w/o pretrained',

        'name': 'w/o pretrained', 

        'mode': 'markers', 

        'marker': {

            'color': cl.scales['9']['div']['Spectral'][0],

            'size': [scale(x) for x in ctrl_bias_metrics['subgroup_size']]

        },

        'text': ctrl_bias_metrics['subgroup']

    })



trace1 = go.Scatter(

{

        'x': ft_bias_metrics['bnsp_auc'], 

        'y': ft_bias_metrics['bpsn_auc'],

        'legendgroup': 'fasttext',

        'name': 'fasttext', 

        'mode': 'markers', 

        'marker': {

            'color': cl.scales['9']['div']['Spectral'][2],

            'size': [scale(x) for x in ft_bias_metrics['subgroup_size']]

        },

        'text': ft_bias_metrics['subgroup']

    })



trace2 = go.Scatter(

{

        'x': gl_bias_metrics['bnsp_auc'], 

        'y': gl_bias_metrics['bpsn_auc'],

        'legendgroup': 'glove',

        'name': 'glove', 

        'mode': 'markers', 

        'marker': {

            'color': cl.scales['9']['div']['Spectral'][4],

            'size': [scale(x) for x in gl_bias_metrics['subgroup_size']]

        },

        'text': gl_bias_metrics['subgroup']

    })



trace3 = go.Scatter(

{

        'x': nb_bias_metrics['bnsp_auc'], 

        'y': nb_bias_metrics['bpsn_auc'],

        'legendgroup': 'numberbatch',

        'name': 'numberbatch', 

        'mode': 'markers', 

        'marker': {

            'color': cl.scales['9']['div']['Spectral'][6],

            'size': [scale(x) for x in nb_bias_metrics['subgroup_size']]

        },

        'text': nb_bias_metrics['subgroup']

    })





layout = go.Layout(

    title= 'Word Embeddings Comparison',

    hovermode = 'closest',

    xaxis = dict(

        title='BNSP-AUC'

    ),

    yaxis = dict(

        title='BPSN-AUC'

    ),

    showlegend = True

)



fig = go.Figure(data=[trace0, trace1, trace2, trace3], layout=layout)

py.iplot(fig)
vec_files = [GLOVE_PATH, FASTTEXT_PATH]

glft_matrix = np.concatenate(

    [build_matrix(tokenizer.word_index, f) for f in vec_files], axis=-1)



# The 600 dimension embedding takes 3h 36min to run compared to 

# ~3h 3min for the other 300 dimension embedding models

# %time glft_preds = train_model(glft_matrix, 'glove+fasttext')
# results_df = pd.concat([pd.DataFrame(glft_preds, columns=['glove+fast']).reset_index(drop=True), results_df], axis=1)

glft_bias_metrics = compute_subgroup_bias_metrics(results_df, IDENTITY_COLUMNS, 'glove+fast', 'target')

overall_auc = auc(results_df['target'], results_df['glove+fast'])

glft_final_bias = compute_final_bias(glft_bias_metrics, overall_auc)

display(glft_bias_metrics)

print(f'Final Metric: {glft_final_bias}')

del(glft_matrix)
bias_sum = nb_final_bias + gl_final_bias + ft_final_bias

nb_weight = nb_final_bias/bias_sum

gl_weight = gl_final_bias/bias_sum

ft_weight = ft_final_bias/bias_sum



# results_df['weighted'] = results_df.apply(lambda x: (ft_weight*x['fasttext']) + (gl_weight*x['glove']) + (nb_weight*x['numberbatch']), axis=1)

w_bias_metrics = compute_subgroup_bias_metrics(results_df, IDENTITY_COLUMNS, 'weighted', 'target')

overall_auc = auc(results_df['target'], results_df['weighted'])

w_final_bias = compute_final_bias(w_bias_metrics, overall_auc)

display(w_bias_metrics)

print(f'Final Metric: {w_final_bias}')

print(f'Glove Weight: {gl_weight}')

print(f'Fasttext Weight: {ft_weight}')

print(f'Numberbatch Weight: {nb_weight}')
meta_matrix = np.divide(fasttext_matrix + glove_matrix + numb_matrix, 3)

# %time meta_preds = train_model(meta_matrix, 'meta')
# results_df = pd.concat([pd.DataFrame(meta_preds, columns=['meta']).reset_index(drop=True), results_df], axis=1)

m_bias_metrics = compute_subgroup_bias_metrics(results_df, IDENTITY_COLUMNS, 'meta', 'target')

overall_auc = auc(results_df['target'], results_df['meta'])

m_final_bias = compute_final_bias(m_bias_metrics, overall_auc)

display(m_bias_metrics)

print(f'Final Metric: {m_final_bias}')

del (meta_matrix)
meta2_matrix = np.divide(fasttext_matrix + glove_matrix, 2)

# %time meta2_preds = train_model(meta2_matrix, 'meta-glove+fasttext')
# results_df = pd.concat([pd.DataFrame(meta2_preds, columns=['meta-glove+fasttext']).reset_index(drop=True), results_df], axis=1)

m2_bias_metrics = compute_subgroup_bias_metrics(results_df, IDENTITY_COLUMNS, 'meta-glove+fasttext', 'target')

overall_auc = auc(results_df['target'], results_df['meta-glove+fasttext'])

m2_final_bias = compute_final_bias(m2_bias_metrics, overall_auc)

display(m2_bias_metrics)

print(f'Final Metric: {m2_final_bias}')

del (meta2_matrix)
nb_weighted = np.divide(numb_matrix, nb_weight)

gl_weighted = np.divide(glove_matrix, gl_weight)

ft_weighted = np.divide(fasttext_matrix, ft_weight)

meta_weighted_matrix = nb_weighted+gl_weighted+ft_weighted

# %time meta_weighted_preds = train_model(meta_weighted_matrix, 'meta-weighted')
# results_df = pd.concat([pd.DataFrame(meta_weighted_preds, columns=['meta-weighted']).reset_index(drop=True), results_df], axis=1)

mw_bias_metrics = compute_subgroup_bias_metrics(results_df, IDENTITY_COLUMNS, 'meta-weighted', 'target')

overall_auc = auc(results_df['target'], results_df['meta-weighted'])

mw_final_bias = compute_final_bias(mw_bias_metrics, overall_auc)

display(mw_bias_metrics)

print(f'Final Metric: {mw_final_bias}')

del (meta_weighted_matrix)
final_bias = {'w/o pretrained': ctrl_final_bias, 'fasttext': ft_final_bias, 'glove': gl_final_bias, 'numberbatch': nb_final_bias, 'weighted': w_final_bias,'glove+fasttext': glft_final_bias,'meta': m_final_bias, 'meta-glove+fasttext': m2_final_bias, 'meta-weighted': mw_final_bias}

final_bias = pd.DataFrame(data = final_bias, index=['final bias score'])

display(final_bias)
trace5 = go.Scatter(

{

        'x': glft_bias_metrics['bnsp_auc'], 

        'y': glft_bias_metrics['bpsn_auc'],

        'legendgroup': 'glove+fasttext',

        'name': 'glove+fasttext', 

        'mode': 'markers', 

        'marker': {

            'color': cl.scales['9']['div']['Spectral'][1],

            'size': [scale(x) for x in glft_bias_metrics['subgroup_size']]

        },

        'text': glft_bias_metrics['subgroup']

    })



trace6 = go.Scatter(

{

        'x': m_bias_metrics['bnsp_auc'], 

        'y': m_bias_metrics['bpsn_auc'],

        'legendgroup': 'meta-embedding',

        'name': 'meta-embedding', 

        'mode': 'markers', 

        'marker': {

            'color': cl.scales['9']['div']['Spectral'][3],

            'size': [scale(x) for x in m_bias_metrics['subgroup_size']]

        },

        'text': m_bias_metrics['subgroup']

    })



trace7 = go.Scatter(

{

        'x': m2_bias_metrics['bnsp_auc'], 

        'y': m2_bias_metrics['bpsn_auc'],

        'legendgroup': 'meta-glove+fasttext',

        'name': 'meta-glove+fasttext', 

        'mode': 'markers', 

        'marker': {

            'color': cl.scales['9']['div']['Spectral'][5],

            'size': [scale(x) for x in m2_bias_metrics['subgroup_size']]

        },

        'text': m2_bias_metrics['subgroup']

    })



trace8 = go.Scatter(

{

        'x': mw_bias_metrics['bnsp_auc'], 

        'y': mw_bias_metrics['bpsn_auc'],

        'legendgroup': 'meta-weighted',

        'name': 'meta-weighted', 

        'mode': 'markers', 

        'marker': {

            'color': cl.scales['9']['div']['Spectral'][7],

            'size': [scale(x) for x in mw_bias_metrics['subgroup_size']]

        },

        'text': mw_bias_metrics['subgroup']

    })



layout = go.Layout(

    title= 'Concatenated and Meta Word Embeddings Comparison',

    hovermode = 'closest',

    xaxis = dict(

        title='BNSP-AUC'

    ),

    yaxis = dict(

        title='BPSN-AUC'

    ),

    showlegend = True

)



fig = go.Figure(data=[trace5, trace6, trace7, trace8], layout=layout)



py.iplot(fig)
layout = go.Layout(

    title= 'Complete Embeddings Comparison',

    hovermode = 'closest',

    xaxis = dict(

        title='BNSP-AUC'

    ),

    yaxis = dict(

        title='BPSN-AUC'

    ),

    showlegend = True

)



fig = go.Figure(data=[trace0, trace1, trace2, trace3, trace5, trace6, trace7, trace8], layout=layout)



py.iplot(fig)
trace0 = go.Scatter(

{

        'x': final_bias['w/o pretrained'], 

        'y': [0],

        'name': 'w/o pretrained', 

        'mode': 'markers', 

        'marker': {

            'color': cl.scales['9']['div']['Spectral'][0],

            'size': 25

        }

    })

trace1 = go.Scatter(

{

        'x': final_bias['fasttext'], 

        'y': [0],

        'name': 'fasttext', 

        'mode': 'markers', 

        'marker': {

            'color': cl.scales['9']['div']['Spectral'][2],

            'size': 25

        }

    })

trace2 = go.Scatter(

{

        'x': final_bias['glove'], 

        'y': [0],

        'name': 'glove', 

        'mode': 'markers', 

        'marker': {

            'color': cl.scales['9']['div']['Spectral'][4],

            'size': 25

        }

    })

trace3 = go.Scatter(

{

        'x': final_bias['numberbatch'], 

        'y': [0],

        'name': 'numberbatch', 

        'mode': 'markers', 

        'marker': {

            'color': cl.scales['9']['div']['Spectral'][6],

            'size': 25

        }

    })

trace5 = go.Scatter(

{

        'x': final_bias['glove+fasttext'], 

        'y': [0],

        'name': 'glove+fasttext', 

        'mode': 'markers', 

        'marker': {

            'color': cl.scales['9']['div']['Spectral'][1],

            'size': 25

        }

    })

trace6 = go.Scatter(

{

        'x': final_bias['meta'], 

        'y': [0],

        'name': 'meta', 

        'mode': 'markers', 

        'marker': {

            'color': cl.scales['9']['div']['Spectral'][3],

            'size': 25

        }

    })

trace7 = go.Scatter(

{

        'x': final_bias['meta-glove+fasttext'], 

        'y': [0],

        'name': 'meta-glove+fasttext', 

        'mode': 'markers', 

        'marker': {

            'color': cl.scales['9']['div']['Spectral'][5],

            'size': 25

        }

    })

trace8 = go.Scatter(

{

        'x': final_bias['meta-weighted'], 

        'y': [0],

        'name': 'meta-weighted', 

        'mode': 'markers', 

        'marker': {

            'color': cl.scales['9']['div']['Spectral'][7],

            'size': 25

        }

    })



layout = go.Layout(

    title= 'Final Bias Score for All Embeddings',

    hovermode = 'closest',

    xaxis = dict(

        title='Final Bias Score'

    ),

    showlegend = True

)



fig = go.Figure(data=[trace0, trace1, trace2, trace3, trace5, trace6, trace7, trace8], layout=layout)



py.iplot(fig)