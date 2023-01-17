import numpy as np 

import pandas as pd 

import os

from sklearn import metrics

from tqdm import tqdm

tqdm.pandas()
TEXT_COL = 'comment_text'

EMB_PATH = '../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec'

train = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv', index_col='id')

test = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv', index_col='id')
# Convert taget and identity columns to booleans. This is need to calculate the competition metric (compare to 'subgroup_size': len(dataset[dataset[subgroup]])) within the competition metric

identity_columns = [

    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',

    'muslim', 'black', 'white', 'psychiatric_or_mental_illness']

def convert_to_bool(df, col_name):

    df[col_name] = np.where(df[col_name] >= 0.5, True, False)

    

def convert_dataframe_to_bool(df):

    bool_df = df.copy()

    for col in ['target'] + identity_columns:

        convert_to_bool(bool_df, col)

    return bool_df



train = convert_dataframe_to_bool(train)
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')



def load_embeddings(embed_dir=EMB_PATH):

    embedding_index = dict(get_coefs(*o.strip().split(" ")) for o in tqdm(open(embed_dir)))

    return embedding_index



def build_embedding_matrix(word_index, embeddings_index, max_features, lower = True, verbose = True):

    embedding_matrix = np.zeros((max_features, 300))

    for word, i in tqdm(word_index.items(),disable = not verbose):

        if lower:

            word = word.lower()

        if i >= max_features: continue

        try:

            embedding_vector = embeddings_index[word]

        except:

            embedding_vector = embeddings_index["unknown"]

        if embedding_vector is not None:

            # words not found in embedding index will be all-zeros.

            embedding_matrix[i] = embedding_vector

    return embedding_matrix



def build_matrix(word_index, embeddings_index):

    embedding_matrix = np.zeros((len(word_index) + 1,300))

    for word, i in word_index.items():

        try:

            embedding_matrix[i] = embeddings_index[word]

        except:

            embedding_matrix[i] = embeddings_index["unknown"]

    return embedding_matrix
from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

import gc



maxlen = 220

max_features = 100000

embed_size = 300

tokenizer = Tokenizer(num_words=max_features, lower=True) #filters = ''

#tokenizer = text.Tokenizer(num_words=max_features)

print('fitting tokenizer')

tokenizer.fit_on_texts(list(train[TEXT_COL]) + list(test[TEXT_COL]))

word_index = tokenizer.word_index

X_train = tokenizer.texts_to_sequences(list(train[TEXT_COL]))

y_train = train['target'].values

X_test = tokenizer.texts_to_sequences(list(test[TEXT_COL]))



X_train = pad_sequences(X_train, maxlen=maxlen)

X_test = pad_sequences(X_test, maxlen=maxlen)





del tokenizer

gc.collect()





embeddings_index = load_embeddings()
embedding_matrix = build_matrix(word_index, embeddings_index)
del embeddings_index

gc.collect()
from keras import backend as K

from keras.engine.topology import Layer

from keras import initializers, regularizers, constraints, optimizers, layers



class Attention(Layer):

    def __init__(self, step_dim,

                 W_regularizer=None, b_regularizer=None,

                 W_constraint=None, b_constraint=None,

                 bias=True, **kwargs):

        self.supports_masking = True

        self.init = initializers.get('glorot_uniform')



        self.W_regularizer = regularizers.get(W_regularizer)

        self.b_regularizer = regularizers.get(b_regularizer)



        self.W_constraint = constraints.get(W_constraint)

        self.b_constraint = constraints.get(b_constraint)



        self.bias = bias

        self.step_dim = step_dim

        self.features_dim = 0

        super(Attention, self).__init__(**kwargs)



    def build(self, input_shape):

        assert len(input_shape) == 3



        self.W = self.add_weight((input_shape[-1],),

                                 initializer=self.init,

                                 name='{}_W'.format(self.name),

                                 regularizer=self.W_regularizer,

                                 constraint=self.W_constraint)

        self.features_dim = input_shape[-1]



        if self.bias:

            self.b = self.add_weight((input_shape[1],),

                                     initializer='zero',

                                     name='{}_b'.format(self.name),

                                     regularizer=self.b_regularizer,

                                     constraint=self.b_constraint)

        else:

            self.b = None



        self.built = True



    def compute_mask(self, input, input_mask=None):

        return None



    def call(self, x, mask=None):

        features_dim = self.features_dim

        step_dim = self.step_dim



        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),

                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))



        if self.bias:

            eij += self.b



        eij = K.tanh(eij)



        a = K.exp(eij)



        if mask is not None:

            a *= K.cast(mask, K.floatx())



        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())



        a = K.expand_dims(a)

        weighted_input = x * a

        return K.sum(weighted_input, axis=1)



    def compute_output_shape(self, input_shape):

        return input_shape[0],  self.features_dim
import keras.layers as L

from keras.models import Model

from keras.optimizers import Adam



def build_model(verbose = False, compile = True):

    sequence_input = L.Input(shape=(maxlen,), dtype='int32')

    embedding_layer = L.Embedding(len(word_index) + 1,

                                300,

                                weights=[embedding_matrix],

                                input_length=maxlen,

                                trainable=False)

    x = embedding_layer(sequence_input)

    x = L.SpatialDropout1D(0.2)(x)

    x = L.Bidirectional(L.CuDNNLSTM(64, return_sequences=True))(x)



    att = Attention(maxlen)(x)

    avg_pool1 = L.GlobalAveragePooling1D()(x)

    max_pool1 = L.GlobalMaxPooling1D()(x)



    x = L.concatenate([att,avg_pool1, max_pool1])



    preds = L.Dense(1, activation='sigmoid')(x)





    model = Model(sequence_input, preds)

    if verbose:

        model.summary()

    if compile:

        model.compile(loss='binary_crossentropy',optimizer=Adam(0.005),metrics=['acc'])

    return model

SUBGROUP_AUC = 'subgroup_auc'

BPSN_AUC = 'bpsn_auc'  # stands for background positive, subgroup negative

BNSP_AUC = 'bnsp_auc'  # stands for background negative, subgroup positive

TOXICITY_COLUMN = 'target'



def compute_auc(y_true, y_pred):

    try:

        return metrics.roc_auc_score(y_true, y_pred)

    except ValueError:

        return np.nan



def compute_subgroup_auc(df, subgroup, label, model_name):

    subgroup_examples = df[df[subgroup]]

    return compute_auc(subgroup_examples[label], subgroup_examples[model_name])



def compute_bpsn_auc(df, subgroup, label, model_name):

    """Computes the AUC of the within-subgroup negative examples and the background positive examples."""

    subgroup_negative_examples = df[df[subgroup] & ~df[label]]

    non_subgroup_positive_examples = df[~df[subgroup] & df[label]]

    examples = subgroup_negative_examples.append(non_subgroup_positive_examples)

    return compute_auc(examples[label], examples[model_name])



def compute_bnsp_auc(df, subgroup, label, model_name):

    """Computes the AUC of the within-subgroup positive examples and the background negative examples."""

    subgroup_positive_examples = df[df[subgroup] & df[label]]

    non_subgroup_negative_examples = df[~df[subgroup] & ~df[label]]

    examples = subgroup_positive_examples.append(non_subgroup_negative_examples)

    return compute_auc(examples[label], examples[model_name])



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
def calculate_overall_auc(df, model_name):

    true_labels = df[TOXICITY_COLUMN]

    predicted_labels = df[model_name]

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
from keras.callbacks import EarlyStopping



#Diasbale SettingWithCopyWarning of pandas, see https://stackoverflow.com/questions/20625582/how-to-deal-with-settingwithcopywarning-in-pandas

pd.options.mode.chained_assignment = None  # default='warn'



class CompetitionMetricEarlyStopping(EarlyStopping):

    

    def __init__(self, model, df, X_valid, *args, **kwargs):

        kwargs['monitor'] = 'val_competition_metric'

        

        self.model = model

        self.df = df

        self.X_valid = X_valid

        self.bias_metrics_df = None

        super().__init__(*args, **kwargs)

            

    def get_monitor_value(self, logs):

        preds = self.model.predict(self.X_valid, batch_size=2048)[:, 0]

        self.df['predictions'] = preds

        bias_metrics_df = compute_bias_metrics_for_model(self.df, identity_columns, 'predictions', TOXICITY_COLUMN)

        self.bias_metrics_df = bias_metrics_df

        val_competition_metric = get_final_metric(bias_metrics_df, calculate_overall_auc(self.df, 'predictions'))

        logs['val_competition_metric'] = val_competition_metric

        return val_competition_metric
from sklearn.model_selection import KFold



splits = list(KFold(n_splits=5).split(X_train,y_train))





from keras.callbacks import EarlyStopping, ModelCheckpoint

import keras.backend as K

import numpy as np

BATCH_SIZE = 2048

NUM_EPOCHS = 100



oof_preds = np.zeros((X_train.shape[0]))

test_preds = np.zeros((X_test.shape[0]))

for fold in [0,1,2,3,4]:

    K.clear_session()

    tr_ind, val_ind = splits[fold]

    model = build_model()

    

    es = CompetitionMetricEarlyStopping(model, train.iloc[val_ind], X_train[val_ind],

                                                   restore_best_weights=False, 

                                                   mode='max',

                                                   verbose=1,

                                                   patience=2)

    # CompetitionMetricEarlyStopping writes the val_competition_metric into the logs (logs is a mutable object and hence is passed by name). Thus, logs['val_competition_metric'] can be accesed, if 

    # CompetitionMetricEarlyStopping.on_epoch_end is executed before ModelCheckpoint.on_epoch_end

    ckpt = ModelCheckpoint(f'gru_{fold}.hdf5', save_best_only = True, monitor='val_competition_metric', mode='max', verbose=1)



    model.fit(X_train[tr_ind],

        y_train[tr_ind]>0.5,

        batch_size=BATCH_SIZE,

        epochs=NUM_EPOCHS,

        validation_data=(X_train[val_ind], y_train[val_ind]>0.5), #remove validation data, if you want to improve the runtime of the kernel 

        callbacks = [es, ckpt])

    model.load_weights(f'gru_{fold}.hdf5')

    print(es.bias_metrics_df)

    oof_preds[val_ind] += model.predict(X_train[val_ind])[:,0]

    test_preds += model.predict(X_test)[:,0]

test_preds /= 5
train['predictions'] = oof_preds

bias_metrics_df = compute_bias_metrics_for_model(train, identity_columns, 'predictions', TOXICITY_COLUMN)

print(bias_metrics_df)

off_competition_metric = get_final_metric(bias_metrics_df, calculate_overall_auc(train, 'predictions')); off_competition_metric
from sklearn.metrics import roc_auc_score

roc_auc_score(y_train>0.5,oof_preds)
submission = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv', index_col='id')

submission['prediction'] = test_preds

submission.reset_index(drop=False, inplace=True)

submission.head()

#%%
submission.to_csv('submission.csv', index=False)