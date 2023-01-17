import numpy as np

import pandas as pd



import warnings

warnings.simplefilter('ignore')



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv("../input/google-quest-challenge/train.csv", index_col='qa_id')

train.shape
test = pd.read_csv("../input/google-quest-challenge/test.csv", index_col='qa_id')

test.shape
target_columns = [

    'question_asker_intent_understanding',

    'question_body_critical',

    'question_conversational',

    'question_expect_short_answer',

    'question_fact_seeking',

    'question_has_commonly_accepted_answer',

    'question_interestingness_others',

    'question_interestingness_self',

    'question_multi_intent',

    'question_not_really_a_question',

    'question_opinion_seeking',

    'question_type_choice',

    'question_type_compare',

    'question_type_consequence',

    'question_type_definition',

    'question_type_entity',

    'question_type_instructions',

    'question_type_procedure',

    'question_type_reason_explanation',

    'question_type_spelling',

    'question_well_written',

    'answer_helpful',

    'answer_level_of_information',

    'answer_plausible',

    'answer_relevance',

    'answer_satisfaction',

    'answer_type_instructions',

    'answer_type_procedure',

    'answer_type_reason_explanation',

    'answer_well_written'

]
y_train = train[target_columns].copy()

x_train = train.drop(target_columns, axis=1)

del train



x_test = test.copy()

del test
x_train.head(1).T
import tensorflow_hub as hub

import tensorflow as tf
# from https://github.com/tensorflow/models/blob/master/official/nlp/bert/tokenization.py



# see https://www.kaggle.com/rtatman/import-functions-from-kaggle-script



from shutil import copyfile





copyfile(src = "../input/tf-bert-tokenization/tokenization.py", dst = "../working/tokenization.py")



from tokenization import *
# from https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1



BERT = '../input/bert-model'



tokenizer = FullTokenizer(BERT + '/assets/vocab.txt', True)
tokenizer.tokenize('Hello world from BERT FullTokenizer!')
# from ??? well known code



def _get_masks(tokens, max_seq_length):

    """Mask for padding"""

    if len(tokens)>max_seq_length:

        raise IndexError("Token length more than max seq length!")

    return [1]*len(tokens) + [0] * (max_seq_length - len(tokens))



def _get_segments(tokens, max_seq_length):

    """Segments: 0 for the first sequence, 1 for the second"""

    if len(tokens)>max_seq_length:

        raise IndexError("Token length more than max seq length!")

    segments = []

    first_sep = True

    current_segment_id = 0

    for token in tokens:

        segments.append(current_segment_id)

        if token == "[SEP]":

            if first_sep:

                first_sep = False

            else:

                current_segment_id = 1

    return segments + [0] * (max_seq_length - len(tokens))



def _get_ids(tokens, tokenizer, max_seq_length):

    """Token ids from Tokenizer vocab"""

    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    input_ids = token_ids + [0] * (max_seq_length-len(token_ids))

    return input_ids
import math





def trim_tokens(t, q, a, max_t, max_q, max_a):

    if (len(t) + len(q) + len(a)) > (max_t + max_q + max_a):

        

        _max_t = max_t

        _max_q = max_q

        _max_a = max_a

        

        if len(t) > _max_t:

            t = t[:_max_t]

        else:

            x = (_max_t - len(t)) / 2.

            _max_q += math.ceil(x)

            _max_a += math.floor(x)

        

        if len(q) > _max_q:

            q = q[:_max_q]

        else:

            _max_a += (_max_q - len(q))

        

        if len(a) > _max_a:

            a = a[:_max_a]

    

    return t,q,a





def make_bert_input(x, max_question_title_length=24, max_question_body_length=242, max_answer_length=242):

    

    max_sequence_length = max_question_title_length + max_question_body_length + max_answer_length + 4

    print('Calculated max sequence length:', max_sequence_length)

    

    input_ids = []

    input_masks = []

    input_segments = []

    

    for idx, row in x[['question_title', 'question_body', 'answer']].iterrows():

        

        # get tokens

        t,q,a = trim_tokens(tokenizer.tokenize(row.question_title),

                            tokenizer.tokenize(row.question_body),

                            tokenizer.tokenize(row.answer),

                            max_question_title_length,

                            max_question_body_length,

                            max_answer_length)

        

        tokens = ['[CLS]'] + t + ['[SEP]'] + q + ['[SEP]'] + a + ['[SEP]']

        # print(tokens)

        

        input_ids.append(_get_ids(tokens, tokenizer, max_sequence_length))

        input_masks.append(_get_masks(tokens, max_sequence_length))

        input_segments.append(_get_segments(tokens, max_sequence_length))

    

    return [

        np.asarray(input_ids, dtype=np.int32),

        np.asarray(input_masks, dtype=np.int32),

        np.asarray(input_segments, dtype=np.int32)

    ]
max_sequence_length = 512



# bert_input_train = make_bert_input(x_train)
def make_model():

    

    input_word_ids = tf.keras.layers.Input(shape=(max_sequence_length,), dtype=tf.int32,

                                           name="input_word_ids")

    input_mask = tf.keras.layers.Input(shape=(max_sequence_length,), dtype=tf.int32,

                                       name="input_mask")

    segment_ids = tf.keras.layers.Input(shape=(max_sequence_length,), dtype=tf.int32,

                                        name="segment_ids")

    

    bert_layer = hub.KerasLayer(BERT, trainable=True)

    

    pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])

    

    tmp = tf.keras.layers.GlobalAveragePooling1D()(sequence_output)

    tmp = tf.keras.layers.Dropout(0.2)(tmp)

    out = tf.keras.layers.Dense(len(target_columns), activation='sigmoid', name='dense_output')(tmp)

    

    model = tf.keras.models.Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)

    

    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5))

    

    return model
from scipy.stats import spearmanr





def mean_spearmanr_correlation_score(y_true, y_pred):

    return np.mean([spearmanr(y_pred[:, idx] + np.random.normal(0, 1e-7, y_pred.shape[0]),

                              y_true[:, idx]).correlation for idx in range(len(target_columns))])
trained_estimators = []
from sklearn.model_selection import KFold

import tensorflow.keras.backend as K

import math





n_splits = 5



scores = []



cv = KFold(n_splits=n_splits, random_state=42)

idx = 1

for train_idx, valid_idx in cv.split(x_train, y_train, groups=x_train.question_body):

    

    x_train_train = x_train.iloc[train_idx]

    y_train_train = y_train.iloc[train_idx]

    x_train_valid = x_train.iloc[valid_idx]

    y_train_valid = y_train.iloc[valid_idx]

    

    K.clear_session()

    

    estimator = make_model()

    estimator.fit(make_bert_input(x_train_train), y_train_train, batch_size=8, epochs=5)

    trained_estimators.append(estimator)

    

    oof_part = estimator.predict(make_bert_input(x_train_valid))

    score = mean_spearmanr_correlation_score(y_train_valid.values, oof_part)

    print('Score:', score)

    

    if not math.isnan(score):

        # trained_estimators.append(estimator)

        scores.append(score)

    

    # limit number of iterations to complete job within 2 hours

    idx += 1

    if idx > 3:

        break





print('Mean score:', np.mean(scores))
len(trained_estimators)
y_pred = []

for estimator in trained_estimators:

    y_pred.append(estimator.predict(make_bert_input(x_test)))
from scipy.stats import rankdata





def blend_by_ranking(data, weights):

    out = np.zeros(data.shape[0])

    for idx,column in enumerate(data.columns):

        out += weights[idx] * rankdata(data[column].values)

    out /= np.max(out)

    return out
submission = pd.read_csv("../input/google-quest-challenge/sample_submission.csv", index_col='qa_id')



out = pd.DataFrame(index=submission.index)

for column_idx,column in enumerate(target_columns):

    

    # collect all predictions for one column

    column_data = pd.DataFrame(index=submission.index)

    for prediction_idx,prediction in enumerate(y_pred):

        column_data[str(prediction_idx)] = prediction[:, column_idx]

    

    out[column] = blend_by_ranking(column_data, np.ones(column_data.shape[1]))
out.head()
out.to_csv("submission.csv")