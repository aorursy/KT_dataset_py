import os
from time import time

import pandas as pd
import numpy as np

# for cross validation
from sklearn.model_selection import StratifiedKFold

# for neural network model
from keras.models import Model
from keras.layers import Embedding, Dropout
from keras.layers import LSTM, Bidirectional, Dense, Input
from keras.layers import concatenate
from keras import backend as K
from keras.preprocessing.sequence import pad_sequences

import keras

# for fast-text vectorization
import gensim

import warnings
warnings.simplefilter(action='ignore')
# config
DATA_DIR = '../input/tweet-sentiment-extraction'
TRAIN_DATA_FILE = 'train.csv'
TEST_DATA_FILE = 'test.csv'
SUBMISSION_FILE = 'submission.csv'

RANDOM_STATE = 0
! ls ../input/traininglstmcheckpoints2
train_data = pd.read_csv(os.path.join(DATA_DIR, TRAIN_DATA_FILE)).fillna('')
test_data = pd.read_csv(os.path.join(DATA_DIR, TEST_DATA_FILE)).fillna('')
corpus = list(train_data['text']) 
corpus.extend(list(test_data['text']))
corpus.extend(['neutral', 'positive', 'negative'])
DIM = 200  # fast-text vector size
ft = gensim.models.FastText(corpus, size=DIM, sg=1)
# Vocabulary
vocab = set()

for context in corpus:
    vocab.update(context)

id2word = {i+1:word for i, word in enumerate(vocab)}
word2id = {word:i for i, word in id2word.items()}

embeddings = np.zeros((len(vocab) + 1, DIM))

for i in range(1, len(vocab)+1):
    try:
        embeddings[i] = ft[id2word[i]]
    except KeyError:
        embeddings[i] = np.random.rand((DIM))
contexts_train = list(train_data['text'])
questions_train = list(train_data['sentiment'])

starts = []
ends = []
for text, selected_text in zip(train_data['text'], train_data['selected_text']):
  start = text.find(selected_text)
  starts.append(start)
  ends.append(start + len(selected_text))

starts = np.array(starts)  
ends = np.array(ends) 

MAX_LEN = max([len(c) for c in corpus])
MAX_LEN_Q = max([len(c) for c in ['neutral', 'positive', 'negative']])  # anything wrong here?

# Context
contexts_le = [[word2id[word] for word in context] for context in contexts_train]
X_train_context = pad_sequences(contexts_le, MAX_LEN, padding='post')

# Questions
questions_le = [[word2id[word] for word in question] for question in questions_train]
X_train_question = pad_sequences(questions_le, MAX_LEN_Q, padding='post')
contexts_test = list(test_data['text'])
questions_test = list(test_data['sentiment'])

contexts_le_test = [[word2id.get(word, 0) for word in context] for context in contexts_test]
X_test_context = pad_sequences(contexts_le_test, MAX_LEN, padding='post')

questions_le_test = [[word2id.get(word, 0) for word in question] for question in questions_test]
X_test_question = pad_sequences(questions_le_test, MAX_LEN_Q, padding='post')
def create_model():
    """Return a model."""
    vocab_size = len(vocab)+1

    # Input 1 - Tweets 
    context_input = Input(shape=(MAX_LEN, ), name='context_input')
    emb_c = Embedding(input_dim=vocab_size,
                      output_dim=200,
                      weights=[embeddings], 
                      input_length=MAX_LEN,
                      trainable=False)(context_input)

    lstm_out_c = Bidirectional(LSTM(50,  return_sequences=True,))(emb_c)
    drop_1 = Dropout(0.1)(lstm_out_c)

    # Input 2 - Sentiment labels
    ques_input = Input(shape=(MAX_LEN_Q, ), name='ques_input')
    emb_q = Embedding(input_dim=vocab_size,
                      output_dim=200,
                      weights=[embeddings], 
                      input_length=MAX_LEN_Q, 
                      trainable=False)(ques_input)
    lstm_out_q = Bidirectional(LSTM(50,return_sequences=True,) )(emb_q)
    drop_2 = Dropout(0.1)(lstm_out_q)

    # Merge inputs
    merge_layer = concatenate([drop_1, drop_2], axis=1)
    biLSTM_s = Bidirectional(LSTM(10,))(merge_layer)
    biLSTM_e = Bidirectional(LSTM(10,))(merge_layer)

    # Output 1
    softmax_1 = Dense(MAX_LEN,
                      activation='softmax',
                      name='start')(biLSTM_s)

    # Output 2
    softmax_2 = Dense(MAX_LEN,
                      activation='softmax',
                      name='end')(biLSTM_e)

    model = Model(inputs=[context_input, ques_input],
                  outputs=[softmax_1, softmax_2]
                  )

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy']
                  )
    
    return model
model = create_model()
model.summary()
def jaccard(top_selected):
    str1, str2 = top_selected
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    if (len(a) == 0) & (len(b) == 0):
        return 0.5
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))
RESUME = False  # FALSE DURING SUBMISSION
test_starts_pred = np.zeros((X_test_context.shape[0], MAX_LEN))
test_ends_pred = np.zeros((X_test_context.shape[0], MAX_LEN))

jac = []  # container for Jaccard scores per fold

# cp_dir = os.path.join('drive', 'My Drive', 'training_lstm_checkpoints')  # in Colab
cp_dir = '../input/training-lstm-checkpoints'
cp_dir = '../input/traininglstmcheckpoints2'
# Commented out because Kaggle raises OSError: [Errno 30] Read-only file system: '../input/training_lstm_checkpoints'
# try:
#     os.mkdir(cp_dir)
# except FileExistsError:
#     pass

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

t0 = time()
for fold, (train_idx, test_idx) in enumerate(skf.split(train_data['text'], train_data['sentiment'])):
    print('>' * 20)
    print(f'>>> FOLD {fold + 1}')
    print('>' * 20)

    K.clear_session()
    # TO-DO: need to load the latest checkpoint.
    # Is re-loading the model contrary to the idea of k-fold validation?
    if RESUME:
        try:
            del model
        except NameError:
            pass
        model = keras.models.load_model(os.path.join(cp_dir, '0-lstm.ckpt'))
    else:
        model = create_model()

    cp_callback = keras.callbacks.ModelCheckpoint(os.path.join(cp_dir, f'{fold}-lstm.ckpt'),
                                                     monitor='val_loss',
                                                     verbose=1,
                                                     save_best_only=True,
                                                     save_weights_only=False,
                                                     mode='auto',
                                                     period=1
                                                     )

  
    training_data=({'context_input': X_train_context[train_idx],
                    'ques_input': X_train_question[train_idx]}, 
                    {'start': starts[train_idx],
                      'end': ends[train_idx]})

    validation_data=({'context_input': X_train_context[test_idx],
                      'ques_input': X_train_question[test_idx]}, 
                    {'start': starts[test_idx],
                      'end': ends[test_idx]})

    # COMMENT OUT DURING SUBMISSION
    #############################################################
#     model.fit(training_data[0],
#               training_data[1], 
#               batch_size=1024,  # 32
#               epochs=2,
#               verbose=1,
#               shuffle=True,
#               callbacks=[cp_callback],
#               validation_data=(validation_data[0],
#                               validation_data[1]
#                               )
#               )
    #############################################################
    
    print('Loading model...')
    model.load_weights(os.path.join(cp_dir, f'{fold}-lstm.ckpt'))

    print('Predicting validation data...')
    preds = model.predict(validation_data[0], verbose=1)
    starts_pred = preds[0].argmax(axis=1)
    ends_pred = preds[1].argmax(axis=1)

    print('Predicting Kaggle test data...')
    test_data_dict = {'context_input': X_test_context,
                  'ques_input': X_test_question}

    test_preds = model.predict(test_data_dict, verbose=1)
    # test_starts_pred += test_preds[0].argmax(axis=1)/skf.n_splits
    # test_ends_pred += test_preds[1].argmax(axis=1)/skf.n_splits
    test_starts_pred += test_preds[0]/skf.n_splits
    test_ends_pred += test_preds[1]/skf.n_splits

    print('Evaluating...')
    res = pd.DataFrame()
    res['pred_starts'] = starts_pred
    res['pred_ends'] = ends_pred
    
    columns = ['text', 'sentiment', 'selected_text']
    res[columns] = train_data[columns].loc[test_idx].reset_index(drop=True)

    res['pred_select'] = res[['text', 'pred_starts', 'pred_ends']].apply(lambda x: x[0][x[1]:x[2]], axis=1)
    # Handle cases where start >= end
    condition = res['pred_starts'] >= res['pred_ends']
    res.loc[:, 'pred_select'][condition] = res.loc[:, 'text'][condition]

    res['score'] = res[['pred_select', 'selected_text']].apply(jaccard, axis=1)

    print(res.groupby('sentiment')['score'].mean())
    mean_jac = res['score'].mean()
    print(f"Mean score in Fold {fold + 1}: {mean_jac}")

    jac.append(mean_jac)

print(f'Done in {(time() - t0)} seconds')

total_score = np.mean(jac)
print('>' * 10)
print(f'Total Jaccard score for 5 folds = {total_score}')
temp_df = pd.DataFrame()
temp_df['pred_starts'] = test_starts_pred.argmax(axis=1)
temp_df['pred_ends'] = test_ends_pred.argmax(axis=1)

# columns = ['text', 'sentiment']
temp_df['text'] = test_data['text']
print(temp_df.head())
temp_df['selected_text'] = temp_df[[
                          'text', 
                          'pred_starts', 
                          'pred_ends']
                                ].apply(lambda x: x[0][x[1]:x[2]], axis=1)

# Handle cases where start >= end
condition = temp_df['pred_starts'] >= temp_df['pred_ends']
temp_df.loc[:, 'selected_text'][condition] = temp_df.loc[:, 'text'][condition]

submission_df = pd.DataFrame() 
submission_df['textID'] = test_data['textID']
submission_df['selected_text'] = temp_df['selected_text']
submission_df.to_csv(SUBMISSION_FILE, index = False)
print(temp_df['pred_starts'].unique())
temp_df['pred_ends'].unique()
test_data.head()
submission_df.head()
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
%matplotlib inline
plt.hist(test_data['text'].map(len), bins=40)
plt.hist(submission_df['selected_text'].map(len), alpha=0.8, bins=40)
plt.title('Lengths of Tweets -vs- Predicted Substrings in testing data\nNeural Network')
plt.xlabel('Char length')
plt.ylabel('How often')
plt.show()
# plt.hist(res['text'].map(len), bins=40)
plt.hist(res['selected_text'].map(len), alpha=0.8, bins=40)
# plt.title('Lengths of Tweets -vs- Selected Substrings in training data\nNeural Network')
plt.title('Distribution of Lengths of TRUE Substrings in Training Data')
plt.xlabel('Char length')
plt.ylabel('How often')
plt.show()
# plt.hist(res['text'].map(len), bins=40)
plt.hist(res['pred_select'].map(len), alpha=0.8, bins=40)
# plt.title('Lengths of Tweets -vs- Predicted Substrings in training data\nNeural Network')
plt.title('Distribution of Lengths of PREDICTED Substrings in Training Data')
plt.xlabel('Char length')
plt.ylabel('How often')
plt.show()
# plt.hist(test_data['text'].map(len),  alpha=1, bins=40)
plt.hist(submission_df['selected_text'].map(len), alpha=0.8, bins=40)
plt.title('Distribution of Lengths of PREDICTED Substrings in Testing Data')
plt.xlabel('Char length')
plt.ylabel('How often')
plt.show()
