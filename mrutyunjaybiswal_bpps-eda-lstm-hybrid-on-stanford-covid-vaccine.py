import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

%matplotlib inline
train = pd.read_json('../input/stanford-covid-vaccine/train.json', lines=True)
test = pd.read_json('../input/stanford-covid-vaccine/test.json', lines=True)
submission = pd.read_csv('../input/stanford-covid-vaccine/sample_submission.csv')
train.head()
test.head()
submission.head()
train.columns
test.columns
set(train.columns).difference(test.columns).difference(submission.columns)
# let's do some cleaning and understand our data more clearly
train = train.drop(['index'], axis=1)
test = test.drop(['index'], axis=1)
train.info()
print(train['sequence'].apply(lambda x: len(x)).value_counts())  # all the sequences have 107 bases
print(train['structure'].apply(lambda x: len(x)).value_counts())  # all the structures have 107 bases
print(train['predicted_loop_type'].apply(lambda x: len(x)).value_counts())  # all the structures have 107 bases
train.head()
test.head()
3005 * 130 + 629 * 107
train['seq_counts'] = train['sequence'].apply(lambda x: Counter(x.upper()))
train['seq_counts']
train['seq_counts'].apply(lambda x: (x.keys(), x.values()))
# doing a bit of feature engieering by taking up the contribution of each code
percentage = []
for i in range(len(train)):
  count = train.iloc[i]['seq_counts']
  percentage.append((count['A']/train.iloc[i]['seq_length'],
                     count['G']/train.iloc[i]['seq_length'],
                     count['C']/train.iloc[i]['seq_length'],
                     count['U']/train.iloc[i]['seq_length']))
  
percentage = pd.DataFrame(percentage, columns=['A_p', 'G_p', 'C_p', 'U_p'])
percentage
pairs = []
all_partners = []
for j in range(len(train)):
    partners = [-1 for i in range(130)]
    pairs_dict = {}
    queue = []
    for i in range(0, len(train.iloc[j]['structure'])):
        if train.iloc[j]['structure'][i] == '(':
            queue.append(i)
        if train.iloc[j]['structure'][i] == ')':
            first = queue.pop()
            try:
                pairs_dict[(train.iloc[j]['sequence'][first], train.iloc[j]['sequence'][i])] += 1
            except:
                pairs_dict[(train.iloc[j]['sequence'][first], train.iloc[j]['sequence'][i])] = 1
                
            partners[first] = i
            partners[i] = first
    
    all_partners.append(partners)
    
    pairs_num = 0
    pairs_unique = [('U', 'G'), ('C', 'G'), ('U', 'A'), ('G', 'C'), ('A', 'U'), ('G', 'U')]
    for item in pairs_dict:
        pairs_num += pairs_dict[item]
    add_tuple = list()
    for item in pairs_unique:
        try:
            add_tuple.append(pairs_dict[item]/pairs_num)
        except:
            add_tuple.append(0)
    pairs.append(add_tuple)
    
pairs = pd.DataFrame(pairs, columns=['U-G', 'C-G', 'U-A', 'G-C', 'A-U', 'G-U'])
pairs
pairs_rate = []

for j in range(len(train)):
    res = dict(Counter(train.iloc[j]['structure']))
    pairs_rate.append(res['('] / 53.5)  # 2 * res['(']/107
    
pairs_rate = pd.DataFrame(pairs_rate, columns=['pairs_rate'])
pairs_rate
loops = []
for j in range(len(train)):
    counts = dict(Counter(train.iloc[j]['predicted_loop_type']))
    available = ['E', 'S', 'H', 'B', 'X', 'I', 'M']
    row = []
    for item in available:
        try:
            row.append(counts[item] / 107)
        except:
            row.append(0)
    loops.append(row)
    
loops = pd.DataFrame(loops, columns=available)
loops
bbps_dir = '../input/stanford-covid-vaccine/bpps'

bbps_fns = os.listdir(bbps_dir)
len(train) + len(test) == len(bbps_fns)
def get_bppm(id_):
    return np.load(os.path.join(bbps_dir, bbps_fns[id_]))


def draw_structure(structure: str):
    pm = np.zeros((len(structure), len(structure)))
    start_token_indices = []
    for i, token in enumerate(structure):
        if token == "(":
            start_token_indices.append(i)
        elif token == ")":
            j = start_token_indices.pop()
            pm[i, j] = 1.0
            pm[j, i] = 1.0
    return pm


def plot_structures(bppm: np.ndarray, pm: np.ndarray):
    fig, axes = plt.subplots(1, 2, figsize=(10, 10))
    axes[0].imshow(bppm)
    axes[0].set_title("BPPM")
    axes[1].imshow(pm)
    axes[1].set_title("structure")
    plt.show()
for _ in range(5):
  idx = np.random.randint(len(bbps_fns))
  fn = bbps_fns[idx]
  df_id = fn.split('.')[0]

  print(fn)
  bbps_ff = get_bppm(idx)
  struct = train[train['id']==df_id]['structure'].values[0] if df_id in train['id'].to_list() else test[test['id']==df_id]['structure'].values[0]
  plot_struct = draw_structure(struct)
  plot_structures(bbps_ff, plot_struct)
target_cols = submission.columns.to_list()[1:]
for col in target_cols:
  print(train[col].apply(lambda x: len(x)).sum()/len(train))

# prediction sequence lenght is 68
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import layers as L
from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold
def tokentoInt(bases):
  return {x:i for i, x in enumerate(bases)}
  pass

print(tokentoInt("".join([x for x in loops.columns])))
def gru_layer(hidden_dim, dropout):
    return L.Bidirectional(L.GRU(hidden_dim, dropout=dropout, return_sequences=True, kernel_initializer='orthogonal'))

def lstm_layer(hidden_dim, dropout):
    return L.Bidirectional(L.LSTM(hidden_dim, dropout=dropout, return_sequences=True, kernel_initializer='orthogonal'))
# source : https://www.kaggle.com/c/stanford-covid-vaccine/discussion/183211
def MCRMSE(y_true, y_pred):
    colwise_mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=1)
    return tf.reduce_mean(tf.sqrt(colwise_mse), axis=1)
def encoding(df, col):
  """
  df: dataframe containing sequences and the features
  col: column to apply encoding
      : valid values are: 'sequence', 'structure' and 'predicted_loop_type'
  """
  try:
    if col == 'sequence':
      seq_encoding = tokentoInt('AGCU')
      
    elif col == 'structure':
      seq_encoding = tokentoInt('(.)')

    elif col == 'predicted_loop_type':
      seq_encoding = tokentoInt("".join([x for x in loops.columns]))

    return np.array(df[col].apply(lambda seq: [seq_encoding[x] for x in seq]).values.tolist())

  except KeyError:
    print('Invalid arguments as col')
from tqdm import tqdm
private_test = test.query("seq_length==130").copy()
public_test = test.query("seq_length==107").copy()

# this split on train set is applied if none of the cv folding aren't applied
train_data = train.query('SN_filter==0')
val_data = train.query('SN_filter==1')
def get_features(df):
  seq_inp = encoding(df, 'sequence')
  struc_inp = encoding(df, 'structure')
  plt_inp = encoding(df, 'predicted_loop_type')
  '''
  bpps_arr = []
  for i in tqdm(range(len(df))):
    idx = df.loc[i]['id']
    bpps_arr.append(np.expand_dims(np.load(os.path.join(bbps_dir, str(idx)+'.npy')), axis=-1))

  cnn_inp = np.array(bpps_arr) # cnn data input
  '''
  return seq_inp, struc_inp, plt_inp #, cnn_inp
train_labels = np.array(train[target_cols].values.tolist()).transpose(0, 2, 1)
train_labels[0, 0, :]
def seq_model(encoding_dict,
              seq_len=107,
              pred_len=68,
              dropout=0.4,
              sp_dropout=0.2,
              embed_size=128,
              hidden_dim=256,
              layers=2,
              gru=False):
  
  # one sequence at a time of len 107 (if training specified)
  input = L.Input(shape=(seq_len, ))

  # apply embedding layer
  embed = L.Embedding(input_dim=len(encoding_dict),
                      output_dim=embed_size)(input)

  '''reshaped = tf.reshape(embed,
                        shape=(-1, embed.shape[1], embed.shape[2] * embed.shape[3]))'''
  hidden = tf.keras.layers.SpatialDropout1D(sp_dropout)(embed)
  # apply bidirectional lstm/gru layers * layers count
  if gru:
    for _ in range(layers):
      hidden = gru_layer(hidden_dim, dropout)(hidden)
  else:
    for _ in range(layers):
      hidden = gru_layer(hidden_dim, dropout)(hidden)
  
  return tf.keras.Model(input, hidden)
  pass
def cnn_model(input_shape=(107, 107), flag=False):
  """
  can be of shape 107*107(train and public set) and 130*130(private set) 
  """
  input = L.Input(shape=(*input_shape, 1))  # images are of 2-D

  # let's just go with 3 layers of CNN
  x = L.Conv2D(kernel_size=(5, 5),
               filters=64,
               strides=(2, 2))(input)
  x = L.MaxPool2D(pool_size=(2, 2))(x)
  x = L.Activation('relu')(x)

  x = L.Conv2D(kernel_size=(3, 3),
               filters=256)(x)
  x = L.MaxPool2D(pool_size=(2, 2))(x)
  x = L.Activation('relu')(x)
  
  x = L.Conv2D(kernel_size=(1, 4), filters=512)(x)
  x = L.Activation('relu')(x)
  
  if flag:
    x = L.Conv2D(kernel_size=(2, 2), filters=512)(x)
    x = L.Activation('relu')(x)
    x = tf.reshape(x, shape=(-1, x.shape[1]*x.shape[2], x.shape[-1], 1))
    return tf.keras.Model(input, x)

  x = tf.reshape(x, shape=(-1, x.shape[1]*x.shape[2], x.shape[-1], 1))
  x = L.Conv2D(kernel_size=(2, 1), filters=1)(x)
  x = L.Activation('relu')(x)




  return tf.keras.Model(input, x)
  pass
def main_model(seq_len=107, pred_len=68, cnn_input_shape=(107, 107), flag=False):
  """
  Consists of four models, one seq_model each for sequence, structure and predicted_loop
  and one CNN for BPPS files.
  """
  # extract from sequences
  Seq_model = seq_model(tokentoInt('AGCU'), seq_len=seq_len, pred_len=pred_len, dropout=0.0)
  Seq_op = Seq_model.output  # for train,  seq_len = 107

  Struct_model = seq_model(tokentoInt('(.)'), seq_len=seq_len, pred_len=pred_len, dropout=0.0)
  Struct_op = Struct_model.output

  PLT_model = seq_model(tokentoInt("".join([x for x in loops.columns])), seq_len=seq_len, pred_len=pred_len, dropout=0.0)
  PLT_op = PLT_model.output
  '''
  # add cnn layer output
  CNN_model = cnn_model(cnn_input_shape, flag=flag)
  CNN_op = CNN_model.output
  CNN_op = tf.reshape(CNN_op, shape=(-1, CNN_op.shape[1], CNN_op.shape[2] * CNN_op.shape[3]))
  
  print(Seq_op.shape, Struct_op.shape, PLT_op.shape, CNN_op.shape)
  '''
  # now we got 4 tensors of shape (BS, 107, 512)
  ip = tf.add_n([Seq_op, Struct_op, PLT_op])/3
  print(ip.shape)
  ip = ip[:, :pred_len]
  ip = L.Dense(5, activation='linear')(ip)
  print(ip.shape)
  return tf.keras.Model(inputs=[Seq_model.input, Struct_model.input, PLT_model.input], outputs=ip)
  pass
model = main_model()
model.summary()
model.compile(loss=MCRMSE,
           optimizer=tf.keras.optimizers.Adam(lr=0.001))
lr_callback = tf.keras.callbacks.ReduceLROnPlateau(patience=5)
sv_lstm = tf.keras.callbacks.ModelCheckpoint(f'lstm.h5')
model.fit(get_features(train), train_labels,
       epochs=75, batch_size=64,
       callbacks=[lr_callback, sv_lstm])
model_long = main_model(seq_len=130, pred_len=130, cnn_input_shape=(130, 130), flag=True)
model_long.summary()
model_long.load_weights('./lstm.h5')
pred_long = model_long.predict(get_features(private_test), verbose=1)
pred_long.shape
model_short = main_model(seq_len=107, pred_len=107, cnn_input_shape=(107, 107), flag=True)
model_short.summary()
model_short.load_weights('./lstm.h5')
pred_short = model_short.predict(get_features(public_test), verbose=1)
pred_short.shape
def format_predictions(public_preds, private_preds):
    preds = []
    
    for df, preds_ in [(public_test, public_preds), (private_test, private_preds)]:
        for i, uid in enumerate(df.id):
            single_pred = preds_[i]

            single_df = pd.DataFrame(single_pred, columns=target_cols)
            single_df['id_seqpos'] = [f'{uid}_{x}' for x in range(single_df.shape[0])]

            preds.append(single_df)

    return pd.concat(preds).groupby('id_seqpos')
df = format_predictions(pred_short, pred_long)
df.first()
submission = df.sum().reset_index()
submission
submission.to_csv('submission.csv', index=False)
