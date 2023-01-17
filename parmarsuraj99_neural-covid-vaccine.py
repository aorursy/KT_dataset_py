import gc



import numpy as np

import pandas as pd



import tensorflow as tf

from tensorflow.keras import layers as L

from tensorflow.keras.models import Model



from sklearn.preprocessing import LabelEncoder
train_df = pd.read_json('/kaggle/input/stanford-covid-vaccine/train.json', lines=True)

test_df = pd.read_json('/kaggle/input/stanford-covid-vaccine/test.json', lines=True)

sample_df = pd.read_csv('/kaggle/input/stanford-covid-vaccine/sample_submission.csv')
train_df.head()
sample_df.head()
train_df.columns
train_df["sequence"].str.split("").apply(lambda x: (np.unique(x), len(x)))
np.unique(train_df["seq_length"].values)
feature_columns = ['sequence', 'structure', 'predicted_loop_type']

target_columns = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']
train_df.head(3)
label_encoders = dict()

for column in feature_columns:

    encoder = LabelEncoder()

    encoder.fit(list(set(train_df[column].apply(list).sum())))

    label_encoders[column]=encoder

    del encoder

    gc.collect()

    
def transform_(df:pd.DataFrame, label_encoders:dict):

    for column in feature_columns:

        df[column+"_n"] = df[column].apply(lambda seq: label_encoders[column].transform(list(seq)))        

    return df
train_df = transform_(train_df, label_encoders)

test_df = transform_(test_df, label_encoders)



feature_columns_n = [column for column in train_df.columns if "e_n" in column];

feature_columns_n



train_x = np.array(train_df[feature_columns_n].values.tolist()).transpose((0, 2, 1))

train_y = np.array(train_df[target_columns].values.tolist()).transpose((0, 2, 1))
def build_model(seq_len=107, pred_len=68, dropout=0.5, embed_dim=100, hidden_dim=128):

    

    inputs = L.Input(shape = (seq_len, 3))

    

    inputs_as = L.Lambda(lambda x: tf.split(x, inputs.shape[-1], axis=-1))(inputs)

    #inputs_as = L.Lambda(lambda x: [tf.expand_dims(x[:, :, i], -1) for i in range(inputs.shape[-1])])(inputs)

    print(len(inputs_as), inputs_as[0].shape)

    

    

    embeddings = []

    for i, (inp_a, col)in enumerate(zip(inputs_as, feature_columns)):

        

        embedding = L.Embedding(input_dim = len(label_encoders[col].classes_),

                               output_dim = embed_dim)(inp_a)

        embedding = L.Reshape((-1, embedding.shape[2] * embedding.shape[3]))(embedding)

        

        embeddings.append(embedding)

        

    embed_cat = L.Concatenate()(embeddings)

    

    """

    lstm1 = L.Bidirectional(L.LSTM(hidden_dim, dropout=dropout, return_sequences=True))(embeddings[0])

    #lstm1 = L.Bidirectional(L.LSTM(hidden_dim//2, dropout=dropout, return_sequences=True))(lstm1)

    

    lstm2 = L.Bidirectional(L.LSTM(hidden_dim, dropout=dropout, return_sequences=True))(embeddings[1])

    #lstm2 = L.Bidirectional(L.LSTM(hidden_dim//2, dropout=dropout, return_sequences=True))(lstm2)

    

    lstm3 = L.Bidirectional(L.LSTM(hidden_dim, dropout=dropout, return_sequences=True))(embeddings[2])

    #lstm3 = L.Bidirectional(L.LSTM(hidden_dim//2, dropout=dropout, return_sequences=True))(lstm3)

    

    cat = L.Add()([lstm1, lstm2, lstm3])

    """

    cat = L.Bidirectional(L.LSTM(hidden_dim, dropout=dropout, return_sequences=True))(embed_cat)

    """

    lstm2 = L.Bidirectional(L.LSTM(hidden_dim, dropout=dropout, return_sequences=True))(lstm1)

    lstm3 = L.Bidirectional(L.LSTM(hidden_dim, dropout=dropout, return_sequences=True))(lstm2)

    """

    

    cat = cat[:, :pred_len]

    

    cat = L.Dense(64, activation="relu")(cat)

    dense = L.Dense(5, activation='linear')(cat)

    

    model = Model(inputs=inputs, outputs = dense)

    model.compile(loss="mse", optimizer='adam')

        

    return model

        
model = build_model()

model.summary()
tf.keras.utils.plot_model(model, show_shapes=True)
model.fit(train_x, train_y, 

          batch_size=64,

          epochs=80,

          callbacks=[

              tf.keras.callbacks.ReduceLROnPlateau(),

              #tf.keras.callbacks.ModelCheckpoint('model.h5')

          ],

          validation_split=0.01)
public_df = test_df.query("seq_length == 107").copy()

private_df = test_df.query("seq_length == 130").copy()



public_df = transform_(public_df, label_encoders)

private_df = transform_(private_df, label_encoders)



public_test_x = np.array(public_df[feature_columns_n].values.tolist()).transpose((0, 2, 1))

private_test_x = np.array(private_df[feature_columns_n].values.tolist()).transpose((0, 2, 1))
model_short = build_model(seq_len=107, pred_len=107)

model_long = build_model(seq_len=130, pred_len=130)



model_short.set_weights(model.get_weights())

model_long.set_weights(model.get_weights())



public_preds = model_short.predict(public_test_x, verbose=1)

private_preds = model_long.predict(private_test_x, verbose=1)
print(public_preds.shape, private_preds.shape)
preds_ls = []



for df, preds in [(public_df, public_preds), (private_df, private_preds)]:

    for i, uid in enumerate(df.id):

        single_pred = preds[i]



        single_df = pd.DataFrame(single_pred, columns=target_columns)

        single_df['id_seqpos'] = [f'{uid}_{x}' for x in range(single_df.shape[0])]



        preds_ls.append(single_df)



preds_df = pd.concat(preds_ls)
preds_df.head()
submission = sample_df[['id_seqpos']].merge(preds_df, on=['id_seqpos'])

submission.to_csv('submission.csv', index=False)