import os

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import keras

from keras.layers import Embedding, Dense, Input, MaxPooling1D, concatenate, Flatten, Dropout, BatchNormalization

from keras.layers import LSTM, Bidirectional, TimeDistributed

from keras.models import Model
base_path = os.path.join('..', 'input')

train_path = os.path.join(base_path, 'train.csv')

test_path = os.path.join(base_path, 'test.csv')

train_df = pd.read_csv(train_path)

test_df = pd.read_csv(test_path)

train_df.sample(3)
from sklearn.preprocessing import LabelEncoder

cat_cols = [x for x in train_df.columns if 'cat' in x]

cont_cols = [x for x in train_df.columns if 'cont' in x]

le_cat_encoder = LabelEncoder()

# fit the encoder based on training and test datasets

le_cat_encoder.fit(np.concatenate([train_df[x] for x in cat_cols]+

                                 [test_df[x] for x in cat_cols]))

print(len(le_cat_encoder.classes_), 'categories')

y_col = 'loss'
all_emb_chan, all_inputs = [], []

for k in cat_cols:

    in_val = Input(shape = (1,), name = k)

    all_emb_chan +=[Embedding(len(le_cat_encoder.classes_)+1, 64)(in_val)]

    all_inputs += [in_val]

concat_layer = concatenate(all_emb_chan, axis = 1) # concatenate all of the columns together



norm_concat_emb = BatchNormalization()(concat_layer)

feature_layer = TimeDistributed(Dense(32))(Dropout(0.5)(norm_concat_emb))

feature_lstm = Bidirectional(LSTM(16))(feature_layer)



cont_input = Input(shape = (len(cont_cols),), name = 'continuous')

bn_cont = BatchNormalization()(cont_input)

cont_feature_layer = Dense(16)(Dropout(0.5)(bn_cont))

full_concat_layer = concatenate([feature_lstm, cont_feature_layer])

full_reduction = Dense(16)(full_concat_layer)



out_layer = Dense(1, activation = 'tanh')(full_reduction)

full_model = Model(inputs = all_inputs+[cont_input], 

                   outputs = [out_layer], name = 'FullModel')

full_model.compile(optimizer = 'adam', loss = 'mae')

print('Using a model with:', full_model.count_params(), 'parameters, in', len(full_model.layers), 'layers')
y_vec = train_df[y_col].copy().values

loss_mean, loss_std = y_vec.mean(), 3*y_vec.std()

y_vec -= loss_mean

y_vec /= loss_std

train_df['loss_norm'] = y_vec.clip(-1,1)
from sklearn.model_selection import train_test_split

t_split_df, v_split_df = train_test_split(train_df, 

                 test_size = 0.2,

                 stratify = pd.qcut(train_df['loss'], 10),

                                         random_state = 2017)

print(t_split_df.shape, v_split_df.shape)
def gen_samples(in_df, batch_size = None, loss_name = 'loss_norm'):

    while True:

        out_df = in_df if batch_size is None else in_df.sample(batch_size)

        feed_dict = {c_name: le_cat_encoder.transform(out_df[c_name].values) for c_name in cat_cols}

        feed_dict['continuous'] = out_df[cont_cols].values

        yield feed_dict, out_df[loss_name].values
loss_history = []
for i in range(10):

    loss_history += [full_model.fit_generator(gen_samples(t_split_df, 32), 

                         steps_per_epoch = 500,

                         epochs = 1,

                         validation_data = next(gen_samples(v_split_df))

                         )]
%%time

valid_vars, valid_loss = next(gen_samples(v_split_df, loss_name = 'loss'))

pred_loss = full_model.predict(valid_vars).ravel()*loss_std+loss_mean
fig, ax1 = plt.subplots(1,1)

ax1.hist(valid_loss-pred_loss)

ax1.set_title('Loss Error: MAE-%2.2f' % (np.mean(np.abs(valid_loss-pred_loss))))

ax1.set_xlabel('Actual - Predicted Loss')
%%time

test_vars, test_id = next(gen_samples(test_df, loss_name = 'id'))

pred_test_loss = full_model.predict(test_vars, verbose = 1).ravel()*loss_std+loss_mean
pd.DataFrame(dict(id = test_id, loss = pred_test_loss)).to_csv('prediction.csv', index = False)