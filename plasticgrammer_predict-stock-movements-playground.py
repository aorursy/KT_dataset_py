import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import itertools

import gc

import os

import sys



sns.set_style('darkgrid')

sns.set_palette('bone')

pd.options.display.float_format = '{:,.3f}'.format



print(os.listdir("../input"))
from kaggle.competitions import twosigmanews



env = twosigmanews.make_env()
%%time

(market, news) = env.get_training_data()
market.isnull().sum()
market.head()
market.describe().T[['mean','min','50%','max']]
market['time'].unique()
market['assetCode'].unique()
market.loc[market['time'] == market['time'].min(),'assetCode'].nunique()
market.groupby('time').count()['assetCode'].hist()
market[['time','assetCode','volume']].head()
market.groupby('assetCode')['volume'].mean().nlargest(5).to_frame()
market[['time','assetCode','open','close']].head()
from sklearn.model_selection import train_test_split



train_indices, val_indices = train_test_split(market.index.values, test_size=0.25, random_state=42)
cat_cols = ['assetCode']

#for i, t in all_data.loc[:, cat_cols].dtypes.iteritems():

#    if t == object:

#        market[i] = pd.factorize(market[i])[0]
def encode(encoder, x):

    len_encoder = len(encoder)

    try:

        id = encoder[x]

    except KeyError:

        id = len_encoder

    return id



encoders = [{} for cat in cat_cols]



for i, cat in enumerate(cat_cols):

    print('encoding %s ...' % cat, end=' ')

    encoders[i] = {l: id for id, l in enumerate(market.loc[train_indices, cat].astype(str).unique())}

    market[cat] = market[cat].astype(str).apply(lambda x: encode(encoders[i], x))

    print('Done')



embed_sizes = [len(encoder) + 1 for encoder in encoders] #+1 for possible unknown assets
from sklearn.preprocessing import StandardScaler



#num_cols = market.select_dtypes(include='number').columns

num_cols = ['volume', 'close', 'open', 

            'returnsClosePrevRaw1', 'returnsOpenPrevRaw1', 'returnsClosePrevMktres1',

            'returnsOpenPrevMktres1', 'returnsClosePrevRaw10', 'returnsOpenPrevRaw10', 

            'returnsClosePrevMktres10', 'returnsOpenPrevMktres10']



market[num_cols] = market[num_cols].fillna(0)

print('scaling numerical columns')



scaler = StandardScaler()

market[num_cols] = scaler.fit_transform(market[num_cols])
from keras.models import Model

from keras.layers import Input, Dense, Embedding, Concatenate, Flatten, BatchNormalization

from keras.losses import binary_crossentropy, mse



categorical_inputs = []

for cat in cat_cols:

    categorical_inputs.append(Input(shape=[1], name=cat))



categorical_embeddings = []

for i, cat in enumerate(cat_cols):

    categorical_embeddings.append(Embedding(embed_sizes[i], 10)(categorical_inputs[i]))



#categorical_logits = Concatenate()([Flatten()(cat_emb) for cat_emb in categorical_embeddings])

categorical_logits = Flatten()(categorical_embeddings[0])

categorical_logits = Dense(32,activation='relu')(categorical_logits)



numerical_inputs = Input(shape=(11,), name='num')

numerical_logits = numerical_inputs

numerical_logits = BatchNormalization()(numerical_logits)



numerical_logits = Dense(128,activation='relu')(numerical_logits)

numerical_logits = Dense(64,activation='relu')(numerical_logits)



logits = Concatenate()([numerical_logits,categorical_logits])

logits = Dense(64,activation='relu')(logits)

out = Dense(1, activation='sigmoid')(logits)



model = Model(inputs = categorical_inputs + [numerical_inputs], outputs=out)

model.compile(optimizer='adam',loss=binary_crossentropy)

model.summary()
def get_input(df, indices):

    X_num = df.loc[indices, num_cols].values

    X = {'num':X_num}

    for cat in cat_cols:

        X[cat] = df.loc[indices, cat_cols].values

    y = (df.loc[indices,'returnsOpenNextMktres10'] >= 0).values

    r = df.loc[indices,'returnsOpenNextMktres10'].values

    u = df.loc[indices, 'universe']

    d = df.loc[indices, 'time'].dt.date

    return X,y,r,u,d



# r, u and d are used to calculate the scoring metric

X_train,y_train,r_train,u_train,d_train = get_input(market, train_indices)

X_valid,y_valid,r_valid,u_valid,d_valid = get_input(market, val_indices)
from keras.callbacks import EarlyStopping, ModelCheckpoint



check_point = ModelCheckpoint('model.hdf5',verbose=True,save_best_only=True)

early_stop = EarlyStopping(patience=5,verbose=True)

model.fit(X_train,y_train.astype(int),

          validation_data=(X_valid,y_valid.astype(int)),

          epochs=2,

          verbose=True,

          callbacks=[early_stop,check_point]) 
from sklearn.metrics import accuracy_score



# distribution of confidence that will be used as submission

model.load_weights('model.hdf5')

confidence_valid = model.predict(X_valid)[:,0]*2 -1

print(accuracy_score(confidence_valid>0,y_valid))

plt.hist(confidence_valid, bins='auto')

plt.title("predicted confidence")

plt.show()
# You can only iterate through a result from `get_prediction_days()` once

# so be careful not to lose it once you start iterating.

days = env.get_prediction_days()
import time

from tqdm import tqdm



n_days = 0

prep_time = 0

prediction_time = 0

packaging_time = 0

predicted_confidences = np.array([])

for (market_obs_df, news_obs_df, predictions_template_df) in days:

    n_days +=1

    #print(n_days,end=' ')

    

    t = time.time()



    market_obs_df['assetCode_encoded'] = market_obs_df[cat].astype(str).apply(lambda x: encode(encoders[i], x))



    market_obs_df[num_cols] = market_obs_df[num_cols].fillna(0)

    market_obs_df[num_cols] = scaler.transform(market_obs_df[num_cols])

    X_num_test = market_obs_df[num_cols].values

    X_test = {'num':X_num_test}

    X_test['assetCode'] = market_obs_df['assetCode_encoded'].values

    

    prep_time += time.time() - t

    

    t = time.time()

    market_prediction = model.predict(X_test)[:,0]*2 -1

    predicted_confidences = np.concatenate((predicted_confidences, market_prediction))

    prediction_time += time.time() -t

    

    t = time.time()

    preds = pd.DataFrame({'assetCode':market_obs_df['assetCode'],'confidence':market_prediction})

    # insert predictions to template

    predictions_template_df = predictions_template_df.merge(preds,how='left').drop('confidenceValue',axis=1).fillna(0).rename(columns={'confidence':'confidenceValue'})

    env.predict(predictions_template_df)

    packaging_time += time.time() - t

env.write_submission_file()

total = prep_time + prediction_time + packaging_time

print(f'Preparing Data: {prep_time:.2f}s')

print(f'Making Predictions: {prediction_time:.2f}s')

print(f'Packing: {packaging_time:.2f}s')

print(f'Total: {total:.2f}s')