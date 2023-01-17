from kaggle.competitions import twosigmanews
import time
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
env = twosigmanews.make_env()
days = env.get_prediction_days()
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

(market_train, news_train) = env.get_training_data()
news_train
def process(market_df, newsdf, train=False):
    market_df['time'] = market_df.time.dt.strftime("%Y%m%d").astype(int)
    
    cat_cols = ['assetCode']
    num_cols = ['volume', 'close', 'open', 'returnsClosePrevRaw1', 'returnsOpenPrevRaw1', 'returnsClosePrevMktres1',
                'returnsOpenPrevMktres1', 'returnsClosePrevRaw10', 'returnsOpenPrevRaw10', 'returnsClosePrevMktres10',
                'returnsOpenPrevMktres10']

    market_df['bartrend'] = market_df['close'] / market_df['open']

    market_df['average'] = (market_df['close'] + market_df['open'])/2
    market_df['pricevolume'] = market_df['volume'] * market_df['close']
    
    # See Raba in https://www.kaggle.com/rabaman/0-64-in-100-lines
    newsdf['time'] = newsdf.time.dt.strftime("%Y%m%d").astype(int)
    newsdf['assetCode'] = newsdf['assetCodes'].map(lambda x: list(eval(x))[0])
    newsdf['position'] = newsdf['firstMentionSentence'] / newsdf['sentenceCount']
    newsdf['coverage'] = newsdf['sentimentWordCount'] / newsdf['wordCount']
    
    droplist = ['sourceTimestamp','firstCreated','sourceId','headline','takeSequence','provider','firstMentionSentence',
                'sentenceCount','bodySize','headlineTag','marketCommentary','subjects','audiences','sentimentClass',
                'assetName', 'assetCodes','urgency','wordCount','sentimentWordCount']
    newsdf.drop(droplist, axis=1, inplace=True)
    market_df.drop(['assetName', 'volume'], axis=1, inplace=True)
    
    # combine multiple news reports for same assets on same day
    newsgp = newsdf.groupby(['time','assetCode'], sort=False).aggregate(np.mean).reset_index()
    
    # join news reports to market data, note many assets will have many days without news data
    market_train = pd.merge(market_df, newsgp, how='left', on=['time', 'assetCode'], copy=False)
    num_cols = [x for x in market_train.columns if x not in cat_cols +['returnsOpenNextMktres10', 'universe']]
    if train:
        print(f"In {(market_train['bartrend'] >= 1.2).sum()} lines price increased by 20% or more...dropping")
        market_train = market_train[market_train['bartrend'] <= 1.2]
        print(f"In {(market_train['bartrend'] <= 0.8).sum()} lines price decreased by 20% or more...dropping")
        market_train = market_train[market_train['bartrend'] >= 0.8]

    market_train[num_cols] = market_train[num_cols].fillna(0)
    
    return market_train, cat_cols, num_cols


market_train, cat_cols, num_cols = process(market_train, news_train, train=True)

scaler = StandardScaler()
market_train[num_cols] = scaler.fit_transform(market_train[num_cols])

train_indices, val_indices = train_test_split(market_train.index.values,test_size=0.25, random_state=23)

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
    encoders[i] = {l: id for id, l in enumerate(market_train.loc[train_indices, cat].astype(str).unique())}
    market_train[cat] = market_train[cat].astype(str).apply(lambda x: encode(encoders[i], x))
    print('Done')

embed_sizes = [len(encoder) + 1 for encoder in encoders] #+1 for possible unknown assets

market_train[num_cols] = market_train[num_cols].fillna(0)
print('scaling numerical columns')
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

numerical_inputs = Input(shape=(31,), name='num')
numerical_logits = numerical_inputs
numerical_logits = BatchNormalization()(numerical_logits)

numerical_logits = Dense(512,activation='sigmoid')(numerical_logits)
numerical_logits = Dense(256,activation='sigmoid')(numerical_logits)
numerical_logits = Dense(128,activation='sigmoid')(numerical_logits)
numerical_logits = Dense(64,activation='sigmoid')(numerical_logits)

logits = Concatenate()([numerical_logits,categorical_logits])
logits = Dense(64,activation='sigmoid')(logits)
out = Dense(1, activation='sigmoid')(logits)

model = Model(inputs = categorical_inputs + [numerical_inputs], outputs=out)
model.compile(optimizer='adam',loss=binary_crossentropy)
def get_input(market_train, indices):
    X_num = market_train.loc[indices, num_cols].values
    X = {'num':X_num}
    for cat in cat_cols:
        X[cat] = market_train.loc[indices, cat_cols].values
    y = (market_train.loc[indices,'returnsOpenNextMktres10'] >= 0).values
    r = market_train.loc[indices,'returnsOpenNextMktres10'].values
    u = market_train.loc[indices, 'universe']
    d = market_train.loc[indices, 'time']
    return X,y,r,u,d

# r, u and d are used to calculate the scoring metric
X_train,y_train,r_train,u_train,d_train = get_input(market_train, train_indices)
X_valid,y_valid,r_valid,u_valid,d_valid = get_input(market_train, val_indices)
from keras.callbacks import EarlyStopping, ModelCheckpoint

check_point = ModelCheckpoint('model.hdf5',verbose=True, save_best_only=True)
early_stop = EarlyStopping(patience=3,verbose=True)
model.fit(X_train,y_train.astype(int),
          validation_data=(X_valid,y_valid.astype(int)),
          epochs=15,
          verbose=True,
          callbacks=[early_stop,check_point]) 
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# distribution of confidence that will be used as submission
model.load_weights('model.hdf5')
confidence_valid = model.predict(X_valid)[:,0]*2 -1
print(accuracy_score(confidence_valid>0,y_valid))
plt.hist(confidence_valid, bins='auto')
plt.title("predicted confidence")
plt.show()
n_days = 0
prep_time = 0
prediction_time = 0
packaging_time = 0
predicted_confidences = np.array([])
for (market_obs_df, news_obs_df, predictions_template_df) in days:
        
    t = time.time()
    
    market_obs_df, cat_cols, num_cols = process(market_obs_df, news_obs_df)
    market_obs_df[num_cols] = scaler.transform(market_obs_df[num_cols])
    
    market_obs_df['assetCode_encoded'] = market_obs_df[cat].astype(str).apply(lambda x: encode(encoders[i], x))

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

total = prep_time + prediction_time + packaging_time
print(f'Preparing Data: {prep_time:.2f}s')
print(f'Making Predictions: {prediction_time:.2f}s')
print(f'Packing: {packaging_time:.2f}s')
print(f'Total: {total:.2f}s')
[x for x in market_train.columns if x not in market_obs_df.columns]
#env.predict(predictions_template_df)
# distribution of confidence as a sanity check: they should be distributed as above
plt.hist(predicted_confidences, bins='auto')
plt.title("predicted confidence")
plt.show()
env.write_submission_file()