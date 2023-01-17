import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.metrics import accuracy_score
from kaggle.competitions import twosigmanews
env = twosigmanews.make_env()
(market_train, _) = env.get_training_data()
cat_cols = ['assetCode']
num_cols = [
            'returnsClosePrevRaw1', 'returnsOpenPrevRaw1', 'returnsClosePrevRaw10','returnsOpenPrevRaw10', # Raw returns
            'returnsOpenPrevMktres1', 'returnsClosePrevMktres1','returnsClosePrevMktres10','returnsOpenPrevMktres10',  # Residualized returns
            'volume', 'close', 'open', # Other
            'close_to_open', 'std1', 'std10', 'std_res1', 'std_res10', 'res_raw1', 'res_raw10', 'raw_res1', 'raw_res10', 'ma10', 'ma50', 'ma200' # Custom Inputs
]

# # Custom Inputs
returns = market_train['returnsClosePrevRaw1']
res_returns = market_train['returnsClosePrevMktres1']

df_close_to_open = np.abs(market_train['close'] / market_train['open'])
std1 =  np.abs(returns) / np.mean(np.abs(market_train['returnsClosePrevRaw1'])) 
std10 =  np.abs(returns) / np.mean(np.abs(market_train['returnsClosePrevRaw10']))

std_res1 = np.abs(res_returns) / np.mean(np.abs(market_train['returnsClosePrevMktres1']))
std_res10 = np.abs(res_returns) / np.mean(np.abs(market_train['returnsClosePrevMktres10']))

res_raw1 = np.abs(res_returns) / np.mean(np.abs(market_train['returnsClosePrevRaw1']))
res_raw10 = np.abs(res_returns) / np.mean(np.abs(market_train['returnsClosePrevRaw10']))

raw_res1 = np.abs(returns) / np.mean(np.abs(market_train['returnsClosePrevMktres1']))
raw_res10 = np.abs(returns) / np.mean(np.abs(market_train['returnsClosePrevMktres10']))
    
N = 10
ma10 = np.convolve(returns, np.ones((N,))/N, mode='valid')
ma10_fix = [sum(ma10)/len(ma10)] * 9
ma10 = np.insert(ma10, 0, ma10_fix)

N = 50
ma50 = np.convolve(returns, np.ones((N,))/N, mode='valid')
ma50_fix = [sum(ma50)/len(ma50)] * 49
ma50 = np.insert(ma50, 0, ma50_fix)

N = 200
ma200 = np.convolve(returns, np.ones((N,))/N, mode='valid')
ma200_fix = [sum(ma200)/len(ma200)] * 199
ma200 = np.insert(ma200, 0, ma200_fix)
from sklearn.model_selection import train_test_split
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

from sklearn.preprocessing import StandardScaler
 
# Add our engineered features to dataset
market_train['close_to_open'] = df_close_to_open
market_train['std1'] = std1
market_train['std10'] = std10
market_train['std_res1']  = std_res1
market_train['std_res10']  = std_res10
market_train['res_raw1'] = res_raw1
market_train['res_raw10'] = res_raw10
market_train['raw_res1'] = raw_res1
market_train['raw_res10'] = raw_res1
market_train['ma10'] = ma10
market_train['ma50'] = ma50
market_train['ma200'] = ma200

# Clean Our Data:
market_train[num_cols] = market_train[num_cols].replace([np.inf, -np.inf], np.nan)
market_train[num_cols] = market_train[num_cols].fillna(0)

market_train['std_res1'] = market_train['std_res1'].replace(0, np.mean(market_train['std_res1']))
market_train['std_res10'] = market_train['std_res10'].replace(0, np.mean(market_train['std_res10']))
market_train['res_raw1'] = market_train['res_raw1'].replace(0, np.mean(market_train['res_raw1']))
market_train['res_raw10'] = market_train['res_raw10'].replace(0, np.mean(market_train['res_raw10']))
market_train['raw_res1'] = market_train['raw_res1'].replace(0, np.mean(market_train['raw_res1']))
market_train['raw_res10'] = market_train['raw_res10'].replace(0, np.mean(market_train['raw_res10']))

market_train['returnsOpenPrevMktres1'] = market_train['returnsOpenPrevMktres1'].replace(0, np.mean(market_train['returnsOpenPrevMktres1']))
market_train['returnsClosePrevMktres1'] = market_train['returnsClosePrevMktres1'].replace(0, np.mean(market_train['returnsClosePrevMktres1']))
market_train['returnsClosePrevMktres10'] = market_train['returnsClosePrevMktres10'].replace(0, np.mean(market_train['returnsClosePrevMktres10']))
market_train['returnsOpenPrevMktres10'] = market_train['returnsOpenPrevMktres10'].replace(0, np.mean(market_train['returnsOpenPrevMktres10']))

print('scaling numerical columns')
scaler = StandardScaler()
market_train[num_cols] = scaler.fit_transform(market_train[num_cols])

from keras.models import Model
from keras.layers import Input, Dense, Embedding, Concatenate, Flatten, BatchNormalization
from keras.losses import binary_crossentropy, mse

categorical_inputs = []
for cat in cat_cols:
    categorical_inputs.append(Input(shape=[1], name=cat))

categorical_embeddings = []
for i, cat in enumerate(cat_cols):
    categorical_embeddings.append(Embedding(embed_sizes[i], 10)(categorical_inputs[i]))

categorical_logits = Flatten()(categorical_embeddings[0])
categorical_logits = Dense(32,activation='relu')(categorical_logits)

numerical_inputs = Input(shape=(23,), name='num')
numerical_logits = numerical_inputs
numerical_logits = BatchNormalization()(numerical_logits)

numerical_logits = Dense(128,activation='relu')(numerical_logits)
numerical_logits = Dense(64,activation='relu')(numerical_logits)

logits = Concatenate()([numerical_logits,categorical_logits])
logits = Dense(64,activation='relu')(logits)
out = Dense(1, activation='sigmoid')(logits)

model = Model(inputs = categorical_inputs + [numerical_inputs], outputs=out)
model.compile(optimizer='adam',loss=binary_crossentropy)
# Lets print our model
model.summary()
def get_input(market_train, indices):
    X_num = market_train.loc[indices, num_cols].values
    X = {'num':X_num}
    for cat in cat_cols:
        X[cat] = market_train.loc[indices, cat_cols].values
    y = (market_train.loc[indices,'returnsOpenNextMktres10'] >= 0).values
    r = market_train.loc[indices,'returnsOpenNextMktres10'].values
    u = market_train.loc[indices, 'universe']
    d = market_train.loc[indices, 'time'].dt.date
    return X,y,r,u,d

# r, u and d are used to calculate the scoring metric
X_train,y_train,r_train,u_train,d_train = get_input(market_train, train_indices)
X_valid,y_valid,r_valid,u_valid,d_valid = get_input(market_train, val_indices)
from keras.callbacks import EarlyStopping, ModelCheckpoint

check_point = ModelCheckpoint('model.hdf5',verbose=True, save_best_only=True)
early_stop = EarlyStopping(patience=5,verbose=True)
model.fit(X_train,y_train.astype(int),
          validation_data=(X_valid,y_valid.astype(int)),
          epochs=1,
          verbose=True,
          callbacks=[early_stop,check_point]) 
# distribution of confidence that will be used as submission
model.load_weights('model.hdf5')
confidence_valid = model.predict(X_valid)[:,0]*2 -1
print(accuracy_score(confidence_valid>0,y_valid))
plt.hist(confidence_valid, bins='auto')
plt.title("predicted confidence")
plt.show()
# calculation of actual metric that is used to calculate final score
r_valid = r_valid.clip(-1,1) # get rid of outliers. Where do they come from??
x_t_i = confidence_valid * r_valid * u_valid
data = {'day' : d_valid, 'x_t_i' : x_t_i}
df = pd.DataFrame(data)
x_t = df.groupby('day').sum().values.flatten()
mean = np.mean(x_t)
std = np.std(x_t)
score_valid = mean / std
print(score_valid)
days = env.get_prediction_days()
# NEEDS WORKS.  PLZ HELP.

n_days = 0
prep_time = 0
prediction_time = 0
packaging_time = 0
predicted_confidences = np.array([])

#------------------------------------------------------------------------------------------------------
# MARKET_OBS_DF NOT ACCESIBLE OUTSIDE OF FOR LOOP THAT CALLS 'DAYS', WHICH IS ONLY CALLABLE ONCE
# FORCED TO INIT CUSTOM VARIABLES INSIDE FOR, BREAKING CODE.

# NEEDS FIX
#------------------------------------------------------------------------------------------------------
for (market_obs_df, news_obs_df, predictions_template_df) in days:
    if days is not None: 
        n_days +=1
        print(n_days,end=' ')
        t = time.time()

        market_obs_df['assetCode_encoded'] = market_obs_df[cat].astype(str).apply(lambda x: encode(encoders[i], x))
        
        # NEED A WAY TO INIT THESE OUTSIDE THIS FOR
        market_obs_df['close_to_open'] = market_train['close_to_open']
        market_obs_df['std1'] = market_train['std1']
        market_obs_df['std10'] = market_train['std10']
        market_obs_df['std_res1']  = market_train['std_res1']
        market_obs_df['std_res10']  = market_train['std_res10']
        market_obs_df['res_raw1'] = market_train['res_raw1']
        market_obs_df['res_raw10'] = market_train['res_raw10']
        market_obs_df['raw_res1'] = market_train['raw_res1']
        market_obs_df['raw_res10'] = market_train['raw_res1']
        market_obs_df['ma10'] = market_train['ma10']
        market_obs_df['ma50'] = market_train['ma50']
        market_obs_df['ma200'] = market_train['ma200']
        
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
# distribution of confidence as a sanity check: they should be distributed as above
plt.hist(predicted_confidences, bins='auto')
plt.title("predicted confidence")
plt.show()