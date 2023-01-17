import os

import warnings



import numpy as np

import pandas as pd



from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

from sklearn.metrics import log_loss



warnings.filterwarnings("ignore")
result = pd.read_csv('../input/google-cloud-ncaa-march-madness-2020-division-1-womens-tournament/WDataFiles_Stage1/WNCAATourneyCompactResults.csv')

result = result.drop(columns=['WLoc', 'NumOT', 'DayNum'])

result.head()
seeds = pd.read_csv('../input/google-cloud-ncaa-march-madness-2020-division-1-womens-tournament/WDataFiles_Stage1/WNCAATourneySeeds.csv')

seeds.Seed = seeds.Seed.map(lambda string : int(string[1:3]))

seeds.head()
Wseeds = seeds.rename(columns={'TeamID':'WTeamID', 'Seed':'WSeed'})

Lseeds = seeds.rename(columns={'TeamID':'LTeamID', 'Seed':'LSeed'})



data = pd.merge(left=result, right=Wseeds, how='left', on=['Season', 'WTeamID'])

data = pd.merge(left=data, right=Lseeds, on=['Season', 'LTeamID'])



data.head()
scores = pd.read_csv('../input/google-cloud-ncaa-march-madness-2020-division-1-womens-tournament/WDataFiles_Stage1/WRegularSeasonCompactResults.csv')

scores.head()
Lscores = scores[['Season', 'WTeamID', 'WScore']].rename(columns={'WTeamID':'TeamID', 'WScore':'Score'})

Wscores = scores[['Season', 'LTeamID', 'LScore']].rename(columns={'LTeamID':'TeamID', 'LScore':'Score'})



result_scores = pd.concat([Wscores, Lscores])

result_scores.head()
season_score = result_scores.groupby(['Season', 'TeamID'])['Score'].sum()
data = pd.merge(data, season_score, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'], how='left')

data = data.rename(columns={'Score':'WScoreT'})

data = pd.merge(data, season_score, left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'], how='left')

data = data.rename(columns={'Score':'LScoreT'})

data = data.drop(columns=['WScore', 'LScore'])

data.head()
Wdata = data.drop(columns=['Season', 

                           #'WTeamID', 

                           #'LTeamID'

                           ])

Wdata.rename(columns={'WSeed':'Seed1', 'LSeed':'Seed2', 

                      'WScoreT':'ScoreT1', 'LScoreT':'ScoreT2',

                      'WTeamID':'TeamID_1', 'LTeamID': 'TeamID_2'}, inplace=True)

Wdata.head()
Ldata = data[['LTeamID', 'WTeamID', 'LSeed', 'WSeed', 'LScoreT', 'WScoreT']]

Ldata.rename(columns={'LTeamID':'TeamID_1', 'WTeamID':'TeamID_2', 

                      'LSeed':'Seed1', 'WSeed':'Seed2', 

                      'LScoreT':'ScoreT1', 'WScoreT':'ScoreT2',}, inplace=True)

Ldata.head()
Wdata['Seed_diff'] = Wdata['Seed1'] - Wdata['Seed2']

Wdata['ScoreT_diff'] = Wdata['ScoreT1'] - Wdata['ScoreT2']

Ldata['Seed_diff'] = Ldata['Seed1'] - Ldata['Seed2']

Ldata['ScoreT_diff'] = Ldata['ScoreT1'] - Ldata['ScoreT2']
Wdata['result'] = 1

Ldata['result'] = 0

train = pd.concat((Wdata, Ldata)).reset_index(drop=True)

train.head()
# Extract year and ID number out of string

test = pd.read_csv('../input/google-cloud-ncaa-march-madness-2020-division-1-womens-tournament/WSampleSubmissionStage1_2020.csv')

test['Season'] = test.ID.map(lambda string : int(string.split('_')[0]))

test['TeamID_1'] = test.ID.map(lambda string : int(string.split('_')[1]))

test['TeamID_2'] = test.ID.map(lambda string : int(string.split('_')[2]))

test = test.drop(columns=['ID'])

test.head()



# Convert test data to the train set's format

test = pd.merge(test, seeds, left_on=['Season', 'TeamID_1'], right_on=['Season', 'TeamID'], how='left')

test.rename(columns={'Seed':'Seed1'}, inplace=True)

test = test.drop('TeamID', axis=1)

test = pd.merge(test, seeds, left_on=['Season', 'TeamID_2'], right_on=['Season', 'TeamID'], how='left')

test.rename(columns={'Seed':'Seed2'}, inplace=True)

test = test.drop('TeamID', axis=1)

test = pd.merge(test, season_score, left_on=['Season', 'TeamID_1'], right_on=['Season', 'TeamID'], how='left')

test.rename(columns={'Score':'ScoreT1'}, inplace=True)

test = pd.merge(test, season_score, left_on=['Season', 'TeamID_2'], right_on=['Season', 'TeamID'], how='left')

test.rename(columns={'Score':'ScoreT2'}, inplace=True)

test['Seed_diff'] = test['Seed1'] - test['Seed2']

test['ScoreT_diff'] = test['ScoreT1'] - test['ScoreT2']

#test = test.drop(columns=['Pred', 'Season', 'TeamID_1', 'TeamID_2'])

test = test.drop(columns=['Pred', 'Season'])

test.head()
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
train.TeamID_1.values
MAX_EMBINT = max(train.TeamID_1.unique())+1
scaler = StandardScaler()
X, y = train.drop('result', axis=1).values, train['result'].values

X_test = test.values
X[:, 2:] = scaler.fit_transform(X[:, 2:])

X_test[:, 2:] = scaler.transform(X_test[:, 2:])
pca = PCA(n_components=2)

x_pca = pca.fit_transform(X[:, 2:])



plt.plot(x_pca[train['result'].values==1, 0], x_pca[train['result'].values==1, 1], '.g', alpha=0.25)

plt.plot(x_pca[train['result'].values==0, 0], x_pca[train['result'].values==0, 1], '.r', alpha=0.25)
!pip install tensorflow-addons
import tensorflow as tf

import tensorflow_addons as tfa
def mish(x):

    return x * tf.keras.backend.softplus(tf.keras.backend.tanh(x))



tf.keras.utils.get_custom_objects().update({'mish': tf.keras.layers.Activation(mish)})
# https://www.kaggle.com/jarnel/clipping-spline-experiment-on-test-predictions



from scipy.interpolate import UnivariateSpline



def spline_model(labels, preds):

    comb = pd.DataFrame({'labels':labels, 'preds':preds})

    comb = comb.sort_values(by='preds').reset_index(drop=True)

    spline_model = UnivariateSpline(comb['preds'].values, comb['labels'].values)

    adjusted = spline_model(preds)

    return spline_model, log_loss(labels, adjusted)
from tensorflow.keras import backend as K



def focal_loss(gamma=1.5, alpha=.5):

	def focal_loss_fixed(y_true, y_pred):

		pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))

		pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

		return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

	return focal_loss_fixed
def get_model():

    feature_inp = tf.keras.layers.Input((6,), name='FeatureInput')

    id1_inp = tf.keras.layers.Input((1,), name='ID1Input')

    id2_inp = tf.keras.layers.Input((1,), name='ID2Input')

    

    emb = tf.keras.layers.Embedding(MAX_EMBINT, 1, input_length=1)

    

    e1 = tf.keras.layers.Flatten()(emb(id1_inp))

    e2 = tf.keras.layers.Flatten()(emb(id2_inp))

    

    e1 = tf.keras.layers.Dropout(0.1)(e1)

    e2 = tf.keras.layers.Dropout(0.1)(e2)

    

    x = tf.keras.layers.Dense(16)(feature_inp)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Activation(mish)(x)

    x = tf.keras.layers.Dropout(0.25)(x)

    

    e = tf.keras.layers.Concatenate()([e1, e2])

    e = tf.keras.layers.Dense(16)(e)

    e = tf.keras.layers.BatchNormalization()(e)

    e = tf.keras.layers.Activation(mish)(e)

    e = tf.keras.layers.Dropout(0.25)(e)

    

    x = tf.keras.layers.Concatenate()([x, e])

    

    x = tf.keras.layers.Dense(16)(x)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Activation(mish)(x)

    x = tf.keras.layers.Dropout(0.25)(x)

    

    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    

    model = tf.keras.Model([feature_inp, id1_inp, id2_inp], x)

    model.compile(optimizer=tfa.optimizers.RectifiedAdam(lr=1e-3),

                  loss=focal_loss(), metrics=['binary_crossentropy'])

    return model
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10)

losses = []

predicts = []

for i, (train_ind,valid_ind) in enumerate(cv.split(X, y)):

    tf.keras.backend.clear_session()

    

    X_train, X_valid = X[train_ind], X[valid_ind]

    y_train, y_valid = y[train_ind], y[valid_ind]

    

    model = get_model()

    if i == 0:

        print(model.summary())

        

    er = tf.keras.callbacks.EarlyStopping(monitor='val_binary_crossentropy', patience=40, restore_best_weights=True)

    model.fit([X_train[:, 2:], X_train[:, 0].astype('int32'), X_train[:, 1].astype('int32')], y_train, 

              epochs=512, batch_size=64, validation_data=[[X_valid[:, 2:], X_valid[:, 0].astype('int32'), X_valid[:, 1].astype('int32')], y_valid], 

              verbose=0, callbacks=[er])

    

    preds = model.predict([X_valid[:, 2:], X_valid[:, 0].astype('int32'), X_valid[:, 1].astype('int32')])

    sm, loss = spline_model(y_valid.flatten(), preds.flatten())

    print(f'Fold {i}: {loss}')

    test_pred = model.predict([X_test[:, 2:], X_test[:, 0].astype('int32'), X_test[:, 1].astype('int32')])

    predicts.append(sm(test_pred))



# Take the average probabilty on 5 folds

predicts = np.asarray(predicts)

predicts = np.mean(predicts, axis=0)
predicts.shape
import seaborn as sns



sns.distplot(predicts)
submission_df = pd.read_csv('../input/google-cloud-ncaa-march-madness-2020-division-1-womens-tournament/WSampleSubmissionStage1_2020.csv')

submission_df['Pred'] = predicts

submission_df.to_csv('submission.csv', index=False)