import pandas as pd

import numpy as np
tourney_result = pd.read_csv('../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MNCAATourneyCompactResults.csv')

tourney_seed = pd.read_csv('../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MNCAATourneySeeds.csv')
# deleting unnecessary columns

tourney_result = tourney_result.drop(['DayNum', 'WScore', 'LScore', 'WLoc', 'NumOT'], axis=1)

tourney_result
tourney_result = pd.merge(tourney_result, tourney_seed, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'], how='left')

tourney_result.rename(columns={'Seed':'WSeed'}, inplace=True)

tourney_result = tourney_result.drop('TeamID', axis=1)



tourney_result = pd.merge(tourney_result, tourney_seed, left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'], how='left')

tourney_result.rename(columns={'Seed':'LSeed'}, inplace=True)

tourney_result = tourney_result.drop('TeamID', axis=1)

tourney_result
def get_seed(x):

    return int(x[1:3])



tourney_result['WSeed'] = tourney_result['WSeed'].map(lambda x: get_seed(x))

tourney_result['LSeed'] = tourney_result['LSeed'].map(lambda x: get_seed(x))

tourney_result
season_result = pd.read_csv('../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MRegularSeasonCompactResults.csv')
season_win_result = season_result[['Season', 'WTeamID', 'WScore']]

season_lose_result = season_result[['Season', 'LTeamID', 'LScore']]



season_win_result.rename(columns={'WTeamID':'TeamID', 'WScore':'Score'}, inplace=True)

season_lose_result.rename(columns={'LTeamID':'TeamID', 'LScore':'Score'}, inplace=True)



season_result = pd.concat((season_win_result, season_lose_result)).reset_index(drop=True)

season_result
season_score = season_result.groupby(['Season', 'TeamID'])['Score'].sum().reset_index()

season_score
tourney_result = pd.merge(tourney_result, season_score, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'], how='left')

tourney_result.rename(columns={'Score':'WScoreT'}, inplace=True)

tourney_result = tourney_result.drop('TeamID', axis=1)



tourney_result = pd.merge(tourney_result, season_score, left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'], how='left')

tourney_result.rename(columns={'Score':'LScoreT'}, inplace=True)

tourney_result = tourney_result.drop('TeamID', axis=1)



tourney_result
tourney_win_result = tourney_result.drop(['Season', 'WTeamID', 'LTeamID'], axis=1)

tourney_win_result.rename(columns={'WSeed':'Seed1', 'LSeed':'Seed2', 'WScoreT':'ScoreT1', 'LScoreT':'ScoreT2'}, inplace=True)

tourney_win_result
tourney_lose_result = tourney_win_result.copy()



tourney_lose_result['Seed1'] = tourney_win_result['Seed2']

tourney_lose_result['Seed2'] = tourney_win_result['Seed1']



tourney_lose_result['ScoreT1'] = tourney_win_result['ScoreT2']

tourney_lose_result['ScoreT2'] = tourney_win_result['ScoreT1']



tourney_lose_result
tourney_win_result['Seed_diff'] = tourney_win_result['Seed1'] - tourney_win_result['Seed2']

tourney_win_result['ScoreT_diff'] = tourney_win_result['ScoreT1'] - tourney_win_result['ScoreT2']



tourney_lose_result['Seed_diff'] = tourney_lose_result['Seed1'] - tourney_lose_result['Seed2']

tourney_lose_result['ScoreT_diff'] = tourney_lose_result['ScoreT1'] - tourney_lose_result['ScoreT2']
tourney_win_result['result'] = 1

tourney_lose_result['result'] = 0

tourney_result = pd.concat((tourney_win_result, tourney_lose_result)).reset_index(drop=True)

tourney_result
test_df = pd.read_csv('../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MSampleSubmissionStage1_2020.csv')
test_df['Season'] = test_df['ID'].map(lambda x: int(x[:4]))

test_df['WTeamID'] = test_df['ID'].map(lambda x: int(x[5:9]))

test_df['LTeamID'] = test_df['ID'].map(lambda x: int(x[10:14]))

test_df
test_df = pd.merge(test_df, tourney_seed, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'], how='left')

test_df.rename(columns={'Seed':'Seed1'}, inplace=True)

test_df = test_df.drop('TeamID', axis=1)



test_df = pd.merge(test_df, tourney_seed, left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'], how='left')

test_df.rename(columns={'Seed':'Seed2'}, inplace=True)

test_df = test_df.drop('TeamID', axis=1)



test_df = pd.merge(test_df, season_score, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'], how='left')

test_df.rename(columns={'Score':'ScoreT1'}, inplace=True)

test_df = test_df.drop('TeamID', axis=1)



test_df = pd.merge(test_df, season_score, left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'], how='left')

test_df.rename(columns={'Score':'ScoreT2'}, inplace=True)

test_df = test_df.drop('TeamID', axis=1)



test_df
test_df['Seed1'] = test_df['Seed1'].map(lambda x: get_seed(x))

test_df['Seed2'] = test_df['Seed2'].map(lambda x: get_seed(x))

test_df['Seed_diff'] = test_df['Seed1'] - test_df['Seed2']

test_df['ScoreT_diff'] = test_df['ScoreT1'] - test_df['ScoreT2']

test_df = test_df.drop(['ID', 'Pred', 'Season', 'WTeamID', 'LTeamID'], axis=1)

test_df
X = tourney_result.drop('result', axis=1)

y = tourney_result.result
from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()





test_df=scaler.fit_transform(test_df)

test_df=test_df.reshape(test_df.shape[0],test_df.shape[1],1)
import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import Sequential

from tensorflow.keras.layers import LSTM, Dropout, MaxPool1D, Flatten, Dense, Dropout, BatchNormalization

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from tensorflow.keras import callbacks



from sklearn.metrics import accuracy_score, roc_auc_score, auc

from sklearn.model_selection import StratifiedKFold

from sklearn import metrics, preprocessing

from tensorflow.keras import backend as K



def NN_model():

  # model

    model=Sequential()

    model.add(Flatten(input_shape=(6,1)))



    model.add(Dense(64, activation='relu'))

    model.add(BatchNormalization())

    model.add(Dropout(0.2))



    model.add(Dense(128, activation='relu'))

    model.add(BatchNormalization())

    model.add(Dropout(0.2))



    model.add(Dense(256, activation='relu'))

    model.add(BatchNormalization())

    model.add(Dropout(0.3))

    

    model.add(Dense(512, activation='relu'))

    model.add(Dropout(0.3))



    model.add(Dense(1,activation='sigmoid'))



    model.compile(optimizer=Adam(lr=1e-3), loss='binary_crossentropy', metrics=['accuracy'])

    return model
import gc

NFOLDS = 50

folds = StratifiedKFold(n_splits=NFOLDS)



columns = X.columns

splits = folds.split(X, y)

y_preds = np.zeros(test_df.shape[0])

y_oof = np.zeros(X.shape[0])



feature_importances = pd.DataFrame()

feature_importances['feature'] = columns

  

for fold_n, (train_index, valid_index) in enumerate(splits):

    

    print('Fold:',fold_n+1)

    X_train, X_valid = X[columns].iloc[train_index], X[columns].iloc[valid_index]

    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

    

    #----------------------------------------------------------#

    # data preparation



    X_train=scaler.fit_transform(X_train)

    X_valid=scaler.fit_transform(X_valid)



    y_train=y_train.to_numpy()

    y_valid=y_valid.to_numpy()



    X_train=X_train.reshape(X_train.shape[0],X_train.shape[1],1)

    X_valid=X_valid.reshape(X_valid.shape[0],X_valid.shape[1],1)



    #----------------------------------------------------------#



    #set early stopping criteria

    

    es = callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0.001, patience=5,verbose=1, mode='max', baseline=None, restore_best_weights=True)

    rlr = callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.5,patience=3, min_lr=1e-6, mode='max', verbose=1)



    model = NN_model() #model creation

    

    model.fit(X_train, y_train, epochs=25, batch_size=32, callbacks=[es, rlr], verbose=0,validation_data=(X_valid, y_valid))



    #----------------------------------------------------------#

    #validation

    

    y_pred_valid = model.predict(X_valid)

    y_oof[valid_index] = y_pred_valid.ravel()

    

#     y_preds += clf.predict(test_df) / NFOLDS

    

    test_fold_preds = model.predict(test_df)

    y_preds += test_fold_preds.ravel()

    

    print(metrics.roc_auc_score(y_valid, y_pred_valid))

    

    del X_train, X_valid, y_train, y_valid

    

    K.clear_session()

    gc.collect()
y_preds /= 50

print('Creating Submission File !!')

submission_df = pd.read_csv('../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MSampleSubmissionStage1_2020.csv')

submission_df['Pred'] = y_preds

submission_df
submission_df['Pred'].hist()
submission_df.to_csv('submission.csv', index=False)