import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import keras

from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout
# Config

OUTPUT_DICT = ''

ID = 'Id'

TARGET_COLS = ['age', 'domain1_var1', 'domain1_var2', 'domain2_var1', 'domain2_var2']

SEED = 42

BASE_PATH = '../input/trends-assessment-prediction'
loading = pd.read_csv(f'{BASE_PATH}/loading.csv')

train = pd.read_csv(f'{BASE_PATH}/train_scores.csv').dropna().reset_index(drop=True)

sample_submission = pd.read_csv(f'{BASE_PATH}/sample_submission.csv')

reveal_site = pd.read_csv(f'{BASE_PATH}/reveal_ID_site2.csv')

fnc = pd.read_csv(f'{BASE_PATH}/fnc.csv')

icn_numbers = pd.read_csv(f'{BASE_PATH}/ICN_numbers.csv')
sample_submission['ID_num'] = sample_submission[ID].apply(lambda x: int(x.split('_')[0]))

test = pd.DataFrame({ID: sample_submission['ID_num'].unique().astype(int)})
train = train.merge(loading, on=ID, how='left')

train = train.merge(fnc, on=ID, how='left')



test = test.merge(loading, on=ID, how='left')

test = test.merge(fnc, on=ID, how='left')
train = train.dropna(how='all').dropna(how='all', axis=1)
X_train = train.drop('Id', axis=1).drop(TARGET_COLS, axis=1)

y_train = train.drop('Id', axis=1)[TARGET_COLS]

X_test = test.drop('Id', axis=1)
np.random.seed(SEED)

epochs= 16

batch_size = 128

verbose = 1



model = Sequential([

               #input

               Dense(2048),

               Activation('relu'),

               Dropout(0.1),

               Dense(2048),

               Activation('relu'),

               Dropout(0.1),

              Dense(256),

              Activation('relu'),

              Dropout(0.1),

              Dense(128),

              Activation('relu'),

              Dropout(0.1),

               #output

               Dense(5),

               Activation('relu'),

        ])





model.compile(loss='mean_squared_error',

              optimizer='adam',

              metrics=['mse'])



hist = model.fit(X_train.values, y_train.values,

                        batch_size = batch_size, epochs = epochs,

                        callbacks = [],

                        verbose=verbose)
prediction_dict = model.predict(X_test)

prediction_dict = pd.DataFrame(prediction_dict)

prediction_dict.columns = y_train.columns
pred_df = pd.DataFrame()



for TARGET in TARGET_COLS:

    tmp = pd.DataFrame()

    tmp[ID] = [f'{c}_{TARGET}' for c in test[ID].values]

    tmp['Predicted'] = prediction_dict[TARGET]

    pred_df = pd.concat([pred_df, tmp])
submission = pd.merge(sample_submission, pred_df, on = 'Id')
submission




submission = pd.merge(sample_submission, pred_df, on = 'Id')[['Id', 'Predicted_y']]

submission.columns = ['Id', 'Predicted']



submission




submission.to_csv('submission.csv', index=False)

submission.head()


