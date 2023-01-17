N_EPOCHS = 3  # Should be more, but the network has to run 16 times in a Kaggle kernel
from datetime import date, timedelta

import pandas as pd

import numpy as np

from sklearn.metrics import mean_squared_error

import lightgbm as lgb

from keras.models import Sequential

from keras.layers.core import Dense, Dropout, Activation

from keras import callbacks

from keras.callbacks import ModelCheckpoint



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
indir = '../input/preparing-data-for-lgbm-or-something-else/'

indir2 = '../input/favorita-grocery-sales-forecasting/'
X_test = pd.read_csv(indir + 'X_test.csv')

X_val = pd.read_csv(indir + 'X_val.csv')

X_train = pd.read_csv(indir + 'X_train.csv')

y_train = np.array(pd.read_csv(indir + 'y_train.csv'))

y_val = np.array(pd.read_csv(indir + 'y_val.csv'))

stores_items = pd.read_csv(indir + 'stores_items.csv', index_col=['store_nbr','item_nbr'])

test_ids = pd.read_csv( indir + 'test_ids.csv',  parse_dates=['date']).set_index(

                        ['store_nbr', 'item_nbr', 'date'] )

items = pd.read_csv( indir2 + 'items.csv' ).set_index("item_nbr")

items = items.reindex( stores_items.index.get_level_values(1) )
model = Sequential()

model.add(Dense(32, kernel_initializer='normal', activation='relu', input_shape=(X_train.shape[1],)))

model.add(Dropout(.2))

model.add(Dense(16, kernel_initializer='normal', activation='relu'))

model.add(Dropout(.1))

model.add(Dense(1, kernel_initializer='normal'))

model.compile(loss = 'mse', optimizer='adam', metrics=['mse'])
val_pred = []

test_pred = []

# wtpath = 'weights.hdf5'  # To save best epoch. But need Keras bug to be fixed first.

sample_weights=np.array( pd.concat([items["perishable"]] * 6) * 0.25 + 1 )

for i in range(16):

    print("=" * 50)

    print("Step %d" % (i+1))

    print("=" * 50)

    y = y_train[:, i]

    xv = np.array(X_val)

    yv = y_val[:, i]

#    bestepoch = ModelCheckpoint( filepath=wtpath, verbose=1, save_best_only=True )

    model.fit( np.array(X_train), y, batch_size = 32, epochs = N_EPOCHS, verbose=2,

               sample_weight=sample_weights, validation_data=(xv,yv) ) 

             #, callbacks=[bestepoch] # bestepoch doesn't work: keras bug

#    model.load_weights( wtpath )

    val_pred.append(model.predict(X_val))

    test_pred.append(model.predict(X_test))
n_public = 5 # Number of days in public test set

weights=pd.concat([items["perishable"]]) * 0.25 + 1

print("Unweighted validation mse: ", mean_squared_error(

    y_val, np.array(val_pred).squeeze(axis=2).transpose()) )

print("Full validation mse:       ", mean_squared_error(

    y_val, np.array(val_pred).squeeze(axis=2).transpose(), sample_weight=weights) )

print("'Public' validation mse:   ", mean_squared_error(

    y_val[:,:n_public], np.array(val_pred).squeeze(axis=2).transpose()[:,:n_public], 

    sample_weight=weights) )

print("'Private' validation mse:  ", mean_squared_error(

    y_val[:,n_public:], np.array(val_pred).squeeze(axis=2).transpose()[:,n_public:], 

    sample_weight=weights) )
y_test = np.array(test_pred).squeeze(axis=2).transpose()

df_preds = pd.DataFrame(

    y_test, index=stores_items.index,

    columns=pd.date_range("2017-08-16", periods=16)

).stack().to_frame("unit_sales")

df_preds.index.set_names(["store_nbr", "item_nbr", "date"], inplace=True)
submission = test_ids.join(df_preds, how="left").fillna(0)

submission["unit_sales"] = np.clip(np.expm1(submission["unit_sales"]), 0, 1000)

submission.to_csv('nn_sub_whatever.csv', float_format='%.4f', index=None)