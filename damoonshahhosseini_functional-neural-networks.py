# Upgrading pip

!/opt/conda/bin/python3.7 -m pip install --upgrade pip

!pip install git+https://github.com/tensorflow/docs
# Imports:

import pandas as pd

import numpy as np



# Importing the needed modules

from tensorflow import keras

from tensorflow.keras import Model, Input

from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ReduceLROnPlateau

from tensorflow.keras.layers import BatchNormalization, Dropout, Dense, Concatenate, Add

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.regularizers import l1, l2, L1L2

from tensorflow.keras.losses import MeanSquaredLogarithmicError

from tensorflow.keras.initializers import TruncatedNormal



# Usefull functions from tensorflow_docs

import tensorflow_docs as tfdocs

import tensorflow_docs.plots

import tensorflow_docs.modeling



# Loss functions used in validation

from sklearn.metrics import mean_absolute_error as MAE, mean_squared_log_error as MSLE



# importing data

X_train = pd.read_csv('/kaggle/input/imputedhousing/imputed_feature_columns.csv')

X_test = pd.read_csv('/kaggle/input/imputedhousing/imputed_feature_columns_test.csv')

y = pd.read_csv('/kaggle/input/imputedhousing/SalePrice.csv')



b1 = pd.read_csv('/kaggle/input/benchmarkshousing/011978.csv')

b2 = pd.read_csv('/kaggle/input/benchmarkshousing/012008.csv')

b3 = pd.read_csv('/kaggle/input/benchmarkshousing/012010.csv')
def validate(y_pred):

    """ Prints out the data validation with respect to the highest submissions. """

    

    for bench in [b1,b2,b3]:

        print('MAE: ', round(MaE(bench, y_pred)), ", MSLE: ", MSLE(bench, y_pred))

    

    print('-----------------------------------')

    print('base-differences:', MSLE(b011, b012))
# Model 01



feature_number = len(X_train.columns)



# Input and feature weighting part

In = Input(shape=(feature_number))

dense21 = Dense(feature_number)(In)

main_feature_dec = Dense(feature_number, activation='softplus')(dense21)



# sub-model 1:

main_dense21 = Dense(4096)(main_feature_dec)

main_dense21 = Dropout(0.3)(BatchNormalization()(Dense(2048)(main_feature_dec)))

main_dense21 = Dropout(0.3)(BatchNormalization()(Dense(1024)(main_feature_dec)))





# Sub-model 1-1:

dense21_1 = Dropout(0.4)(Dense(512)(main_dense21))

dense21_2 = Dense(128, activation='relu')(dense21_1)

bacth21_1 = BatchNormalization()(dense21_2)



# Sub-model 1-2:

dense21_3 = Dense(512)(main_dense21)

dense21_4 = Dense(128, activation='relu')(dense21_3)

bacth21_2 = BatchNormalization()(dense21_4)



# Sub-model 1-3:

dense21_5 = Dense(512)(main_dense21)

dense21_6 = Dense(128, activation='relu')(dense21_5)

bacth21_3 = BatchNormalization()(dense21_6)



# Interaction parts of sub-model 1

mid_dense21_1 = Dense(256)(Concatenate()([bacth21_1, bacth21_2]))

mid_dense21_2 = Dense(256)(bacth21_3)



mid_dense_combo = Dropout(0.4)(BatchNormalization()(Dense(500)(Concatenate()([mid_dense21_1, mid_dense21_2]))))



Addition = Concatenate()([bacth21_1, bacth21_2, bacth21_3])



# Prediciton of sub-model 1

pred1 = Dense(1, name='P1')(Concatenate()([Addition, mid_dense_combo]))



# Sub model 2:

dense22 = Dense(100)(main_feature_dec)



feature_dec2_1 = Dense(feature_number, activation='softplus')(dense22)

dense_feat2_1 = Dense(512)(feature_dec2_1)

feature_dec2_2 = Dense(feature_number, activation='softplus')(dense22)

dense_feat2_2 = Dense(512)(feature_dec2_2)

feature_dec2_3 = Dense(feature_number, activation='softplus')(dense22)

dense_feat2_3 = Dense(512)(feature_dec2_3)



# Precition 2:

sum_of_features = Add()([feature_dec2_1, feature_dec2_2, feature_dec2_3])

pred2 = Dense(100, activation='relu')(sum_of_features)

pred2 = Dropout(0.4)(BatchNormalization()(Dense(500)(pred2)))

pred2 = Dense(1, name='P2')(pred2)



features_combo = Concatenate()([dense_feat2_1, dense_feat2_2, dense_feat2_3])

# one layer

mid_dense22_1 = Dense(100)( features_combo)

# two layers

mid_dense22_2 = Dense(500, activation='relu')(features_combo)

mid_dense22_2 = Dense(100)(mid_dense22_2)

# three layers

mid_dense22_3 = Dense(1000, activation='relu')(features_combo)

mid_dense22_3 = Dense(500, activation='tanh')(mid_dense22_3)

mid_dense22_3 = Dense(100)(mid_dense22_3)

# five layers

mid_dense22_4 = Dropout(0.5)(Dense(2000, activation='relu')(features_combo))

mid_dense22_4 = Dense(1000, activation='tanh')(mid_dense22_4)

mid_dense22_4 = Dense(500, activation='relu')(mid_dense22_4)

mid_dense22_4 = Dense(100)(mid_dense22_4)



final_dense22_1 = Concatenate()([mid_dense22_1, mid_dense22_2, mid_dense22_3, mid_dense22_4])

final_dense22_1 = BatchNormalization()(Dense(1000, activation='relu')(final_dense22_1))

final_dense22_1 = BatchNormalization()(Dense(100)(final_dense22_1))

pred3 = Dense(1, name='P3')(final_dense22_1)



# Weighting different predections



p1p2 = Dense(1)(Concatenate()([pred1, pred2]))

p1p3 = Dense(1)(Concatenate()([pred1, pred3]))

p2p3 = Dense(1)(Concatenate()([pred3, pred2]))



predictions = Concatenate()([pred1, pred2, pred3, p1p2, p1p3, p2p3])

combo_pred = Dense(1,name="PG")(predictions)



model = Model(inputs=[In], outputs=[pred1, pred2, pred3, combo_pred], name='model-01')



# model.summary()
keras.utils.plot_model(model, "/kaggle/working/model01.png", show_shapes=True)
# def scheduler(epoch, lr):

#     if lr > 1e-6:

#         return lr * (0.98 ** (epoch // 15))

#     else:

#         return lr



model.compile(loss=MeanSquaredLogarithmicError(),optimizer=Adam(0.01))



hist = model.fit(X_train, y, epochs= 5000, batch_size=32,

          callbacks=[

              tfdocs.modeling.EpochDots(),

              EarlyStopping(monitor='val_loss',

                           patience=100,

                           mode='min',

                           restore_best_weights=True

              ),

              ReduceLROnPlateau(monitor='val_loss', factor=0.96, patience=5,

                  verbose=1, mode='auto', min_delta=1e-5, cooldown=0, min_lr=1e-15

              )

          ], 

          verbose=False,

          validation_split=0.3, shuffle=True)
d = pd.DataFrame(hist.history)

d['Epoch'] = range(0,d.shape[0])
import matplotlib.pyplot as plt

import pylab as plot

params = {'legend.fontsize': 25,

          'legend.handlelength': 3}

plot.rcParams.update(params)



fig, (ax1,ax2, ax3) = plt.subplots(1,3, figsize=(50,20))



fig.legend(["blue", "orange"], prop={"size":20})

d.plot(x="Epoch",y=d.columns.to_list()[1:5], ax=ax1)

d.plot(x="Epoch",y=d.columns.to_list()[6:-2], ax=ax2)

d.plot(x="Epoch",y='lr', ax=ax3)
fig, axes = plt.subplots(4,2, figsize=(40,30), sharey='col')

d.plot(x="Epoch",y=['P1_loss'], ax=axes[0][0])

d.plot(x="Epoch",y=['val_P1_loss'], ax=axes[0][1])

d.plot(x="Epoch",y=['P2_loss'], ax=axes[1][0])

d.plot(x="Epoch",y=['val_P2_loss'], ax=axes[1][1])

d.plot(x="Epoch",y=['P3_loss'], ax=axes[2][0])

d.plot(x="Epoch",y=['val_P3_loss'], ax=axes[2][1])

d.plot(x="Epoch",y=['PG_loss'], ax=axes[3][0])

d.plot(x="Epoch",y=['val_PG_loss'], ax=axes[3][1])
# All predictions by model01

preds = model.predict(X_test)
# ID used for the submission

ID = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv').Id
for index, pred in enumerate(preds):

    output = pd.DataFrame({'Id': ID,

                      'SalePrice': pd.DataFrame(preds[index])[0]})

    output.to_csv(f'/kaggle/working/sub_{index}.csv', index=False)