!pip install pyts
# IMPORTINGS

import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from tqdm.notebook import tqdm

import gc

gc.enable()



# Sklearn

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import KFold

from sklearn.metrics import mean_absolute_error



from pyts.image import GramianAngularField



# Keras

from keras import applications

from keras.models import Sequential

from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense

from keras.optimizers import Adam
# GLOBAL VARIABLES

PATH = '/kaggle/input/predict-volcanic-eruptions-ingv-oe'

N_SPLITS = 5

IMG_SIZE = 28

CHANNELS = 10

LR = 1e-5

EPOCHS = 30

BS = 64

DROP = 0.5

SEED = 42





# HELPING FUNCTIONS

def tseries_to_img(df_list, type_d, method='difference'):

    """

    Function that transforms segments time series into images using GramianAngularField

    

    Segments are preprocessed as follows:

    - Missing data in channels are filled with the mean

    - If a channel is completely missing --> fill with zeroes

    

    Once the data of all sensors are converted to an image, all images are stacked horizontally

    

    """

    

    df_list.index = df_list['segment_id']

    df_img = np.zeros((df_list.shape[0], IMG_SIZE, IMG_SIZE*CHANNELS, 1))



    

    for r,seg in enumerate(tqdm(df_list['segment_id'].values.tolist())):

        seg_df = pd.read_csv(os.path.join(PATH, type_d, str(seg)+'.csv'))

        

        for r2,sens in enumerate(range(CHANNELS)):

            seg_sens = seg_df['sensor_'+str(sens+1)].values.reshape(1,-1)

            

            if np.isnan(seg_sens).sum() < seg_sens.shape[1]:

                seg_sens[np.isnan(seg_sens)] = np.mean(seg_sens[~np.isnan(seg_sens)])

            else:

                seg_sens[np.isnan(seg_sens)] = 0

                

            gadf = GramianAngularField(image_size=IMG_SIZE, method=method)

            seg_sens_gadf = gadf.fit_transform(seg_sens)

            

            df_img[r, :, 0+IMG_SIZE*r2:IMG_SIZE*(r2+1), 0] = seg_sens_gadf[0,:,:]

        

    return df_img

    

    

def create_cnn():

    """

    Function that creates a CNN model using Keras

    """

    model = Sequential()

    model.add(Conv2D(32, (3,3), activation='relu', kernel_initializer='he_uniform',

                     input_shape=(IMG_SIZE, IMG_SIZE*CHANNELS, 1)))

    model.add(BatchNormalization())

    model.add(MaxPooling2D())

    model.add(Dropout(DROP))

    model.add(Flatten())

    model.add(Dense(16, activation='relu', kernel_initializer='he_uniform'))

    model.add(Dense(1, activation='linear'))

    model.compile(optimizer=Adam(lr=LR), loss='mae', metrics=['mae'])

    print(model.summary())

    return model

    



def crossvalidate_model(kf, X_train_f, y_train_f, X_test_f, plt_hist=True):

    """

    Function to crossvalidate model

    """

    

    iofs_preds = np.zeros((X_train_f.shape[0],1))

    oofs_preds = np.zeros((X_train_f.shape[0],1))

    test_preds = np.zeros((X_test_f.shape[0],1))

    

    for k, (trn_idx, val_idx) in enumerate(kf.split(X_train_f, y_train_f)):



        X_trn_cv, y_trn_cv = X_train_f[trn_idx,:], y_train_f[trn_idx,:]

        X_val_cv, y_val_cv = X_train_f[val_idx,:], y_train_f[val_idx,:]

        

        sc = MinMaxScaler(feature_range=(-1,1))

        sc.fit(y_trn_cv)

        y_trn_cv_sc = sc.transform(y_trn_cv)

        y_val_cv_sc = sc.transform(y_val_cv)

        

        model = create_cnn()

        history = model.fit(X_trn_cv, y_trn_cv_sc, 

                            epochs=EPOCHS, batch_size=BS, 

                            validation_data=(X_val_cv, y_val_cv_sc),

                            verbose=1)

        

        if plt_hist:

            fig, ax = plt.subplots(1,2, figsize=(10,5))

            ax[0].plot(history.history['loss'])

            ax[0].plot(history.history['val_loss'])

            ax[1].plot(history.history['mae'])

            ax[1].plot(history.history['val_mae'])

            plt.show()

        

        y_pred_trn_cv = sc.inverse_transform(model.predict(X_trn_cv))

        y_pred_val_cv = sc.inverse_transform(model.predict(X_val_cv))

        y_pred_test = sc.inverse_transform(model.predict(X_test_f))

        

        print("Fold {} train MAE: {}".format(k+1, mean_absolute_error(y_trn_cv, y_pred_trn_cv)))

        print("Fold {} val MAE: {}".format(k+1, mean_absolute_error(y_val_cv, y_pred_val_cv)))

        

        iofs_preds[trn_idx] = y_pred_trn_cv

        oofs_preds[val_idx] = y_pred_val_cv

        test_preds += y_pred_test / kf.get_n_splits()

        

    print("Overall train MAE: {}".format(mean_absolute_error(y_train_f, iofs_preds)))

    print("Overall val MAE: {}".format(mean_absolute_error(y_train_f, oofs_preds)))

    

    return iofs_preds, oofs_preds, test_preds
# Load data

train_df = pd.read_csv(os.path.join(PATH,'train.csv'))

sub = pd.read_csv(os.path.join(PATH,'sample_submission.csv'))
# Prepare data

df_list_train = train_df.copy()

y_train = df_list_train['time_to_eruption'].values.reshape(-1,1)

X_train = tseries_to_img(df_list_train, type_d='train', method='difference')



df_list_test = sub.copy()

X_test = tseries_to_img(df_list_test, type_d='test', method='difference')
# Plot some training and test images

fig, ax = plt.subplots(4,1, figsize=(20,10))

ax[0].imshow(X_train[0,:,:,0])

ax[0].set_title('Segment '+str(df_list_train['segment_id'].iloc[0]))

ax[1].imshow(X_train[1,:,:,0])

ax[1].set_title('Segment '+str(df_list_train['segment_id'].iloc[1]))

ax[2].imshow(X_test[0,:,:,0])

ax[2].set_title('Segment '+str(df_list_test['segment_id'].iloc[0]))

ax[3].imshow(X_test[1,:,:,0])

ax[3].set_title('Segment '+str(df_list_test['segment_id'].iloc[1]))

plt.show()



del df_list_train, df_list_test

gc.collect()
# Crossvalidate model

kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

iofs_preds, oofs_preds, test_preds = crossvalidate_model(kf, X_train, y_train, X_test, plt_hist=False)



# Save predictions for test set

sub['time_to_eruption'] = test_preds

sub.to_csv('submission.csv', index=False)
# nulls_df = pd.DataFrame(index=train_df['segment_id'], columns=['channels_nulls'])

# nulls_seg = []

# nulls_nulls = []

# for s in tqdm(train_df['segment_id']):

#     s_df = pd.read_csv(os.path.join(PATH, 'train', str(s)+'.csv'))

#     if s_df.isnull().sum().any():

#         tmp = s_df.isnull().sum()

#         nulls_df.loc[s,'channels_nulls'] = str(tmp[tmp!=0].to_dict())

#         nulls_seg.append(tmp[tmp!=0].index.tolist())

#         nulls_nulls.append(tmp[tmp!=0].tolist())

        

# nulls_df = nulls_df.loc[nulls_df['channels_nulls'].notnull(),:]

# nulls_seg = [c1 for c2 in n_seg for c1 in c2]

# nulls_nulls = [c1 for c2 in n_nulls for c1 in c2]