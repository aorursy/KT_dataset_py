import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as bk
import tensorflow.keras.layers as ly
import tensorflow.keras.models as ml
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
import tensorflow_addons as tfa
from sklearn.model_selection import train_test_split
import xgboost
%%time
train_df=pd.read_csv('/kaggle/input/predict-volcanic-eruptions-ingv-oe/train.csv')
n_f=12
total_data=np.empty((train_df.shape[0],n_f*10))
time_=np.empty((train_df.shape[0],1))
for i_,seg_ in enumerate(train_df['segment_id']):
    the_df=pd.read_csv(f'/kaggle/input/predict-volcanic-eruptions-ingv-oe/train/{seg_}.csv').fillna(0)
    total_data[i_,:]=np.concatenate((the_df.abs().mean().to_numpy(),
                                    the_df.std().to_numpy(),
                                    the_df.mean().to_numpy(),
                                    the_df.var().to_numpy(),
                                    the_df.min().to_numpy(),
                                    the_df.max().to_numpy(),
                                    the_df.median().to_numpy(),
                                    the_df.quantile([0.1,0.25,0.5,0.75,0.9]).to_numpy().reshape(1,-1)[0]))
    time_[i_,0]=train_df.loc[i_,'time_to_eruption']
%%time
sample_submission_df=pd.read_csv('../input/predict-volcanic-eruptions-ingv-oe/sample_submission.csv')
n_f=12
total_data_test_=np.empty((sample_submission_df.shape[0],n_f*10))
for i_,seg_ in enumerate(sample_submission_df['segment_id']):
    the_df=pd.read_csv(f'/kaggle/input/predict-volcanic-eruptions-ingv-oe/test/{seg_}.csv').fillna(0)
    total_data_test_[i_,:]=np.concatenate((the_df.abs().mean().to_numpy(),
                                    the_df.std().to_numpy(),
                                    the_df.mean().to_numpy(),
                                    the_df.var().to_numpy(),
                                    the_df.min().to_numpy(),
                                    the_df.max().to_numpy(),
                                    the_df.median().to_numpy(),
                                    the_df.quantile([0.1,0.25,0.5,0.75,0.9]).to_numpy().reshape(1,-1)[0]))
del the_df
def create_my_model():
    model = ml.Sequential()
    model.add(ly.Input(total_data.shape[1]))
    model.add(ly.BatchNormalization())
    model.add(tfa.layers.WeightNormalization(ly.Dense(1000,activation='relu')))
    model.add(ly.BatchNormalization())
    model.add(ly.Dropout(0.7))
    model.add(tfa.layers.WeightNormalization(ly.Dense(1,activation='relu')))


    model.compile(optimizer=tfa.optimizers.AdamW(lr = 1, weight_decay = 1e-5, clipvalue = 900),loss='mean_absolute_error')
    return model
cb_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, min_lr=1e-7, patience=2, verbose=1, mode='min')
cb_early = EarlyStopping(monitor="val_loss", mode="min", restore_best_weights=True, patience= 5, verbose = 1)
model=create_my_model()
X_train1, X_val, y_train1, y_val = train_test_split(total_data, time_, test_size=0.1, random_state=3)
model.fit(X_train1,y_train1,batch_size=8,epochs=600,verbose=1,validation_data=(X_val,y_val),callbacks=[cb_lr,cb_early])
sample_submission_df1=pd.read_csv('../input/predict-volcanic-eruptions-ingv-oe/sample_submission.csv')
sample_submission_df['time_to_eruption']=model.predict(total_data_test_)
sample_submission_df.to_csv('nn_res.csv',index=False)
#remove `tree_method` if you dont have gpu as processor
model1 = xgboost.XGBRegressor(n_estimators=100000,tree_method='gpu_hist',max_depth=8,learning_rate=0.05,alpha=0.1,SUBSAMPLE=0.6)
X_train1, X_val, y_train1, y_val = train_test_split(total_data, time_, test_size=0.1, random_state=3)
eval_set = [(X_val, y_val)]
model1.fit(X_train1, y_train1,early_stopping_rounds=5,eval_metric='mae', eval_set=eval_set, verbose=True)
sample_submission_df1=pd.read_csv('../input/predict-volcanic-eruptions-ingv-oe/sample_submission.csv')
sample_submission_df1['time_to_eruption']=model1.predict(total_data_test_)[:,None]
sample_submission_df1.to_csv('xgb_res.csv',index=False)
