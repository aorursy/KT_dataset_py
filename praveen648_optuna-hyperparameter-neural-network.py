import os



import matplotlib.pyplot as plt

from matplotlib import pyplot



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import tensorflow as tf

from tensorflow.keras.layers import Conv2D, UpSampling2D, LeakyReLU, Concatenate,Dense,Input,Flatten,BatchNormalization,Dropout

from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TensorBoard

from tensorflow.keras import Model



from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import log_loss



from random import randint



import optuna

TRAIN_FEATURES = pd.read_csv('/kaggle/input/lish-moa/train_features.csv')

TEST_FEATURES = pd.read_csv('/kaggle/input/lish-moa/test_features.csv')

TRAIN_TARGETS = pd.read_csv('/kaggle/input/lish-moa/train_targets_scored.csv')

TEST_TARGETS = pd.read_csv('/kaggle/input/lish-moa/sample_submission.csv')
TRAIN_FEATURES.loc[(TRAIN_FEATURES['cp_type'] == "trt_cp"), 'cp_type'] = 0

TRAIN_FEATURES.loc[(TRAIN_FEATURES['cp_type'] == "ctl_vehicle"), 'cp_type'] = 1



TRAIN_FEATURES.loc[(TRAIN_FEATURES['cp_dose'] == "D1"), 'cp_dose'] = 0

TRAIN_FEATURES.loc[(TRAIN_FEATURES['cp_dose'] == "D2"), 'cp_dose'] = 1



TRAIN_FEATURES[["cp_dose", "cp_type"]] = TRAIN_FEATURES[["cp_dose", "cp_type"]].apply(pd.to_numeric)



TEST_FEATURES.loc[(TEST_FEATURES['cp_type'] == "trt_cp"), 'cp_type'] = 0

TEST_FEATURES.loc[(TEST_FEATURES['cp_type'] == "ctl_vehicle"), 'cp_type'] = 1



TEST_FEATURES.loc[(TEST_FEATURES['cp_dose'] == "D1"), 'cp_dose'] = 0

TEST_FEATURES.loc[(TEST_FEATURES['cp_dose'] == "D2"), 'cp_dose'] = 1



TEST_FEATURES[["cp_dose", "cp_type"]] = TEST_FEATURES[["cp_dose", "cp_type"]].apply(pd.to_numeric)
feature_variables = list(set([i for i in TEST_FEATURES.columns])-set(['cp_type','sig_id']))

target_variables =  list(set([i for i in TEST_TARGETS.columns])-set(['sig_id']))
Dataset = pd.merge(TRAIN_FEATURES,TRAIN_TARGETS,on='sig_id')

print(Dataset.shape)

Dataset.drop(Dataset[Dataset['cp_type']==1].index,inplace=True)

print(Dataset.shape)
Test_Dataset = pd.merge(TEST_FEATURES,TEST_TARGETS,on='sig_id')

Test_Dataset.loc[Test_Dataset['cp_type']==1,target_variables]=0
X_train = Dataset[feature_variables].to_numpy()

Y_train = Dataset[target_variables].to_numpy()

X_test = Test_Dataset[feature_variables].to_numpy()
from sklearn.model_selection import train_test_split

x_train,x_val,y_train,y_val = train_test_split(X_train,Y_train,test_size=0.25, random_state=123)
from sklearn.preprocessing import StandardScaler

#Standardise All Columns

scaler = StandardScaler()

scaler.fit(x_train)

x_train = scaler.transform(x_train)

x_val = scaler.transform(x_val)
from sklearn.cluster import KMeans



kmeans = KMeans(n_clusters=10)



kmeans.fit(x_train)
x_train = np.append(x_train,np.expand_dims(kmeans.predict(x_train),axis=1),axis=1)

x_val = np.append(x_val,np.expand_dims(kmeans.predict(x_val),axis=1),axis=1)

X_test = np.append(X_test,np.expand_dims(kmeans.predict(X_test),axis=1),axis=1)
# def objective(trial,x_train=x_train,y_train=y_train,input_feature_shape=len(feature_variables),\

#              output_feature_shape=len(target_variables)):

    

#     #create and validation dataset

#     x_train,x_val,y_train,y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=randint(1,1000))

    

#     np.random.seed(42)     #seed everything

    

#     #Define Model

    

#     model_input = Input(shape = input_feature_shape)

    

#     x = Dense(units = trial.suggest_int("units1",128,2048,log=True),\

#               activation=trial.suggest_categorical("activation1",["relu","sigmoid"]))(model_input)

#     x = BatchNormalization()(x)

#     x = Dropout(rate = trial.suggest_float("rate1",0.1,0.5,log=True))(x)

    

#     x = Dense(units = trial.suggest_int("units2",128,2048,log=True),\

#               activation=trial.suggest_categorical("activation2",["relu","sigmoid"]))(model_input)

#     x = BatchNormalization()(x)

#     x = Dropout(rate = trial.suggest_float("rate2",0.1,0.5,log=True))(x)

    

#     x = Dense(units = trial.suggest_int("units3",128,2048,log=True),\

#               activation=trial.suggest_categorical("activation3",["relu","sigmoid"]))(model_input)

#     x = BatchNormalization()(x)

#     x = Dropout(rate = trial.suggest_float("rate3",0.1,0.5,log=True))(x)

    

#     Output = Dense(output_feature_shape,activation='sigmoid')(x)

    

#     #define model

#     model = Model(model_input,Output,name = 'BaseLine')

    

#     #Define callbacks

    

#     factor = trial.suggest_float("factor",0.4,0.9,log=True)

#     patience = trial.suggest_int("patience",2,9,log=True)

#     min_lr = trial.suggest_float("min_lr",0.00001,0.0001,log=True)

#     lr = trial.suggest_float("lr",0.00001,0.0001,log=True)

#     batch_size = trial.suggest_int("batch_size",128,512,log=True)

#     epochs = trial.suggest_int("epochs",128,512,log=True)

    

#     def callbacks(file_path):

#         reduce_learning_rate = ReduceLROnPlateau(monitor='val_loss',

#                                              factor=factor,

#                                              patience=patience,

#                                              cooldown=1,

#                                              min_lr=min_lr,

#                                              verbose=1)

#         checkpoint = ModelCheckpoint(filepath = file_path,monitor='val_loss',

#                                      mode='min',save_best_only=True,verbose=1)



#         early = EarlyStopping(monitor="val_loss", mode="min", patience= patience)



#         return [reduce_learning_rate,checkpoint,early]

    

#     #file path for callbacks

#     file_path = model.name+'best_weights.hd5'

#     callbacks_list = callbacks(file_path = file_path)

    

#     #optimiser    

#     optimizer = tf.keras.optimizers.Adam(lr=lr, amsgrad=True)

#     #compile the model

#     model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=optimizer)

    

#     history=model.fit(x_train,y_train,epochs= epochs, batch_size=batch_size, callbacks = callbacks_list)



#     predictions = model.predict(x_val)

    

    

#     logloss = tf.keras.losses.BinaryCrossentropy()



#     log_loss_metric = logloss(y_val, predictions).numpy()

    

#     return log_loss_metric
# study = optuna.create_study(direction="minimize",storage='sqlite:///example.db')

# study.optimize(objective, n_trials=250, timeout=25000)
# print("Number of finished trials: {}".format(len(study.trials)))



# print("Best trial:")

# trial = study.best_trial



# print("  Value: {}".format(trial.value))



# print("  Params: ")

# for key, value in trial.params.items():

#     print("    {}: {}".format(key, value))



input_feature_shape=len(feature_variables)

output_feature_shape=len(target_variables)



model_input = Input(shape = input_feature_shape+1)



x = Dense(units = 600,activation="relu")(model_input)

x = BatchNormalization()(x)

x = Dropout(rate = 0.1737384812871308)(x)



x = Dense(units = 298,activation="sigmoid")(model_input)

x = BatchNormalization()(x)

x = Dropout(rate = 0.20010328415413295)(x)



x = Dense(units = 1099,activation="sigmoid")(model_input)

x = BatchNormalization()(x)

x = Dropout(rate = 0.3234867319850344)(x)





Output = Dense(output_feature_shape,activation='sigmoid')(x)



#define model

model = Model(model_input,Output,name = 'BaseLine')



model.summary()
factor= 0.8250037987063858

patience=2

min_lr= 5.101088055532695e-05

lr=6.353131263848553e-05

batch_size=256

epochs=150



def callbacks(file_path):

    reduce_learning_rate = ReduceLROnPlateau(monitor='val_loss',

                                         factor=factor,

                                         patience=patience,

                                         cooldown=1,

                                         min_lr=min_lr,

                                         verbose=1)

    checkpoint = ModelCheckpoint(filepath = file_path,monitor='val_loss',

                                 mode='min',save_best_only=True,verbose=1)



    early = EarlyStopping(monitor="val_loss", mode="min", patience= patience)



    return [reduce_learning_rate,checkpoint,early]



file_path = model.name+'best_weights.hd5'

callbacks_list = callbacks(file_path = file_path)



optimizer = tf.keras.optimizers.Adam(lr=lr, amsgrad=True)

#compile the model

model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=optimizer)



history=model.fit(x_train,y_train,epochs= epochs, batch_size=batch_size, callbacks = callbacks_list)
model.evaluate(x_val, y_val, batch_size=128)
y_pred=model.predict(X_test)

Test_Dataset[target_variables] = y_pred

Test_Dataset.loc[Test_Dataset['cp_type']==1,target_variables]=0
(Test_Dataset[pd.read_csv('../input/lish-moa/sample_submission.csv').columns]).to_csv('submission.csv',index=False)