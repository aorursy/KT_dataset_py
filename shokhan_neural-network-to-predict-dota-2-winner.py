import os

import pandas as pd



PATH_TO_DATA = '../input'



df_train_features = pd.read_csv(os.path.join(PATH_TO_DATA, 

                                             'train_features.csv'), 

                                    index_col='match_id_hash')



df_train_targets = pd.read_csv(os.path.join(PATH_TO_DATA, 

                                            'train_targets.csv'), 

                                   index_col='match_id_hash')



df_test_features = pd.read_csv(os.path.join(PATH_TO_DATA, 

                                            'test_features.csv'), 

                                   index_col='match_id_hash')
X = df_train_features

y = df_train_targets['radiant_win'].astype('int') #models prefer numbers instead of True/False labels

test = df_test_features
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, y, 

                                                      test_size=0.25, random_state=17, 

                                                      shuffle=True, stratify=y)
X_train.shape, X_valid.shape, y_train.shape, y_valid.shape
#we will use Sequential model, where we can add layers easily

from keras.models import Sequential



#now we import different layer types to use in our model

from keras.layers import Dense, Activation, Dropout, BatchNormalization



#we will have to optimize the model, so

from keras import optimizers



#tools to control overfitting

from keras.callbacks import EarlyStopping
#is keras on gpu?

from keras import backend as K

K.tensorflow_backend._get_available_gpus()
def auc(y_true, y_pred):

    auc = tf.metrics.auc(y_true, y_pred)[1]

    K.get_session().run(tf.local_variables_initializer())

    return auc
model = Sequential() #we define the model



#first layer

model.add(Dense(512,input_dim=X_train.shape[1])) #basicly,245x512 matrix of weights

model.add(BatchNormalization()) #makes NN learning easier

model.add(Activation('relu')) #applying non-linear transformation

model.add(Dropout(0.2)) #say no to overfitting



#second layer

model.add(Dense(256)) #here we don't have to specify its input_dim, it knows it from previous layer automaticly

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(Dropout(0.2))



#third layer

model.add(Dense(256)) #here we don't have to specify its input_dim, it knows it from previous layer automaticly

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(Dropout(0.2))



#you can experiment with the number of layers as well...



#now the final layer to convert everything to one number(probability)

model.add(Dense(1, activation='sigmoid'))
model.summary()
#Adam optimizer, there others as well, like RMSProp

adam = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999)



#EarlyStop if things does not improve for some time

earlystop = EarlyStopping(monitor="val_auc", patience=20, verbose=1, mode='max')
#Telling the model what to do...

import tensorflow as tf

model.compile(optimizer=adam,loss='binary_crossentropy',metrics = [auc])  
history = model.fit(X_train, y_train, validation_data = (X_valid,y_valid),\

                       epochs=20,batch_size=64,verbose=1,callbacks=[earlystop])
# Plot the loss and auc curves for training and validation 

import matplotlib.pyplot as plt



fig, ax = plt.subplots(2,1)

ax[0].plot(history.history['loss'], color='b', label="Training loss")

ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])

legend = ax[0].legend(loc='best', shadow=True)



ax[1].plot(history.history['auc'], color='b', label="Training auc")

ax[1].plot(history.history['val_auc'], color='r',label="Validation auc")

legend = ax[1].legend(loc='best', shadow=True)
#simply as it

predictions = model.predict(test)[:,0]
predictions
df_submission = pd.DataFrame({'radiant_win_prob': predictions}, 

                                 index=df_test_features.index)

import datetime

submission_filename = 'submission_{}.csv'.format(

    datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

df_submission.to_csv(submission_filename)

print('Submission saved to {}'.format(submission_filename))