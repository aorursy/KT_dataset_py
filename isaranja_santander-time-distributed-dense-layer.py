import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# plotting

import matplotlib

import matplotlib.pyplot as plt



# preprocessing

from sklearn.preprocessing import StandardScaler # normalization



# keras packages

from keras.models import Sequential, Model

from keras.layers import Input, Dense, Dropout, BatchNormalization, Flatten



from keras import optimizers



# model selection and metrics

from sklearn.model_selection import train_test_split

from sklearn import metrics



# evaluation

from sklearn.metrics import confusion_matrix, accuracy_score , roc_auc_score



#

import tensorflow as tf
from keras import backend as K

K.tensorflow_backend._get_available_gpus()
test_df = pd.read_csv("../input/test.csv")

train_df = pd.read_csv("../input/train.csv")

submission = pd.read_csv("../input/sample_submission.csv")
train_df = train_df.assign(isTrain = True)

test_df = test_df.assign(isTrain=False)



full_df = pd.concat([train_df,test_df],sort=False)
scaler = StandardScaler()

X=scaler.fit_transform(full_df.loc[:,'var_0':'var_199'].values)
x = X[full_df.isTrain]

y = full_df[full_df.isTrain].target

x_test = X[~full_df.isTrain]



x_train, x_val, y_train, y_val = train_test_split(x, y , test_size=0.33, stratify=y, random_state=42)
x_train = x_train.reshape(-1,200,1)

x_val = x_val.reshape(-1,200,1)

x_test = x_test.reshape(-1,200,1)
# focal loss 

def focal_loss(alpha=0.25,gamma=5.0):

    def focal_crossentropy(y_true, y_pred):

        bce = K.binary_crossentropy(y_true, y_pred)

        

        y_pred = K.clip(y_pred, K.epsilon(), 1.- K.epsilon())

        p_t = (y_true*y_pred) + ((1-y_true)*(1-y_pred))

        alpha_t =  y_true*alpha + ((1-alpha)*(1-y_true))



        # compute the final loss and return

        return K.mean(alpha_t*K.pow((1-p_t), gamma)*bce, axis=-1)

    return focal_crossentropy
def auc(y_true, y_pred):

    auc = tf.metrics.auc(y_true, y_pred,num_thresholds=50000)[1]

    K.get_session().run(tf.local_variables_initializer())

    return auc
model_nn = Sequential()

model_nn.add(Dense(4, input_shape = (200,1),activation='relu'))

model_nn.add(BatchNormalization())

model_nn.add(Dense(16,activation='relu'))

model_nn.add(BatchNormalization())

model_nn.add(Flatten())

model_nn.add(Dense(1, activation='sigmoid'))



model_nn.compile(loss=[focal_loss(alpha=.25, gamma=2.0)], 

              optimizer='adam',

              metrics=['accuracy',auc])



model_nn.summary()
from keras.callbacks import EarlyStopping, ModelCheckpoint

# Set callback functions to early stop training and save the best model so far

callbacks = [EarlyStopping(monitor='val_loss', patience=100)]
epochs = 200

history = model_nn.fit(x_train,

                    y_train,

                    epochs=epochs,

                    verbose=2,

                    batch_size=2048,

                    callbacks=callbacks,

                    validation_data= (x_val,y_val),

                    shuffle=True)
y_pred = model_nn.predict(x_val)

predictions = np.round(y_pred)



print(" Confusion metrix ")

print(confusion_matrix(y_val,predictions))

print(" Prediction accuracy:%2f%% Prediction auc:%2f%% Baseline accuracy:%2f%%  " %((accuracy_score(y_val,predictions)*100.00) ,roc_auc_score(y_val,y_pred)*100.00 ,((1-np.sum(y_val)/len(predictions))*100)))



x_axis = range(0,np.max(history.epoch)+1)



fig, axes = plt.subplots(ncols=2,nrows=1, figsize=(20,5))



axes[0].plot(x_axis,history.history['loss'],label='Train')

axes[0].plot(x_axis,history.history['val_loss'],label='Test')

axes[0].legend()

axes[0].set_ylabel('loss')

axes[0].set_title('nn log loss')



# plot classificaton error

axes[1].plot(x_axis,history.history['auc'],label='Train')

axes[1].plot(x_axis,history.history['val_auc'],label='Val')

axes[1].legend()

axes[1].set_ylabel('Classification auc')

axes[1].set_title('nn Classification auc')



plt.show()
submission = full_df.loc[~full_df.isTrain,['ID_code','target']]

submission['target'] = model_nn.predict(x_test)

submission.to_csv('submission.csv',index=False)