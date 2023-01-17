import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from keras.models import Input,Model,Sequential,load_model

from keras.layers import Activation,Add,BatchNormalization,Conv2D,Dropout

from keras.layers import Dense,GlobalAveragePooling2D,MaxPooling2D

from keras.optimizers import adam



import os

print(os.listdir("../input"))



import keras

from PIL import Image

import numpy as np



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
%%time

df_train = pd.read_csv('../input/aerial-cactus-identification/train.csv')

df_test = pd.read_csv('../input/aerial-cactus-identification/sample_submission.csv')

df_train.head(3)
def plot_roc_auc(truelabel, pred):

    fpr, tpr, thresholds = sklearn.metrics.roc_curve(truelabel, pred)

    auc = sklearn.metrics.auc(fpr, tpr)

#     print(auc)



    plt.plot(fpr, tpr, label='ROC curve (auc = %.6f)'%auc)

    plt.fill_between(x=fpr, y1=tpr,facecolor='yellow', alpha=0.5 )

    plt.legend()

    plt.title('ROC curve')

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.grid(True)

#     plt.show()

    return auc
# TRAIN_DATA_PATH ='../input/train/train'

TRAIN_DATA_PATH ='../input/aerial-cactus-identification/train/train'

sample = TRAIN_DATA_PATH +'/'+ os.listdir(TRAIN_DATA_PATH)[0]

# TEST_DATA_PATH  = '../input/test/test'

TEST_DATA_PATH  = '../input/aerial-cactus-identification/test/test'

# os.listdir('../input/test/test')

np.array(Image.open(sample)).shape
fig, ax = plt.subplots(4, 8, figsize=(12,6))

for i in range(32):

    ax[i//8][i%8].tick_params(labelbottom=False, labelleft=False, bottom=False, left=False,) 

    target = df_train.iloc[i]['has_cactus']

    ax[i//8][i%8].set_title(f'{i} -> {target}')

    ax[i//8][i%8].imshow(np.array(Image.open(TRAIN_DATA_PATH +'/'+ df_train.iloc[i]['id'])),)

plt.tight_layout()    
sns.countplot(df_train.has_cactus)
print ('target 0:1->',len(df_train[df_train.has_cactus==0])/len(df_train[df_train.has_cactus==1]))
%%time

tmp = [] 

for i in range(len(df_train)):

    tmp.append(np.array(Image.open(TRAIN_DATA_PATH +'/'+ df_train.iloc[i]['id'])))

X_train, y_train = np.array(tmp)/255, df_train['has_cactus']

tmp = [] 

for i in range(len(df_test)):

    tmp.append(np.array(Image.open(TEST_DATA_PATH +'/'+ df_test.iloc[i]['id'])))

X_test = np.array(tmp)/255

del tmp

print(X_train.shape, y_train.shape)

print(X_test.shape)
def build_model(input_shape):

    model =Sequential()

    model.add(Conv2D(64,(3,3),padding='same',input_shape=(input_shape)))

    model.add(Activation('relu'))

    model.add(BatchNormalization(scale=False))

    model.add(Conv2D(64,(3,3),padding='same'))

    model.add(Activation('relu'))

    model.add(BatchNormalization(scale=False))    

    model.add(Conv2D(64,(3,3),padding='same'))

    model.add(Activation('relu'))

    model.add(BatchNormalization(scale=False))        

    model.add(MaxPooling2D())

    model.add(Dropout(0.5))

    

    model.add(Conv2D(128,(3,3),padding='same'))

    model.add(Activation('relu'))

    model.add(BatchNormalization(scale=False))

    model.add(Conv2D(128,(3,3),padding='same'))

    model.add(Activation('relu'))

    model.add(BatchNormalization(scale=False)) 

    model.add(Conv2D(128,(3,3),padding='same'))

    model.add(Activation('relu'))

    model.add(BatchNormalization(scale=False))        

    model.add(MaxPooling2D())

    model.add(Dropout(0.5))



    model.add(Conv2D(256,(3,3),padding='same'))

    model.add(Activation('relu'))

    model.add(BatchNormalization(scale=False))

    model.add(Conv2D(256,(3,3),padding='same'))

    model.add(Activation('relu'))

    model.add(BatchNormalization(scale=False)) 

    model.add(Conv2D(256,(3,3),padding='same'))

    model.add(Activation('relu'))

    model.add(BatchNormalization(scale=False))          

    

    model.add(GlobalAveragePooling2D())

    model.add(Dense(256))

    model.add(Activation('relu'))    

    model.add(Dropout(0.5))

    

    model.add(Dense(1))

    model.add(Activation('sigmoid'))

    model.compile('adam',loss='binary_crossentropy',metrics=['accuracy']) 

#     model.add(Dense(2))

#     model.add(Activation('softmax'))    

#     model.compile('adam',loss='categorical_crossentropy',metrics=['accuracy']) 

    

    return model 
#data count a:100 b:200 c:300 weights a:3 b:1.5 c:1 

1.0 / len(df_train[df_train.has_cactus==0]) * len(df_train[df_train.has_cactus==1]) 
%%time

import sklearn

from sklearn.preprocessing import *

from sklearn.model_selection import train_test_split,KFold,StratifiedKFold

import keras.backend as K

from sklearn.metrics import *

from keras.preprocessing.image import ImageDataGenerator

histories = []

oof_pred = np.zeros(len(df_train))

sub_pred = np.zeros(len(df_test))



class_weights = {} 

weights = [3.010082493125573,#3.01,

           1.0]

len(df_train[df_train.has_cactus==0]) 

for i in range(2): 

    class_weights[i] = weights[i] 

print('class_weights:',class_weights)



checkpoint_name = '/checkpoint.file'

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

BATCH_SIZE = 64#128

EPOCHS = 128



for fold_id, (train_index, val_index) in enumerate(skf.split(X_train, y_train)):

    print(f'fold id: {fold_id}')

    X_tr, y_tr = X_train[train_index], y_train[train_index]

    X_val, y_val = X_train[val_index], y_train[val_index]



    callbacks=[

        keras.callbacks.ModelCheckpoint(

            checkpoint_name, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False),

        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=5, verbose=1,min_delta=0.00005, ),

        keras.callbacks.EarlyStopping(monitor='val_loss', patience=15)

    ]   



    datagen = ImageDataGenerator(

#             width_shift_range=0.1,#2,

            height_shift_range=0.1,

            horizontal_flip=True

    )

    K.clear_session()

    model = build_model(X_train.shape[1:])   

    model.summary()    

    

    histories.append(

#         model.fit(X_tr, y_tr, batch_size=16, epochs=128,#64, 

#                   validation_data=(X_val, y_val), 

#                   class_weight=class_weights,verbose=2, callbacks=callbacks)

        model.fit_generator(

            datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),

            steps_per_epoch=int(np.ceil(len(X_train) / BATCH_SIZE)), validation_data=(X_val, y_val), 

            epochs=EPOCHS, class_weight=class_weights, 

            callbacks=callbacks, verbose=2)

    )

    

    model = load_model(checkpoint_name)

    oof_pred[val_index] = model.predict(X_val).flatten()

    sub_pred += model.predict(X_test).flatten() / skf.n_splits

    print(roc_auc_score(y_val, oof_pred[val_index]))

    plot_roc_auc(y_val, oof_pred[val_index])

    del callbacks

sub_pred = np.clip(sub_pred,0.0,1.0)    
sub_pred.min(),sub_pred.max()
plt.hist(sub_pred)

plt.show()



plt.title('auc count (between 0.80 - 0.20)')

plt.hist(sub_pred[(sub_pred<0.80) & (sub_pred>0.20)], bins=100)

plt.show()



plt.title('auc count (between 0.70 - 0.30)')

plt.hist(sub_pred[(sub_pred<0.70) & (sub_pred>0.30)], bins=100)

plt.show()



plt.title('auc count (between 0.60 - 0.40)')

plt.hist(sub_pred[(sub_pred<0.60) & (sub_pred>0.40)], bins=100)

plt.show()
fig, ax = plt.subplots(4, 8, figsize=(12,6))

for i in range(32):

    ax[i//8][i%8].tick_params(labelbottom=False, labelleft=False, bottom=False, left=False,) 

    target = df_train.iloc[i]['has_cactus']

    pred = oof_pred[i]

    ax[i//8][i%8].set_title(f'{i}->{target}/{pred:.2f}')

    ax[i//8][i%8].imshow(np.array(Image.open(TRAIN_DATA_PATH +'/'+ df_train.iloc[i]['id'])),)

plt.tight_layout()    
fig, ax = plt.subplots(4, 8, figsize=(12,6))

for i in range(32):

    ax[i//8][i%8].tick_params(labelbottom=False, labelleft=False, bottom=False, left=False,)    

    target = sub_pred[i]

    ax[i//8][i%8].set_title(f'{i} -> {target:.2f}')

    ax[i//8][i%8].imshow(np.array(Image.open(TEST_DATA_PATH +'/'+ df_test.iloc[i]['id'])),)

plt.tight_layout()   
print('Ambiguous image index:',np.where((sub_pred<0.80) & (sub_pred>0.20))[:32][0])
fig, ax = plt.subplots(4, 8, figsize=(12,6))

for i, idx in enumerate(np.where((sub_pred<0.80) & (sub_pred>0.20))[:32][0]):

    ax[i//8][i%8].tick_params(labelbottom=False, labelleft=False, bottom=False, left=False,) 

    target = sub_pred[idx]

    ax[i//8][i%8].set_title(f'{i} -> {target:.3f}')

    ax[i//8][i%8].imshow(np.array(Image.open(TEST_DATA_PATH +'/'+ df_test.iloc[idx]['id'])),)

plt.tight_layout()   
submission = pd.read_csv('../input/aerial-cactus-identification/sample_submission.csv')

submission['has_cactus'] = sub_pred

submission.to_csv('submission.csv', index=False)



submission.head(50)