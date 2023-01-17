# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


#load Data Here
print("Fetching Training Data...")
train = pd.read_csv("../input/train.csv")

print("Fetching Testing Data...")
test = pd.read_csv("../input/test.csv")
Ytrain = np.array(train['label'])
Xtrain = np.array(train.iloc[:,1:]).reshape(-1,28,28,1)
Xtest = np.array(test).reshape(-1,28,28,1)

#normalize X
Xtrain = Xtrain/255.0
Xtest = Xtest/255.0

from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

def getModel():
    
    mi = Input(shape=(28,28,1), name="images")

    x = Conv2D(32, kernel_size=(5, 5), activation='relu')((BatchNormalization(momentum=0))(mi))
    x = Conv2D(32, kernel_size=(3, 3), activation='relu')(x)
    x = (MaxPooling2D(pool_size=(2, 2)))(x)
    x = Dropout(0.2)(x)
    
    x = Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
    x = Conv2D(128, kernel_size=(3, 3), activation='relu')(x)
    x = (MaxPooling2D(pool_size=(2, 2)))(x)
    x = Dropout(0.2)(x)
    
    x = Flatten()(x)
    
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    main_output = Dense(10, activation="softmax", name='main_output')(x)
    
    model = Model(mi,main_output)
    
    myoptim=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
       
    model.compile(loss='categorical_crossentropy',
                  optimizer=myoptim,
                  metrics=['accuracy'])
    
    return model
model = getModel()
model.summary()
from sklearn.model_selection import StratifiedShuffleSplit

sss = StratifiedShuffleSplit(n_splits=5)
i=0

for train_index, cv_index in sss.split(Xtrain,Ytrain):

    i += 1

    X_train, X_cv = Xtrain[train_index], Xtrain[cv_index]
    y_train, y_cv = Ytrain[train_index], Ytrain[cv_index]
    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, 10)
    Y_cv = np_utils.to_categorical(y_cv, 10)
        
    # Data Info, etc
    nb_classes = 10
    nb_epoch = 100
    batch_size = 32
    
    earlyStopping = EarlyStopping(monitor='val_loss', patience=6, verbose=0, mode='min')
    mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.05, patience=2, verbose=1, epsilon=1e-4, mode='min')
    history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=0, callbacks=[earlyStopping, mcp_save,reduce_lr_loss],validation_data=(X_cv,Y_cv))

    model.load_weights(filepath = '.mdl_wts.hdf5')

    score = model.evaluate(X_cv, Y_cv, verbose=2)
    print('CV Train loss:', score[0])
    print('CV Train accuracy:', score[1])

    pred = model.predict(Xtest, batch_size=batch_size, verbose=0)

    pred_val = np.argmax(pred,axis=1)
    
    submission = pd.DataFrame({'ImageId': np.arange(1,len(Xtest[:,...])+1),'Label': pred_val})
    print(submission.head(10))
    
    #submission.to_csv('../input/Submission_'+str(i)+'.csv', index=False)
    
    break # remove to do all folds
