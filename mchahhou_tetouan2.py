import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import accuracy_score



from keras.callbacks import EarlyStopping, ModelCheckpoint



from keras.layers import Input, Dense, Conv2D, Flatten

from keras.models import Model

from keras.optimizers import SGD, Adam



from keras.utils import np_utils
#et voici le résultat de la transformation.

np_utils.to_categorical([0,1,2,1,0])



#On peut voir 3 colonnes, chaquue colonne correspond à une classe, (première colonne est la classe 0,

#deuxième colonne est la classe 1, troisième colonne est la classe 2).

#On voit que la première colonne contient des 1 pour les lignes appartenant à

#la classe 0 (première et dernière ligne), etc



#cette technique de transformation est appelée le One Hot Encoding (ohe)
df = pd.read_csv('../input/mnist-digit-recognizer/train.csv')



dfX = df.drop('label', axis=1).values

dfY = df.label.values

dfX = dfX/255.



dfY_ohe = np_utils.to_categorical(dfY) #Application du One Hot Encoding

print(dfY_ohe.shape)# Nous avons bien 10 vecteurs
def create_dense_model():

    inpt = Input ( (784,) )

    

    x = Dense(128, activation='relu', name='couche1')(inpt)

    x = Dense(128, activation='relu', name='couche2')(x)

    

    x = Dense(10, activation='softmax', name='output')(x) # la couche de sortie contient 10 neurones

    #chaque neurones va apprendre à prédire la classe qui lui correspond

    #Dans un problème multi_classification, on utilise l'activation softmax au lieu de la sigmoid

    

    model = Model( inpt, x )

    return model



model = create_dense_model()

model.summary()





cv = StratifiedKFold(n_splits=2)

for train_idx, test_idx in cv.split(dfX, dfY):

    model = create_dense_model()

    model.compile( loss='mse' , optimizer=Adam(), metrics=['accuracy'])

    

    es = EarlyStopping(patience=5, monitor='val_accuracy', mode='max')

    mc = ModelCheckpoint('./weights.h5', monitor='val_accuracy', mode='max', save_best_only=True)

    

    trainX = dfX[train_idx]

    trainY = dfY_ohe[train_idx] #on utilise dfY_ohe et non dfY

    

    testX  = dfX[test_idx]    

    testY  = dfY_ohe[test_idx]

    

    model.fit( trainX, trainY, validation_data=[testX, testY], callbacks = [es,mc],

              epochs=1000)

    

    

    model.load_weights('./weights.h5')#On charge les meilleurs poids sauvegardés par le ModelCheckpoint

    #on prédit le Test

    preds = model.predict(testX)

    score_test = accuracy_score( dfY[test_idx], np.argmax(preds, axis=1) )#j'expliquerai au cours

    print (' LE SCORE DE TEST : ', score_test)

    print('')
def create_cnn_model():

    inpt = Input ( (28, 28, 1) )

    

    x = Conv2D(filters=16, kernel_size=(4,4), strides=(2, 2), activation='relu')(inpt)

    x = Conv2D(filters=32, kernel_size=(4,4), strides=(2, 2), activation='relu')(x)

    x = Conv2D(filters=64, kernel_size=(4,4), strides=(2, 2), activation='relu')(x)

    x = Flatten()(x)

    x = Dense(128, activation='relu')(x)  

    x = Dense(10, activation='softmax', name='output')(x)     

    model = Model( inpt, x )

    return model



model = create_cnn_model()

model.summary()





cv = StratifiedKFold(n_splits=2)

for train_idx, test_idx in cv.split(dfX, dfY):

    model = create_cnn_model()

    model.compile( loss='mse' , optimizer=Adam(), metrics=['accuracy'])

    

    es = EarlyStopping(patience=5, monitor='val_accuracy', mode='max')

    mc = ModelCheckpoint('./weights.h5', monitor='val_accuracy', mode='max', save_best_only=True)

    

    trainX = dfX[train_idx]

    trainY = dfY_ohe[train_idx] #on utilise dfY_ohe et non dfY

    trainX = np.reshape( trainX, (-1,28,28, 1) )#on transforme les vecteurs de pixels en matrices représentant les images

    

    testX  = dfX[test_idx]    

    testY  = dfY_ohe[test_idx]

    testX = np.reshape( testX, (-1,28,28, 1) )

    

    model.fit( trainX, trainY, validation_data=[testX, testY], callbacks = [es,mc],

              epochs=1000)

    

    

    model.load_weights('./weights.h5')#On charge les meilleurs poids sauvegardés par le ModelCheckpoint

    #on prédit le Test

    preds = model.predict(testX)

    score_test = accuracy_score( dfY[test_idx], np.argmax(preds, axis=1) )#j'expliquerai au cours

    print (' LE SCORE DE TEST : ', score_test)

    print('')