!tar xzf /kaggle/input/files-ships-2020/ships.tgz  # les images dans des répertoires
types = ['coastguard', 'containership', 'corvette', 'cruiser', 'cv', 'destroyer', 'methanier', 'smallfish', 'submarine', 'tug']

types_id = {t:i for (i,t) in enumerate(types)}



batch_size = 16
from keras.preprocessing.image import ImageDataGenerator 



train_datagen = ImageDataGenerator(

        rescale=1./255,

        horizontal_flip=True,

        validation_split=0.1)



train_generator = train_datagen.flow_from_directory(

        'ships_scaled',

        target_size=(128, 192 ),

        batch_size=batch_size,

        subset="training")



validation_generator = train_datagen.flow_from_directory(

        'ships_scaled',

        target_size=(128, 192 ),

        batch_size=batch_size,

        subset="validation")
from keras.models import Model

from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Activation



inputs = Input(shape=(128, 192, 3), name='cnn_input')



x = Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)

#... pleins de lignes



x = Flatten()(x)

# quelques lignes



outputs = Dense(10, activation="softmax")(x)



model = Model(inputs, outputs)



model.compile(optimizer='rmsprop',   # pas obligatoirement le meilleur algo pour converger

              loss='categorical_crossentropy',

              metrics=['accuracy']

              )
model.fit_generator(

    train_generator,

    steps_per_epoch = train_generator.samples // batch_size,

    validation_data = validation_generator, 

    validation_steps = validation_generator.samples // batch_size,

    epochs = 1)   # 10 permet d'avoir une idée mais probablement pas suffisant pour un beau résultat 
# une autre cellule de fit_generator est possible pour continuer
import numpy as np

import pandas as pd

from keras.utils import np_utils



ships = np.load('/kaggle/input/files-ships-2020/ships_test.npz', allow_pickle=True)

X_test = ships['X']

Y_test = ships['Y']



X_test = X_test.astype('float32') / 255

Y_test_cat = np_utils.to_categorical(Y_test).astype('bool')
score = model.evaluate(X_test, Y_test_cat, verbose=0)

print('Test score:', score[0])

print('Test accuracy:', score[1])
from sklearn.metrics import classification_report, confusion_matrix



res = model.predict(X_test).argmax(axis=1)

confu = confusion_matrix(Y_test, res)

pd.DataFrame({types[i][:3]:confu[:,i] for i in range(len(types))}, index=types)
print(classification_report(Y_test, res, target_names=types))
ships = np.load('/kaggle/input/files-ships-2020/ships_competition.npz', allow_pickle=True)

X_test = ships['X']

X_test = X_test.astype('float32') / 255
# predict results

res = model.predict(X_test).argmax(axis=1)

df = pd.DataFrame({"Category":res})

df.to_csv("reco_nav.csv", index_label="Id")
!head reco_nav.csv
import os

os.chdir(r'/kaggle/working')

from IPython.display import FileLink

FileLink(r'reco_nav.csv')
!rm -rf ships_scaled/