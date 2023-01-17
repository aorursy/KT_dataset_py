!tar xf /kaggle/input/files-ships-2020/ships.tgz  # les images dans des répertoires!tar vxf /kaggle/input/files-ships-2020/ships.tgz  # les images dans des répertoires
types = ['coastguard', 'containership', 'corvette', 'cruiser', 'cv', 'destroyer', 'methanier', 'smallfish', 'submarine', 'tug']

types_id = {t:i for (i,t) in enumerate(types)}



batch_size = 64
from keras.preprocessing.image import ImageDataGenerator 



#After trying different experiments, i just realized that the training data plays the most important role.

#Data Augmentation

train_datagen = ImageDataGenerator(

        rescale=1./255,

        horizontal_flip=True,

        validation_split=0.1,

        rotation_range=6,

        shear_range=8,

        zoom_range=[0.9, 1.5],  

        brightness_range=[0.7, 1.2],

        width_shift_range=0.2,

        height_shift_range=0.3,

        channel_shift_range = 50,

)



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
# from matplotlib import pyplot as plt

# #128, 192, 3

# X_batch, y_batch = next(train_generator)

# plt.figure(figsize=(20, 15))

# for i in range(9):

#     print(y_batch[i])

#     plt.subplot(331 + i)

#     plt.imshow(X_batch[i])

# plt.show()
from keras.models import Model

from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Activation, concatenate

from keras.regularizers import l1_l2



inputs = Input(shape=(128, 192, 3), name='cnn_input')

#... pleins de lignes



#with 2 stacked convolutional (two feature maps are stacked along the depth), the model is able to detect the details of the ships. 



#block 1

#feature maps: extracting edges

x = Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)

x = BatchNormalization()(x)

x = MaxPooling2D(pool_size=(2, 2))(x) #downsampling



#block 2: 2 stacked convolutional layers

x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)

x = BatchNormalization()(x)

x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)

x = BatchNormalization()(x)

x = MaxPooling2D(pool_size=(2, 2))(x)  #downsampling



#block 3: 2 stacked convolutional layers



x = Conv2D(192, (3, 3), padding='same', activation='relu')(x)

x = BatchNormalization()(x)

x = Conv2D(192, (3, 3), padding='same', activation='relu')(x)

x = BatchNormalization()(x)

x = MaxPooling2D(pool_size=(2, 2))(x)  #downsampling



x = Flatten()(x)



#Fully connected layers



x = Dense(256, activation='relu')(x)  

x = BatchNormalization()(x)

x = Dropout(0.3)(x) #avoid overfitting



x = Dense(256, activation='relu')(x) 

x = BatchNormalization()(x)

x = Dropout(0.4)(x) #avoid overfitting



outputs = Dense(10, activation="softmax")(x)



model = Model(inputs, outputs)



model.compile(optimizer='rmsprop',   # pas obligatoirement le meilleur algo pour converger

              loss='categorical_crossentropy',

              metrics=['accuracy']

              )
model.summary()
from sklearn.utils import class_weight

import numpy as np

#The current dataset is imbalanced => too few data for cv, corvette, submarine (the 3 lowest accuracies in previous experiments)

#generate class weights for dataset

class_weights = class_weight.compute_class_weight('balanced', np.unique(train_generator.classes), train_generator.classes)

print(class_weights)
from keras.callbacks import EarlyStopping, ModelCheckpoint



history = model.fit_generator(

        train_generator,

        steps_per_epoch = train_generator.samples // batch_size,

        validation_data = validation_generator, 

        validation_steps = validation_generator.samples // batch_size,

        callbacks = [

#             EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=10),

            ModelCheckpoint("best_model.hdf5", monitor='val_accuracy', verbose=1, save_best_only=True),

        ],

        class_weight = class_weights,

        epochs = 90)   # 10 permet d'avoir une idée mais probablement pas suffisant pour un beau résultat 
# une autre cellule de fit_generator est possible pour continuer
from matplotlib import pyplot as plt

plt.figure(figsize=(8, 6))



plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'val'], loc='upper left')

plt.show()
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
from keras.models import load_model

best_model = load_model('/kaggle/working/best_model.hdf5')

score = best_model.evaluate(X_test, Y_test_cat, verbose=0)

print('Test score:', score[0])

print('Test accuracy:', score[1])

res = best_model.predict(X_test).argmax(axis=1)

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
from IPython.display import HTML

def create_download_link(title = "Download CSV file", filename = "data.csv"):  

    html = '<a href={filename}>{title}</a>'

    html = html.format(title=title,filename=filename)

    return HTML(html)



# create a link to download the dataframe which was saved with .to_csv method

create_download_link(title="Download reco_nav.csv", filename='reco_nav.csv')
res = best_model.predict(X_test).argmax(axis=1)

df = pd.DataFrame({"Category":res})

df.to_csv("reco_nav_best_model.csv", index_label="Id")

create_download_link(title="Download reco_nav_best_model.csv", filename='reco_nav_best_model.csv')
!head reco_nav_best_model.csv
create_download_link(title="Download best_model.hdf5", filename='best_model.hdf5')
!rm ships_scaled -rf
!ls /kaggle/working