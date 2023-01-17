# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/input/pokemonclassification/null/PokemonData'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import keras
from keras.layers import Dense,Dropout,BatchNormalization,MaxPooling2D,Conv2D,Input
from keras import Model
from keras.layers import Flatten
from keras.preprocessing.image import ImageDataGenerator
imgDataGen = ImageDataGenerator(validation_split=0.25,rescale=1./255)
img_width = 256
img_height = 256
batch_size = 25
train_step_epoch = 207
val_step_epoch = 65
train = imgDataGen.flow_from_directory('../input/pokemonclassification/PokemonData',target_size=(img_width,img_height),subset='training',batch_size=batch_size)
val = imgDataGen.flow_from_directory('../input/pokemonclassification/PokemonData',target_size=(img_width,img_height),subset='validation',batch_size=batch_size)
inp = Input(shape=(img_width,img_height,3))
layer1 = Conv2D(64,kernel_size=(4,4),activation='relu')(inp)
layer2 = BatchNormalization()(layer1)
layer3 = MaxPooling2D(pool_size=(2,2))(layer2)
layer4 = Dropout(0.2)(layer3)

layer5 = Conv2D(32,kernel_size=(4,4),activation='relu')(layer4)
layer6 = BatchNormalization()(layer5)
layer7 = MaxPooling2D(pool_size=(2,2))(layer6)
layer8 = Dropout(0.3)(layer7)
layer9 = Flatten()(layer8)
out = Dense(256,activation='relu')(layer9)
out = Dropout(0.5)(out)
out = Dense(150,activation='softmax')(out)
model = Model(inp,out)
model.compile(
    optimizer=keras.optimizers.RMSprop(learning_rate=0.0001),
    loss=keras.losses.categorical_crossentropy,
    metrics=['acc'])
model.summary()
result = model.fit(
    train,
    validation_data=val,
    steps_per_epoch=train_step_epoch,
    validation_steps=val_step_epoch,
    epochs = 50)
classpokemon = train.class_indices
classpokemon
# ../input/pokemontestdata/test
import os
from keras.preprocessing import image
import numpy as np
folder = os.listdir("../input/pokemontestdata/test")
dir_smaple= '../input/pokemontestdata/test'
for file in folder:
    test_image =image.load_img(os.path.join(dir_smaple,file),target_size=(256,256))
    test_image=image.img_to_array(test_image) 
    test_image= np.expand_dims(test_image,axis=0)
    result = model.predict(test_image)
    result = np.argmax(result)
    print(list(classpokemon.keys())[list(classpokemon.values()).index(result)],result,end=" ") 
    print(file)
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save('model')

import zipfile
def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file))
zipf = zipfile.ZipFile('model.zip', 'w', zipfile.ZIP_DEFLATED)
zipdir('./model', zipf)
zipf.close()