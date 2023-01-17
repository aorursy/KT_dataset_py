import os
import cv2
import json
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from enum import Enum
import matplotlib
%matplotlib inline
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

!pip install pycountry-convert
import pycountry_convert

import keras
import tensorflow as tf
from keras.metrics import top_k_categorical_accuracy
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True 
sess = tf.Session(config=config)
set_session(sess) 
batch_size = 340*3
valid_percent = 1
data_files = 100
size = 64 # image size
num_classes = 340

class_paths = os.listdir("../input/quickdraw-doodle-recognition/train_simplified/")
cat_names = [item[:-4] for item in class_paths] # take file names, remove '.csv' extension
cat_names.sort(key=lambda x: str.lower(x)) # sort names of classes regardless capital letters
UNKNOWN_COUNTRY = 'YY' # code of unknown country

# mode indicating whether to consider area context
class ContextMode(Enum):
    no_context = 1
    area_context = 2    
valid_countries_dict = pycountry_convert.map_countries(cn_name_format="default")
valid_country_codes = list(set([v['alpha_2'] for k, v in valid_countries_dict.items()]))
all_country_codes= set(valid_country_codes) | set([UNKNOWN_COUNTRY])
def ValidateCountry(country_code):    
    return country_code if country_code in all_country_codes else UNKNOWN_COUNTRY
our_mapping = pd.read_csv('../input/area-context-country-clustering/area_mapping.csv')
print(our_mapping.head())

our_dict = pd.Series(our_mapping.groups.values, index = our_mapping.alpha3).to_dict()

def get_area(iso_code):
    try:
        return our_dict[iso_code]
    except KeyError:
        return 0
num_groups = max(our_mapping.groups)+1
print(num_groups)
#---drawing images: ref [1,2]--------------------------------------
def draw_cv2(raw_strokes, size=32, lw=6):    
    BASE_SIZE = 256
    img = np.zeros((BASE_SIZE, BASE_SIZE), np.uint8)    
    for stroke_no, stroke in enumerate(raw_strokes):
        line_intensity = 255 - min(stroke_no, 10) * 10
        for i in range(len(stroke[0]) - 1):
            _ = cv2.line(img, (stroke[0][i], stroke[1][i]), 
                         (stroke[0][i + 1], stroke[1][i + 1]), line_intensity, lw)            
    if size != BASE_SIZE:
        return cv2.resize(img, (size, size))
    return img
def image_generator(batch_size, isTraining):
    while True:
        if isTraining: # if valid_percent==1 -> training will use 99 files of 100
            index_table = np.random.permutation(data_files-valid_percent)
        else: # if valid_percent==1 -> validation will use 1 file of 100
            index_table = np.random.permutation(valid_percent)+(data_files-valid_percent)
        
        _path = '../input/shuffle-and-filter/'
        for k in index_table:                         
            filename = os.path.join(_path,'train_k{}.csv.gz'.format(k))
            for chunk in pd.read_csv(filename, chunksize=batch_size, 
                                     usecols=['countrycode','y','drawing','word'], 
                                     keep_default_na=False):             
                ch_size = len(chunk)                
                if(ch_size<batch_size):
                    continue # generator will reach to the beginning 
                    # of another file for full length batch
                else:
                    yield chunk
def test_gen():
    _path = '../input/quickdraw-doodle-recognition/test_simplified.csv'        
    while True:        
        for _chunk in pd.read_csv(_path, chunksize=batch_size, 
                                  usecols=['drawing','countrycode'], 
                                  keep_default_na=False):            
            
            # I made it as it helps overcomming some shape problems with
            # predict_generator, but one has to take care of selecting
            # appropriate range of lines for the output predictions
            if (len(_chunk)<batch_size):
                _chunk = pd.concat([_chunk, _chunk[0:batch_size-len(_chunk)]])                                               
            yield _chunk            

gen_TEST = test_gen()
areas_oneh = OneHotEncoder().fit(np.arange(num_groups).reshape(-1,1))
class GeneratorMode(Enum):
    training = 1
    validation = 2
    testing = 3
    
def main_generator(whatGeneratorMode, whatContextMode=ContextMode.no_context):
    while True:
        
        #---getting and simple filtering of data--------------------------
        if whatGeneratorMode==GeneratorMode.training:
            results = next(train_generator)            
            
        elif whatGeneratorMode==GeneratorMode.validation:
            results = next(valid_generator)            
            
        elif whatGeneratorMode==GeneratorMode.testing:
            results = next(gen_TEST)                                        
                    
                    
        #---drawing images: ref [1,2]--------------------------------------
        results['drawing'] = results['drawing'].apply(json.loads)        
        x = np.zeros((batch_size, size, size))
        for i, raw_strokes in enumerate(results.drawing.values):
            x[i] = draw_cv2(raw_strokes, size=size)
    
        x = x / 255.
        x = x.reshape((batch_size, size, size, 1)).astype(np.float32)        
                
        
        if whatGeneratorMode!=GeneratorMode.testing:                            
            #---converting labels to 1-hot---------------------------------
            y = keras.utils.to_categorical(results.y, num_classes=num_classes)
                
        #---output: - yielded 'x' depends on the selected context mode----
        if whatContextMode==ContextMode.no_context:
            if whatGeneratorMode==GeneratorMode.testing:                            
                yield x
            else:
                yield x, y
        elif whatContextMode==ContextMode.area_context:
            _countries = results['countrycode'].apply(ValidateCountry).values
            _areas = np.array(list(map(get_area, _countries)))
            xac = areas_oneh.transform(_areas.reshape(-1,1)).toarray()                        
            
            if whatGeneratorMode==GeneratorMode.testing:                         
                yield [x, xac]
            else:
                yield [x, xac], y                
def in_top_3(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)
def create_model(whatContextMode=ContextMode.no_context, alpha=1.0, show=False):
    
    from keras.models import Model
    from keras.layers import Input, concatenate, Reshape, Activation, Conv2D
    from keras.layers import Dropout, Flatten, GlobalAveragePooling2D
    from keras.optimizers import Adam     
    from keras.applications import MobileNet
    from keras import backend as K
    K.clear_session()
        
    if whatContextMode.value==ContextMode.no_context.value:        
        my_model = MobileNet(input_shape=(size, size, 1), weights=None, alpha=alpha, classes=num_classes)

        my_model.compile(optimizer=Adam(), loss='categorical_crossentropy',
                  metrics=['accuracy',in_top_3])

    elif whatContextMode.value==ContextMode.area_context.value:
        
        dropout=0.001        
        shape0 = (1, 1, int(1024*alpha))
        shape1 = (1, 1, int(1024*alpha+num_groups))        
        Input_area = Input(shape=(num_groups,))     
        
        my_model = MobileNet(input_shape=(size, size, 1), weights=None, alpha=alpha, include_top=False)

        x = my_model.get_layer('conv_pw_13_relu').output # call last layer of the model

        # following implementation of architecture of top layers of MobileNet taken from:
        # https://github.com/fchollet/deep-learning-models/blob/master/mobilenet.py
        #--------- Start of implementation: ---------
        x = GlobalAveragePooling2D()(x)
        x = Reshape(shape0, name='reshape_0')(x)
        
        #---------------adding area context (not in original model)--
        x = Flatten()(x)
        x = concatenate([x, Input_area])        
        x = Reshape(shape1, name='reshape_1')(x)
        #------------------------------------------------------------
        
        x = Dropout(dropout, name='dropout')(x)
        x = Conv2D(num_classes, (1, 1), padding='same', name='conv_preds')(x)
        x = Activation('softmax', name='act_softmax')(x)
        preds = Reshape((num_classes,), name='reshape_2')(x)
        #--------- End of implementation ----------
        
        my_model = Model(inputs=[my_model.input, Input_area], outputs=preds)                
        my_model.compile(optimizer=Adam(), loss='categorical_crossentropy',
                  metrics=['accuracy',in_top_3])
    else:
        print('Context not recognized')
        return
    
    if show:
        my_model.summary()

    return my_model
ourContextMode=ContextMode.area_context
create_model(whatContextMode=ourContextMode, alpha=0.5, show=True)
from numpy.random import seed as nseed
nseed(2018)
from tensorflow import set_random_seed
set_random_seed(2018)
ourContextMode = ContextMode.area_context

train_generator = image_generator(batch_size=batch_size, isTraining=True)
valid_generator = image_generator(batch_size=batch_size, isTraining=False)

train_gen=main_generator(whatGeneratorMode=GeneratorMode.training,   
                         whatContextMode=ourContextMode)
valid_gen=main_generator(whatGeneratorMode=GeneratorMode.validation, 
                         whatContextMode=ourContextMode)    

model = create_model(whatContextMode=ourContextMode)
train_epoch=44
train_steps=1000
validation_steps=34

from keras.callbacks import ReduceLROnPlateau
cb = [ReduceLROnPlateau(monitor='val_acc', factor=0.5, 
                        patience=5, mode='max', cooldown=3, verbose=0)]

history = model.fit_generator(
        generator=train_gen, steps_per_epoch=train_steps, 
        validation_data=valid_gen, validation_steps=validation_steps,
        callbacks=cb, 
        epochs=train_epoch, verbose=1)
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))
options = ['val_loss','val_acc','val_in_top_3','loss','acc','in_top_3']
print(ourContextMode.name)
for i, opt in enumerate(options):    
    ax = axs[i//3, i%3]        
    ax.plot(np.arange(train_epoch)+1, history.history[opt], marker='o')
    ax.set_xlabel('epochs')
    ax.set_title(opt)    
    ax.grid()
plt.tight_layout()

_test_gen = main_generator(whatGeneratorMode=GeneratorMode.testing, 
                           whatContextMode=ourContextMode)
predictions = model.predict_generator(_test_gen, steps=np.ceil(112199/batch_size), verbose=1)
print(np.shape(predictions))
def top3cats(predictions):
    return pd.DataFrame(np.argsort(-predictions, axis=1)[:, :3], 
                        columns=['word','word2','word3'] )

test_simp = pd.read_csv('../input/quickdraw-doodle-recognition/test_simplified.csv', 
                        nrows=np.shape(predictions)[0])
categories_dict = {_id: cat_name.replace(' ', '_') for _id, cat_name in enumerate(cat_names)}

output = top3cats(predictions).replace(categories_dict)
output = (output.word + ' ').str.cat([output.word2 + ' ', output.word3])
output.head()
output = pd.concat([test_simp['key_id'], output.loc[0:112198]], axis=1)
output.tail()
output.to_csv(ourContextMode.name + '_myResults.csv',index=False) 