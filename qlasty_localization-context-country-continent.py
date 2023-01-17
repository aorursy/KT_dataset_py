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
num_classes = 340 #30 will be set later
items_in_class = 75
batch_size = 2048
training_percent = 0.4 # how much data to take (40%)
valid_percent = 0.05 # size of the validation set (5%)
size = 32 # size of image: 32x32px

my_class_path = "../input/train_simplified/"
class_paths = os.listdir("../input/train_simplified/")
cat_names = [item[:-4] for item in class_paths] # take file names, remove '.csv' extension
cat_names.sort(key=lambda x: str.lower(x)) # sort names of classes regardless capital letters

UNKNOWN_COUNTRY = 'YYY'
UNKNOWN_CONTINENT = 'XXX'

cntntns = ['AF', 'AS', 'EU', 'NA', 'OC', 'SA']+[UNKNOWN_CONTINENT] # list of continents

# mode indicating whether to consider localization context
class ContextMode(Enum):
    no_context = 1
    country_context = 2
    continent_context = 3
data_countries_codes = []
file_lengths = []
recog_stat_list = []

for _name in tqdm(class_paths[0:num_classes]):
    # use keep_default_na=False to prevent pandas parsing country code 'NA' as nan
    df = pd.read_csv(my_class_path+_name, keep_default_na=False)
    file_lengths.append(len(df)-1) # number of samples in file (header not counted)
    
    recog_stat = df['recognized'].value_counts()
    recog_stat_list.append(100.0*recog_stat[1]/(recog_stat[0]+recog_stat[1])) # % of recognized images in class
    
    ccode_stat = df['countrycode'].unique()    
    set1 = set(data_countries_codes)
    set2 = set(ccode_stat)
    any_new = set2-set1
    data_countries_codes += list(any_new) # list of unique country two letter codes               
    
# set size of validation set for each class
valid_lengths=[int(leng*valid_percent) for leng in file_lengths]
    
print('Average file length: {:.0f}+-{:.0f} lines\nRecognized images: {:.0f}+-{:.0f}%\nUnique countries in data: {}'.format(np.mean(file_lengths), np.std(file_lengths), np.mean(recog_stat_list), np.std(recog_stat_list), len(data_countries_codes)))
myhist=plt.hist(file_lengths, bins=50)
myhist=plt.ylabel('number of classes[n]')
myhist=plt.xlabel('items in class [n]')
valid_countries_dict = pycountry_convert.map_countries(cn_name_format="default")
valid_country_codes = list(set([value['alpha_2'] for key, value in valid_countries_dict.items()]))
all_country_codes= set(valid_country_codes) | set(data_countries_codes) | set([UNKNOWN_COUNTRY])
def ValidateCountry(country_code):    
    return country_code if country_code in all_country_codes else UNKNOWN_COUNTRY

def ValidateContinent(country_code):
    try:
        return pycountry_convert.country_alpha2_to_continent_code(country_code)
    except:
        return UNKNOWN_CONTINENT
data_countries_codes = [ValidateCountry(_cntry) for _cntry in data_countries_codes]
cont_codes = [ValidateContinent(_cntry) for _cntry in data_countries_codes]
print("First 10 countries:    {}".format(data_countries_codes[0:10]))
print("Associated continents: {}\n".format(cont_codes[0:10]))
print("Unique continents:     {}".format(set(cont_codes))) # has to match defined cntntns list
#---drawing images: ref [1,2]--------------------------------------
def draw_cv2(raw_strokes):
    size = 32
    lw = 6
    BASE_SIZE = 256
    img = np.zeros((BASE_SIZE, BASE_SIZE), np.uint8)    
    for stroke_no, stroke in enumerate(raw_strokes):
        line_intensity = 255 - min(stroke_no, 10) * 10
        for i in range(len(stroke[0]) - 1):
            _ = cv2.line(img, (stroke[0][i], stroke[1][i]), (stroke[0][i + 1], stroke[1][i + 1]), line_intensity, lw)            
    if size != BASE_SIZE:
        return cv2.resize(img, (size, size))
    return img
num_classes = 30

def items_gen(_id, isTraining):
    _path = my_class_path + cat_names[_id] + '.csv'
    
    if isTraining: # skip the first valid_percent lines of the file
        start_index, end_index = 1, valid_lengths[_id]        
        if training_percent<1: # or select only last training_ratio lines
            start_index, end_index = 1, int(file_lengths[_id]*(1-training_percent))                    
    else: # validation: skip the last 1-valid_percent lines of the file
        start_index, end_index = valid_lengths[_id], file_lengths[_id]
    
    while True:
        # skiprows-> generator will chunk within the specified range, depending on its type [training/validation]
        for _chunk in pd.read_csv(_path, chunksize=items_in_class +1, usecols=['drawing', 'recognized', 'word', 'countrycode'],skiprows=range(start_index, end_index), keep_default_na=False):
            yield _chunk
gen_list_TRAIN = [items_gen(i, isTraining=True ) for i in range(num_classes)]
gen_list_VALID = [items_gen(i, isTraining=False) for i in range(num_classes)]
def test_gen():
    _path = '../input/test_simplified.csv'        
    while True:        
        for _chunk in pd.read_csv(_path, chunksize=batch_size, usecols=['drawing','countrycode'], keep_default_na=False):
            yield _chunk            

gen_TEST = test_gen()
labels_encoder = LabelEncoder()
ccodes_encoder = LabelEncoder()
cntnts_encoder = LabelEncoder()
labels_encoder.fit(cat_names[0:num_classes])
ccodes_encoder.fit(list(all_country_codes))
cntnts_encoder.fit(cntntns)
#------------------------------------------
labels_oneh = OneHotEncoder().fit(np.arange(num_classes).reshape(-1,1))
ccodes_oneh = OneHotEncoder().fit(np.arange(len(list(all_country_codes))).reshape(-1,1))
cntnts_oneh = OneHotEncoder().fit(np.arange(len(cntntns)).reshape(-1,1))
class GeneratorMode(Enum):
    training = 1
    validation = 2
    testing = 3
    
def main_generator(whatGeneratorMode, whatContextMode=ContextMode.no_context):
    while True:
        
        #---getting and simple filtering of data--------------------------
        if whatGeneratorMode==GeneratorMode.training:
            results = [next(gen_list_TRAIN[id]) for id in range(num_classes)]                
            results = pd.concat(results)
            results = results[results.recognized == True]
            
        elif whatGeneratorMode==GeneratorMode.validation:
            results = [next(gen_list_VALID[id]) for id in range(num_classes)]                
            results = pd.concat(results)
            results = results[results.recognized == True]
            
        elif whatGeneratorMode==GeneratorMode.testing:
            results = next(gen_TEST)                        
                                 
        #---shuffling and batch size setting------------------------------
        #-(depending on the number of not recognized samples, we need to--
        #-add or subtract samples from the concatenated dataframe)--------
        _itms = len(results)        
        results = results.sample(frac=1, random_state=2018).reset_index(drop=True)
        if(_itms>batch_size):
            results=results[0:batch_size]
        elif(_itms<batch_size):
            results = pd.concat([results, results[0:batch_size-_itms]])
                        
        #---drawing images: ref [1,2]--------------------------------------
        results['drawing'] = results['drawing'].apply(json.loads)        
        x = np.zeros((batch_size, size, size))
        for i, raw_strokes in enumerate(results.drawing.values):
            x[i] = draw_cv2(raw_strokes)
    
        x = x / 255.
        x = x.reshape((batch_size, size, size, 1)).astype(np.float32)        
        
        if whatGeneratorMode!=GeneratorMode.testing:                            
            #---converting labels to 1-hot---------------------------------
            _all_labels = results['word'].values
            _all_labels = labels_encoder.transform(_all_labels)        
            y = labels_oneh.transform(_all_labels.reshape(-1,1)).toarray()
                
        #---output: - yielded 'x' depends on the selected context mode----
        if whatContextMode==ContextMode.no_context:
            if whatGeneratorMode==GeneratorMode.testing:                            
                yield x
            else:
                yield x, y
        elif whatContextMode==ContextMode.country_context:
            _countries = results['countrycode'].apply(ValidateCountry).values
            _countries = ccodes_encoder.transform(_countries)        
            xcc = ccodes_oneh.transform(_countries.reshape(-1,1)).toarray()                        
            
            if whatGeneratorMode==GeneratorMode.testing:                            
                yield [x, xcc]
            else:
                yield [x, xcc], y            
        elif whatContextMode==ContextMode.continent_context:
            _countries = results['countrycode'].apply(ValidateCountry).values
            _continents = [ValidateContinent(_cntry) for _cntry in _countries]                    
            _continents = cntnts_encoder.transform(_continents)        
            xct = cntnts_oneh.transform(_continents.reshape(-1,1)).toarray()            
            
            if whatGeneratorMode==GeneratorMode.testing:                
                yield [x, xct]
            else:                
                yield [x, xct], y   
ourContextMode=ContextMode.continent_context
train_gen=main_generator(whatGeneratorMode=GeneratorMode.training, whatContextMode=ourContextMode)
xx,yy=next(train_gen)
#print('x shape: {}'.format(np.shape(xx)))
print('x[0] shape: {}'.format(np.shape(xx[0])))
print('x[1] shape: {}'.format(np.shape(xx[1])))
print('y shape:    {}'.format(np.shape(yy)))
def insight_function(xx, yy, whatContextMode, selected_word='bear'):
    if whatContextMode==ContextMode.no_context:
        print('No context')
        return
    
    context = cntntns if whatContextMode==whatContextMode.continent_context else list(all_country_codes)

    certain_image_list=[ind for ind in range(batch_size) if cat_names[np.argmax(yy[ind])] == selected_word]
    certain_image_context=[context[np.argmax(xx[1][_ind])] for _ind in certain_image_list]
    proper_order=sorted(range(len(certain_image_context)), key=lambda c: certain_image_context[c])

    print("{}:".format(selected_word))    
    fig, axs = plt.subplots(nrows=9, ncols=7, sharex=True, sharey=True, figsize=(12, 12))
    for i,index in enumerate(proper_order):
        ax = axs[i//9, i%7]
        selected_example=certain_image_list[index]
        ax.imshow(xx[0][selected_example,:,:,0], cmap=plt.cm.gray_r)
        ax.set_title(context[np.argmax(xx[1][selected_example])]) 
        ax.axis('off')
        if i==len(proper_order) or i==9*7:
            break
    plt.tight_layout()
insight_function(xx, yy, ourContextMode)
def in_top_3(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)
def create_model(whatContextMode=ContextMode.no_context, denseLayerNeurons=512, show=False):
    
    from keras.models import Model
    from keras.layers import Conv2D, MaxPooling2D, Input, concatenate
    from keras.layers import Dense, Dropout, Flatten, BatchNormalization
    from keras.optimizers import Adam 
    from keras.initializers import glorot_normal
    from keras import backend as K
    K.clear_session()

    Input_image = Input(shape=(size, size, 1))
    
    if whatContextMode==ContextMode.country_context:
        Input_context = Input(shape=(len(all_country_codes),))
    elif whatContextMode==ContextMode.continent_context:
        Input_context = Input(shape=(len(cntntns),))
    
    x = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same',\
               kernel_initializer=glorot_normal(seed=2018))(Input_image)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same',\
              kernel_initializer=glorot_normal(seed=2018))(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same',\
              kernel_initializer=glorot_normal(seed=2018))(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same',\
              kernel_initializer=glorot_normal(seed=2018))(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Flatten()(x)
    
    # if image context mode allowed, concat context with image summary from CNN
    if whatContextMode!=ContextMode.no_context:
        x = concatenate([x, Input_context])

    x = Dense(denseLayerNeurons, activation='relu',\
              kernel_initializer=glorot_normal(seed=2018))(x)
    x = BatchNormalization()(x)
    x = Dropout(seed=2018, rate=0.5)(x)
    x = Dense(denseLayerNeurons, activation='relu',\
             kernel_initializer=glorot_normal(seed=2018))(x)
    x = BatchNormalization()(x)
    out = Dense(num_classes, activation='softmax',\
               kernel_initializer=glorot_normal(seed=2018))(x)

    if whatContextMode==ContextMode.no_context:
        my_model = Model(inputs = Input_image, outputs = out)        
    else:
        my_model = Model(inputs = [Input_image, Input_context], outputs = out)        

    my_model.compile(optimizer=Adam(), loss='categorical_crossentropy',
              metrics=['accuracy',in_top_3])
    
    if show:
        my_model.summary()
    
    return my_model
create_model(whatContextMode=ourContextMode, denseLayerNeurons=1024, show=True)
from numpy.random import seed as nseed
nseed(2018)
from tensorflow import set_random_seed
set_random_seed(2018)

def reset_seeds():    
    nseed(2018)
    set_random_seed(2018)
# average file length * training_ratio is around 58 000, dividing by around 70 items of class in a batch (for 30 classes and 2048 batch size) gives 830 steps/epoch
# average validation set size is 0.05*140 000=7000, dividing by around 70 items of class in a batch (for 30 classes and 2048 batch size) gives 100 steps
test_epoch=10
test_steps=620
test_validation_steps=100
def nice_evaluator(whatContextMode, denseLayerNeurons=512):    
    reset_seeds()    
    gen_list_TRAIN =[items_gen(i, True ) for i in range(num_classes)]
    gen_list_VALID =[items_gen(i, False) for i in range(num_classes)]
    train_gen=main_generator(whatGeneratorMode=GeneratorMode.training,  whatContextMode=whatContextMode)
    valid_gen=main_generator(whatGeneratorMode=GeneratorMode.validation,whatContextMode=whatContextMode)

    model = create_model(whatContextMode=whatContextMode, denseLayerNeurons=denseLayerNeurons)
    history = model.fit_generator(
            generator=train_gen, steps_per_epoch=test_steps,
            validation_data=valid_gen, validation_steps=test_validation_steps,
            epochs=test_epoch, verbose=2)
    
    return {'hist': history.history, 'model': model, 'context': whatContextMode}
dict1 = nice_evaluator(whatContextMode=ContextMode.no_context,        denseLayerNeurons=256)
dict2 = nice_evaluator(whatContextMode=ContextMode.country_context,   denseLayerNeurons=256)
dict3 = nice_evaluator(whatContextMode=ContextMode.continent_context, denseLayerNeurons=256)
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))
options = ['val_loss','val_acc','val_in_top_3','loss','acc','in_top_3']
for i, opt in enumerate(options):    
    ax = axs[i//3, i%3]        
    ax.plot(np.arange(test_epoch)+1, dict1['hist'][opt],marker='o', label='no context')
    ax.plot(np.arange(test_epoch)+1, dict2['hist'][opt],marker='x', label='country context')
    ax.plot(np.arange(test_epoch)+1, dict3['hist'][opt],marker='d', label='continent context')    
    ax.set_xlabel('epochs')
    ax.set_title(opt)
    ax.legend()
    ax.grid()
plt.tight_layout()

model_to_test = dict3

test_gen = main_generator(whatGeneratorMode=GeneratorMode.testing, whatContextMode=model_to_test['context'])
predictions = model_to_test['model'].predict_generator(test_gen, steps=5, verbose=1) #steps=np.ceil(112199/batch_size) # for all data
print(np.shape(predictions))
def top3cats(predictions):
    return pd.DataFrame(np.argsort(-predictions, axis=1)[:, :3], columns=['word','word2','word3'] )

test_simp = pd.read_csv('../input/test_simplified.csv', nrows=np.shape(predictions)[0])
categories_dict = {_id: cat_name.replace(' ', '_') for _id, cat_name in enumerate(cat_names)}

output = top3cats(predictions).replace(categories_dict)

output = (output.word + ' ').str.cat([output.word2 + ' ', output.word3])
output.head()
output = pd.concat([test_simp['key_id'], output], axis=1)
output.tail()
output.to_csv('myResults.csv',index=False) # But note we trained only 30 classes here