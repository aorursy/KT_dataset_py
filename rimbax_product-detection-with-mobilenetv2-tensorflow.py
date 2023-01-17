import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import tensorflow as tf
!nvidia-smi
train_path = '../input/shopee-product-detection-student/train/train/train/'
test_path = '../input/shopee-product-detection-student/test/test/test/'

broken_fnames = []
for label in os.listdir(train_path):
    label_path = train_path + label + '/'
    for filename in os.listdir(label_path):
        if len(filename) > 36:
            print(label_path + filename)
            broken_fnames.append(label_path + filename)
            
print()
for filename in os.listdir(test_path):
    if len(filename) > 36:
        print(test_path + filename)
        broken_fnames.append(test_path + filename)
        
f = open('broken-file-names.txt', 'w')
f.write('\n'.join(broken_fnames))
f.close()
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 128
SEED = 0

def get_set():
    train_path = '../input/shopee-product-detection-student/train/train/train/'
    test_path = '../input/shopee-product-detection-student/test/test/'

    train_gen = ImageDataGenerator(rescale=1./255, validation_split=3007./105390)
    train_set = train_gen.flow_from_directory(train_path, target_size=IMAGE_SIZE, \
                                              batch_size=BATCH_SIZE, seed=SEED, \
                                              subset='training')
    val_set = train_gen.flow_from_directory(train_path, target_size=IMAGE_SIZE, \
                                            batch_size=BATCH_SIZE, seed=SEED, \
                                            subset='validation')

    test_gen = ImageDataGenerator(rescale=1./255)
    test_set = train_gen.flow_from_directory(test_path, target_size=IMAGE_SIZE, \
                                             batch_size=BATCH_SIZE, seed=SEED, \
                                             shuffle=False, class_mode=None)
    
    return train_set, val_set, test_set

train_set, val_set, test_set = get_set()
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model

def get_model():
    base = MobileNetV2(input_shape=IMAGE_SIZE+(3,), include_top=False, \
                       pooling='avg', weights='imagenet')
    base.trainable = False
    dense = Dense(42, activation='softmax', name='dense')(base.output)

    model = Model(inputs=base.inputs, outputs=dense, name='mobilenetv2')
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    return model

model = get_model()
EPOCHS = 3

hist = model.fit(train_set, epochs=EPOCHS, batch_size=BATCH_SIZE, \
                 validation_data=val_set, shuffle=False)
model.save('model-mobilenetv2.hdf5')
loss, acc = model.evaluate(val_set, batch_size=BATCH_SIZE)
print('Validation acc (percent): %.2f' % (100 * acc))
def generate_prediction(model, save_name):
    subm = pd.read_csv('../input/shopee-product-detection-student/test.csv')
    subm = subm.sort_values(by='filename', ignore_index=True)
    
    fnames = sorted(os.listdir('../input/shopee-product-detection-student/test/test/test'))
    unbroken_index = np.where(np.vectorize(len)(np.array(fnames)) == 36)[0]
    
    y_pred = model.predict(test_set, batch_size=BATCH_SIZE)
    pred = y_pred.argmax(axis=1)
    pred = pred[unbroken_index]
    subm['category'] = pred
    subm['category'] = subm['category'].apply(lambda x : '%02d' % x) # zero pad
    
    subm.to_csv(save_name, index=False)
    return subm
from tensorflow.keras.models import load_model

model = load_model('model-mobilenetv2.hdf5')
subm = generate_prediction(model, './submission.csv')
subm