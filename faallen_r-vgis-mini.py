import numpy as np
import pandas as pd
import cv2
from glob import glob
import os
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
import random
import ast
import csv
df = pd.read_csv('../input/test_simplified.csv')
#df = pd.read_csv('../input/train_simplified/airplane.csv')

#country_codes = set(df['countrycode'])
#df = df.sort_values(by='countrycode')
pd.set_option('display.max_columns', None)
print(df.head(5))


IMG_SIZE = 64
IMG_BASE_SIZE = 256
BATCH_SIZE = 512
TRAIN_CSV_PATH_LIST = glob('../input/train_simplified/*.csv')
SKIP_RECORD = 0
RECORD_RANGE = 5000
TRAIN_CSV_PATH_LIST[:5]
len(TRAIN_CSV_PATH_LIST)
class_list = []
for item in TRAIN_CSV_PATH_LIST:
    classname = os.path.basename(item).split('.')[0]
    class_list.append(classname)
    
item = pd.read_csv('../input/train_simplified/angel.csv')
print(item.head(10))

# Prints the number of images for each class
# Total number of images: 49707579
# Total number of classes: 340
total = 0
if False: # Set True to count number of images per class
    for i in class_list:
        item = pd.read_csv('../input/train_simplified/{}.csv'.format(i))
        print(len(item))
        total +=len(item)
    print('Total number of images {}'.format(total))
class_list = sorted(class_list)
class_list[:5]
#Number of classes
len(class_list)

word_encoder = LabelEncoder()
word_encoder.fit(class_list)
word_encoder.transform(class_list[:5])
def my_one_hot_encoder(word):
    return to_categorical(word_encoder.transform([word]),num_classes=340).reshape(340)
test_y = my_one_hot_encoder('The Eiffel Tower')
test_y
test_y.shape
from keras.preprocessing.image import ImageDataGenerator
def train_generator(path_list, img_size, batch_size, lw=6):
    while True:
        csv_path_list = random.choices(path_list, k=batch_size)
        x = np.zeros((batch_size, img_size, img_size, 1))
        y = np.zeros((batch_size, 340))
        for j in range(batch_size):
            csv_path = csv_path_list[j]
            f = open(csv_path, 'r')
            reader = csv.reader(f)
            for _ in range(SKIP_RECORD+1):
                __ = next(reader)
            i = 0
            s = np.random.randint(RECORD_RANGE)
            for row in reader:
                if i == s:
                    drawing = row[1]
                    break
                else:
                    i += 1
            f.close()
            lst = ast.literal_eval(drawing)
            img = np.zeros((IMG_BASE_SIZE, IMG_BASE_SIZE), np.uint8)
            for t, stroke in enumerate(lst):
                color = 255 - min(t, 10) * 13
                for i in range(len(stroke[0]) - 1):
                    _ = cv2.line(img, (stroke[0][i], stroke[1][i]), (stroke[0][i + 1], stroke[1][i + 1]), color, lw)
            if img_size != IMG_BASE_SIZE:
                x[j, :, :, 0] = cv2.resize(img, (img_size, img_size))/255
            else:
                x[j, :, :, 0] = img/255
            classname = os.path.basename(csv_path).split('.')[0]
            y_tmp = my_one_hot_encoder(classname)
            y[j] = y_tmp
            
            data_gen_args = dict(featurewise_center=True,
                     featurewise_std_normalization=True,
                     rotation_range=90,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     zoom_range=0.2)
            image_datagen = ImageDataGenerator(**data_gen_args)
            image_datagen.fit(x)
        yield x, y
datagen = train_generator(path_list=TRAIN_CSV_PATH_LIST, img_size=IMG_SIZE, batch_size=BATCH_SIZE, lw=6)

import matplotlib.pyplot as plt

if False: # Set True to Display images
    fig, axs = plt.subplots(nrows=9, ncols=3, sharex=True, sharey=True, figsize=(16, 48))
    drawings = []
    for i, instance in enumerate(class_list):
        if i>8:
            break
        item = pd.read_csv('../input/train_simplified/{}.csv'.format(instance))
        item = item[item.recognized]
        item['timestamp'] = pd.to_datetime(item.timestamp)
        item = item.sort_values(by='timestamp', ascending=False)[-3:]
        item['drawing'] = item['drawing'].apply(ast.literal_eval)
            
        for j, drawing in enumerate(item.drawing):
            ax = axs[i, j]
            for x, y in drawing:
                ax.plot(x, -np.array(y), lw=3)
            ax.axis('off')
    fig.savefig('items.png', dpi=200)
    plt.show();
x, y = next(datagen)
x.shape, y.shape, x.min(), x.max(), y.min(), y.max(), y.sum(), y[0].sum()
VAL_IMAGES_PER_CLASS = 20
VAL_CLASS = 340
VAL_SKIP_RECORD = SKIP_RECORD + RECORD_RANGE
def create_val_set(path_list, val_class, val_images_per_class, img_size, lw=6):
    csv_path_list = random.sample(path_list, k=val_class)
    x = np.zeros((val_class*val_images_per_class, img_size, img_size, 1))
    y = np.zeros((val_class*val_images_per_class, 340))
    z = []
    for k in range(val_class):
        csv_path = csv_path_list[k]
        f = open(csv_path, 'r')
        reader = csv.reader(f)
        for _ in range(VAL_SKIP_RECORD+1):
            __ = next(reader)
        s = 0
        for row in reader:
            if s == val_images_per_class:
                break
            else:
                drawing = row[1]
                lst = ast.literal_eval(drawing)
                img = np.zeros((IMG_BASE_SIZE, IMG_BASE_SIZE), np.uint8)
                for t, stroke in enumerate(lst):
                    color = 255 - min(t, 10) * 13
                    for i in range(len(stroke[0]) - 1):
                        _ = cv2.line(img, (stroke[0][i], stroke[1][i]), (stroke[0][i + 1], stroke[1][i + 1]), color, lw)
                if img_size != IMG_BASE_SIZE:
                    x[k*val_images_per_class+s, :, :, 0] = cv2.resize(img, (img_size, img_size))/255
                else:
                    x[k*val_images_per_class+s, :, :, 0] = img/255
                classname = os.path.basename(csv_path).split('.')[0]
                y_tmp = my_one_hot_encoder(classname)
                y[k*val_images_per_class+s,:] = y_tmp
                z_tmp = (row[0], classname)
                z.append(z_tmp)
                s += 1
        f.close()
    return x, y, z
valid_x, valid_y, valid_z = create_val_set(path_list=TRAIN_CSV_PATH_LIST, val_class=VAL_CLASS,
                                  val_images_per_class=VAL_IMAGES_PER_CLASS, img_size=IMG_SIZE, lw=6)
print(valid_z[0])
valid_x.shape, valid_y.shape, valid_x.min(), valid_x.max(), valid_y.min(), valid_y.max(), valid_y.sum(), valid_y[0].sum()
from keras import Model
from keras.layers import GlobalAveragePooling2D
from keras.layers import Dense
from keras.layers import Dropout
#from keras.applications import VGG16
from keras.applications import ResNet50
#from keras.applications import InceptionResNetV2
from keras import optimizers

def get_model(input_shape):
    #model = VGG16(input_shape=input_shape, include_top=True, weights=None, classes=340)
    model = ResNet50(input_shape=input_shape, include_top=True, weights=None, classes=340)
    #model = InceptionResNetV2(input_shape=input_shape, include_top=True, weights=None, classes=340)
    return model
model = get_model(input_shape=(IMG_SIZE,IMG_SIZE,1))
model.summary()
# Values greater than lr=0.003 cause significant drop in performance.
c = optimizers.adam(lr = 0.003)
model.compile(loss='categorical_crossentropy', optimizer=c, metrics=['categorical_accuracy'])
history = model.fit_generator(datagen, epochs=1, steps_per_epoch=30, verbose=1, validation_data=(valid_x, valid_y))
test_df = pd.read_csv('../input/test_simplified.csv')
test_df.shape
test_df.head()
def create_test_data(img_size, lw=6):
    x = np.zeros((test_df.shape[0], img_size, img_size, 1))
    for j in range(test_df.shape[0]):
        drawing = test_df.loc[j,'drawing']
        lst = ast.literal_eval(drawing)
        img = np.zeros((IMG_BASE_SIZE, IMG_BASE_SIZE), np.uint8)
        for t, stroke in enumerate(lst):
            color = 255 - min(t, 10) * 13
            for i in range(len(stroke[0]) - 1):
                _ = cv2.line(img, (stroke[0][i], stroke[1][i]), (stroke[0][i + 1], stroke[1][i + 1]), color, lw)
        if img_size != IMG_BASE_SIZE:
            x[j, :, :, 0] = cv2.resize(img, (img_size, img_size))/255
        else:
            x[j, :, :, 0] = img/255
    return x
test_x = create_test_data(img_size=IMG_SIZE, lw=6)
test_x.shape
test_pred = model.predict(test_x, batch_size=128, verbose=1)
import warnings
warnings.filterwarnings('ignore')
pred_rows = []
for i in range(test_df.shape[0]):
    test_top3 = test_pred[i].argsort()[::-1][:3]
    test_top3_words = word_encoder.inverse_transform(test_top3).tolist()
    test_top3_words = [k.replace(' ', '_') for k in test_top3_words]
    pred_words = test_top3_words[0] + ' ' + test_top3_words[1] + ' ' + test_top3_words[2]
    pred_rows += [{'key_id': test_df.loc[i, 'key_id'], 'word': pred_words}] 
sub = pd.DataFrame(pred_rows)[['key_id', 'word']]
sub.to_csv('submission.csv', index=False)
sub.head()
mini_x, mini_y, mini_z = create_val_set(path_list=TRAIN_CSV_PATH_LIST, val_class=VAL_CLASS,
                                  val_images_per_class=VAL_IMAGES_PER_CLASS, img_size=IMG_SIZE, lw=6)
mini_pred = model.predict(valid_x)
pred_true = []
pred_false = []
for i in range(len(mini_z)):
    test_top3 = mini_pred[i].argsort()[::-1][:3]
    test_top3_words = word_encoder.inverse_transform(test_top3).tolist()
    #test_top3_words = [k.replace(' ', '_') for k in test_top3_words]

    if mini_z[i][-1] in test_top3_words:
        pred_true.append((mini_x, mini_z[i]))
    else:
        pred_false.append((mini_x, mini_z[i]))
print(len(pred_true))
print(len(pred_false))
    
from collections import Counter
tmp_class = []
tmp_draw = []
for i in pred_true:
    tmp_class.append(i[1][1])
    tmp_draw.append(i[0][:])
print(Counter(tmp_class))
if False: # Set True to Display images
    fig, axs = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True, figsize=(16, 48))
    drawings = []
    
    for i, instance in enumerate(tmp_draw):
        if i>1:
            break
        for j, drawing in enumerate(instance):
            if j>2:
                break
            ax = axs[i, j]
            ax.plot(drawing[0], drawing[1], lw=3)
            ax.axis('off')
    fig.savefig('items.png', dpi=200)
    plt.show();