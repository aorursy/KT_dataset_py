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
IMG_SIZE = 64
IMG_BASE_SIZE = 256
BATCH_SIZE = 512
TRAIN_CSV_PATH_LIST = glob('../input/train_simplified/*.csv')
SKIP_RECORD = 0
RECORD_RANGE = 10000
TRAIN_CSV_PATH_LIST[:5]
len(TRAIN_CSV_PATH_LIST)
class_list = []
for item in TRAIN_CSV_PATH_LIST:
    classname = os.path.basename(item).split('.')[0]
    class_list.append(classname)
class_list = sorted(class_list)
class_list[:5]
len(class_list)
word_encoder = LabelEncoder()
word_encoder.fit(class_list)
word_encoder.transform(class_list[:5])
def my_one_hot_encoder(word):
    return to_categorical(word_encoder.transform([word]),num_classes=340).reshape(340)
test_y = my_one_hot_encoder('The Eiffel Tower')
test_y
test_y.shape
def train_generator(path_list, img_size, batch_size, lw=6):
    while True:
        csv_path_list = random.choices(path_list, k=batch_size)
        x = np.zeros((batch_size, img_size, img_size, 3))
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
                x[j, :, :, 1] = cv2.resize(img, (img_size, img_size))/255
                x[j, :, :, 2] = cv2.resize(img, (img_size, img_size))/255
            else:
                x[j, :, :, 0] = img/255
                x[j, :, :, 1] = img/255
                x[j, :, :, 2] = img/255
            classname = os.path.basename(csv_path).split('.')[0]
            y_tmp = my_one_hot_encoder(classname)
            y[j] = y_tmp
        yield x, y
datagen = train_generator(path_list=TRAIN_CSV_PATH_LIST, img_size=IMG_SIZE, batch_size=BATCH_SIZE, lw=6)
x, y = next(datagen)
x.shape, y.shape, x.min(), x.max(), y.min(), y.max(), y.sum(), y[0].sum()
VAL_IMAGES_PER_CLASS = 20
VAL_CLASS = 170
VAL_SKIP_RECORD = SKIP_RECORD + RECORD_RANGE
def create_val_set(path_list, val_class, val_images_per_class, img_size, lw=6):
    csv_path_list = random.sample(path_list, k=val_class)
    x = np.zeros((val_class*val_images_per_class, img_size, img_size, 3))
    y = np.zeros((val_class*val_images_per_class, 340))
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
                    x[k*val_images_per_class+s, :, :, 1] = cv2.resize(img, (img_size, img_size))/255
                    x[k*val_images_per_class+s, :, :, 2] = cv2.resize(img, (img_size, img_size))/255
                else:
                    x[k*val_images_per_class+s, :, :, 0] = img/255
                    x[k*val_images_per_class+s, :, :, 1] = img/255
                    x[k*val_images_per_class+s, :, :, 2] = img/255
                classname = os.path.basename(csv_path).split('.')[0]
                y_tmp = my_one_hot_encoder(classname)
                y[k*val_images_per_class+s,:] = y_tmp
                s += 1
        f.close()
    return x, y
valid_x, valid_y = create_val_set(path_list=TRAIN_CSV_PATH_LIST, val_class=VAL_CLASS,
                                  val_images_per_class=VAL_IMAGES_PER_CLASS, img_size=IMG_SIZE, lw=6)
valid_x.shape, valid_y.shape, valid_x.min(), valid_x.max(), valid_y.min(), valid_y.max(), valid_y.sum(), valid_y[0].sum()
def calc_map3_per_image(true_label, pred_3label):
    if true_label == pred_3label[0]:
        score = 1
    elif true_label == pred_3label[1]:
        score = 1/2
    elif true_label == pred_3label[2]:
        score = 1/3
    else:
        score = 0
    return score
def calc_map3_allimage(y_trues, y_preds):
    num = y_trues.shape[0]
    scores = list()
    for i in range(num):
        true_label = y_trues[i].argsort()[::-1][0]
        pred_3label = y_preds[i].argsort()[::-1][:3]
        score = calc_map3_per_image(true_label, pred_3label)
        scores.append(score)
    return np.mean(scores)
import tensorflow as tf
def my_metric(y_trues, y_preds):
    return tf.py_func(calc_map3_allimage, [y_trues, y_preds], tf.float64)
from keras import Model
from keras.layers import GlobalAveragePooling2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.applications import ResNet50
from keras import optimizers

def get_model(input_shape):
    base_model = ResNet50(input_shape=input_shape, include_top=False, weights=None)
    for l in base_model.layers:
        l.trainable = True
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(340, activation='softmax')(x)
    model = Model(base_model.input, x)
    
    return model
model = get_model(input_shape=(IMG_SIZE,IMG_SIZE,3))
model.summary()
c = optimizers.adam(lr = 0.001)
model.compile(loss='categorical_crossentropy', optimizer=c, metrics=[my_metric])
history = model.fit_generator(datagen, epochs=40, steps_per_epoch=30, verbose=1, validation_data=(valid_x, valid_y))
test_df = pd.read_csv('../input/test_simplified.csv')
test_df.shape
test_df.head()
def create_test_data(img_size, lw=6):
    x = np.zeros((test_df.shape[0], img_size, img_size, 3))
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
            x[j, :, :, 1] = cv2.resize(img, (img_size, img_size))/255
            x[j, :, :, 2] = cv2.resize(img, (img_size, img_size))/255
        else:
            x[j, :, :, 0] = img/255
            x[j, :, :, 1] = img/255
            x[j, :, :, 2] = img/255
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