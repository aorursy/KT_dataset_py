import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import cv2
print(os.listdir("../input"))
img_rows, img_cols, img_chans = 100, 100, 3
input_shape = (img_rows, img_cols, img_chans)
num_classes = 10
batch_size = 64
!mkdir train
!mkdir test
from shutil import copyfile
print('Preparing Test Folder...') 
datasets = ['c', 'augc']
for dataset in datasets:
    src_test_file_paths = glob.glob('../input/testing-' + dataset + '/*.*')
    dst_test_folder = 'test/test/'
    if not os.path.exists(dst_test_folder):
        os.makedirs(dst_test_folder)
    n = len(src_test_file_paths)
    i = 0
    for src_test_file_path in src_test_file_paths:
        filename = os.path.basename(src_test_file_path)
        dst_test_file_path = dst_test_folder + filename
#         image = cv2.imread(src_test_file_path)
#         image = cv2.resize(image, (img_rows, img_cols))
#         cv2.imwrite(dst_test_file_path, image)
        copyfile(src_test_file_path, dst_test_file_path)
        i += 1
        print('\rProcessed {}/{}'.format(i, n), end='')
    print()
print('Preparing Train Folder...')
datasets = ['c'] 
for dataset in datasets:
    csv_file = '../input/training-'+ dataset +'.csv'
    src_folder = '../input/training-' + dataset + '/'
    dst_folder = 'train/'
    
    label = pd.read_csv(csv_file)
    n, m = label.shape
    
    for index, row in label.iterrows():
        src_file = src_folder + row.filename
        dst_file = dst_folder + str(row.digit) + '/' + row.filename
        if not os.path.exists(dst_folder + str(row.digit) + '/'):
            os.makedirs(dst_folder + str(row.digit) + '/')
#         image = cv2.imread(src_file)
#         image = cv2.resize(image, (img_rows, img_cols))
#         cv2.imwrite(dst_file, image)
        copyfile(src_file, dst_file)
        print('\rProcessed {}/{}'.format(index + 1, n), end='')
    print()
!ls train
!ls test
import keras
from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
def get_model():
    model = Sequential()
    model.add(Conv2D(8, (5, 5), input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(16, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    
    return model
simple_train_datagen = ImageDataGenerator(rescale=1./255)
simple_train_generator = simple_train_datagen.flow_from_directory(
        'train',
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode='categorical')
simple_model = get_model()
history = simple_model.fit_generator(
        simple_train_generator,
        epochs=20, verbose=1)
aug_train_datagen = ImageDataGenerator(rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    fill_mode='nearest')
aug_train_generator = aug_train_datagen.flow_from_directory(
        'train',
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode='categorical')
augmented_model = get_model()
history = augmented_model.fit_generator(
        aug_train_generator,
        epochs=20, verbose=1)
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
        'test',
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode=None, shuffle=False)
def predict(test_generator, model):
    pred_probs = model.predict_generator(test_generator, verbose=1)
    pred_label = []
    for i in range(len(pred_probs)):
        label = np.argmax(pred_probs[i])
        pred_label.append(label)
    pred_key = os.listdir('test/test/')
    pred_key.sort()
    pred_df = pd.DataFrame({'key': pred_key, 'label': pred_label})
    return pred_df
simple_pred_df = predict(test_generator, simple_model)
augmented_pred_df = predict(test_generator, augmented_model)
n, m = simple_pred_df.shape
image_per_row = 10
image_per_col = 10
N = image_per_row * image_per_col
random_index = np.random.randint(0, n, (N))
fig = plt.figure(figsize=(2 * image_per_col, 2 * image_per_row))
for i in range(N):
    idx = random_index[i]
    simple_result = simple_pred_df.iloc[idx]
    augmented_result = augmented_pred_df.iloc[idx]
    image_file = 'test/test/' + simple_result['key']
    image = cv2.imread(image_file)
    plt.subplot(image_per_row, image_per_col, i + 1)
    plt.axis('off')
    plt.imshow(image)
    plt.title(str(simple_result['label']) + '/' + str(augmented_result['label']))
plt.show()
n, m = simple_pred_df.shape
image_per_row = 10
image_per_col = 10
N = image_per_row * image_per_col
fig = plt.figure(figsize=(2 * image_per_col, 2 * image_per_row))
i = 0
while i < N:
    idx = np.random.randint(0, n)
    simple_result = simple_pred_df.iloc[idx]
    augmented_result = augmented_pred_df.iloc[idx]
    if(simple_result['label'] == augmented_result['label']):
        continue
    image_file = 'test/test/' + simple_result['key']
    image = cv2.imread(image_file)
    plt.subplot(image_per_row, image_per_col, i + 1)
    plt.axis('off')
    plt.imshow(image)
    plt.title(str(simple_result['label']) + '/' + str(augmented_result['label']))
    i += 1
plt.show()
!rm -r train
!rm -r test