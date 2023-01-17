import numpy as np

import pandas as pd

import os

import cv2

import matplotlib.pyplot as plt



from sklearn.model_selection import KFold



from tensorflow.keras.layers import Dropout, Flatten, Dense, InputLayer

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, ReduceLROnPlateau

from tensorflow.keras import Sequential

from tensorflow.keras.applications.vgg16 import VGG16

from tensorflow.keras.applications.resnet import ResNet50

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.backend import clear_session

import tensorflow as tf
INPUT_SHAPE = (128, 128, 3) # Image Dimensions

BATCH_SIZE = 128

DROPOUT_RATE = 0.5

EPOCHS = 24

LR = 0.0001 # Learning Rate

REG_STRENGTH = 0.01 # Regularization Strength

NFOLDS = 5 # No of folds for cross validation

WORKERS = 4 # Multithreading no of threads

MAXQ = 10 # Max Queue size for multithreading

THRES = [0.2] * 17 # Threshold for truth value of label, applied on sigmoid output.



TRAIN_PATH = '/kaggle/input/planet-understanding-the-amazon-from-space/train-jpg'

TEST_PATH = '/kaggle/input/planet-understanding-the-amazon-from-space/test-jpg-v2'



TRAIN_CSV_PATH = '/kaggle/input/planet-understanding-the-amazon-from-space/train_v2.csv'

TEST_CSV_PATH = '/kaggle/input/planet-understanding-the-amazon-from-space/sample_submission_v2.csv'
df_train = pd.read_csv(TRAIN_CSV_PATH)

df_test = pd.read_csv(TEST_CSV_PATH)



df_train['image_name'] = df_train['image_name'].astype(str) + '.jpg'

df_test['image_name'] = df_test['image_name'].astype(str) + '.jpg'



df_test['tags'] = df_test['tags'].apply(lambda x: x.split(' '))



print(df_train.head())

print(df_test.head())



X_train_files = np.array(df_train['image_name'].tolist())

X_train_files.reshape((X_train_files.shape[0], 1))



y_train = np.array(df_train['tags'].tolist())
labels = []



for tag in df_train['tags'].values:

    labels_in_tag = tag.split(' ')

    for label in labels_in_tag:

        if label not in labels:

            labels.append(label)

        

labels.sort()

print(labels)
plt.figure(figsize=(12, 12))

res = [32, 64, 128, 256]

NIMGS = 5



for i in range(len(res)):

    for j in range(NIMGS):

        img = cv2.imread(os.path.join(TRAIN_PATH,df_train['image_name'][j+1]))

        img = cv2.resize(img, (res[i], res[i]))

        plt.subplot(len(res), NIMGS, i*NIMGS+j+1)

        plt.imshow(img)

        plt.title(df_train['tags'][j+1] + "\n" + str(res[i]) + "x" + str(res[i]), rotation=18)

        plt.axis('off')

    

plt.show()
plt.figure(figsize=(8, 8))



labels_count = {}



for tag in df_train['tags'].values:

    labels_in_tag = tag.split(' ')

    for label in labels_in_tag:

        if label in labels_count:

            labels_count[label] += 1

        else:

            labels_count[label] = 0

            

min_label = min(labels_count, key=labels_count.get)

max_label = max(labels_count, key=labels_count.get)

print(min_label+" is tagged least no of times: "+str(labels_count[min_label]))

print(max_label+" is tagged max no of times: "+str(labels_count[max_label]))

            

plt.bar(range(len(labels_count)), list(labels_count.values()), align='center')

plt.xticks(range(len(labels_count)), list(labels_count.keys()), rotation=90)



plt.show()
def create_model():

    model = Sequential()

    model.add(InputLayer(INPUT_SHAPE))

    model.add(VGG16(weights='imagenet', include_top=False))

    model.add(Flatten())

#     model.add(Dense(4096, activation='relu'))

#     model.add(Dropout(DROPOUT_RATE))

#     model.add(Dense(4096, activation='relu'))

#     model.add(Dropout(DROPOUT_RATE))

    model.add(Dense(17, activation='sigmoid'))

    return model



clear_session()



model = create_model()

model.summary()
def f2_score(y_true, y_pred):

    y_true = tf.cast(y_true, "int32")

    y_pred = tf.cast(tf.round(y_pred), "int32") # implicit 0.5 threshold via tf.round

    y_correct = y_true * y_pred

    sum_true = tf.reduce_sum(y_true, axis=1)

    sum_pred = tf.reduce_sum(y_pred, axis=1)

    sum_correct = tf.reduce_sum(y_correct, axis=1)

    precision = sum_correct / sum_pred

    recall = sum_correct / sum_true

    f_score = 5 * precision * recall / (4 * precision + recall)

    f_score = tf.where(tf.math.is_nan(f_score), tf.zeros_like(f_score), f_score)

    return tf.reduce_mean(f_score)
num_fold = 0



y_test = []



folds = KFold(n_splits=NFOLDS, shuffle=True, random_state=1).split(X_train_files, y_train)



for train_index, val_index in folds:

    X_train_files_fold = X_train_files[train_index]

    y_train_fold = y_train[train_index]

    X_val_files_fold = X_train_files[val_index]

    y_val_fold = np.array(y_train[val_index])

    

    train_df = pd.DataFrame(list(zip(X_train_files_fold, y_train_fold)), columns = ['image_name', 'tags'])

    val_df = pd.DataFrame(list(zip(X_val_files_fold, y_val_fold)), columns = ['image_name', 'tags'])

    

    train_df['tags'] = train_df['tags'].apply(lambda x: x.split(' '))

    val_df['tags'] = val_df['tags'].apply(lambda x: x.split(' '))



    train_datagen = ImageDataGenerator(

        rescale=1./255,

        width_shift_range=0.2,

        height_shift_range=0.2,

        horizontal_flip=True,

        vertical_flip=True

    )

    

    train_generator = train_datagen.flow_from_dataframe(

        train_df,

        directory=TRAIN_PATH,

        x_col='image_name',

        y_col='tags',

        target_size=(INPUT_SHAPE[0], INPUT_SHAPE[1]),

        class_mode='categorical',

        batch_size=BATCH_SIZE,

        classes=labels,

    )

    

    val_datagen = ImageDataGenerator(

        rescale=1./255

    )

    

    val_generator = val_datagen.flow_from_dataframe(

        val_df,

        directory=TRAIN_PATH,

        x_col='image_name',

        y_col='tags',

        target_size=(INPUT_SHAPE[0], INPUT_SHAPE[1]),

        class_mode='categorical',

        batch_size=BATCH_SIZE,

        classes=labels,

    )

    

    test_datagen = ImageDataGenerator(

        rescale=1./255

    )

    

    test_generator = test_datagen.flow_from_dataframe(

        df_test,

        directory=TEST_PATH,

        x_col='image_name',

        y_col='tags',

        target_size=(INPUT_SHAPE[0], INPUT_SHAPE[1]),

        class_mode='categorical',

        batch_size=BATCH_SIZE,

        classes=labels,

        shuffle=False,

    )



    model_path_of_fold = os.path.join('', 'weights_of_fold_' + str(num_fold) + '.h5')

    

    clear_session()

    model = create_model()

    

    adam = Adam(learning_rate=LR)

    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=[f2_score])

    

    callbacks = [

        ModelCheckpoint(model_path_of_fold, monitor='val_f2_score', save_best_only=True, mode='max'),

        ReduceLROnPlateau(monitor='loss', factor=0.1, patience=3, mode='min', min_lr=0.000001)

    ]

    

    model.fit_generator(train_generator, epochs=EPOCHS, validation_data=val_generator, callbacks=callbacks,

                       workers=WORKERS, use_multiprocessing=True, max_queue_size=MAXQ)



    model.load_weights(model_path_of_fold)



    p_test = model.predict_generator(test_generator, workers=WORKERS, use_multiprocessing=True, max_queue_size=MAXQ)

    y_test.append(p_test)
result = np.array(y_test[0])

for i in range(1, NFOLDS):

    result += np.array(y_test[i])

result /= NFOLDS

result = pd.DataFrame(result, columns = labels)

result.head()
preds = []

for i in range(result.shape[0]):

    a = result.ix[[i]]

    a = a.apply(lambda x: x > THRES, axis=1)

    a = a.transpose()

    a = a.loc[a[i] == True]

    ' '.join(list(a.index))

    preds.append(' '.join(list(a.index)))

    

df_test['tags'] = preds

df_test['image_name'] = df_test['image_name'].astype(str).str.slice(stop=-4)

df_test.to_csv('submit.csv', index=False)