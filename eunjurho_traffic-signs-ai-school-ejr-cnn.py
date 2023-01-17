SEED_VALUE = 42



import random

random.seed(SEED_VALUE)

import numpy as np

np.random.seed(SEED_VALUE)

import tensorflow as tf

tf.random.set_seed(SEED_VALUE)

import keras

from keras.preprocessing.image import ImageDataGenerator

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



print(tf.__version__)

print(keras.__version__)
IMAGE_PATH = '../input/german-traffic-sign-recognition-benchmark-cropped/gtsrb-preprocessed/'

META_PATH = '../input/gtsrb-german-traffic-sign/'



import os

print(os.listdir(IMAGE_PATH))

print(os.listdir(META_PATH))
df_train = pd.read_csv(META_PATH + 'Train.csv')

df_train['Path'] = df_train['Path'].str.lower()

df_train['ClassId'] = df_train['ClassId'].apply(str)



print(df_train.shape)

df_train.head(5)
df_test = pd.read_csv(META_PATH + 'Test.csv')

df_test['Path'] = df_test['Path'].str.lower()

df_test['ClassId'] = df_test['ClassId'].apply(str)



print(df_test.shape)

df_test.head(5)
BATCH_SIZE = 24

IMG_ROWS = 24

IMG_COLS = 24

NUM_CLASS = 43
from sklearn.model_selection import train_test_split



(df_train, df_validation) = train_test_split(df_train, test_size=0.3, random_state=SEED_VALUE)



# Train 디렉토리로부터 학습 데이터셋과 검증 데이터셋을 나눠서 불러오기

train_datagen = ImageDataGenerator(rescale=1./255)



train_generator = train_datagen.flow_from_dataframe(

        df_train,

        directory=IMAGE_PATH,

        x_col='Path',

        y_col='ClassId',

        shuffle=True,

        seed=SEED_VALUE,

        target_size=(IMG_ROWS, IMG_COLS),

        color_mode='rgb',

        class_mode='categorical',    

        batch_size=BATCH_SIZE)



validation_generator = train_datagen.flow_from_dataframe(

        df_validation,

        directory=IMAGE_PATH,

        x_col='Path',

        y_col='ClassId',

        shuffle=True,

        seed=SEED_VALUE,

        target_size=(IMG_ROWS, IMG_COLS),

        color_mode='rgb',

        class_mode='categorical',    

        batch_size=BATCH_SIZE)
# 학습 데이터셋이 잘 읽어지는지 확인

x_train_batch, y_train_batch = train_generator.next()

print('train y data shape: {}'.format(y_train_batch.shape))

print('train x data shape: {}'.format(x_train_batch.shape))



f, ax = plt.subplots(1, 5, figsize=(10, 40))

ax[0].imshow(x_train_batch[0])

ax[0].set_title(np.argmax(y_train_batch[0]))

ax[1].imshow(x_train_batch[1])

ax[1].set_title(np.argmax(y_train_batch[1]))

ax[2].imshow(x_train_batch[2])

ax[2].set_title(np.argmax(y_train_batch[2]))

ax[3].imshow(x_train_batch[3])

ax[3].set_title(np.argmax(y_train_batch[3]))

ax[4].imshow(x_train_batch[4])

ax[4].set_title(np.argmax(y_train_batch[4]))

plt.show()

    

train_generator.reset()
# 검증 데이터셋이 잘 읽어지는지 확인

x_valid_batch, y_valid_batch = validation_generator.next()

print('validation y data shape: {}'.format(y_valid_batch.shape))

print('validation x data shape: {}'.format(x_valid_batch.shape))



f, ax = plt.subplots(1, 5, figsize=(10, 40))

ax[0].imshow(x_valid_batch[0])

ax[0].set_title(np.argmax(y_valid_batch[0]))

ax[1].imshow(x_valid_batch[1])

ax[1].set_title(np.argmax(y_valid_batch[1]))

ax[2].imshow(x_valid_batch[2])

ax[2].set_title(np.argmax(y_valid_batch[2]))

ax[3].imshow(x_valid_batch[3])

ax[3].set_title(np.argmax(y_valid_batch[3]))

ax[4].imshow(x_valid_batch[4])

ax[4].set_title(np.argmax(y_valid_batch[4]))

plt.show()



validation_generator.reset()
from keras.models import Sequential

from keras.layers import Conv2D

from keras.layers import MaxPooling2D

from keras.layers import Dense

from keras.layers import Flatten



model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same',activation='relu', input_shape=(IMG_ROWS, IMG_COLS, 3)))

model.add(Conv2D(64, (3, 3), padding='same',activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dense(NUM_CLASS, activation='softmax'))



model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit_generator(train_generator, 

                    validation_data=validation_generator,

                    epochs=3)
print(history.history.keys())



f, ax = plt.subplots(1, 2, figsize=(15, 5))



ax[0].plot(history.history['loss'])

ax[0].plot(history.history['val_loss'])

ax[0].set_xlabel('Epochs')

ax[0].set_ylabel('Loss')

ax[0].legend(('train_loss', 'val_loss'))



ax[1].plot(history.history['accuracy'])

ax[1].plot(history.history['val_accuracy'])

ax[1].set_xlabel('Epochs')

ax[1].set_ylabel('Accuracy')

ax[1].legend(('train_accuracy', 'val_accuracy'))



plt.show()
test_datagen = ImageDataGenerator(rescale=1./255)



test_generator = test_datagen.flow_from_dataframe(

        df_test,

        directory=IMAGE_PATH,

        x_col='Path',

        y_col='ClassId',

        shuffle=False,

        target_size=(IMG_ROWS, IMG_COLS),

        color_mode='rgb',

        class_mode='categorical',    

        batch_size=BATCH_SIZE)
score = model.evaluate_generator(test_generator, verbose=1)

print('Test dataset accuracy: {}'.format(score[1]))
test_generator.reset()



pred = model.predict_generator(test_generator)

pred = np.argmax(pred, axis=1)
# https://github.com/keras-team/keras/blob/master/keras/preprocessing/image.py#L521-L526



generator_idx_to_label_map = {v: k for k, v in test_generator.class_indices.items()}
from sklearn.metrics import classification_report



measures_by_class_str = classification_report(test_generator.classes, pred, target_names=list(generator_idx_to_label_map.values()))

print(measures_by_class_str)



# precision: tp / (tp + fp), positive 가 아닌 것을 positive 로 분류하지 않는 능력.

# recall: tp / p, positive 인 것을 모두 식별해낼 수 있는 능력.

# f1-score: precision 과 recall 의 조화 평균.

# support: 실제 데이터셋에서 출현 횟수.
# 클래스별 metrics 리포트 포맷을 pandas dataframe 으로 변환



measures_by_class = classification_report(test_generator.classes, pred, target_names=list(generator_idx_to_label_map.values()), output_dict=True)



class_ids = []

precisions = []

recalls = []

f1_scores = []

supports = []



for class_id, measures in measures_by_class.items():

    if (class_id.isdigit()):

        class_ids.append(class_id)

        precisions.append(measures['precision'])

        recalls.append(measures['recall'])

        f1_scores.append(measures['f1-score'])

        supports.append(measures['support'])



df_report = pd.DataFrame(list(zip(class_ids, precisions, recalls, f1_scores, supports)),

               index=class_ids,

               columns=['ClassId', 'Precision', 'Recall', 'F1-score', 'Support'])



print(df_report.shape)

df_report.head()
# 클래스별 metrics 리포트에 클래스별 accuracy 계산해서 추가



from sklearn.metrics import confusion_matrix



cm = confusion_matrix(test_generator.classes, pred)

cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]



df_report['Accuracy'] = pd.Series(cm.diagonal(), index=list(generator_idx_to_label_map.values()))



print(df_report.shape)

df_report.head()
f, ax = plt.subplots(3, 1, figsize=(20, 15))



order_by_class_size = df_train.ClassId.value_counts().index



sns.countplot(x='ClassId', data=df_train, ax=ax[0], order=order_by_class_size, palette="GnBu_d")

ax[0].set_title('Number of images by class within training dataset')



sns.barplot(x=df_report.index, y=df_report['Accuracy'], ax=ax[1], order=order_by_class_size, palette="GnBu_d")

ax[1].set_title('Accuracy by class')



sns.barplot(x=df_report.index, y=df_report['F1-score'], ax=ax[2], order=order_by_class_size, palette="GnBu_d")

ax[2].set_title('F1-score by class')



plt.show()