%matplotlib inline



import numpy as np 

import pandas as pd 

import cv2

import ast

from matplotlib import pyplot as plt

import keras





import keras

from keras.layers import Conv2D, MaxPooling2D

from keras.layers import Dense, Dropout, Flatten, Activation

from keras.metrics import categorical_accuracy, top_k_categorical_accuracy, categorical_crossentropy

from keras.models import Sequential

from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from keras.optimizers import Adam

from keras.applications import MobileNet

from keras.applications.mobilenet import preprocess_input



import os

print(os.listdir("../input/quickdraw-doodle-recognition"))
train_simplified_path = '../input/quickdraw-doodle-recognition/train_simplified/'

test_simplified_path = '../input/quickdraw-doodle-recognition/test_simplified.csv'



BATCHSIZE = 128

SIZE = 96

NCLASSES = 340

VALIDSAMPLES = 100
def apk(actual, predicted, k=10):

    """

    Computes the average precision at k.

    This function computes the average prescision at k between two lists of

    items.

    Parameters

    ----------

    actual : list

             A list of elements that are to be predicted (order doesn't matter)

    predicted : list

                A list of predicted elements (order does matter)

    k : int, optional

        The maximum number of predicted elements

    Returns

    -------

    score : double

            The average precision at k over the input lists

    """

    if len(predicted)>k:

        predicted = predicted[:k]



    score = 0.0

    num_hits = 0.0



    for i,p in enumerate(predicted):

        if p in actual and p not in predicted[:i]:

            num_hits += 1.0

            score += num_hits / (i+1.0)



    if not actual:

        return 0.0



    return score / min(len(actual), k)



def mapk(actual, predicted, k=10):

    """

    Computes the mean average precision at k.

    This function computes the mean average prescision at k between two lists

    of lists of items.

    Parameters

    ----------

    actual : list

             A list of lists of elements that are to be predicted 

             (order doesn't matter in the lists)

    predicted : list

                A list of lists of predicted elements

                (order matters in the lists)

    k : int, optional

        The maximum number of predicted elements

    Returns

    -------

    score : double

            The mean average precision at k over the input lists

    """

    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])
data = pd.read_csv(train_simplified_path + 'roller coaster.csv', index_col='key_id', nrows=10)

data['drawing'] = data['drawing'].apply(ast.literal_eval)

data['word'] = data['word'].apply(lambda x: x.replace(' ', '_'))

data.head()
def draw_cv2(raw_strokes, size=256, lw=6, time_color=True, base_size=256):

    img = np.zeros((base_size, base_size), np.uint8)

    for t, stroke in enumerate(raw_strokes):

        for i in range(len(stroke[0]) - 1):

            color = 255 - min(t, 10) * 13 if time_color else 255

            _ = cv2.line(img, (stroke[0][i], stroke[1][i]),

                         (stroke[0][i + 1], stroke[1][i + 1]), color, lw)

    if size != base_size:

        return cv2.resize(img, (size, size))

    else:

        return img
plt.title('Sample image of a roller coaster in grayscale')

plt.imshow(255 - draw_cv2(data.iloc[9]['drawing']), cmap=plt.cm.gray)

plt.show()
train_files = os.listdir(train_simplified_path)

train_count = []

for file in train_files:

    with open(train_simplified_path + file) as f:

        for (count, _) in enumerate(f, 1):

            pass

        train_count.append(count)

train_count = np.sort(train_count)
print('Number of classes: ', len(train_count))

print('Minimum count of a label: ', min(train_count))

print('Maximum count of a label: ', max(train_count))

plt.title('The count of each of the labels sorted')

plt.scatter(range(0, len(train_count)), train_count)

plt.show()
# class index map

train_files = os.listdir(train_simplified_path)

class_index = {}

for i, file in enumerate(train_files):

    class_index[file[:-4].replace(' ', '_')] = i
def train_generator(size=SIZE, files_list=train_files, class_index=class_index, batch_size=BATCHSIZE, n_classes=NCLASSES, lw=6, time_color=True):

    while True:

        ind = VALIDSAMPLES

        while ind < 113610:

            files = [files_list[i] for i in np.random.randint(n_classes, size=batch_size)]

            x = np.zeros((batch_size, size, size, 1))

            y = np.zeros((batch_size, n_classes))

            for i, file in enumerate(files):

                df = pd.read_csv(train_simplified_path + file, skiprows=range(1,ind), nrows=1)

                df['drawing'] = df['drawing'].apply(ast.literal_eval)

                df['word'] = df['word'].apply(lambda x: class_index[x.replace(' ', '_')])

                y[i,:] = keras.utils.to_categorical(df['word'], num_classes=n_classes)

                x[i,:,:,0] = draw_cv2(df.iloc[0]['drawing'], size=size, lw=lw, time_color=time_color)

            x = preprocess_input(x).astype(np.float32)

            ind += 1

            yield x, y
def validation_data(file_list=train_files, size=SIZE, class_index=class_index, n_samples_per_class=VALIDSAMPLES, n_classes=NCLASSES, lw=6, time_color=True):

    x = np.zeros((len(file_list) * n_samples_per_class, size, size, 1))

    y = np.zeros((len(file_list) * n_samples_per_class, n_classes))

    for i, file in enumerate(file_list):

        df = pd.read_csv(train_simplified_path + file, nrows=n_samples_per_class)

        df['drawing'] = df['drawing'].apply(ast.literal_eval)

        df['word'] = df['word'].apply(lambda x: class_index[x.replace(' ', '_')])

        y[i*n_samples_per_class:(i+1)*n_samples_per_class,:] = keras.utils.to_categorical(df['word'], num_classes=n_classes)

        for j, raw_strokes in enumerate(df.drawing.values):

            x[i*n_samples_per_class+j,:,:,0] = draw_cv2(raw_strokes, size=size, lw=lw, time_color=time_color)

    x = preprocess_input(x).astype(np.float32)

    return x, y
x_validation, y_validation = validation_data()

print(x_validation.shape, y_validation.shape)
train_data_generator = train_generator()
# metric

def top_3_categorical_accuracy(y_true, y_pred):

    return top_k_categorical_accuracy(y_true, y_pred, k=3)
model = MobileNet(input_shape=(SIZE, SIZE, 1), weights=None, classes=NCLASSES)

model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=[top_3_categorical_accuracy])

print(model.summary())
callbacks = [

    ReduceLROnPlateau(monitor='val_top_3_categorical_accuracy', factor=0.5, patience=2, mode='max', min_lr=1e-5, verbose=1),

    ModelCheckpoint('model.h5', monitor='val_top_3_categorical_accuracy', mode='max', save_best_only=True, save_weights_only=True)

]



hist = model.fit_generator(

    train_data_generator, steps_per_epoch=500, epochs=15, verbose=1,

    validation_data=(x_validation, y_validation),

    callbacks = callbacks

)
print(hist.history.keys())

#  "top 3 Accuracy"

plt.plot(hist.history['top_3_categorical_accuracy'])

plt.plot(hist.history['val_top_3_categorical_accuracy'])

plt.title('model top 3 accuracy')

plt.ylabel('top 3 accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper left')

plt.show()

# "Loss"

plt.plot(hist.history['loss'])

plt.plot(hist.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper left')

plt.show()
# index to class mapping

index_class = {v: k for k, v in class_index.items()}
prediction = model.predict(x_validation)

y_true = []

for i in np.argmax(y_validation, axis=1):

    y_true.append(index_class[i])

y_pred = []

for i in range(prediction.shape[0]):

    pred = ''

    for j in prediction[i].argsort()[-3:][::-1]:

        pred += index_class[j] + ' '

    y_pred.append(pred[:-1])



del x_validation

del y_validation

print('Mean average precision (k=3) on the validation set: ', mapk(y_pred, y_true, k=3))
test = pd.read_csv(test_simplified_path, index_col='key_id')

x_test = np.zeros((len(test), SIZE, SIZE, 1))

test['drawing'] = test['drawing'].apply(ast.literal_eval)

for i, raw_strokes in enumerate(test.drawing.values):

    x_test[i,:,:,0] = draw_cv2(raw_strokes, size=SIZE, lw=6, time_color=True)

    x_test[i,:,:,0] = preprocess_input(x_test[i,:,:,0]).astype(np.float32)
y_pred = []

prediction = model.predict(x_test)

for i in range(prediction.shape[0]):

    pred = ''

    for j in prediction[i].argsort()[-3:][::-1]:

        pred += index_class[j] + ' '

    y_pred.append(pred[:-1])

test['word']  = y_pred

test.drop(['drawing', 'countrycode'], axis=1, inplace=True)

test.to_csv('submission.csv')