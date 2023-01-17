# import python standard library

import json, gc, os



# import data manipulation library

import numpy as np

import pandas as pd



# import data visualization library

import matplotlib.pyplot as plt

from tqdm import tqdm



# import image processing library

import cv2



# import tensorflow model class

from tensorflow import keras

from tensorflow.keras.applications import MobileNet

from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D

from tensorflow.keras.models import load_model, Sequential



# import sklearn model selection

from sklearn.model_selection import train_test_split



# import tensorflow model evaluation classification metrics

from tensorflow.keras.metrics import top_k_categorical_accuracy
# numpy options

np.random.seed(seed=58)
# list training and testing data directory

os.listdir('../input/')
# acquiring training and testing data

df_train = pd.read_csv('../input/quick-draw-doodle-recognition-challenge-shufflecsv/train_k0.csv.gz', nrows=100)

df_test = pd.read_csv('../input/quickdraw-doodle-recognition/test_simplified.csv', nrows=2)
# visualize head of the training data

df_train.head(n=5)
# visualize tail of the testing data

df_test.tail(n=5)
# dataframe columns name

names = ['countrycode', 'drawing', 'key_id', 'recognized', 'timestamp', 'word']



# class files and dictionary

files = sorted(os.listdir('../input/quickdraw-doodle-recognition/train_simplified/'), reverse=False)

class_dict = {file[:-4].replace(" ", "_"): i for i, file in enumerate(files)}

classreverse_dict = {v: k for k, v in class_dict.items()}



# combine training and testing dataframe

df_train = df_train.drop(['shuffle'], axis=1)

df_train['datatype'], df_test['datatype'] = 'training', 'testing'

df_train = df_train[['key_id', 'countrycode', 'drawing', 'datatype', 'word', 'recognized']]

df_test['word'], df_test['recognized'] = '', True

df_data = pd.concat([df_train, df_test], ignore_index=True)
# data dimensions

chunksize = 680

img_size = 64

num_channels = 1

num_classes = 340

num_shuffles = 50



# flat dimensions

img_size_flat = img_size * img_size * num_channels
def drawplot(draw: list, label: list, figsize: tuple = (4, 3), ncols: int = 5, nrows: int = None) -> plt.figure:

    """ Return a draw image plot applied for an image data in vector format.

    

    Args:

        draw (list): The draw image data.

        label (list): The label of an image data.

        figsize (tuple): The matplotlib figure size width and height in inches. Default to (4, 3).

        ncols (int): The number of columns for axis in the figure. Default to 5.

        nrows (int): The number of rows for axis in the figure. Default to None.

    

    Returns:

        plt.figure: The plot figure.

    """

    

    if nrows is None: nrows = (len(label) - 1) // ncols + 1

    

    fig, axes = plt.subplots(figsize=(figsize[0]*ncols , figsize[1]*nrows), ncols=ncols, nrows=nrows)

    axes = axes.flatten()

    for i in label.index:

        for j in range(len(draw[i])): _ = axes[i - label.index[0]].plot(draw[i][j][0], draw[i][j][1])

        axes[i - label.index[0]].invert_yaxis()

        axes[i - label.index[0]].set_title(label[i])

    return fig
def imageplot(image: list, label: list, size: tuple, figsize: tuple = (4, 3), ncols: int = 5, nrows: int = None) -> plt.figure:

    """ Return an image plot applied for an image data in grayscale picture (m, n) format, RGB picture (m, n, 3) format and RGBA picture (m, n, 4) format.

    

    Args:

        image (list): The image data.

        label (list): The label of an image data.

        size (tuple): The tuple of an image size.

        figsize (tuple): The matplotlib figure size width and height in inches. Default to (4, 3).

        ncols (int): The number of columns for axis in the figure. Default to 5.

        nrows (int): The number of rows for axis in the figure. Default to None.

    

    Returns:

        plt.figure: The plot figure.

    """

    

    if nrows is None: nrows = (len(label) - 1) // ncols + 1

    

    fig, axes = plt.subplots(figsize=(figsize[0]*ncols , figsize[1]*nrows), ncols=ncols, nrows=nrows)

    axes = axes.flatten()

    _ = [axes[i].imshow(image[i].reshape(size), interpolation='spline16') for i in range(len(label))]

    return fig
def draw2pixel(draw: list) -> np.ndarray:

    """ Return a draw image to pixel image data.

    

    Args:

        draw (list): The draw image data.

    

    Returns:

        np.ndarray: The draw image to pixel image data.

    """

    

    image, xmin, xmax, ymin, ymax = np.zeros((256, 256)), 255, 0, 255, 0

    for k, stroke in enumerate(draw):

        for i in range(len(stroke[0])-1):

            xmin, xmax = min(xmin, stroke[0][i], stroke[0][i + 1]), max(xmax, stroke[0][i], stroke[0][i + 1])

            ymin, ymax = min(ymin, stroke[1][i], stroke[1][i + 1]), max(ymax, stroke[1][i], stroke[1][i + 1])

            color = (255.0 - min(k, 10) * 13) / 255.0

            _ = cv2.line(image, (stroke[0][i], stroke[1][i]), (stroke[0][i + 1], stroke[1][i + 1]), color=color, thickness=5)

    if xmin == xmax: xmin, xmax = 0, 255

    if ymin == ymax: ymin, ymax = 0, 255

    return cv2.resize(image[ymin:ymax, xmin:xmax], (img_size, img_size))
def feature_extraction(df_data: pd.DataFrame) -> pd.DataFrame:

    """ Return the feature exploration, engineering and cleansing.

    

    Args:

        df_data (pd.DataFrame): The data to extract features.

    

    Returns:

        pd.DataFrame: The extracted features dataframe.

    """

    

    # feature extraction: drawing

    df_data['drawing'] = df_data['drawing'].apply(lambda x: json.loads(x))

    

    # feature extraction: word

    df_data['word'] = df_data['word'].apply(lambda x: -1 if x == '' else class_dict[x.replace(' ', '_')])

    

    # feature extraction: drawing to pixel

    df_data['pixel'] = df_data['drawing'].apply(lambda x: draw2pixel(x))

    

    return df_data
def feature_extraction2(df_data: pd.DataFrame) -> pd.DataFrame:

    """ Return the feature exploration, engineering and cleansing.

    

    Args:

        df_data (pd.DataFrame): The data to extract features.

    

    Returns:

        pd.DataFrame: The extracted features dataframe.

    """

    

    # feature extraction: remove countrycode, drawing and datatype

    df_data = df_data.drop(['countrycode', 'drawing', 'datatype'], axis=1)

    

    return df_data
# feature extraction: step 1

df_data = feature_extraction(df_data)
# feature exploration: image

_ = drawplot(df_data.loc[:19, 'drawing'], df_data.loc[:19, 'word'])
# feature exploration: image

_ = imageplot(df_data.loc[:19, 'pixel'], df_data.loc[:19, 'word'], (img_size, img_size))
# feature extraction: step 2

df_data = feature_extraction2(df_data)
# describe data dataframe

df_data.describe(include='all')
# verify dtypes object

df_data.info()
# memory clean-up

del df_data, df_train, df_test

gc.collect()
def train_generator() -> tuple:

    """ Return training data generator.

    

    Returns:

        tuple: The training data tuple.

    """

    

    while True:

        for k in np.random.permutation(range(num_shuffles - 1)):

            for df_data in pd.read_csv('../input/quick-draw-doodle-recognition-challenge-shufflecsv/train_k%d.csv.gz' %k, chunksize=chunksize):

                # feature extraction: drawing

                df_data['drawing'] = df_data['drawing'].apply(lambda x: json.loads(x))

                

                # feature extraction: word

                df_data['word'] = df_data['word'].apply(lambda x: -1 if x == '' else class_dict[x.replace(' ', '_')])

                

                # feature extraction: drawing to pixel

                x = np.zeros((df_data.shape[0], img_size, img_size, 1))

                for i, drawing in enumerate(df_data['drawing'].values): x[i, :, :, 0] = draw2pixel(drawing)

                y = keras.utils.to_categorical(df_data['word'], num_classes=num_classes)

                yield x, y



# training data generator

gen_train = train_generator()
# testing (validating) data

df_data = pd.read_csv('../input/quick-draw-doodle-recognition-challenge-shufflecsv/train_k%d.csv.gz' %(num_shuffles - 1), nrows=34000)



# feature extraction: drawing

df_data['drawing'] = df_data['drawing'].apply(lambda x: json.loads(x))



# feature extraction: word

df_data['word'] = df_data['word'].apply(lambda x: -1 if x == '' else class_dict[x.replace(' ', '_')])



# feature extraction: drawing to pixel

x_validate = np.zeros((df_data.shape[0], img_size, img_size, 1))

for i, drawing in enumerate(df_data['drawing'].values): x_validate[i, :, :, 0] = draw2pixel(drawing)

y_validate = keras.utils.to_categorical(df_data['word'], num_classes=num_classes)
# memory clean-up

del df_data

gc.collect()
def top_3_categorical_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:

    """ Return top 3 categorical accuracy.

    

    Args:

        y_true (np.ndarray): The ground truth (correct) labels.

        y_pred (np.ndarray): The predicted labels.

    

    Returns:

        float: The top 3 categorical accuracy.

    """

    

    return top_k_categorical_accuracy(y_true, y_pred, k=3)
# mobilenet model setup

model_mobilenet = MobileNet(input_shape=(img_size, img_size, 1), alpha=1.0, dropout=1e-3, weights=None, classes=num_classes)

model_mobilenet.summary()
# mobilenet model setup

model_mobilenet.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[top_3_categorical_accuracy])



# mobilenet model fit

hist = model_mobilenet.fit_generator(gen_train, steps_per_epoch=800, epochs=96, verbose=2, validation_data=(x_validate, y_validate))

model_hist = pd.DataFrame(hist.history)



# mobilenet model metrics

model_mobilenet_score = model_mobilenet.evaluate(x_validate, y_validate, verbose=1)

print('mobilenet\n  top 3 categorical accuracy score: %0.4f' %model_mobilenet_score[1])
# plot the model history

fig, axes = plt.subplots(figsize=(20, 10), ncols=1, nrows=2)

axes = axes.flatten()

model_hist.plot(y='top_3_categorical_accuracy', kind='line', ax=axes[0])

model_hist.plot(y='val_top_3_categorical_accuracy', kind='line', ax=axes[0])

model_hist.plot(y='loss', kind='line', ax=axes[1])

model_hist.plot(y='val_loss', kind='line', ax=axes[1])

for axis in axes: axis.set_xlabel('epoch')
# mobilenet model save

model_mobilenet.save('model_mobilenet.h5')
# memory clean-up

del x_validate, y_validate

gc.collect()
# acquiring testing data

df_test = pd.read_csv('../input/quickdraw-doodle-recognition/test_simplified.csv')



# feature extraction: drawing

df_test['drawing'] = df_test['drawing'].apply(lambda x: json.loads(x))
# prepare testing data and compute the observed value

x_test = np.zeros((df_test.shape[0], img_size, img_size, 1))

for i, drawing in enumerate(df_test['drawing'].values): x_test[i, :, :, 0] = draw2pixel(drawing)

y_test = np.argsort(-model_mobilenet.predict(x_test, verbose=1))[:, 0:3]

df_word = pd.DataFrame({'top 1': y_test[:, 0], 'top 2': y_test[:, 1], 'top 3': y_test[:, 2]})

df_word = df_word.replace(classreverse_dict)

df_word['submission'] = df_word['top 1'] + ' ' + df_word['top 2'] + ' ' + df_word['top 3']
# submit the results

out = pd.DataFrame({'key_id': df_test['key_id'], 'word': df_word['submission']})

out.to_csv('submission.csv', index=False)
# visualize head of the submitted results

out.head(n=5)