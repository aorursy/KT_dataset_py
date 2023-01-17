%matplotlib inline

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"



import os, ast, cv2

import matplotlib.pyplot as plt

import dask.dataframe    as dd



import pandas as pd

import numpy  as np

from tqdm import tqdm

from ast  import literal_eval
DP_DIR    = '../input/shuffle-csvs/'

INPUT_DIR = '../input/quickdraw-doodle-recognition/'



BASE_SIZE = 256

NCSVS     = 100

NCATS     = 340



def f2cat(filename: str) -> str:

    return filename.split('.')[0]



def list_all_categories():

    files = os.listdir(os.path.join(INPUT_DIR, 'train_simplified'))

    return sorted([f2cat(f) for f in files], key=str.lower)
#

# 데이터를 읽어오기 전

# dask.dataframe으로 불러와

# 간단하게 구조만 확인

#

ddf = dd.read_csv('../input/quickdraw-doodle-recognition/train_simplified/a*.csv')
ddf
ddf.head(10)
ddf.tail(10)
# dask.compute 시 멀티프로세싱 옵션을 주어 빠르게 연산할 수 있게끔

row = ddf.loc[1].compute(scheduler='processes', num_workers=4)

row.iloc[0]
stroke = row.iloc[0]['drawing']

title  = 'Unrecognized ' + row.iloc[0]['word']
print(stroke)

print(type(stroke))
print(literal_eval(stroke))

print(literal_eval(stroke)[0])

print(type(literal_eval(stroke)[0]))
#

# stroke 데이터를 기반으로 이미지를 그린다

# time_color를 이용하여

# 획 순서와 방향을 확인할 수 있도록

#

def draw_cv2(raw_strokes, size=256, lw=6, time_color=True):

    img = np.zeros((BASE_SIZE, BASE_SIZE), np.uint8)

    for t, stroke in enumerate(raw_strokes):

        for i in range(len(stroke[0]) - 1):

            color = 255 - min(t, 10) * 13 if time_color else 255

            _ = cv2.line(img, (stroke[0][i], stroke[1][i]),

                         (stroke[0][i + 1], stroke[1][i + 1]), color, lw)

    if size != BASE_SIZE:

        return cv2.resize(img, (size, size))

    else:

        return img



plt.imshow(draw_cv2(literal_eval(stroke)), cmap='bone')

plt.title(title)

plt.show()
import tensorflow as tf

from tensorflow import keras

from tensorflow.keras.layers  import Conv2D, MaxPooling2D

from tensorflow.keras.layers  import Dense, Dropout, Flatten, Activation

from tensorflow.keras.metrics import categorical_accuracy, top_k_categorical_accuracy, categorical_crossentropy

from tensorflow.keras.models  import Sequential, load_model

from tensorflow.keras.callbacks    import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from tensorflow.keras.optimizers   import Adam

from tensorflow.keras.applications import MobileNetV2

from tensorflow.keras.applications.mobilenet import preprocess_input
def apk(actual, predicted, k=3):

    """

    Source: https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py

    """

    if len(predicted) > k:

        predicted = predicted[:k]

    score = 0.0

    num_hits = 0.0

    for i, p in enumerate(predicted):

        if p in actual and p not in predicted[:i]:

            num_hits += 1.0

            score += num_hits / (i + 1.0)

    if not actual:

        return 0.0

    return score / min(len(actual), k)



def mapk(actual, predicted, k=3):

    """

    Source: https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py

    """

    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])



def preds2catids(predictions):

    return pd.DataFrame(np.argsort(-predictions, axis=1)[:, :3], columns=['a', 'b', 'c'])



def top_3_accuracy(y_true, y_pred):

    return top_k_categorical_accuracy(y_true, y_pred, k=3)
#

# 이미지 데이터 생성 및 전처리

# 과적합을 피하기위해 제공된 학습셋을 셔플하여 사용하였다

# 

def image_generator_xd(size, batchsize, ks, lw=6, time_color=True):

    while True:

        for k in np.random.permutation(ks):

            

            filename = os.path.join(DP_DIR, 'train_k{}.csv.gz'.format(k))

            for df in pd.read_csv(filename, chunksize=batchsize):

                df['drawing'] = df['drawing'].apply(ast.literal_eval)

                x = np.zeros((len(df), size, size, 1))

                

                for i, raw_strokes in enumerate(df.drawing.values):

                    x[i, :, :, 0] = draw_cv2(raw_strokes, size=size, lw=lw,

                                             time_color=time_color)

                x = preprocess_input(x).astype(np.float32)

                y = keras.utils.to_categorical(df.y, num_classes=NCATS)

                yield x, y



def df_to_image_array_xd(df, size, lw=6, time_color=True):

    df['drawing'] = df['drawing'].apply(ast.literal_eval)

    x = np.zeros((len(df), size, size, 1))

    for i, raw_strokes in enumerate(df.drawing.values):

        x[i, :, :, 0] = draw_cv2(raw_strokes, size=size, lw=lw, time_color=time_color)

    x = preprocess_input(x).astype(np.float32)

    return x
STEPS = 1000

EPOCHS = 20

size = 80

batchsize = 300
# 마지막 파일을 valid set으로 사용

valid_df = pd.read_csv(os.path.join(DP_DIR, 'train_k{}.csv.gz'.format(NCSVS - 1)), nrows=34000)

x_valid = df_to_image_array_xd(valid_df, size)

y_valid = keras.utils.to_categorical(valid_df.y, num_classes=NCATS)

print(x_valid.shape, y_valid.shape)
train_datagen = image_generator_xd(size=size, batchsize=batchsize, ks=range(NCSVS - 1))
# 학습셋 확인

x, y = next(train_datagen)

n = 3

fig, axs = plt.subplots(nrows=n, ncols=n, sharex=True, sharey=True, figsize=(9, 9))

for i in range(n**2):

    ax = axs[i // n, i % n]

    (-x[i]+1)/2

    ax.imshow((-x[i, :, :, 0] + 1)/2, cmap=plt.cm.bone)

    ax.axis('off')

plt.tight_layout()

plt.show();
model = MobileNetV2(input_shape=(size, size, 1), alpha=1., weights=None, classes=NCATS)

model.compile(optimizer=Adam(lr=0.002), loss='categorical_crossentropy',

              metrics=[categorical_crossentropy, categorical_accuracy, top_3_accuracy])

model.summary()
filepath = "model.h5"

checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=0, save_best_only=True, mode='min')

callbacks = [EarlyStopping(patience=5, verbose=0), ReduceLROnPlateau(monitor='val_categorical_accuracy', factor=0.5, patience=5,

                      min_delta=0.005, mode='max', cooldown=3, verbose=1), checkpoint

]

hists = []

hist = model.fit_generator(

    train_datagen, steps_per_epoch=STEPS, epochs=EPOCHS, verbose=1,

    callbacks = callbacks

)

hists.append(hist)
valid_predictions = model.predict(x_valid, batch_size=128, verbose=1)

map3 = mapk(valid_df[['y']].values, preds2catids(valid_predictions).values)

print('Map3: {:.3f}'.format(map3))
y_loss  = hist.history['loss']



x_len = np.arange(len(y_loss))

plt.plot(x_len, y_loss, marker='.',  c='blue', label='Trainset_loss')



plt.legend(loc='upper right')

plt.grid()

plt.xlabel('epoch')

plt.ylabel('loss')

plt.show()
model_conv1D = load_model('../input/quickdraw/model_cnn_1.hdf5', custom_objects = {'top_3_accuracy':top_3_accuracy})

print(model_conv1D.summary())



del model_conv1D
model_conv2D_m = load_model('../input/quickdraw/model_cnn_2.hdf5', custom_objects = {'top_3_accuracy':top_3_accuracy})

print(model_conv2D_m.summary())



del model_conv2D_m
del train_datagen, valid_predictions, hists, x_valid, y_valid
test = pd.read_csv(os.path.join(INPUT_DIR, 'test_simplified.csv'))

test.head()

x_test = df_to_image_array_xd(test, size)

print(test.shape, x_test.shape)
test_predictions = model.predict(x_test, batch_size=128, verbose=1)
top3 = preds2catids(test_predictions)

top3.head()

top3.shape
cats = list_all_categories()

id2cat = {k: cat.replace(' ', '_') for k, cat in enumerate(cats)}

top3cats = top3.replace(id2cat)

top3cats.head()

top3cats.shape
test['word'] = top3cats['a'] + ' ' + top3cats['b'] + ' ' + top3cats['c']

submission = test[['key_id', 'word']]

submission.to_csv('submission.csv'.format(int(map3 * 10**4)), index=False)

submission.head()

submission.shape