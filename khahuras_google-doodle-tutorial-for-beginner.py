## Import các gói thư viện cần thiết để sử dụng xuyên suốt notebook này



# Các gói cơ bản

import numpy as np # thư viện tính toán đại số tuyến tính

import pandas as pd # thư viện tiện lợi để làm việc trên các "dataframe", hay gọi là file bảng (.csv)

import os # thư viện liên quan tới hệ thống (kiểm tra thư mục hiện tại xem có những file gì...)

print(os.listdir("../input")) # chạy dòng này sẽ hiện ra các file/folder trong thư mục hiện hành (nhấn Ctrl+Enter sẽ chạy cả ô code này)



# Tiến hành gán tên một số thư mục cho tiện sử dụng

DP_DIR = '../input/doodle-kha/'

DP_DIR_2 = '../input/doodle-kha-2/'

INPUT_DIR = '../input/quickdraw-doodle-recognition/'

#MODEL_DIR = '../input/doodle-modelweights/'
# Các gói nâng cao

from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation, Conv2D, MaxPooling2D # các lớp trong neural network

from tensorflow.keras.metrics import categorical_accuracy, top_k_categorical_accuracy, categorical_crossentropy # Các hàm mục tiêu

from tensorflow.keras.models import Sequential # dạng mô hình (Sequential là mô hình tuần tự, tức là khi sử dụng ta sẽ add tuần tự các layers, hồi sau sẽ rõ)

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint # các hàm gọi về trong quá trình huấn luyện

from tensorflow.keras.optimizers import Adam # giải thuật Adam cho việc huấn luyện

from tensorflow.keras.applications import MobileNet # mô hình MobileNet, đã được nhóm nghiên cứu khác phát triển, ta chỉ việc sử dụng

from tensorflow.keras.applications.mobilenet import preprocess_input



#%matplotlib inline

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

import ast

import json

import datetime as dt

import cv2

import tensorflow as tf

from tensorflow import keras

from tqdm import tqdm_notebook as tqdm
def f2cat(filename: str) -> str:

    return filename.split('.')[0]



def list_all_categories():

    files = os.listdir(os.path.join(INPUT_DIR, 'train_simplified'))

    return sorted([f2cat(f) for f in files], key=str.lower)



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



def draw_cv2(raw_strokes, size=256, lw=6, time_color=True):

    img = np.zeros((BASE_SIZE, BASE_SIZE), np.uint8)

    for t, stroke in enumerate(raw_strokes):

        for i in range(len(stroke[0]) - 1):

            color = 255 - min(t, 10) * 13 if time_color else 255

            _ = cv2.line(img, (stroke[0][i], stroke[1][i]),

                         (stroke[0][i + 1], stroke[1][i + 1]), color, lw)

    if size != BASE_SIZE: return cv2.resize(img, (size, size))

    else: return img



def image_generator_xd(size, batchsize, ks, lw=6, time_color=True):

    while True:

        for k in np.random.permutation(ks):

            if k < 32: filename = os.path.join(DP_DIR, 'train_kha{}.csv/train_kha{}.csv'.format(k,k))

            else: filename = os.path.join(DP_DIR_2, 'train_kha{}.csv/train_kha{}.csv'.format(k,k))

            for df in pd.read_csv(filename, chunksize=batchsize):

                df['drawing'] = df['drawing'].apply(ast.literal_eval)

                x = np.zeros((len(df), size, size, 1))

                for i, raw_strokes in enumerate(df.drawing.values):

                    x[i, :, :, 0] = draw_cv2(raw_strokes, size=size, lw=lw,

                                             time_color=time_color)

                x = preprocess_input(x).astype(np.float16)

                y = keras.utils.to_categorical(df.y, num_classes=NCATS)

                yield x, y



def df_to_image_array_xd(df, size, lw=6, time_color=True):

    df['drawing'] = df['drawing'].apply(json.loads) #  json.loads , ast.literal_eval

    x = np.zeros((len(df), size, size, 1))

    for i, raw_strokes in enumerate(df.drawing.values):

        x[i, :, :, 0] = draw_cv2(raw_strokes, size=size, lw=lw, time_color=time_color)

    x = preprocess_input(x).astype(np.float32)

    return x
MODEL_WEIGHTS = None

#MODEL_WEIGHTS = '../input/mbn-crossentropy-size64/model_first32csv-epoch4-size64-crossentropy'

OUTPUT_NAME = 'epoch1'

TRAIN_KS = range(10)



batchsize = 512

STEPS =   993000*len(TRAIN_KS)//batchsize  # 23500000//batchsize # steps per epoch (1 csv ~ 990K rows, 24 csvs ~ 23,500,000)

EPOCHS = 1

size = 64 # image size after resized (less or equal 256)

SEED = 1991



BASE_SIZE = 256 # original image size before downsampling

NCATS = 340 # no. classes
np.random.seed(seed=SEED)

tf.set_random_seed(seed=SEED)



model = MobileNet(input_shape=(size, size, 1), alpha=1., weights=None, classes=NCATS)

model.compile(optimizer=Adam(lr=0.002), loss='categorical_crossentropy',

              metrics=[categorical_crossentropy, categorical_accuracy, top_3_accuracy])

print(model.summary())



if MODEL_WEIGHTS is not None:

    model.load_weights(MODEL_WEIGHTS + '.h5')



train_datagen = image_generator_xd(size=size, batchsize=batchsize, ks=TRAIN_KS)
hists = []

hist = model.fit_generator(

    train_datagen, steps_per_epoch=STEPS, epochs=EPOCHS, verbose=1,

    validation_data=None,

   # callbacks = callbacks

)

hists.append(hist)
model.save_weights("model_" + OUTPUT_NAME +".h5")

print("Model weights saved.")
test = pd.read_csv(os.path.join(INPUT_DIR, 'test_simplified.csv'))

x_test = df_to_image_array_xd(test, size)

print(test.shape, x_test.shape)

print('Test array memory {:.2f} GB'.format(x_test.nbytes / 1024.**3 ))
test_predictions = model.predict(x_test, batch_size=128, verbose=1)



top3 = preds2catids(test_predictions)



cats = list_all_categories()

id2cat = {k: cat.replace(' ', '_') for k, cat in enumerate(cats)}

top3cats = top3.replace(id2cat)



test_predictions

np.save('test_prediction_'+OUTPUT_NAME, test_predictions)
test['word'] = top3cats['a'] + ' ' + top3cats['b'] + ' ' + top3cats['c']

submission = test[['key_id', 'word']]

submission.to_csv('Submission_'+OUTPUT_NAME, index=False)