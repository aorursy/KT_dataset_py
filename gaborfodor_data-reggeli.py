%matplotlib inline
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import ast
import os
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation
from tensorflow.keras.metrics import categorical_accuracy, top_k_categorical_accuracy, categorical_crossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.figure_factory as ff
from IPython.core.display import HTML
start = dt.datetime.now()
DP_DIR = '../input/shuffle-animal-csvs/'
INPUT_DIR = '../input/quickdraw-doodle-recognition/'
# INPUT_DIR = '../input/'
BW_DIR = '../input/black-white-cnn-animals/'
GS_DIR = '../input/greyscale-mobilenet-animals/'
BASE_SIZE = 256
NCSVS = 100
np.random.seed(seed=1987)
tf.set_random_seed(seed=1987)

def f2cat(filename: str) -> str:
    return filename.split('.')[0]

class Simplified():
    def __init__(self, input_path='./input'):
        self.input_path = input_path

    def list_all_categories(self):
        files = os.listdir(os.path.join(self.input_path, 'train_simplified'))
        return [f2cat(f) for f in files]

    def read_training_csv(self, category, nrows=None, usecols=None, drawing_transform=False):
        df = pd.read_csv(os.path.join(self.input_path, 'train_simplified', category + '.csv'),
                         nrows=nrows, parse_dates=['timestamp'], usecols=usecols)
        if drawing_transform:
            df['drawing'] = df['drawing'].apply(ast.literal_eval)
        return df
s = Simplified(INPUT_DIR)
animals = ['ant', 'bat', 'bear', 'bee', 'bird', 'butterfly', 'camel', 'cat', 'cow',
           'crab', 'crocodile', 'dog', 'dolphin', 'dragon', 'duck', 'elephant', 'fish',
           'flamingo', 'frog', 'giraffe', 'hedgehog', 'horse', 'kangaroo', 'lion',
           'lobster', 'monkey', 'mosquito', 'mouse', 'octopus', 'owl', 'panda',
           'parrot', 'penguin', 'pig', 'rabbit', 'raccoon', 'rhinoceros', 'scorpion',
           'sea turtle', 'shark', 'sheep', 'snail', 'snake', 'spider', 'squirrel',
           'swan', 'teddy-bear', 'tiger', 'whale', 'zebra']
NCATS = len(animals)
df = s.read_training_csv('owl', nrows=100, drawing_transform=True)
df.head()

drawing = df.drawing.values[0]
print(drawing)
print('--------------------------------------')
print('This drawing has {} strokes.'.format(len(drawing)))
print('i, [[xs], [ys]]')
for i, stroke in enumerate(drawing):
    print(i, stroke)
n = 10
fig, axs = plt.subplots(nrows=n, ncols=n, sharex=True, sharey=True, figsize=(16, 10))
for i, row in df[: n * n].iterrows():
    ax = axs[i // n, i % n]
    for x, y in row.drawing:
        color = 'green' if row.recognized else 'red'
        ax.plot(x, -np.array(y), lw=3, color=color)
    ax.axis('off')
plt.suptitle('Recognized and unrecognized owls')
plt.show();
def draw_cv2(raw_strokes, size=256, lw=6):
    img = np.zeros((BASE_SIZE, BASE_SIZE), np.uint8)
    for stroke in raw_strokes:
        for i in range(len(stroke[0]) - 1):
            _ = cv2.line(img, (stroke[0][i], stroke[1][i]), (stroke[0][i + 1], stroke[1][i + 1]), 255, lw)
    if size != BASE_SIZE:
        return cv2.resize(img, (size, size))
    else:
        return img

def image_generator(size, batchsize, ks, lw=6):
    while True:
        for k in np.random.permutation(ks):
            filename = os.path.join(DP_DIR, 'train_k{}.csv.gz'.format(k))
            for df in pd.read_csv(filename, chunksize=batchsize):
                df['drawing'] = df['drawing'].apply(ast.literal_eval)
                x = np.zeros((len(df), size, size))
                for i, raw_strokes in enumerate(df.drawing.values):
                    x[i] = draw_cv2(raw_strokes, size=size, lw=lw)
                x = x / 255.
                x = x.reshape((len(df), size, size, 1)).astype(np.float32)
                y = keras.utils.to_categorical(df.y, num_classes=NCATS)
                yield x, y

def df_to_image_array(df, size, lw=6):
    df['drawing'] = df['drawing'].apply(ast.literal_eval)
    x = np.zeros((len(df), size, size))
    for i, raw_strokes in enumerate(df.drawing.values):
        x[i] = draw_cv2(raw_strokes, size=size, lw=lw)
    x = x / 255.
    x = x.reshape((len(df), size, size, 1)).astype(np.float32)
    return x

def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)

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
size = 32
model = Sequential()
model.add(Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu',
                 input_shape=(size, size, 1)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(NCATS, activation='softmax'))
model.compile(optimizer=Adam(lr=0.002), loss='categorical_crossentropy',
              metrics=[categorical_crossentropy, categorical_accuracy, top_3_accuracy])
print(model.summary())
model.load_weights(filepath=os.path.join(BW_DIR, 'bw_animal_cnn.h5'))
valid_df = pd.read_csv(os.path.join(DP_DIR, 'train_k{}.csv.gz'.format(NCSVS - 1)), nrows=10**5)
x_valid = df_to_image_array(valid_df, size)
y_valid = keras.utils.to_categorical(valid_df.y, num_classes=NCATS)
print(x_valid.shape, y_valid.shape)
print('Validation array memory {:.2f} GB'.format(x_valid.nbytes / 1024.**3 ))
valid_predictions = model.predict(x_valid, batch_size=128, verbose=1)
map3 = mapk(valid_df[['y']].values, preds2catids(valid_predictions).values)
print('Top1 Accuracy: {:.3f}'.format(np.mean(valid_df.y.values == np.argmax(valid_predictions, 1))))
print('Map@3: {:.3f}'.format(map3))
predicted_cat = np.argmax(valid_predictions, 1)
cmx = confusion_matrix(valid_df.y.values, np.argmax(valid_predictions, 1))
k = 20
layout = dict(
    title = 'BW CNN Confusion Matrix',
    xaxis= dict(title='Predicted Class', ticklen=5, zeroline=False, gridwidth=2),
    yaxis=dict(title='Ground Truth', ticklen=5, gridwidth=2),
    width = 800,
    height = 800,
    margin=go.layout.Margin(l=200, r=50, b=50, t=250, pad=4),
)
fig = ff.create_annotated_heatmap(
    z=cmx[:k, :k],
    x=list(animals[:k]),
    y=list(animals[:k]),
    colorscale='Blues',
    reversescale=True,
    showscale=True,
    font_colors = ['#efecee', '#3c3636'])
fig['layout'].update(layout)
py.iplot(fig, filename='bw_confusion')
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
size = 64
model = MobileNet(input_shape=(size, size, 1), alpha=1., weights=None, classes=NCATS)
model.compile(optimizer=Adam(lr=0.002), loss='categorical_crossentropy',
              metrics=[categorical_crossentropy, categorical_accuracy, top_3_accuracy])
print(model.summary())
model.load_weights(filepath=os.path.join(GS_DIR, 'gs_animal_mobile.h5'))
valid_df = pd.read_csv(os.path.join(DP_DIR, 'train_k{}.csv.gz'.format(NCSVS - 1)), nrows=10**5)
x_valid = df_to_image_array_xd(valid_df, size)
y_valid = keras.utils.to_categorical(valid_df.y, num_classes=NCATS)
print(x_valid.shape, y_valid.shape)
print('Validation array memory {:.2f} GB'.format(x_valid.nbytes / 1024.**3 ))
bw_hist_df = pd.read_csv(os.path.join(BW_DIR, 'bw_cnn_history.csv'))
gs_hist_df = pd.read_csv(os.path.join(GS_DIR, 'gs_mobile_history.csv'))
data = [
    go.Scatter(
        x=gs_hist_df.index.values,
        y=gs_hist_df.val_categorical_accuracy.values,
        mode='lines',
        name='64x64 MobileNet',
        line=dict(width=4, color='#5ac995')
    ),
    go.Scatter(
        x=bw_hist_df.index,
        y=bw_hist_df.val_categorical_accuracy.values,
        mode='lines',
        name='32x32 Simple CNN',
        line=dict(width=4, color='#007FB4')
    ),
]
layout = go.Layout(
    title='Validation Performance',
    xaxis=dict(title='Epoch', ticklen=5, zeroline=False, gridwidth=2),
    yaxis=dict(title='Accuracy', ticklen=5, gridwidth=2),
    showlegend=True
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='users')
valid_predictions = model.predict(x_valid, batch_size=128, verbose=1)
map3 = mapk(valid_df[['y']].values, preds2catids(valid_predictions).values)
print('Top1 Accuracy: {:.3f}'.format(np.mean(valid_df.y.values == np.argmax(valid_predictions, 1))))
print('Map@3: {:.3f}'.format(map3))
predicted_cat = np.argmax(valid_predictions, 1)
cmx = confusion_matrix(valid_df.y.values, np.argmax(valid_predictions, 1))
k = 20
layout = dict(
    title = 'GS Mobilenet Confusion Matrix',
    xaxis= dict(title='Predicted Class', ticklen=5, zeroline=False, gridwidth=2),
    yaxis=dict(title='Ground Truth', ticklen=5, gridwidth=2),
    width = 800,
    height = 800,
    margin=go.layout.Margin(l=200, r=50, b=50, t=250, pad=4),
)
fig = ff.create_annotated_heatmap(
    z=cmx[:k, :k],
    x=list(animals[:k]),
    y=list(animals[:k]),
    colorscale='Greens',
    reversescale=True,
    showscale=True,
    font_colors = ['#efecee', '#3c3636'])
fig['layout'].update(layout)
py.iplot(fig, filename='bw_confusion')
end = dt.datetime.now()
print('Latest run {}.\nTotal time {}s'.format(end, (end - start).seconds))
HTML('''
<h1 id="documentation-for-individual-models">Available pretrained keras models</h1>
<table>
<thead>
<tr>
<th>Model</th>
<th align="right">Size</th>
<th align="right">Top-1 Accuracy</th>
<th align="right">Top-5 Accuracy</th>
<th align="right">Parameters</th>
<th align="right">Depth</th>
</tr>
</thead>
<tbody>
<tr>
<td><a href="https://keras.io/applications/#xception">Xception</a></td>
<td align="right">88 MB</td>
<td align="right">0.790</td>
<td align="right">0.945</td>
<td align="right">22,910,480</td>
<td align="right">126</td>
</tr>
<tr>
<td><a href="https://keras.io/applications/#vgg16">VGG16</a></td>
<td align="right">528 MB</td>
<td align="right">0.713</td>
<td align="right">0.901</td>
<td align="right">138,357,544</td>
<td align="right">23</td>
</tr>
<tr>
<td><a href="https://keras.io/applications/#vgg19">VGG19</a></td>
<td align="right">549 MB</td>
<td align="right">0.713</td>
<td align="right">0.900</td>
<td align="right">143,667,240</td>
<td align="right">26</td>
</tr>
<tr>
<td><a href="https://keras.io/applications/#resnet50">ResNet50</a></td>
<td align="right">99 MB</td>
<td align="right">0.749</td>
<td align="right">0.921</td>
<td align="right">25,636,712</td>
<td align="right">168</td>
</tr>
<tr>
<td><a href="https://keras.io/applications/#inceptionv3">InceptionV3</a></td>
<td align="right">92 MB</td>
<td align="right">0.779</td>
<td align="right">0.937</td>
<td align="right">23,851,784</td>
<td align="right">159</td>
</tr>
<tr>
<td><a href="https://keras.io/applications/#inceptionresnetv2">InceptionResNetV2</a></td>
<td align="right">215 MB</td>
<td align="right">0.803</td>
<td align="right">0.953</td>
<td align="right">55,873,736</td>
<td align="right">572</td>
</tr>
<tr>
<td><a href="https://keras.io/applications/#mobilenet">MobileNet</a></td>
<td align="right">16 MB</td>
<td align="right">0.704</td>
<td align="right">0.895</td>
<td align="right">4,253,864</td>
<td align="right">88</td>
</tr>
<tr>
<td><a href="https://keras.io/applications/#mobilenetv2">MobileNetV2</a></td>
<td align="right">14 MB</td>
<td align="right">0.713</td>
<td align="right">0.901</td>
<td align="right">3,538,984</td>
<td align="right">88</td>
</tr>
<tr>
<td><a href="https://keras.io/applications/#densenet">DenseNet121</a></td>
<td align="right">33 MB</td>
<td align="right">0.750</td>
<td align="right">0.923</td>
<td align="right">8,062,504</td>
<td align="right">121</td>
</tr>
<tr>
<td><a href="https://keras.io/applications/#densenet">DenseNet169</a></td>
<td align="right">57 MB</td>
<td align="right">0.762</td>
<td align="right">0.932</td>
<td align="right">14,307,880</td>
<td align="right">169</td>
</tr>
<tr>
<td><a href="https://keras.io/applications/#densenet">DenseNet201</a></td>
<td align="right">80 MB</td>
<td align="right">0.773</td>
<td align="right">0.936</td>
<td align="right">20,242,984</td>
<td align="right">201</td>
</tr>
<tr>
<td><a href="https://keras.io/applications/#nasnet">NASNetMobile</a></td>
<td align="right">23 MB</td>
<td align="right">0.744</td>
<td align="right">0.919</td>
<td align="right">5,326,716</td>
<td align="right">-</td>
</tr>
<tr>
<td><a href="https://keras.io/applications/#nasnet">NASNetLarge</a></td>
<td align="right">343 MB</td>
<td align="right">0.825</td>
<td align="right">0.960</td>
<td align="right">88,949,818</td>
<td align="right">-</td>
</tr>
</tbody>
</table>
<p>The top-1 and top-5 accuracy refers to the model's performance on the ImageNet validation dataset.</p>
''')