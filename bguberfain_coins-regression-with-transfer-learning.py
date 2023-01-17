import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from pathlib import Path

from PIL import Image

from tqdm import tqdm_notebook

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler



%matplotlib inline
input_path = Path('/kaggle/input/br-coins/regression_dataset/all/')

im_size = 320
image_files = list(input_path.glob('*.jpg'))
def read_file(fname):

    # Read image

    im = Image.open(fname)



    # Resize

    im.thumbnail((im_size, im_size))



    # Convert to numpy array

    im_array = np.asarray(im)



    # Get target

    target = int(fname.stem.split('_')[0])



    return im_array, target
images = []

targets = []



for image_file in tqdm_notebook(image_files):

    image, target = read_file(image_file)

    

    images.append(image)

    targets.append(target)
X = (np.array(images).astype(np.float32) / 127.5) - 1

y = np.array(targets)
X.shape, y.shape
i = 555

plt.imshow(np.uint8((X[i] + 1) * 127.5))

plt.title(str(y[i]));
scaler = StandardScaler()

y = scaler.fit_transform(y[:, np.newaxis])[:, 0]
X_train, X_valid, y_train, y_valid, fname_train, fname_valid = train_test_split(

    X, y, image_files, test_size=0.2, random_state=42)
im_width = X.shape[2]

im_height = X.shape[1]



im_width, im_height
del X, y, images, targets
from keras.layers import Dense, Input

from keras.models import load_model, Model

from keras.optimizers import Adam

from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
model_classification = load_model('../input/coins-classification/best.model')
model_classification.summary()
cnn_output_layer = model_classification.get_layer('global_average_pooling2d_1')

cnn_output_layer
cnn = Model(model_classification.input, cnn_output_layer.output, name='cnn_classification')

cnn.summary()
inp = Input((im_height, im_width, 3))



x = cnn(inp)



x = Dense(256, activation='relu')(x)

out = Dense(1)(x)



model = Model(inp, out)



model.summary()
optim = Adam(lr=1e-3)

model.compile(optim, 'mse')
callbacks = [

    ReduceLROnPlateau(patience=5, factor=0.1, verbose=True, min_lr=5e-6),

    ModelCheckpoint('best.model', save_best_only=True),

    EarlyStopping(patience=12, min_delta=1e-4)

]



history = model.fit(X_train, y_train, epochs=2000, validation_data=(X_valid, y_valid), batch_size=32,

                   callbacks=callbacks)
df_history = pd.DataFrame(history.history)
ax = df_history[['loss', 'val_loss']].plot()

# ax.set_ylim(0.0150, 0.02)
model.load_weights('best.model')
model.evaluate(X_valid, y_valid)
y_pred = model.predict(X_valid)
y_pred_cls = scaler.inverse_transform(y_pred)[:, 0]
y_pred_cls = (y_pred_cls / 5).round() * 5
y_cls = scaler.inverse_transform(y_valid[:, np.newaxis])[:, 0]
np.isclose(y_pred_cls, y_cls).mean()
plt.figure(figsize=(10, 7))

plt.scatter(y_cls, y_pred_cls, s=2, alpha=0.5)

plt.xlabel("True")

plt.ylabel("Pred")
errors = np.where(~np.isclose(y_pred_cls, y_cls))[0]

errors
i = np.random.choice(errors)

plt.figure(figsize=(10, 10))

im = Image.open(fname_valid[i])

plt.imshow(np.uint8(im), interpolation='bilinear')

plt.title('Idx: {} Class: {}, Predicted: {}'.format(i, y_cls[i], y_pred_cls[i]));