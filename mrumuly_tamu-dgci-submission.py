import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf # neural nets
import matplotlib.pyplot as plt # plotting
from matplotlib.colors import rgb_to_hsv
from imageio import imread
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import mean_squared_error
train_list = pd.read_csv("../input/train.csv")
train_y = train_list.DGCI.values

train_X = np.zeros((len(train_y), 400, 300, 3))
for i, imageId in enumerate(train_list.Id.values):
    image = imread("../input/archive/{}.jpg".format(imageId))
    if image.shape == (300, 400, 3):
        image = np.rot90(image)
    train_X[i] = rgb_to_hsv(image)/255

sub_size = 100

# show random example of image
plt.imshow(train_X[np.random.randint(len(train_X))])
# Split each image into multiple 50x50 non-overlapping images. 
# Take advantage of repetition, reduce size of network while preserving all information (as opposed to cropping). Preserves mapping to labels
def fragment(X, y=None, size=50):
    # compute index mappings
    n_across = int(X.shape[1]//size)
    n_down = int(X.shape[2]//size)
    expansion = int(n_across*n_down)
    new_len = int(X.shape[0]*expansion)
    
    # image expansion
    fX = np.zeros((new_len,size, size, 3))
    for i in range(len(X)):
        for m in range(n_across):
            for n in range(n_down):
                fX[int(i*expansion+m*n_down+n),:,:,:] = X[i, int(m*size):int((m+1)*size),int(n*size):int((n+1)*size),:]
    
    # label expansion
    if not y is None:
        fy = np.zeros(new_len)
        for i in range(len(y)):
            fy[i*expansion:(i+1)*expansion] = np.full(expansion, y[i])
        return expansion, fX, fy
    return expansion, fX
# produce fresh model
def make_model():
  model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=64, kernel_size=(7,7), padding='Same', activation='relu', input_shape=(sub_size,sub_size,3)),
    tf.keras.layers.AveragePooling2D(pool_size=(2,2), strides=(2,2)),
    tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding='Same', activation='relu'),
    tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding='Same', activation='relu'),
    tf.keras.layers.AveragePooling2D(pool_size=(2,2), strides=(2,2)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation=tf.keras.activations.linear)
  ])
  model.compile(
    optimizer=tf.keras.optimizers.Nadam(lr=0.0001),
    loss='mean_squared_error'
  )
  return model

# annealer
lr_annealing = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', patience=2, factor=0.5, min_lr=0.000001)

# data generator
def make_datagen():
    dg = tf.keras.preprocessing.image.ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=0,
        zoom_range = 0.0,
        width_shift_range=0.0,
        height_shift_range=0.0,
        horizontal_flip=False,
        vertical_flip=False
    )
    return dg
def regression_cross_val_score(
  estimator,
  X,
  y,
  datagen,
  cv=10,
  fit_params={},
  convert=lambda x:x,
  batch_size=86
):
  score = 0
  for train, test in ShuffleSplit(cv).split(X,y):
    train_exp, X_train, y_train = fragment(X[train], y[train], sub_size)
    test_exp, X_test, y_test = fragment(X[test], y[test], sub_size)
    model = estimator()
    data = datagen()
    data.fit(X_train)
    model.fit_generator(
        data.flow(X_train, y_train, batch_size=batch_size),
        validation_data=(X_test, y_test),
        steps_per_epoch=X_train.shape[0] // batch_size,
        **fit_params
    )
    
    s_part = mean_squared_error(y[test], convert(np.median(np.reshape(model.predict(X_test), (-1, test_exp)), axis=1)))
    print(s_part)
    score += s_part/cv
  return score # mean square error
# regression_cross_val_score(
#     make_model,
#     train_X,
#     train_y,
#     make_datagen,
#     cv = 4,
#     fit_params={
#         'epochs':20,
#         'callbacks':[lr_annealing],
#     },
#     batch_size=1024
# )
_, exp_train_X, exp_train_y = fragment(train_X, train_y, sub_size)
del train_X
del train_y
pos = np.arange(exp_train_X.shape[0])
np.random.shuffle(pos)
exp_train_X[pos] = exp_train_X
exp_train_y[pos] = exp_train_y

datagen = make_datagen()
datagen.fit(exp_train_X)

lr_anneal = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='loss', patience=2, factor=0.5, min_lr=0.000001)

model = make_model()
model.fit_generator(
        datagen.flow(exp_train_X, exp_train_y, batch_size=86),
        steps_per_epoch=exp_train_X.shape[0] // 86,
        epochs=20,
        callbacks=[lr_anneal]
    )
# del exp_train_X
# del exp_train_y
test_list = pd.read_csv("../input/sample.csv")

test_X = np.zeros((len(test_list.Id.values), 400, 300, 3))
for i, imageId in enumerate(test_list.Id.values):
    image = imread("../input/archive/{}.jpg".format(imageId))
    if image.shape == (300, 400, 3):
        image = np.rot90(image)
    test_X[i] = rgb_to_hsv(image[:400,:300,:])/255

test_exp, exp_test_X = fragment(test_X, size=sub_size)
prediction = np.median(np.reshape(model.predict(exp_test_X), (-1, test_exp)), axis=1)
print(prediction.shape)
test_list['DGCI'] = prediction
test_list.to_csv('submission.csv', index=False)
plt.hist(prediction)