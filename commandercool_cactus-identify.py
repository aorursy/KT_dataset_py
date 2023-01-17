import pandas as pd



import numpy as np



import matplotlib.pyplot as plt



from PIL import Image



from sklearn.metrics import confusion_matrix, roc_auc_score

from sklearn.model_selection import train_test_split



from keras.layers import Dense, Flatten, Dropout, Concatenate, BatchNormalization, GlobalMaxPooling2D, GlobalAveragePooling2D, Input

from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from keras.preprocessing.image import ImageDataGenerator

from keras.applications.densenet import DenseNet121

from keras.models import Sequential

from keras.models import Model

from keras import applications

from keras import optimizers



from tqdm import tqdm



import seaborn as sns



import os



%matplotlib inline
SEED = 99



HEIGHT = 32

WIDTH = 32



FULL_PATH = os.path.join("..", "input")

TRAIN_DIR = os.path.join(FULL_PATH, "train", "train")

TEST_DIR = os.path.join(FULL_PATH, "test", "test")

LABELS = os.path.join(FULL_PATH, "train.csv")
def plot_loss(history):

    plt.plot(history.history['loss'])

    plt.plot(history.history['val_loss'])

    plt.title('Model Loss')

    plt.ylabel('Loss')

    plt.xlabel('Epoch')

    plt.legend(['Train', 'Test'], loc='upper left')

    plt.show()
def plot_acc(history):

    plt.plot(history.history['acc'])

    plt.plot(history.history['val_acc'])

    plt.title('Model Accuracy')

    plt.ylabel('Accuracy')

    plt.xlabel('Epoch')

    plt.legend(['Train', 'Test'], loc='upper left')

    plt.show()
train = pd.read_csv(LABELS)
fig = plt.figure(figsize=(25, 8))

train_imgs = os.listdir(TRAIN_DIR)

for idx, img in enumerate(np.random.choice(train_imgs, 20)):

    ax = fig.add_subplot(4, 20//4, idx+1, xticks=[], yticks=[])

    im = Image.open(os.path.join(TRAIN_DIR, img))

    plt.imshow(im)

    lab = train.loc[train['id'] == img, 'has_cactus'].values[0]

    ax.set_title(f'Label: {lab}')
x_train, x_test, y_train, y_test = train_test_split(train['id'], 

                                                    train['has_cactus'], 

                                                    test_size=0.2,

                                                    random_state=3)
X_train = []

for img in tqdm(x_train):

    path = os.path.join(TRAIN_DIR, img)

    X_train.append(plt.imread(path))



X_test = []

for img in tqdm(x_test):

    path = os.path.join(TRAIN_DIR, img)

    X_test.append(plt.imread(path))



    

X_train = np.array(X_train)

X_test = np.array(X_test)



X_train = X_train.astype('float32')

X_test = X_test.astype('float32')



X_train /= 255

X_test /= 255
augmentations = ImageDataGenerator(

    vertical_flip=True,

    horizontal_flip=True,

    zoom_range=0.1)



augmentations.fit(X_train)
inputs = Input((32,32,3))

model_base = DenseNet121(weights='imagenet', include_top=False, input_shape=(32,32,3))

x = model_base(inputs)

x_1 = GlobalMaxPooling2D()(x)

x_2 = GlobalAveragePooling2D()(x)

x_3 = Flatten()(x)

x = Concatenate(axis=-1)([x_1,x_2,x_3])

x = Dropout(0.5)(x)

x = Dense(256,activation='relu')(x)

x = Dense(1,activation='sigmoid')(x)



for layer in model_base.layers:

    layer.trainable = True



model = Model(inputs,x)

model.summary()
early_stop = EarlyStopping(

    monitor='val_loss',

    min_delta=0.001,

    patience=10,

    verbose=1,

    mode='auto'

)



reduce = ReduceLROnPlateau(

    monitor='val_loss',

    factor=0.5,

    patience=10,

    verbose=1, 

    mode='auto',

    cooldown=1 

)



callbacks = [early_stop, reduce]
# opt = optimizers.SGD(lr=1e-3)

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
epochs = 50



history = model.fit_generator(

    augmentations.flow(X_train, 

                       y_train, 

                       batch_size=16),

    validation_data=(X_test, y_test),

    steps_per_epoch=150,

    validation_steps=150,

    epochs=epochs,

    callbacks=callbacks,

    verbose=1

)
plot_loss(history)
plot_acc(history)
[loss, accuracy] = model.evaluate(X_test, y_test)

print('Test Set Accuracy: ', str(accuracy*100), "%")
sample = pd.read_csv('../input/sample_submission.csv')

images_test = []

for images in tqdm(sample['id']):

    img = plt.imread('../input/test/test/' + images)

    images_test.append(img)



images_test = np.array(images_test)

images_test = images_test.astype(np.float32)

images_test /= 255
prediction = model.predict(images_test)
submission = pd.read_csv(os.path.join(FULL_PATH, "sample_submission.csv"))

submission['has_cactus'] = prediction
submission.to_csv('sample_submission.csv', index = False)
submission.head(10)
train_pred = model.predict(X_train, verbose= 1)

valid_pred = model.predict(X_test, verbose= 1)



train_acc = roc_auc_score(np.round(train_pred), y_train)

valid_acc = roc_auc_score(np.round(valid_pred), y_test)



confusion_matrix(np.round(valid_pred), y_test)