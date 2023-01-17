import cv2

from PIL import Image

import numpy as np

from keras import layers

from keras.applications import ResNet152, InceptionV3

from keras.callbacks import Callback, ModelCheckpoint

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.optimizers import Adam

import matplotlib.pyplot as plt

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.metrics import cohen_kappa_score, accuracy_score

import scipy

import tensorflow as tf

from tqdm import tqdm



%matplotlib inline
SEED = 42

np.random.seed(SEED)

tf.random.set_seed(SEED)
BATCH_SIZE = 16

LEARNING_RATE = 5e-05

EPOCHS = 40
train_df = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')
def preprocess_image(image_path, desired_size=224):

    im = Image.open(image_path)

    im = im.resize((desired_size, )*2, resample=Image.LANCZOS)

    

    return im
N = train_df.shape[0]

x_train = np.empty((N, 224, 224, 3), dtype=np.uint8)



for i, image_id in enumerate(tqdm(train_df['id_code'])):

    x_train[i, :, :, :] = preprocess_image(

        f'../input/aptos2019-blindness-detection/train_images/{image_id}.png'

    )
y_train = pd.get_dummies(train_df['diagnosis']).values
y_train_multi = np.empty(y_train.shape, dtype=y_train.dtype)

y_train_multi[:, 4] = y_train[:, 4]



for i in range(3, -1, -1):

    y_train_multi[:, i] = np.logical_or(y_train[:, i], y_train_multi[:, i+1])
x_train, x_val, y_train, y_val = train_test_split(

    x_train, y_train_multi, 

    test_size=0.15, 

    random_state=SEED

)
def create_datagen():

    return ImageDataGenerator(

        zoom_range=0.15,

        fill_mode='constant',

        cval=0.,

        horizontal_flip=True,

        vertical_flip=True,

    )
class Metrics(Callback):

    def on_train_begin(self, logs={}):

        self.val_kappas = []



    def on_epoch_end(self, epoch, logs={}):

        X_val, y_val = self.validation_data[:2]

        y_val = y_val.sum(axis=1) - 1

        

        y_pred = self.model.predict(X_val) > 0.5

        y_pred = y_pred.astype(int).sum(axis=1) - 1



        _val_kappa = cohen_kappa_score(

            y_val,

            y_pred, 

            weights='quadratic'

        )



        self.val_kappas.append(_val_kappa)



        print(f"val_kappa: {_val_kappa:.4f}")

        

        if _val_kappa == max(self.val_kappas):

            print("Validation Kappa has improved. Saving model.")

            self.model.save('model.h5')



        return
resnet_backbone = ResNet152(

    weights='imagenet',

    include_top=False,

    input_shape=(224,224,3)

)



inception_backbone = InceptionV3(

    weights='imagenet',

    include_top=False,

    input_shape=(224,224,3)

)



kappa_metrics = Metrics()
def build_model(backbone=inception_backbone, lr=0.00005):

    model = Sequential()

    model.add(backbone)

    model.add(layers.GlobalAveragePooling2D())

    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(5, activation='sigmoid'))

    

    model.compile(

        loss='binary_crossentropy',

        optimizer=Adam(lr=lr),

        metrics=['accuracy']

    )

    

    return model
def train_model(model, epochs):

    history = model.fit_generator(

        data_generator,

        steps_per_epoch=x_train.shape[0] / BATCH_SIZE,

        epochs=epochs,

        validation_data=(x_val, y_val),

        callbacks=[kappa_metrics]

    )



    return history



def print_output(history):

    history_df = pd.DataFrame(history.history)

    history_df[['loss', 'val_loss']].plot()

    history_df[['accuracy', 'val_accuracy']].plot()



#    plt.plot(kappa_metrics.val_kappas)
print('creating datagenerator')

data_generator = create_datagen().flow(x_train, y_train, batch_size=16, seed=SEED)

print('datagenerator done, building model')

resnet_model = build_model(backbone=resnet_backbone, lr=5e-05)

print('model done, training model')

print(f'ResNet152, {BATCH_SIZE}, {LEARNING_RATE}, sigmoid')

history_resnet = train_model(resnet_model, EPOCHS)

print_output(history_resnet)
print('creating datagenerator')

data_generator = create_datagen().flow(x_train, y_train, batch_size=16, seed=SEED)

print('datagenerator done, building model')

inception_model = build_model(backbone=inception_backbone, lr=5e-05)

print('model done, training model')

print(f'InceptionV3, {BATCH_SIZE}, {LEARNING_RATE}, sigmoid')



history_inception = train_model(inception_model, 12)

print_output(history_inception)
def generate_contingency(c1, c2, x, y):

    y2 = y.sum(axis=1) - 1



    c1_pred = c1.predict(x) > 0.5

    c1_pred = c1_pred.astype(int).sum(axis=1) - 1



    c2_pred = c2.predict(x) > 0.5

    c2_pred = c2_pred.astype(int).sum(axis=1) - 1

    

    c1c2_correct = 0

    c1_correct = 0

    c2_correct = 0

    none_correct = 0



    for i, sample in enumerate(x):

        if ((y2[i] == c1_pred[i]) and (y2[i] == c2_pred[i])):

            c1c2_correct += 1

        elif ((y2[i] == c1_pred[i]) and (y2[i] != c2_pred[i])):

            c1_correct += 1

        elif ((y2[i] != c1_pred[i]) and (y2[i] == c2_pred[i])):

            c2_correct += 1

        elif ((y2[i] != c1_pred[i]) and (y2[i] != c2_pred[i])):

            none_correct += 1

    

    table = [[c1c2_correct, c1_correct],

		 [c2_correct, none_correct]]

    total = c1c2_correct + c1_correct + c2_correct + none_correct

    

    

    print(f'c1c2_correct: {c1c2_correct}, c1_correct: {c1_correct}, c2_correct: {c2_correct}, none_correct: {none_correct}, total: {total}')

    return table
table = generate_contingency(resnet_model, inception_model, x_val, y_val)
# Example of calculating the mcnemar test

from statsmodels.stats.contingency_tables import mcnemar



# calculate mcnemar test

result = mcnemar(table, exact=True)

# summarize the finding

print('statistic=%.3f, p-value=%.3f' % (result.statistic, result.pvalue))

# interpret the p-value

alpha = 0.05

if result.pvalue > alpha:

	print('Same proportions of errors (fail to reject H0)')

else:

	print('Different proportions of errors (reject H0)')