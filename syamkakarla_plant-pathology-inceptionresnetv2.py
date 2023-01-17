# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('whitegrid')


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import regularizers, optimizers

import os

!ls ../input/plant-pathology-2020-fgvc7
train_df = pd.read_csv('../input/plant-pathology-2020-fgvc7/train.csv')
test_df =  pd.read_csv('../input/plant-pathology-2020-fgvc7/test.csv')
train_df.shape, test_df.shape
train_df.head()
test_df.head()
train_df.image_id = train_df.image_id.apply(lambda x: x+'.jpg')
test_df.image_id = test_df.image_id.apply(lambda x: x+'.jpg')
print("Train Images:{}\nTest Images: {}".format(train_df.image_id.shape, test_df.image_id.shape))
train_df.head()
test_df.head()
datagen=ImageDataGenerator( horizontal_flip=True,
                            vertical_flip=True,
                            rotation_range=10,
                            width_shift_range=0.1,
                            height_shift_range=0.1,
                            zoom_range=.1,
                            fill_mode='nearest',
                            shear_range=0.1,
                            rescale=1/255,
                            brightness_range=[0.5, 1.5]
                          )

train_generator=datagen.flow_from_dataframe(
dataframe=train_df,
directory="../input/plant-pathology-2020-fgvc7/images/",
x_col="image_id",
y_col= ['healthy', 'multiple_diseases', 'rust', 'scab'],
batch_size=16,
seed=42,
shuffle=True,
class_mode="other",
target_size=(299, 299))
test_datagen=ImageDataGenerator(rescale=1./255.)

test_generator=test_datagen.flow_from_dataframe(
dataframe=test_df,
directory="../input/plant-pathology-2020-fgvc7/images/",
x_col="image_id",
y_col=None,
batch_size=16,
seed=42,
shuffle=False,
class_mode=None,
target_size=(299, 299))
img, lab = next(train_generator)
img.shape
fig = plt.figure(figsize = (10,10))
for i in range(1, 1+9):
    fig.add_subplot(3,3,i)
    plt.imshow(img[i].reshape(img_rows, img_cols, img_chn))
    plt.axis('off')
    plt.title(','.join([i for i,j in zip(train_df.columns[1:].tolist(), lab[i].tolist()) if j == 1]))
# Building model

from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
net = InceptionResNetV2(weights= 'imagenet', include_top=False, input_shape= (img_rows,img_cols,img_chn))
x = net.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(64, activation="relu")(x)
x = Dense(32, activation="relu")(x)
predictions = Dense(4, activation= 'softmax')(x)
model = Model(inputs = net.input, outputs = predictions)
model.summary()
model_check = ModelCheckpoint('best_model.h5', monitor='accuracy', verbose=0, save_best_only=True, mode='max')

early = EarlyStopping(monitor='accuracy', min_delta=0, patience=10, verbose=0, mode='max', restore_best_weights=True)

reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=10, min_lr=0.001)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
n_epochs = 60
history =  model.fit(train_generator, steps_per_epoch=50, epochs = n_epochs, verbose = 1, 
                         callbacks = [model_check, early, reduce_lr])
hist_df = pd.DataFrame(data = history.history)
hist_df.to_csv('train_log.csv')
hist_df.head()
loss, acc = model.evaluate_generator(train_generator, verbose=1)
print('Accuracy: ', acc, '\nLoss    : ', loss)
import plotly.express as px
import plotly.graph_objects as go
fig = go.Figure()
ind = np.arange(1, len(history.history['accuracy'])+1)
fig.add_trace(go.Scatter(x=ind, mode='lines+markers', y=hist_df['accuracy'], marker=dict(color="dodgerblue"), name="Accyracy"))
    
fig.add_trace(go.Scatter(x=ind, mode='lines+markers', y=hist_df['loss'], marker=dict(color="darkorange"),name="Loss"))
    
fig.update_layout(title_text='Accuracy and Loss', yaxis_title='Value', xaxis_title="Epochs", template="plotly_white")

fig.show()
%%time
pred = model.predict(test_generator, verbose = 1)

SUB_PATH = "../input/plant-pathology-2020-fgvc7/sample_submission.csv"

sub = pd.read_csv(SUB_PATH)
sub.loc[:, 'healthy':] = pred
sub.to_csv('Submission_1.csv', index=False)
sub.head()