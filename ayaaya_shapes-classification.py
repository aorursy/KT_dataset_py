!pip install efficientnet
import pandas as pd

import numpy as np



import os

import cv2



from tqdm import tqdm_notebook as tqdm

import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split

from keras_preprocessing.image import ImageDataGenerator



from tensorflow.keras.models import Model

from tensorflow.keras.layers import Dense

import efficientnet.tfkeras as efn



import warnings

warnings.filterwarnings("ignore")
current_path = r'../input/four-shapes/shapes/'

circle_paths = os.listdir(os.path.join(current_path, 'circle'))

triangle_paths = os.listdir(os.path.join(current_path, 'triangle'))
print(f'We got {len(circle_paths)} circles and {len(triangle_paths)} triangles')
circles = pd.DataFrame()

triangles = pd.DataFrame()



for n,i in enumerate(tqdm(range(len(circle_paths)))):

    circle_path = os.path.join(current_path, 'circle' ,circle_paths[i])

    circles.loc[n,'path'] = circle_path

    circles.loc[n, 'circle'] = 1

    circles.loc[n, 'triangle'] = 0

    

for n,i in enumerate(tqdm(range(len(triangle_paths)))):

    triangle_path = os.path.join(current_path, 'triangle' ,triangle_paths[i])

    triangles.loc[n,'path'] = triangle_path

    triangles.loc[n, 'circle'] = 0

    triangles.loc[n, 'triangle'] = 1

    

data = pd.concat([circles, triangles], axis=0).sample(frac=1.0, random_state=42).reset_index(drop=True)
plt.figure(figsize=(16,16))



for i in range(36):

    plt.subplot(6,6,i+1)

    img = cv2.imread(data.path[i])

    plt.imshow(img)

    plt.title(data.iloc[i,1:].sort_values().index[1])

    plt.axis('off')
train, test = train_test_split(data, test_size=.3, random_state=42)



train.shape, test.shape
example = train.sample(n=1).reset_index(drop=True)

example_data_gen = ImageDataGenerator(

    rescale=1./255,

    horizontal_flip=True,

    vertical_flip=True,

)



example_gen = example_data_gen.flow_from_dataframe(example,

                                                  target_size=(200,200),

                                                  x_col="path",

                                                  y_col=['circle','triangle'],

                                                  class_mode='raw',

                                                  shuffle=False,

                                                  batch_size=32)



plt.figure(figsize=(20, 20))

for i in range(0, 9):

    plt.subplot(3, 3, i+1)

    for X_batch, _ in example_gen:

        image = X_batch[0]

        plt.imshow(image)

        plt.axis('off')

        break
test_data_gen= ImageDataGenerator(rescale=1./255)



train_data_gen= ImageDataGenerator(

    rescale=1./255,

    horizontal_flip=True,

    vertical_flip=True,

)
train_generator=train_data_gen.flow_from_dataframe(train,

                                                      target_size=(200,200),

                                                      x_col="path",

                                                      y_col=['circle','triangle'],

                                                      class_mode='raw',

                                                      shuffle=False,

                                                      batch_size=32)
test_generator=test_data_gen.flow_from_dataframe(test,

                                                  target_size=(200,200),

                                                  x_col="path",

                                                  y_col=['circle','triangle'],

                                                  class_mode='raw',

                                                  shuffle=False,

                                                  batch_size=1)
def get_model():

    base_model = efn.EfficientNetB0(weights='imagenet', include_top=False, pooling='avg', input_shape=(200, 200, 3))

    x = base_model.output

    predictions = Dense(2, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])

    return model
model = get_model()

model.fit_generator(train_generator,

                    epochs=1,

                    steps_per_epoch=train_generator.n/32,

                    )
model.evaluate(test_generator)
pred_test = np.argmax(model.predict(test_generator, verbose=1), axis=1)
plt.figure(figsize=(24,24))



for i in range(100):

    plt.subplot(10,10,i+1)

    img = cv2.imread(test.reset_index(drop=True).path[i])

    plt.imshow(img)

    plt.title(test.reset_index(drop=True).iloc[0,1:].index[pred_test[i]])

    plt.axis('off')