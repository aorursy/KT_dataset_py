import re

import gc

import glob

import keras

import pandas as pd

import numpy  as np



import tensorflow as tf

import plotly.express as px

import matplotlib.pyplot as plt

import plotly.graph_objects as go

from plotly.subplots import make_subplots

from sklearn.metrics import roc_curve, auc, multilabel_confusion_matrix





import keras.backend as K

from keras.models     import Sequential

from keras.layers     import Dense, Dropout, GlobalMaxPooling2D

from keras.optimizers import Adam, SGD

from keras.applications import MobileNetV2

from keras.callbacks    import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, CSVLogger

from sklearn.model_selection   import train_test_split

from keras.preprocessing.image import ImageDataGenerator



np.random.seed(451)
# General parameters

batch_size = 16

image_size = 224

epochs     = 50
# Read and prepare data

raw_data = pd.read_csv('../input/japanese-female-facial-expression-dataset-jaffe/data.csv')

raw_data['filepath'] = '../input/japanese-female-facial-expression-dataset-jaffe/' + raw_data['filepath']

raw_data.fillna('UNKNOWN', inplace=True)

raw_data.sample(3)
label_count_df = pd.DataFrame(raw_data['facial_expression'].value_counts()).reset_index()
fig = px.bar(label_count_df,

             y='index',

             x='facial_expression',

             orientation='h',

             color='index',

             title='Label Distribution',

             opacity=0.8,

             color_discrete_sequence=px.colors.diverging.curl,

             template='plotly_dark'

            )

fig.update_xaxes(range=[0,35])

fig.show()
def plot_samples(df, label_list):

    for label in label_list:

        query_string = "facial_expression == '{}'".format(label)

        df_label = df.query(query_string).reset_index(drop=True)

        

        fig = plt.figure(figsize=(18,15))

        plt.subplot(1,4,1)

        plt.imshow(plt.imread(df_label.loc[0,'filepath']),cmap='gray')

        plt.title(label.capitalize())

        

        plt.subplot(1,4,2)

        plt.imshow(plt.imread(df_label.loc[1,'filepath']),cmap='gray')

        plt.title(label.capitalize())

        

        plt.subplot(1,4,3)

        plt.imshow(plt.imread(df_label.loc[2,'filepath']),cmap='gray')

        plt.title(label.capitalize())

        

        plt.subplot(1,4,4)

        plt.imshow(plt.imread(df_label.loc[3,'filepath']),cmap='gray')

        plt.title(label.capitalize())

        

        plt.show()
plot_samples(raw_data, ['happiness', 'surprise', 'neutral', 'disgust', 'angry', 'fear'])
# Create train and testing sets

train, test = train_test_split(raw_data,

                               test_size = 0.3,

                               stratify=raw_data['facial_expression'],

                               random_state=451

                              )
train_generator = ImageDataGenerator(

                    rescale     = 1./255,

                    shear_range = 0.1,

                    zoom_range  = 0.1,

                    width_shift_range  = 0.1,

                    height_shift_range = 0.1,

                    horizontal_flip    = True)



test_generator = ImageDataGenerator(rescale=1./255)
train_gen = train_generator.flow_from_dataframe(dataframe = train,

                                    class_mode  = 'categorical',

                                    x_col       = 'filepath',

                                    y_col       = 'facial_expression',

                                    shuffle     = True,

                                    batch_size  = batch_size,

                                    target_size = (image_size, image_size),

                                    seed=451)









test_gen  = test_generator.flow_from_dataframe(dataframe = test,

                                    class_mode='categorical',

                                    x_col='filepath',

                                    y_col='facial_expression',

                                    shuffle     = False,

                                    batch_size  = batch_size,

                                    target_size = (image_size, image_size),

                                    seed=451)

# Create and compile model

model = Sequential()

model.add(MobileNetV2(input_shape=(image_size, image_size, 3), weights='imagenet', include_top=False))

model.add(GlobalMaxPooling2D())

model.add(Dropout(0.5))

model.add(Dense(7,activation='softmax'))

model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

model.summary()
training_hist = model.fit(train_gen,

                          epochs = epochs,

                            )
train_df = pd.DataFrame(training_hist.history).reset_index()
fig = px.area(train_df,

            x='index',

            y='loss',

            template='plotly_dark',

            color_discrete_sequence=['rgb(18, 115, 117)'],

            title='Training Loss x Epoch Number',

           )



fig.update_yaxes(range=[0,2])

fig.show()
results = model.evaluate_generator(test_gen)

preds   = model.predict_generator(test_gen)

print('The current model achieved a categorical accuracy of {}%!'.format(round(results[1]*100,2)))
summarized_confusion_matrix = np.sum(multilabel_confusion_matrix(pd.get_dummies(test['facial_expression']), preds >= 0.5),axis=0)
fig = px.imshow(summarized_confusion_matrix,

                template ='plotly_dark',

                color_continuous_scale = px.colors.sequential.Blugrn

                )





fig.update_layout(title_text='Confusion Matrix', title_x=0.5)

fig.show()