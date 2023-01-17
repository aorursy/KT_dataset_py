import pandas as pd

import numpy as np

import sys

import os as os

import random

from pathlib import Path

import imageio

import skimage

import skimage.io

import skimage.transform

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.graph_objs as go

import plotly.figure_factory as ff

from plotly import tools

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)

import scipy

from sklearn.model_selection import train_test_split

from sklearn import metrics

from keras import optimizers

from keras.models import Sequential

from keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout, BatchNormalization,LeakyReLU

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler

from keras.utils import to_categorical

import tensorflow
IMAGE_PATH = '../input/honey-bee-annotated-images/bee_imgs/bee_imgs/'

IMAGE_WIDTH = 100

IMAGE_HEIGHT = 100

IMAGE_CHANNELS = 3

RANDOM_STATE = 2018

TEST_SIZE = 0.2

VAL_SIZE = 0.2

CONV_2D_DIM_1 = 16

CONV_2D_DIM_2 = 16

CONV_2D_DIM_3 = 32

CONV_2D_DIM_4 = 64

MAX_POOL_DIM = 2

KERNEL_SIZE = 3

BATCH_SIZE = 32

NO_EPOCHS_1 = 5

NO_EPOCHS_2 = 15

NO_EPOCHS_3 = 50

PATIENCE = 5

VERBOSE = 1
os.listdir("../input/honey-bee-annotated-images/")

import pandas as pd

bee_data = pd.read_csv("../input/honey-bee-annotated-images/bee_data.csv")
bee_data.shape
bee_data.sample(100).head()
def missing_data(data):

    total = data.isnull().sum().sort_values(ascending = False)

    percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)

    return pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data(bee_data)
image_files = list(os.listdir(IMAGE_PATH))

print("Number of image files: {}".format(len(image_files)))
file_names = list(bee_data['file'])

print("Matching image names: {}".format(len(set(file_names).intersection(image_files))))
def read_image_sizes(file_name):

    image = skimage.io.imread(IMAGE_PATH + file_name)

    return list(image.shape)
m = np.stack(bee_data['file'].apply(read_image_sizes))

df = pd.DataFrame(m,columns=['w','h','c'])

bee_data = pd.concat([bee_data,df],axis=1, sort=False)
traceW = go.Box(

    x = bee_data['w'],

    name="Width",

     marker=dict(

                color='rgba(238,23,11,0.5)',

                line=dict(

                    color='red',

                    width=1.2),

            ),

    orientation='h')

traceH = go.Box(

    x = bee_data['h'],

    name="Height",

    marker=dict(

                color='rgba(11,23,245,0.5)',

                line=dict(

                    color='blue',

                    width=1.2),

            ),

    orientation='h')

data = [traceW, traceH]

layout = dict(title = 'Width & Heights of images',

          xaxis = dict(title = 'Size', showticklabels=True), 

          yaxis = dict(title = 'Image dimmension'),

          hovermode = 'closest',

         )

fig = dict(data=data, layout=layout)

iplot(fig, filename='width-height')

tmp = bee_data.groupby(['zip code'])['location'].value_counts()

df = pd.DataFrame(data={'Images': tmp.values}, index=tmp.index).reset_index()

df
bee_data = bee_data.replace({'location':'Athens, Georgia, USA'}, 'Athens, GA, USA')
tmp = bee_data.groupby(['zip code'])['location'].value_counts()

df = pd.DataFrame(data={'Images': tmp.values}, index=tmp.index).reset_index()

df['code'] = df['location'].map(lambda x: x.split(',', 2)[1])

df
trace = go.Bar(

        x = df['location'],

        y = df['Images'],

        marker=dict(color="Tomato"),

        text=df['location']

    )

data = [trace]

    

layout = dict(title = 'Number of bees images per location',

          xaxis = dict(title = 'Subspecies', showticklabels=True, tickangle=15), 

          yaxis = dict(title = 'Number of images'),

          hovermode = 'closest'

         )

fig = dict(data = data, layout = layout)

iplot(fig, filename='images-location')

#list of locations

locations = (bee_data.groupby(['location'])['location'].nunique()).index



def draw_category_images(var,cols=5):

    categories = (bee_data.groupby([var])[var].nunique()).index

    f, ax = plt.subplots(nrows=len(categories),ncols=cols, figsize=(2*cols,2*len(categories)))

    # draw a number of images for each location

    for i, cat in enumerate(categories):

        sample = bee_data[bee_data[var]==cat].sample(cols)

        for j in range(0,cols):

            file=IMAGE_PATH + sample.iloc[j]['file']

            im=imageio.imread(file)

            ax[i, j].imshow(im, resample=True)

            ax[i, j].set_title(cat, fontsize=9)  

    plt.tight_layout()

    plt.show()

    

draw_category_images("location")
bee_data['date_time'] = pd.to_datetime(bee_data['date'] + ' ' + bee_data['time'])

bee_data["year"] = bee_data['date_time'].dt.year

bee_data["month"] = bee_data['date_time'].dt.month

bee_data["day"] = bee_data['date_time'].dt.day

bee_data["hour"] = bee_data['date_time'].dt.hour

bee_data["minute"] = bee_data['date_time'].dt.minute



tmp = bee_data.groupby(['date_time', 'hour'])['location'].value_counts()

df = pd.DataFrame(data={'Images': tmp.values}, index=tmp.index).reset_index()

hover_text = []

for index, row in df.iterrows():

    hover_text.append(('Date/time: {}<br>'+

                      'Hour: {}<br>'+

                      'Location: {}<br>'+

                      'Images: {}').format(row['date_time'],

                                            row['hour'],

                                            row['location'],

                                            row['Images']))

df['hover_text'] = hover_text

locations = (bee_data.groupby(['location'])['location'].nunique()).index

data = []

for location in locations:

    dfL = df[df['location']==location]

    trace = go.Scatter(

        x = dfL['date_time'],y = dfL['hour'],

        name=location,

        marker=dict(

            symbol='circle',

            sizemode='area',

            sizeref=0.2,

            size=dfL['Images'],

            line=dict(

                width=2

            ),),

        mode = "markers",

        text=dfL['hover_text'],

    )

    data.append(trace)

    

layout = dict(title = 'Number of bees images per date, approx. hour and location',

          xaxis = dict(title = 'Date', showticklabels=True), 

          yaxis = dict(title = 'Hour'),

          hovermode = 'closest'

         )

fig = dict(data = data, layout = layout)



iplot(fig, filename='images-date_time')



draw_category_images("hour")
tmp = bee_data.groupby(['subspecies'])['year'].value_counts()

df = pd.DataFrame(data={'Images': tmp.values}, index=tmp.index).reset_index()

df
trace = go.Bar(

        x = df['subspecies'],

        y = df['Images'],

        marker=dict(color="Green"),

        text=df['subspecies']

    )

data = [trace]

    

layout = dict(title = 'Number of bees images per subspecies',

          xaxis = dict(title = 'Subspecies', showticklabels=True, tickangle=15), 

          yaxis = dict(title = 'Number of images'),

          hovermode = 'closest'

         )

fig = dict(data = data, layout = layout)

iplot(fig, filename='images-subspecies')
draw_category_images("subspecies")
tmp = bee_data.groupby(['subspecies'])['location'].value_counts()

df = pd.DataFrame(data={'Images': tmp.values}, index=tmp.index).reset_index()



piv = pd.pivot_table(df, values="Images",index=["subspecies"], columns=["location"], fill_value=0)

m = piv.values



trace = go.Heatmap(z = m, y= list(piv.index), x=list(piv.columns),colorscale='Rainbow',reversescale=False)

    

data=[trace]

layout = dict(title = "Number of images per subspecies and location",

              xaxis = dict(title = 'Location',

                        showticklabels=True,

                           tickangle = 45,

                        tickfont=dict(

                                size=10,

                                color='black'),

                          ),

              yaxis = dict(title = 'Subspecies', 

                        showticklabels=True, 

                           tickangle = 45,

                        tickfont=dict(

                            size=10,

                            color='black'),

                      ), 

              hovermode = 'closest',

              showlegend=False,

                  width=600,

                  height=600,

             )

fig = dict(data = data, layout = layout)

iplot(fig, filename='images-location_subspecies')



tmp = bee_data.groupby(['subspecies'])['hour'].value_counts()

df = pd.DataFrame(data={'Images': tmp.values}, index=tmp.index).reset_index()

piv = pd.pivot_table(df, values="Images",index=["subspecies"], columns=["hour"], fill_value=0)

m = piv.values

trace = go.Heatmap(z = m, y= list(piv.index), x=list(piv.columns),colorscale='Rainbow',reversescale=False)

    

data=[trace]

layout = dict(title = "Number of images per subspecies and hour",

              xaxis = dict(title = 'Hour',

                        showticklabels=True,

                           tickangle = 0,

                        tickfont=dict(

                                size=10,

                                color='black'),

                          ),

              yaxis = dict(title = 'Subspecies', 

                        showticklabels=True, 

                           tickangle = 45,

                        tickfont=dict(

                            size=10,

                            color='black'),

                      ), 

              hovermode = 'closest',

              showlegend=False,

                  width=600,

                  height=600,

             )

fig = dict(data = data, layout = layout)

iplot(fig, filename='images-location_subspecies')

def draw_trace_box(dataset,var, subspecies):

    dfS = dataset[dataset['subspecies']==subspecies];

    trace = go.Box(

        x = dfS[var],

        name=subspecies,

        marker=dict(

                    line=dict(

                        color='black',

                        width=0.8),

                ),

        text=dfS['subspecies'], 

        orientation = 'h'

    )

    return trace



subspecies = (bee_data.groupby(['subspecies'])['subspecies'].nunique()).index

def draw_group(dataset, var, title,height=500):

    data = list()

    for subs in subspecies:

        data.append(draw_trace_box(dataset, var, subs))

        

    layout = dict(title = title,

              xaxis = dict(title = 'Size',showticklabels=True),

              yaxis = dict(title = 'Subspecies', showticklabels=True, tickfont=dict(

                family='Old Standard TT, serif',

                size=8,

                color='black'),), 

              hovermode = 'closest',

              showlegend=False,

                  width=600,

                  height=height,

             )

    fig = dict(data=data, layout=layout)

    iplot(fig, filename='subspecies-image')





draw_group(bee_data, 'w', "Width of images per subspecies")

draw_group(bee_data, 'h', "Height of images per subspecies")

def draw_trace_scatter(dataset, subspecies):

    dfS = dataset[dataset['subspecies']==subspecies];

    trace = go.Scatter(

        x = dfS['w'],y = dfS['h'],

        name=subspecies,

        mode = "markers",

        marker = dict(opacity=0.8),

        text=dfS['subspecies'], 

    )

    return trace



subspecies = (bee_data.groupby(['subspecies'])['subspecies'].nunique()).index

def draw_group(dataset, title,height=600):

    data = list()

    for subs in subspecies:

        data.append(draw_trace_scatter(dataset, subs))

        

    layout = dict(title = title,

              xaxis = dict(title = 'Width',showticklabels=True),

              yaxis = dict(title = 'Height', showticklabels=True, tickfont=dict(

                family='Old Standard TT, serif',

                size=8,

                color='black'),), 

              hovermode = 'closest',

              showlegend=True,

                  width=800,

                  height=height,

             )

    fig = dict(data=data, layout=layout)

    iplot(fig, filename='subspecies-image')



draw_group(bee_data,  "Width and height of images per subspecies")

tmp = bee_data.groupby(['health'])['year'].value_counts()

df = pd.DataFrame(data={'Images': tmp.values}, index=tmp.index).reset_index()

df
trace = go.Bar(

        x = df['health'],

        y = df['Images'],

        marker=dict(color="Red"),

        text=df['health']

    )

data = [trace]

    

layout = dict(title = 'Number of bees images per health',

          xaxis = dict(title = 'Health', showticklabels=True, tickangle=15), 

          yaxis = dict(title = 'Number of images'),

          hovermode = 'closest'

         )

fig = dict(data = data, layout = layout)

iplot(fig, filename='images-health')

tmp = bee_data.groupby(['subspecies'])['health'].value_counts()

df = pd.DataFrame(data={'Images': tmp.values}, index=tmp.index).reset_index()

df
piv = pd.pivot_table(df, values="Images",index=["subspecies"], columns=["health"], fill_value=0)

m = piv.values

trace = go.Heatmap(z = m, y= list(piv.index), x=list(piv.columns),colorscale='Rainbow',reversescale=False)

    

data=[trace]

layout = dict(title = "Number of images per subspecies and health",

              xaxis = dict(title = 'Subspecies',

                        showticklabels=True,

                           tickangle = 45,

                        tickfont=dict(

                                size=10,

                                color='black'),

                          ),

              yaxis = dict(title = 'Health', 

                        showticklabels=True, 

                           tickangle = 45,

                        tickfont=dict(

                            size=10,

                            color='black'),

                      ), 

              hovermode = 'closest',

              showlegend=False,

                  width=600,

                  height=600,

             )

fig = dict(data = data, layout = layout)

iplot(fig, filename='images-health_subspecies')

tmp = bee_data.groupby(['health', 'location'])['subspecies'].value_counts()

df = pd.DataFrame(data={'Images': tmp.values}, index=tmp.index).reset_index()

hover_text = []

for index, row in df.iterrows():

    hover_text.append(('Subspecies: {}<br>'+

                      'Health: {}<br>'+

                      'Location: {}<br>'+

                      'Images: {}').format(row['subspecies'],

                                            row['health'],

                                            row['location'],

                                            row['Images']))

df['hover_text'] = hover_text

subspecies = (bee_data.groupby(['subspecies'])['subspecies'].nunique()).index

data = []

for subs in subspecies:

    dfL = df[df['subspecies']==subs]

    trace = go.Scatter(

        x = dfL['location'],y = dfL['health'],

        name=subs,

        marker=dict(

            symbol='circle',

            sizemode='area',

            sizeref=0.2,

            size=dfL['Images'],

            line=dict(

                width=2

            ),),

        mode = "markers",

        text=dfL['hover_text'],

    )

    data.append(trace)

    

layout = dict(title = 'Number of bees images per location, health and subspecies',

          xaxis = dict(title = 'Location', showticklabels=True), 

          yaxis = dict(title = 'Health', tickangle=45),

          hovermode = 'closest'

         )

fig = dict(data = data, layout = layout)

iplot(fig, filename='images-subspecies-health-location')

#plot image for each health catagory

draw_category_images("health")
tmp = bee_data.groupby(['pollen_carrying'])['year'].value_counts()

df = pd.DataFrame(data={'Images': tmp.values}, index=tmp.index).reset_index()

df
tmp = bee_data.groupby(['pollen_carrying'])['subspecies'].value_counts()

df = pd.DataFrame(data={'Images': tmp.values}, index=tmp.index).reset_index()

df[df['pollen_carrying']==True]
draw_category_images("pollen_carrying")
train_df, test_df = train_test_split(bee_data, test_size=TEST_SIZE, random_state=RANDOM_STATE)



train_df, val_df = train_test_split(train_df, test_size=VAL_SIZE, random_state=RANDOM_STATE)



print("Train set rows: {}".format(train_df.shape[0]))

print("Test  set rows: {}".format(test_df.shape[0]))

print("Val   set rows: {}".format(val_df.shape[0]))
#A function for reading images from the image files, scale all images to 100 x 100 x 3 (channels).



def read_image(file_name):

    image = skimage.io.imread(IMAGE_PATH + file_name)

    image = skimage.transform.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), mode='reflect')

    return image[:,:,:IMAGE_CHANNELS]





#A function to create the dummy variables corresponding to the categorical target variable.



def categories_encoder(dataset, var='subspecies'):

    X = np.stack(dataset['file'].apply(read_image))

    y = pd.get_dummies(dataset[var], drop_first=False)

    return X, y



X_train, y_train = categories_encoder(train_df)

X_val, y_val = categories_encoder(val_df)

X_test, y_test = categories_encoder(test_df)
model1=Sequential()

model1.add(Conv2D(CONV_2D_DIM_1, kernel_size=KERNEL_SIZE, input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT,IMAGE_CHANNELS), activation='relu', padding='same'))

model1.add(MaxPool2D(MAX_POOL_DIM))

model1.add(Conv2D(CONV_2D_DIM_2, kernel_size=KERNEL_SIZE, activation='relu', padding='same'))

model1.add(Flatten())

model1.add(Dense(y_train.columns.size, activation='softmax'))

model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



model1.summary()
image_generator = ImageDataGenerator(

        featurewise_center=False,

        samplewise_center=False,

        featurewise_std_normalization=False,

        samplewise_std_normalization=False,

        zca_whitening=False,

        rotation_range=180,

        zoom_range = 0.1, 

        width_shift_range=0.1,

        height_shift_range=0.1, 

        horizontal_flip=True,

        vertical_flip=True)

image_generator.fit(X_train)
train_model1  = model1.fit_generator(image_generator.flow(X_train, y_train, batch_size=BATCH_SIZE),

                        epochs=NO_EPOCHS_1,

                        validation_data=[X_val, y_val],

                        steps_per_epoch=len(X_train)/BATCH_SIZE)
score = model1.evaluate(X_test, y_test, verbose=0)

print('Test loss:', score[0])

print('Test accuracy:', score[1])
def test_accuracy_report(model):

    predicted = model.predict(X_test)

    test_predicted = np.argmax(predicted, axis=1)

    test_truth = np.argmax(y_test.values, axis=1)

    print(metrics.classification_report(test_truth, test_predicted, target_names=y_test.columns)) 

    test_res = model.evaluate(X_test, y_test.values, verbose=0)

    print('Loss function: %s, accuracy:' % test_res[0], test_res[1])

test_accuracy_report(model1)
model2=Sequential()

model2.add(Conv2D(CONV_2D_DIM_1, kernel_size=KERNEL_SIZE, input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT,IMAGE_CHANNELS), activation='relu', padding='same'))

model2.add(MaxPool2D(MAX_POOL_DIM))

model2.add(Dropout(0.4))

model2.add(Conv2D(CONV_2D_DIM_2, kernel_size=KERNEL_SIZE, activation='relu', padding='same'))

model2.add(Dropout(0.4))

model2.add(Flatten())

model2.add(Dense(y_train.columns.size, activation='softmax'))

model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



model2.summary()
train_model2  = model2.fit_generator(image_generator.flow(X_train, y_train, batch_size=BATCH_SIZE),

                        epochs=NO_EPOCHS_2,

                        validation_data=[X_val, y_val],

                        steps_per_epoch=len(X_train)/BATCH_SIZE)
test_accuracy_report(model2)