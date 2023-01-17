# Importing required libraries



import numpy as np

import pandas as pd

np.random.seed(1)



import os

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.offline as pyo

import plotly.graph_objs as go

import plotly.express as px

pyo.init_notebook_mode(connected = True)

from plotly.subplots import make_subplots



import cv2



import keras

from keras.preprocessing import image

from keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf

tf.random.set_seed(1)

from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten

from keras.models import Sequential

from keras.optimizers import RMSprop, Adam

from keras.callbacks import EarlyStopping, ModelCheckpoint



from keras.applications import VGG16

from keras.models import load_model

os.getcwd()
os.listdir('/kaggle/input/intel-image-classification')
os.listdir('/kaggle/input/intel-image-classification/seg_train/seg_train')
train = '/kaggle/input/intel-image-classification/seg_train/seg_train'

val = '/kaggle/input/intel-image-classification/seg_test/seg_test'

test = '/kaggle/input/intel-image-classification/seg_pred'
train_datgen = ImageDataGenerator(rescale=1./255,

                                     rotation_range=30,

                                     width_shift_range=0.2,

                                     height_shift_range=0.2,

                                     shear_range=0.2,

                                     zoom_range=0.2,

                                     horizontal_flip=True,

                                     fill_mode='nearest')



train_generator = train_datgen.flow_from_directory(train,target_size=(150, 150),batch_size=50,

                                                                         classes=['sea', 'forest', 'mountain', 'glacier', 'buildings', 'street'],

                                                                        class_mode='categorical')





val_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(val,target_size=(150, 150),batch_size=30,

                                                                         classes=['sea', 'forest', 'mountain', 'glacier', 'buildings', 'street'],

                                                                        class_mode='categorical')

n=1



fig,ax = plt.subplots(6,6,figsize=(15,15))

i = 0

for path in os.listdir(train):

    img_path = os.path.join(train,path,os.listdir(os.path.join(train,path))[n])

    img = image.load_img(img_path,target_size=(150,150))

    img = image.img_to_array(img)

    #flow requires (batch,width,height,channel), our current image shape is (width,height,channel) so we need to add batch

    img = img.reshape((1,)+img.shape)

    # alternatively this can be done with np.expand_dims()

    j=0

    for batch in train_datgen.flow(img,batch_size=1):

        sub = ax[i,j]

        sub.imshow(image.array_to_img(batch[0]))

        sub.axis('off')

        j+=1

        if j%6 == 0:

            break

    i+=1
data = {'Train':dict([(path,len(os.listdir(os.path.join(train,path)))) for path in os.listdir(train)]),

       'Validation':dict([(path,len(os.listdir(os.path.join(val,path)))) for path in os.listdir(val)])}

data = pd.DataFrame(data)

data
fig, (ax1,ax2) = plt.subplots(1,2,figsize=(20,6))

sns.barplot(x = 'Train', y = 'index', data = data.reset_index(),ax=ax1)

sns.barplot(x = 'Validation', y = 'index', data = data.reset_index(),ax=ax2)
data = data.reset_index()

data.columns = ['Category','Train_value','Validation_Value']

labels = data['Category']

train = data['Train_value']

val = data['Validation_Value']



fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]],subplot_titles=['Train Ratio', 'Validation Ratio'])

fig.add_trace(go.Pie(labels=labels, values=train, name="Train"),

              1, 1)

fig.add_trace(go.Pie(labels=labels, values=val, name="Validation"),

              1, 2)

fig.update_traces(textposition='inside', textinfo='percent+label',hole=.4, hoverinfo="label+percent+name")

fig.show()
model = Sequential()

model.add(Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3),padding='same'))

model.add(MaxPooling2D(pool_size=(2, 2),strides=2))

model.add(Conv2D(64,(3,3),activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2),strides=2))

model.add(Conv2D(128,(3,3),activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2),strides=2))

model.add(Conv2D(256,(3,3),activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2),strides=2))

model.add(Flatten())

model.add(Dense(768,activation='relu'))

model.add(Dropout(0.3))

model.add(Dense(64,activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(6,activation='softmax'))



model.summary()
early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='auto')

optimizer = RMSprop(learning_rate=0.0001)



model.compile(optimizer,loss='categorical_crossentropy',metrics=['acc'])
history = model.fit_generator(train_generator,steps_per_epoch=560,epochs=100,callbacks=[early_stopping],

                              validation_data=val_generator,validation_steps=100)
def loss_accuracy_plot(history):

    accuracy = history.history['acc']

    loss = history.history['loss']

    val_accuracy = history.history['val_acc']

    val_loss = history.history['val_loss']

    epochs = history.epoch

    

    fig = make_subplots(rows=2, cols=1)

    

    fig.add_trace(go.Scatter(x=epochs, y=loss, name='Train Loss',

                             line=dict(color='royalblue', width=4)),row=1,col=1)

    fig.add_trace(go.Scatter(x=epochs, y=val_loss, name = 'Validation Loss',

                             line=dict(color='firebrick', width=4, dash='dot')),row=1,col=1)

    

        

    fig.add_trace(go.Scatter(x=epochs, y=accuracy, name='Train Accuracy',

                             line=dict(color='royalblue', width=4)),row=2,col=1)

    fig.add_trace(go.Scatter(x=epochs, y=val_accuracy, name = 'Validation Accuracy',

                             line=dict(color='firebrick', width=4, dash='dot')),row=2,col=1)

    

    fig.update_layout(height=800, width=800, title_text="Loss and Accuracy")

    fig.show()
loss_accuracy_plot(history)
conv_base = VGG16(weights='imagenet',

                 include_top=False,

                 input_shape=(150,150,3))
model2 = Sequential()

model2.add(conv_base)

model2.add(Flatten())

model2.add(Dense(256,activation='relu'))

model2.add(Dropout(0.2))

model2.add(Dense(6,activation='softmax'))





early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='auto',verbose=1)

checkpoint1 = ModelCheckpoint('check_1.h5',monitor='val_acc',verbose=1,save_best_only=True,mode='auto')

checkpoint2 = ModelCheckpoint('check_2.h5',monitor='val_loss',verbose=1,save_best_only=True,mode='auto')



optimizer = RMSprop(learning_rate=0.0001)



model2.compile(optimizer,loss='categorical_crossentropy',metrics=['acc'])
history2 = model2.fit_generator(train_generator,steps_per_epoch=560,epochs=100,callbacks=[early_stopping,checkpoint1,checkpoint2],

                              validation_data=val_generator,validation_steps=100)
loss_accuracy_plot(history2)
saved_model = load_model('check_2.h5')
pred_class = dict([(v,k) for (k,v) in train_generator.class_indices.items()])

pred_class
def result_plotter(rows=3,cols=3):

    

    fig, ax = plt.subplots(rows,cols,figsize=((cols*2)+1,(rows*2)+1))



    for i in range(rows):

        for j in range(cols):

            img_path = os.path.join(test,'seg_pred',os.listdir(test+'/seg_pred')[(i*cols)+j])

            img = image.load_img(img_path,target_size=(150,150))

            img = image.img_to_array(img)

            img = img.reshape((1,)+img.shape)

            pred = saved_model.predict(img)

            pred = np.argmax(pred)





            sub = ax[i,j]

            img = plt.imread(img_path)

            cv2.rectangle(img,(0,150),(150,130),thickness=-1, color=(255, 255, 255))

            sub.text(75,140,pred_class[pred],ha='center',va='center',size=12)

            sub.imshow(img)

            sub.axis('off')

#             sub.set_title(pred_class[pred],fontdict={'fontsize':'x-large'})
result_plotter(rows=10,cols=10)