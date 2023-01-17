import os

import pandas as pd

import numpy as np

from keras.models import Sequential, Model

from keras.layers import Dense, Conv2D, Flatten, Activation, Dropout

from keras.preprocessing.image import ImageDataGenerator

from keras import applications, optimizers

import cv2



import matplotlib.pyplot as plt
path = '../input/severstal-steel-defect-detection/'
tr = pd.read_csv(path + 'train.csv')

print(tr.shape)

tr.head()
df = tr[tr['ImageId_ClassId'].apply(lambda x: x.split('_')[1] == '1')].reset_index(drop=True)

df['ImageId_ClassId'] = df['ImageId_ClassId'].apply(lambda x: x.split('_')[0])

df = df.drop('EncodedPixels', axis=1)



print(len(df))

df.head()
imgs_templ = ['000789191.jpg','00d7ae946.jpg','01d590c5f.jpg','01e501f99.jpg','023353d24.jpg',\

              '031614d60.jpg','03395a3da.jpg','063b5dcbe.jpg','06a86ee90.jpg','07cb85a8d.jpg','07e8fca73.jpg',\

              '08e21ba66.jpg','047681252.jpg','092c1f666.jpg','0a3bbea4d.jpg','0a46cc4bf.jpg','0a65bd8d4.jpg',\

              '0a76ac9b8.jpg','0b3a0fabe.jpg','0b50b417a.jpg','0d0c21687.jpg','0d22de6d4.jpg','0e09ff3bd.jpg',\

              '0e3ade070.jpg','0d0c21687.jpg','0d22de6d4.jpg','0ef4bff49.jpg','0faa71251.jpg','0fac62a3e.jpg',\

              '100de36e9.jpg','109fbcecf.jpg','110e63bfa.jpg']

len(imgs_templ)
plt.imshow(plt.imread(path + 'train_images/'+ imgs_templ[31]))
df_trn = pd.concat([df[~df['ImageId_ClassId'].isin(imgs_templ)][:50], pd.DataFrame(imgs_templ, columns=['ImageId_ClassId'])], ignore_index=True)

df_trn['IsTemp'] = '0' 

df_trn['IsTemp'][df_trn['ImageId_ClassId'].isin(imgs_templ)] = '1'



print(df_trn['IsTemp'].value_counts())

print(len(df_trn))

df_trn.head()
img = plt.imread(path + 'train_images/'+ df_trn[0:1]['ImageId_ClassId'].values[0])

plt.imshow(img)
img_size = 256

batch_size = 16
train_datagen=ImageDataGenerator(

    rescale=1./255,

    horizontal_flip = True,

    vertical_flip = True

)



train_generator=train_datagen.flow_from_dataframe(

    dataframe=df_trn,

    directory=path + 'train_images',

    x_col="ImageId_ClassId",

    y_col="IsTemp",

    batch_size=batch_size,

    shuffle=True,

    class_mode="binary",

    target_size=(img_size,img_size)

    )
base_model = applications.VGG16(weights=None, input_shape=(img_size, img_size, 3), include_top=False)

base_model.load_weights('../input/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')

base_model.trainable = False
x = base_model.output

x = Flatten()(x)

x = Dense(1024, activation="relu")(x)

x = Dropout(0.5)(x)

predictions = Dense(1, activation="sigmoid")(x)

model = Model(input = base_model.input, output = predictions)



model.compile(loss='binary_crossentropy', optimizer = optimizers.adam(lr=0.0001), metrics=['accuracy'])
%%time

model.fit_generator(

        train_generator,

        steps_per_epoch=100,

        epochs=2,

        verbose=1)
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_dataframe(  

        dataframe=df,

        directory = path + 'train_images',    

        x_col="ImageId_ClassId",

        target_size = (img_size,img_size),

        batch_size = 1,

        shuffle = False,

        class_mode = None

        )
%%time

test_generator.reset()

predict = model.predict_generator(test_generator, steps = len(test_generator.filenames))

len(predict)
df_p = df.copy(deep=True)

df_p['Pred']=predict.round()

img_tmpl = df_p[df_p['Pred'] == 1]['ImageId_ClassId'].values

print(len(img_tmpl))

img_tmpl
fig=plt.figure(figsize=(20, 12))

columns = 4

rows = 10

for i in range(1, columns*rows +1):   

    img = cv2.imread(path + 'train_images/'+ img_tmpl[i+100])

    fig.add_subplot(rows, columns, i).set_title(img_tmpl[i+100])

    plt.axis('off')

    plt.imshow(img)

plt.show()