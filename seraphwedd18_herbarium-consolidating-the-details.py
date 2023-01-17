import time

start_time = time.time()
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        if filename.endswith('.jpg'):

            break

        print(os.path.join(dirname, filename))
sample_sub = pd.read_csv('../input/herbarium-2020-fgvc7/sample_submission.csv')

display(sample_sub)
import json, codecs

with codecs.open("../input/herbarium-2020-fgvc7/nybg2020/train/metadata.json", 'r',

                 encoding='utf-8', errors='ignore') as f:

    train_meta = json.load(f)

    

with codecs.open("../input/herbarium-2020-fgvc7/nybg2020/test/metadata.json", 'r',

                 encoding='utf-8', errors='ignore') as f:

    test_meta = json.load(f)
display(train_meta.keys())
train_df = pd.DataFrame(train_meta['annotations'])

display(train_df)
train_cat = pd.DataFrame(train_meta['categories'])

train_cat.columns = ['family', 'genus', 'category_id', 'category_name']

display(train_cat)
train_img = pd.DataFrame(train_meta['images'])

train_img.columns = ['file_name', 'height', 'image_id', 'license', 'width']

display(train_img)
train_reg = pd.DataFrame(train_meta['regions'])

train_reg.columns = ['region_id', 'region_name']

display(train_reg)
train_df = train_df.merge(train_cat, on='category_id', how='outer')

train_df = train_df.merge(train_img, on='image_id', how='outer')

train_df = train_df.merge(train_reg, on='region_id', how='outer')
print(train_df.info())

display(train_df)
na = train_df.file_name.isna()

keep = [x for x in range(train_df.shape[0]) if not na[x]]

train_df = train_df.iloc[keep]
dtypes = ['int32', 'int32', 'int32', 'int32', 'object', 'object', 'object', 'object', 'int32', 'int32', 'int32', 'object']

for n, col in enumerate(train_df.columns):

    train_df[col] = train_df[col].astype(dtypes[n])

print(train_df.info())

display(train_df)
test_df = pd.DataFrame(test_meta['images'])

test_df.columns = ['file_name', 'height', 'image_id', 'license', 'width']

print(test_df.info())

display(test_df)
train_df.to_csv('full_train_data.csv', index=False)

test_df.to_csv('full_test_data.csv', index=False)
print("Total Unique Values for each columns:")

print("{0:10s} \t {1:10d}".format('train_df', len(train_df)))

for col in train_df.columns:

    print("{0:10s} \t {1:10d}".format(col, len(train_df[col].unique())))
family = train_df[['family', 'genus', 'category_name']].groupby(['family', 'genus']).count()

display(family.describe())
from tensorflow.keras.models import Model

from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten, BatchNormalization, Input, concatenate

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.utils import plot_model

from sklearn.model_selection import train_test_split as tts



in_out_size = (120*120) + 3 #We will resize the image to 120*120 and we have 3 outputs

def xavier(shape, dtype=None):

    return np.random.rand(*shape)*np.sqrt(1/in_out_size)



def fg_model(shape, lr=0.001):

    '''Family-Genus model receives an image and outputs two integers indicating both the family and genus index.'''

    i = Input(shape)

    

    x = Conv2D(3, (3, 3), activation='relu', padding='same', kernel_initializer=xavier)(i)

    x = Conv2D(3, (5, 5), activation='relu', padding='same', kernel_initializer=xavier)(x)

    x = MaxPool2D(pool_size=(3, 3), strides=(3,3))(x)

    x = BatchNormalization()(x)

    x = Dropout(0.5)(x)

    x = Conv2D(16, (5, 5), activation='relu', padding='same', kernel_initializer=xavier)(x)

    #x = Conv2D(16, (5, 5), activation='relu', padding='same', kernel_initializer=xavier)(x)

    x = MaxPool2D(pool_size=(5, 5), strides=(5,5))(x)

    x = BatchNormalization()(x)

    x = Dropout(0.5)(x)

    x = Flatten()(x)

    

    o1 = Dense(310, activation='softmax', name='family', kernel_initializer=xavier)(x)

    

    o2 = concatenate([o1, x])

    o2 = Dense(3678, activation='softmax', name='genus', kernel_initializer=xavier)(o2)

    

    o3 = concatenate([o1, o2, x])

    o3 = Dense(32094, activation='softmax', name='category_id', kernel_initializer=xavier)(o3)

    

    x = Model(inputs=i, outputs=[o1, o2, o3])

    

    opt = Adam(lr=lr, amsgrad=True)

    x.compile(optimizer=opt, loss=['sparse_categorical_crossentropy', 

                                   'sparse_categorical_crossentropy', 

                                   'sparse_categorical_crossentropy'],

                 metrics=['accuracy'])

    return x



model = fg_model((120, 120, 3))

model.summary()

plot_model(model, to_file='full_model_plot.png', show_shapes=True, show_layer_names=True)
from tensorflow.keras.preprocessing.image import ImageDataGenerator



train_datagen = ImageDataGenerator(featurewise_center=False,

                                     featurewise_std_normalization=False,

                                     rotation_range=180,

                                     width_shift_range=0.1,

                                     height_shift_range=0.1,

                                     zoom_range=0.2)
m = train_df[['file_name', 'family', 'genus', 'category_id']]

fam = m.family.unique().tolist()

m.family = m.family.map(lambda x: fam.index(x))

gen = m.genus.unique().tolist()

m.genus = m.genus.map(lambda x: gen.index(x))

display(m)
train, verif = tts(m, test_size=0.2, shuffle=True, random_state=17)

train = train[:40000]

verif = verif[:10000]

shape = (120, 120, 3)

epochs = 2

batch_size = 32



model = fg_model(shape, 0.007)



#Disable the last two output layers for training the Family

for layers in model.layers:

    if layers.name == 'genus' or layers.name=='category_id':

        layers.trainable = False



#Train Family for 2 epochs

model.fit_generator(train_datagen.flow_from_dataframe(dataframe=train,

                                                      directory='../input/herbarium-2020-fgvc7/nybg2020/train/',

                                                      x_col="file_name",

                                                      y_col=["family", "genus", "category_id"],

                                                      target_size=(120, 120),

                                                      batch_size=batch_size,

                                                      class_mode='multi_output'),

                    validation_data=train_datagen.flow_from_dataframe(

                        dataframe=verif,

                        directory='../input/herbarium-2020-fgvc7/nybg2020/train/',

                        x_col="file_name",

                        y_col=["family", "genus", "category_id"],

                        target_size=(120, 120),

                        batch_size=batch_size,

                        class_mode='multi_output'),

                    epochs=epochs,

                    steps_per_epoch=len(train)//batch_size,

                    validation_steps=len(verif)//batch_size,

                    verbose=1,

                    workers=8,

                    use_multiprocessing=False)



#Reshuffle the inputs

train, verif = tts(m, test_size=0.2, shuffle=True, random_state=17)

train = train[:40000]

verif = verif[:10000]



#Make the Genus layer Trainable

for layers in model.layers:

    if layers.name == 'genus':

        layers.trainable = True

        

#Train Family and Genus for 2 epochs

model.fit_generator(train_datagen.flow_from_dataframe(dataframe=train,

                                                      directory='../input/herbarium-2020-fgvc7/nybg2020/train/',

                                                      x_col="file_name",

                                                      y_col=["family", "genus", "category_id"],

                                                      target_size=(120, 120),

                                                      batch_size=batch_size,

                                                      class_mode='multi_output'),

                    validation_data=train_datagen.flow_from_dataframe(

                        dataframe=verif,

                        directory='../input/herbarium-2020-fgvc7/nybg2020/train/',

                        x_col="file_name",

                        y_col=["family", "genus", "category_id"],

                        target_size=(120, 120),

                        batch_size=batch_size,

                        class_mode='multi_output'),

                    epochs=epochs,

                    steps_per_epoch=len(train)//batch_size,

                    validation_steps=len(verif)//batch_size,

                    verbose=1,

                    workers=8,

                    use_multiprocessing=False)



#Reshuffle the inputs

train, verif = tts(m, test_size=0.2, shuffle=True, random_state=17)

train = train[:40000]

verif = verif[:10000]



#Make the category_id layer Trainable

for layers in model.layers:

    if layers.name == 'category_id':

        layers.trainable = True

        

#Train them all for 2 epochs

model.fit_generator(train_datagen.flow_from_dataframe(dataframe=train,

                                                      directory='../input/herbarium-2020-fgvc7/nybg2020/train/',

                                                      x_col="file_name",

                                                      y_col=["family", "genus", "category_id"],

                                                      target_size=(120, 120),

                                                      batch_size=batch_size,

                                                      class_mode='multi_output'),

                    validation_data=train_datagen.flow_from_dataframe(

                        dataframe=verif,

                        directory='../input/herbarium-2020-fgvc7/nybg2020/train/',

                        x_col="file_name",

                        y_col=["family", "genus", "category_id"],

                        target_size=(120, 120),

                        batch_size=batch_size,

                        class_mode='multi_output'),

                    epochs=epochs,

                    steps_per_epoch=len(train)//batch_size,

                    validation_steps=len(verif)//batch_size,

                    verbose=1,

                    workers=8,

                    use_multiprocessing=False)



'''

for i in range(epochs):

    n = 1

    for X, Y in train_datagen.flow_from_dataframe(dataframe=train,

                                                  directory='../input/herbarium-2020-fgvc7/nybg2020/train/',

                                                  x_col="file_name",

                                                  y_col=["family", "genus", "category_id"],

                                                  target_size=(120, 120),

                                                  batch_size=batch_size,

                                                  class_mode='multi_output'):

        model.train_on_batch(X, Y, reset_metrics=False)

        loss, fam_loss, gen_loss, cat_loss, fam_acc, gen_acc, cat_acc = model.evaluate(X, Y, verbose=False)

        if n%10==0:

            print(f"For epoch {i} batch {n}: {loss}, {fam_loss}, {gen_loss}, {cat_loss}, {fam_acc}, {gen_acc}, {cat_acc}")

            for layers in model.layers:

                if layers.name == 'family' and fam_acc>0.90:

                    layers.trainable=False

                elif layers.name == 'genus':

                    if fam_acc>0.75:

                        layers.trainable=True

                    else:

                        layers.trainable=False

                elif layers.name == 'category_id':

                    if fam_acc>0.75 and gen_acc>0.5:

                        layers.trainable=True

                    else:

                        layers.trainable=False

        n += 1

'''
model.save('fg_model.h5')
batch_size = 32

test_datagen = ImageDataGenerator(featurewise_center=False,

                                  featurewise_std_normalization=False)



generator = test_datagen.flow_from_dataframe(

        dataframe = test_df.iloc[:10000], #Limiting the test to the first 10,000 items

        directory = '../input/herbarium-2020-fgvc7/nybg2020/test/',

        x_col = 'file_name',

        target_size=(120, 120),

        batch_size=batch_size,

        class_mode=None,  # only data, no labels

        shuffle=False)



family, genus, category = model.predict_generator(generator, verbose=1)
sub = pd.DataFrame()

sub['Id'] = test_df.image_id

sub['Id'] = sub['Id'].astype('int32')

sub['Predicted'] = np.concatenate([np.argmax(category, axis=1), 23718*np.ones((len(test_df.image_id)-len(category)))], axis=0)

sub['Predicted'] = sub['Predicted'].astype('int32')

display(sub)

sub.to_csv('category_submission.csv', index=False)
sub['Predicted'] = np.concatenate([np.argmax(family, axis=1), np.zeros((len(test_df.image_id)-len(family)))], axis=0)

sub['Predicted'] = sub['Predicted'].astype('int32')

display(sub)

sub.to_csv('family_submission.csv', index=False)
sub['Predicted'] = np.concatenate([np.argmax(genus, axis=1), np.zeros((len(test_df.image_id)-len(genus)))], axis=0)

sub['Predicted'] = sub['Predicted'].astype('int32')

display(sub)

sub.to_csv('genus_submission.csv', index=False)
end_time = time.time()

total = end_time - start_time

h = total//3600

m = (total%3600)//60

s = total%60

print("Total time spent: %i hours, %i minutes, and %i seconds" %(h, m, s))