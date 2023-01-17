from IPython.display import display, HTML

display(HTML(data="""
<style>
    div#notebook-container    { width: 95%; }
    div#menubar-container     { width: 65%; }
    div#maintoolbar-container { width: 99%; }
</style>
"""))
#to use different versions
!pip install efficientnet
#!pip install --upgrade tensorflow-gpu
#!pip install --upgrade efficientnet
import numpy as np
import pandas as pd
import os
import json, codecs
import tensorflow as tf
from efficientnet.keras import EfficientNetB0
from kaggle_datasets import KaggleDatasets
print(tf.__version__)
# Detect hardware, return appropriate distribution strategy
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection. No parameters necessary if TPU_NAME environment variable is set. On Kaggle this is always the case.
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy() # default distribution strategy in Tensorflow. Works on CPU and single GPU.

print("REPLICAS: ", strategy.num_replicas_in_sync)
#name = !ls /kaggle/input
#name
#name = !ls /kaggle/input/
#GCS_DS_PATH = KaggleDatasets().get_gcs_path('herbarium-2020-fgvc7') # you can list the bucket with "!gsutil ls $GCS_DS_PATH"
for dirname,_,filenames in os.walk("../input/herbarium-2020-fgvc7"):
    for filename in filenames:
        if filename.endswith('.jpg'):
            break
        print(os.path.join(dirname,filename))
with codecs.open("../input/herbarium-2020-fgvc7/nybg2020/train/metadata.json", 'r',
                 encoding='utf-8', errors='ignore') as f:
    train_meta = json.load(f)
    
with codecs.open("../input/herbarium-2020-fgvc7/nybg2020/test/metadata.json", 'r',
                 encoding='utf-8', errors='ignore') as f:
    test_meta = json.load(f)
train_meta.keys()
train_df = pd.DataFrame(train_meta['annotations'])
train_df
train_cat = pd.DataFrame(train_meta['categories'])
train_cat.columns = ['family', 'genus', 'category_id', 'category_name']
train_cat
train_img = pd.DataFrame(train_meta['images'])
train_img.columns = ['file_name', 'height', 'image_id', 'license', 'width']
train_img
train_reg = pd.DataFrame(train_meta['regions'])
train_reg.columns = ['region_id', 'region_name']
train_reg
train_df = train_df.merge(train_cat, on='category_id', how='outer')
train_df = train_df.merge(train_img, on='image_id', how='outer')
train_df = train_df.merge(train_reg, on='region_id', how='outer')
train_df
train_df.info()
na = train_df.file_name.isna()
keep = [x for x in range(train_df.shape[0]) if not na[x]]
train_df = train_df.iloc[keep]
dtypes = ['int32', 'int32', 'int32', 'int32', 'object', 'object', 'object', 'object', 'int32', 'int32', 'int32', 'object']
for n, col in enumerate(train_df.columns):
    train_df[col] = train_df[col].astype(dtypes[n])
print(train_df.info())
train_df
test_df = pd.DataFrame(test_meta['images'])
test_df.columns = ['file_name', 'height', 'image_id', 'license', 'width']
print(test_df.info())
test_df
print("Total Unique Values for each columns:")
print("{0:10s} \t {1:10d}".format('train_df', len(train_df)))
for col in train_df.columns:
    print("{0:10s} \t {1:10d}".format(col, len(train_df[col].unique())))
family = train_df[['family', 'genus', 'category_name']].groupby(['family', 'genus']).count()
display(family.describe())
from keras.models import Model
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten, BatchNormalization, Input, concatenate
from keras.optimizers import Adam
from keras.utils import plot_model
from sklearn.model_selection import train_test_split as tts



def xavier(shape, dtype=None):
    return np.random.rand(*shape)*np.sqrt(1/in_out_size)



def fg_model(shape,lr):
    
    actual_shape = shape
    i = Input(actual_shape)
    x = EfficientNetB0(weights='imagenet', include_top=False, input_shape=actual_shape, pooling='max')(i)
    #x = Flatten()(x)
    o1 = Dense(310, name="family", activation='softmax')(x)
    o2 = concatenate([x,o1])
    o2 = Dense(3678, name="genus", activation='softmax')(o2)
    o3 = concatenate([x,o1,o2])
    o3 = Dense(32094, name="category_id", activation='softmax')(o3)
    model = Model(inputs=i,outputs=[o1,o2,o3])
    
    model.layers[1].trainable = False
    model.get_layer('genus').trainable = False
    model.get_layer('category_id').trainable = False
    
    opt = Adam(lr=lr, amsgrad=True)
    model.compile(optimizer=opt, loss=['sparse_categorical_crossentropy', 
                                   'sparse_categorical_crossentropy', 
                                   'sparse_categorical_crossentropy'],
                 metrics=['accuracy'])
    return model


#plot_model(model, to_file='full_model_plot.png', show_shapes=True, show_layer_names=True)
from keras.preprocessing.image import ImageDataGenerator

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
train = train[:80000]
verif = verif[:20000]
shape = (224,224, 3)
epochs = 8
batch_size = 32

#model = fg_model(shape, 0.007)
#model.summary()
model = fg_model((224,224,3), 0.007)
model.summary()
#Disable the last two output layers for training the Family
for layers in model.layers:
    if layers.name == 'genus' or layers.name=='category_id':
        layers.trainable = False
#Train Family for 2 epochs
model.fit_generator(train_datagen.flow_from_dataframe(dataframe=train,
                                                      directory='../input/herbarium-2020-fgvc7/nybg2020/train/',
                                                      x_col="file_name",
                                                      y_col=["family", "genus", "category_id"],
                                                      target_size=(224,224),
                                                      batch_size=batch_size,
                                                      class_mode='multi_output'),
                    validation_data=train_datagen.flow_from_dataframe(
                        dataframe=verif,
                        directory='../input/herbarium-2020-fgvc7/nybg2020/train/',
                        x_col="file_name",
                        y_col=["family", "genus", "category_id"],
                        target_size=(224,224),
                        batch_size=batch_size,
                        class_mode='multi_output'),
                    epochs=epochs,
                    steps_per_epoch=len(train)//batch_size,
                    validation_steps=len(verif)//batch_size,
                    verbose=1,
                    workers=8,
                    use_multiprocessing=False)

model.save_weights("weights.h5")
model.save("model.h5")
#Reshuffle the inputs
train, verif = tts(m, test_size=0.2, shuffle=True, random_state=17)
train = train#[:500000]
verif = verif#[:100000]
#Make the Genus layer Trainable
for layers in model.layers:
    if layers.name == 'genus':
        layers.trainable = True
        
#Train Family and Genus for 2 epochs
model.fit_generator(train_datagen.flow_from_dataframe(dataframe=train,
                                                      directory='../input/herbarium-2020-fgvc7/nybg2020/train/',
                                                      x_col="file_name",
                                                      y_col=["family", "genus", "category_id"],
                                                      target_size=(224,224),
                                                      batch_size=batch_size,
                                                      class_mode='multi_output'),
                    validation_data=train_datagen.flow_from_dataframe(
                        dataframe=verif,
                        directory='../input/herbarium-2020-fgvc7/nybg2020/train/',
                        x_col="file_name",
                        y_col=["family", "genus", "category_id"],
                        target_size=(224,224),
                        batch_size=batch_size,
                        class_mode='multi_output'),
                    epochs=epochs,
                    steps_per_epoch=len(train)//batch_size,
                    validation_steps=len(verif)//batch_size,
                    verbose=1,
                    workers=4,
                    use_multiprocessing=False)

model.save_weights("weights.h5")
model.save("model.h5")
#Reshuffle the inputs
train, verif = tts(m, test_size=0.2, shuffle=True, random_state=17)
train = train#[:500000]
verif = verif#[:100000]

#Make the category_id layer Trainable
for layers in model.layers:
    if layers.name == 'category_id':
        layers.trainable = True
        
#Train them all for 2 epochs
model.fit_generator(train_datagen.flow_from_dataframe(dataframe=train,
                                                      directory='../input/herbarium-2020-fgvc7/nybg2020/train/',
                                                      x_col="file_name",
                                                      y_col=["family", "genus", "category_id"],
                                                      target_size=(224,224),
                                                      batch_size=batch_size,
                                                      class_mode='multi_output'),
                    validation_data=train_datagen.flow_from_dataframe(
                        dataframe=verif,
                        directory='../input/herbarium-2020-fgvc7/nybg2020/train/',
                        x_col="file_name",
                        y_col=["family", "genus", "category_id"],
                        target_size=(224,224),
                        batch_size=batch_size,
                        class_mode='multi_output'),
                    epochs=epochs,
                    steps_per_epoch=len(train)//batch_size,
                    validation_steps=len(verif)//batch_size,
                    verbose=1,
                    workers=4,
                    use_multiprocessing=False)
model.save_weights("weights.h5")
model.save("model.h5")

