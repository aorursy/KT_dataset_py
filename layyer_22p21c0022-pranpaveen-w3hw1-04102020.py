import numpy as np # linear algebra
import pandas as pd 
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import os
import pandas as pd
tf.__version__
!pwd
train_label = pd.read_csv('/kaggle/input/super-ai-image-classification/train/train/train.csv')
train_label.head()
train_label.category.plot.hist()
class0 = train_label[train_label.category == 0]
class1 = train_label[train_label.category == 1]
class0 = class0.sample(frac=1)
class1 = class1.sample(frac=1)
n_split = 0.8
df_train = pd.concat([ class0[:int(n_split*len(class0))], class1[:int(n_split*len(class1))] ]).reset_index()
df_val = pd.concat([ class0[int(n_split*len(class0)):], class1[int(n_split*len(class1)):] ]).reset_index()
train_path = '/kaggle/input/super-ai-image-classification/train/train/images'
test_path = '/kaggle/input/super-ai-image-classification/val/val/images'
df_train.head()
df_val.head()
from shutil import copyfile,rmtree
if os.path.isdir('train_data'):
    rmtree('train_data')
os.makedirs('train_data')
if not os.path.isdir('train_data/class0'):
    os.makedirs('train_data/train')
    os.makedirs('train_data/train/class0')
    os.makedirs('train_data/train/class1')
for i in range(len(df_train)):
    file = df_train.iloc[i]["id"]
    src = os.path.join(train_path,file)
    if df_train.iloc[i]['category'] == 0:
        des = 'train_data/train/class0/'+file
    elif df_train.iloc[i]['category'] == 1:
        des = 'train_data/train/class1/'+file
    copyfile(src,des)

os.makedirs('train_data/test')
os.makedirs('train_data/test/class0')
os.makedirs('train_data/test/class1')
for i in range(len(df_val)):
    file = df_val.iloc[i]["id"]
    src = os.path.join(train_path,file)
    if df_val.iloc[i]['category'] == 0:
        des = 'train_data/test/class0/'+file
    elif df_val.iloc[i]['category'] == 1:
        des = 'train_data/test/class1/'+file
    copyfile(src,des)
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    zoom_range=0.3
)
val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
)
os.listdir('./train_data')
train_generator = train_datagen.flow_from_directory('./train_data/train',target_size=(224,224),batch_size=32,class_mode="binary") 
val_generator = val_datagen.flow_from_directory('./train_data/test',target_size=(224,224),batch_size=32,class_mode="binary",shuffle=False) 
pretrain = tf.keras.applications.ResNet101V2(weights='imagenet',include_top=False,input_shape=(224,224,3))
len(pretrain.layers)
#pretrain.trainable = False
for layer in pretrain.layers:
    layer.trainable = False
    if layer == pretrain.get_layer('conv4_block23_out'):
        break
pretrain.output
last_pretrain = pretrain.layers[-1].output
x = tf.keras.layers.GlobalAveragePooling2D()(last_pretrain)
x = tf.keras.layers.Dropout(0.25)(x)
x = tf.keras.layers.Dense(1024,activation='relu')(x)
x = tf.keras.layers.Dropout(0.25)(x)
x =  tf.keras.layers.Dense(1,activation='sigmoid')(x)
model = tf.keras.Model(pretrain.input,x)
#model.summary()
model.compile(optimizer='adam',loss=tf.keras.losses.BinaryCrossentropy(),metrics=['acc'])
class_weight = {0:1,1:3}
learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)
history = model.fit(train_generator ,validation_data=val_generator,epochs=20,
                    steps_per_epoch=len(train_generator),class_weight=class_weight,
                   callbacks=[learning_rate_reduction])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
for layer in pretrain.layers:
    layer.trainable = True
learning_rate_reduction2 = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_acc', 
                                            patience=2, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=1e-8)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-6),loss=tf.keras.losses.BinaryCrossentropy(),metrics=['acc'])
history2 = model.fit(train_generator ,validation_data=val_generator,epochs=5,steps_per_epoch=len(train_generator),class_weight=class_weight)
plt.plot(history2.history['acc'])
plt.plot(history2.history['val_acc'])
pred_val = model.predict(val_generator)
pred_val = (pred_val>0.5).astype(int).reshape(-1)
from sklearn.metrics import classification_report
print(classification_report(val_generator.labels,pred_val))
model.save('super_ai_hw3.h5')
pred_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255
)
val_gen = pred_datagen.flow_from_directory('/kaggle/input/super-ai-image-classification/val/val/',target_size=(224,224),batch_size=32,shuffle=False,class_mode="binary") 
pred = model.predict_generator(val_gen)
pred_en = (pred>0.5).astype(int).reshape(-1)
pred_en
val_name = [x.split('/')[1] for x in val_gen.filenames]
#val_name
len(pred_en)
val = pd.DataFrame()
val["id"] = val_name
val["category"] = pred_en
val.head()
val.to_csv('val_submit.csv', index=False)
rmtree('train_data')