# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/"))

print(os.listdir("../input/chest_xray/chest_xray/"))



# Any results you write to the current directory are saved as output.
data_dir = '../input/chest_xray/chest_xray'

os.listdir(data_dir)
train_dir = os.path.join(data_dir, 'train')



val_dir = os.path.join(data_dir, 'val')



test_dir = os.path.join(data_dir, 'test')



train_dir, val_dir, test_dir, os.listdir(train_dir), os.listdir(val_dir), os.listdir(test_dir)
# not enough validation images

# borrow some from training



# delete folders

!rm -rf 'train'

!rm -rf 'val'

!rm -rf 'test'



# new folders

if not os.path.exists('train'):

    os.mkdir('train')

    os.mkdir('train/ok')

    os.mkdir('train/pneu')

if not os.path.exists('val'):

    os.mkdir('val')

    os.mkdir('val/ok')

    os.mkdir('val/pneu')

if not os.path.exists('test'):

    os.mkdir('test')

    os.mkdir('test/ok')

    os.mkdir('test/pneu')

!echo train

!ls -l train

!echo val

!ls -l val

!echo test

!ls -l test

# copy file

#from shutil import copyfile

#copyfile(src, dst)



# old folders

old_tr_ok = '../input/chest_xray/chest_xray/train/NORMAL'

old_tr_bad = '../input/chest_xray/chest_xray/train/PNEUMONIA'

old_val_ok = '../input/chest_xray/chest_xray/val/NORMAL'

old_val_bad = '../input/chest_xray/chest_xray/val/PNEUMONIA'

old_test_ok = '../input/chest_xray/chest_xray/test/NORMAL'

old_test_bad = '../input/chest_xray/chest_xray/test/PNEUMONIA'



import shutil



def copy_all_to(src_dir, dst_dir):

    # copy file, duplicates!

    i=0

    for f in os.listdir(src_dir):

        #print('copy',f,'from',src_dir,'to',dst_dir)

        shutil.copyfile(src_dir+'/'+f, dst_dir+'/'+f)

        i+=1

    print('copied',i,src_dir,dst_dir)

    

copy_all_to(old_tr_ok, 'train/ok')

copy_all_to(old_tr_bad, 'train/pneu')

copy_all_to(old_val_ok, 'val/ok')

copy_all_to(old_val_bad, 'val/pneu')

copy_all_to(old_test_ok, 'test/ok')

copy_all_to(old_test_bad, 'test/pneu')



def move_from_to_n(src_dir,dst_dir, n):

    # move files, no duplicates!

    fs = os.listdir(src_dir)

    assert len(fs) >= n

    i=0

    targ = fs[-n:]

    print('targ', targ)

    for f in targ:

        print('move_from_to_n', i, f, src_dir, dst_dir)

        os.rename(src_dir+'/'+f, dst_dir+'/'+f)

        i+=1

    print('copied',i,src_dir,dst_dir)

    

n=50



val_ok_sz = len(os.listdir('val/ok'))

val_bad_sz = len(os.listdir('val/pneu'))

print('val size', val_ok_sz, val_bad_sz)



# too few val/ok and val/pneu

if val_ok_sz < n:

    src, dst = 'train/ok', 'val/ok'

    move_from_to_n(src,dst, n)

if val_ok_sz < n:

    src, dst = 'train/pneu', 'val/pneu'

    move_from_to_n(src,dst, n)





val_ok_sz = len(os.listdir('val/ok'))

val_bad_sz = len(os.listdir('val/pneu'))

print('val size', val_ok_sz, val_bad_sz)



train_dir = os.path.join('train')

val_dir = os.path.join( 'val')

test_dir = os.path.join( 'test')



train_dir, val_dir, test_dir
from PIL import Image
train_data = []



train_ok_dir = os.path.join(train_dir, 'ok')

for f in os.listdir(train_ok_dir):

    if '.jpeg' in f:

        train_data.append((os.path.join(train_ok_dir,f), 0))



train_bad_dir = os.path.join(train_dir, 'pneu')

for f in os.listdir(train_bad_dir):

    if '.jpeg' in f:

        train_data.append((os.path.join(train_bad_dir,f), 1))



len(train_data), train_data[0], train_data[-1]
test_data = []



test_ok_dir = os.path.join(test_dir, 'ok')

for f in os.listdir(test_ok_dir):

    if '.jpeg' in f:

        test_data.append((os.path.join(test_ok_dir,f), 0))



test_bad_dir = os.path.join(test_dir, 'pneu')

for f in os.listdir(test_bad_dir):

    if '.jpeg' in f:

        test_data.append((os.path.join(test_bad_dir,f), 1))



len(test_data), test_data[0], test_data[-1]
img = Image.open(train_data[0][0])



img
img.size
img_sm = img.resize((150,150))

img_sm
img_sm.size
label = train_data[0][1]

img.size, label
from keras.preprocessing.image import ImageDataGenerator
train_data_gen = ImageDataGenerator(rotation_range=20, 

                                    zoom_range=0.2,

                                    width_shift_range=0.2, 

                                    height_shift_range=0.2,

                                    rescale=1./255)



val_data_gen = ImageDataGenerator(rescale=1./255)

train_generator = train_data_gen.flow_from_directory(

    train_dir,

    target_size=(150,150),

    batch_size=20,

    color_mode='grayscale',

    class_mode='binary')
val_generator = val_data_gen.flow_from_directory(

    val_dir,

    target_size=(150,150),

    batch_size=20,

    color_mode='grayscale',

    class_mode='binary')
train_generator.labels
# val

val_generator.labels
train_samp = train_generator.next()



train_samp[0].shape, len(train_samp), len(train_samp[0])
from keras import layers

from keras import models
last_conv_layer=None
def make_model():

    global last_conv_layer

    model = models.Sequential()

    

    model.add(layers.Conv2D(32, (3,3), activation='relu',

                                input_shape=(150,150,1)))

    model.add(layers.MaxPooling2D((2,2)))

    

    model.add(layers.Conv2D(64, (3,3), activation='relu'))

    model.add(layers.MaxPooling2D((2,2)))

    

    model.add(layers.Conv2D(128, (3,3), activation='relu'))

    model.add(layers.MaxPooling2D((2,2)))

    

    last_conv_layer = layers.Conv2D(128, (3,3), activation='relu')

    model.add(last_conv_layer)

    model.add(layers.MaxPooling2D((2,2)))

    

    model.add(layers.Flatten())

    

    model.add(layers.Dropout(0.5))

    

    last_layer=32

    model.add(layers.Dense(last_layer, activation='relu'))

    model.add(layers.Dense(1, activation='sigmoid'))



    return model
model = make_model()

model.summary()
last_conv_layer.name
from keras import optimizers
model.compile(loss='binary_crossentropy',

             optimizer='adam',

             metrics=['acc'])
import keras.callbacks

## model checkpointing

filepath='weights.val_loss-{val_loss:.2f}.val_acc-{val_acc:.2f}.epoch-{epoch:02d}.hdf5'

checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)

callbacks_list = [checkpoint]
#assert False "TODO: need validation generator"

history = model.fit_generator(

                train_generator,

                steps_per_epoch=250,

                epochs=50,

                validation_data=val_generator,

                validation_steps=50,

                callbacks=callbacks_list)

# p137
!ls -lh *.hdf5
#model.save('pneumonia_chest_xray_sm.h5')
!ls -lh *.hdf5 | head -1
# delete all the other checkpoints

#!rm *.hdf5
# save model

model_json = model.to_json()

with open("model.json", "w") as json_file:

    json_file.write(model_json)
!ls -lh model.json
import matplotlib.pyplot as plt
acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(1, len(acc)+1)



plt.plot(epochs, acc, 'bo', label='Training Acc.')

plt.plot(epochs, val_acc, 'b', label='Validation Acc.')

plt.title('Training and validation accuracy')

plt.legend()



plt.figure()



plt.plot(epochs, loss, 'bo', label='Training Loss')

plt.plot(epochs, val_loss, 'b', label='Validation Loss')

plt.title('Training and validation loss')

plt.legend()



plt.show()

from keras import backend as K
print('test files', len(os.listdir('test/pneu/')))

x = os.listdir('test/pneu/')[0]

x = 'test/pneu/'+x

print(x)



pic = Image.open( x )



def preproc_im(x, crop=None):

    print('preproc_im', x)

    x = Image.open( x )

    

    if crop:

        w, h = pic.size

        cx,cy,cw,ch = crop

        x_crop = pic.crop((cx, cy, w-cw, h-ch )) # 0, 180, w, h-50

    

    x = x.resize((150,150))

    x = np.array(x).astype('float32')

    x /= 255.

    x=x.reshape(150, 150, 1)

    print(x.shape)



    #make it 4d

    x = np.array([x])

    print(x.shape)

    #print(x)

    return x





pic
nx = preproc_im(x)

nx.shape





# pred = model.predict(x)

pred = model.predict(nx)



pred
# p 174



pneumonia_output = model.output[:1]

print('pneumonia_output', pneumonia_output.shape)



# last_conv_layer.name

# last_conv_layer = model.get_layer('conv2d_8')

last_conv_layer = model.get_layer(last_conv_layer.name)

print('last_conv_layer', last_conv_layer)



grads = K.gradients(pneumonia_output, last_conv_layer.output)[0]

print('grads', grads.shape)



pooled_grads = K.mean(grads, axis=(0,1,3))

print('pooled_grads', pooled_grads.shape)



iterate = K.function([model.input],

                    [pooled_grads, last_conv_layer.output[0]])



#pooled_grads_value, conv_layer_output_value = iterate([x_crop])

pooled_grads_value, conv_layer_output_value = iterate([nx])

print('pooled_grads_value', pooled_grads_value.shape)

print('conv_layer_output_value', conv_layer_output_value.shape)



for i in range(15):

    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]



heatmap = np.mean(conv_layer_output_value, axis=-1)
heatmap = np.maximum(heatmap, 0)

heatmap /= np.max(heatmap)



plt.matshow(heatmap, cmap='hot')
len(test_data)
test_data_gen = ImageDataGenerator(rescale=1./255)



test_generator = test_data_gen.flow_from_directory(

    test_dir,

    target_size=(150,150),

    batch_size=20,

    color_mode='grayscale',

    class_mode='binary')
test_datagen = ImageDataGenerator(rescale=1./255)



test_generator = test_datagen.flow_from_directory(

        test_dir,

        target_size=(150, 150),

        color_mode="grayscale",

        shuffle = False,

        class_mode='categorical',

        batch_size=1)



filenames = test_generator.filenames

nb_samples = len(filenames)



predict = model.predict_generator(test_generator,steps = nb_samples)
rights, wrongs =0,0

for i,pre in enumerate(predict):

    name=filenames[i]

    rnd = round(pre[0])

    dir_truth={'ok':0,'pneu':1 }

    right=False

    if 'ok' in name: right=(rnd==0)

    if 'pneu' in name: right=(rnd==1)

    if right:

        rights+=1

    else:

        wrongs+=1

    print(i, pre, rnd, right,name )
rights, wrongs
rights/len(predict), wrongs/len(predict)
# delete images so files<500, error
!ls -l 
!rm -rf val/ test/ train/
!ls -lh