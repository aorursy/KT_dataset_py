import numpy as np
import pandas as pd
import shutil
import os
import csv
from tabulate import tabulate

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate
from keras.models import Model
from keras.optimizers import SGD

# for reproducible results
import tensorflow as tf
import random as rn
np.random.seed(42)
rn.seed(12345)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                              inter_op_parallelism_threads=1)
from keras import backend as K
tf.set_random_seed(1234)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)
# for reproducible results

class_sz = 400
image_sz = 64
batch_sz = 14
split = True
epochs_num = 30
lr = 0.0001
np.random.seed(123)
all_image_dir = '/kaggle/input/dataset/dataset'

train_csv = 'train.csv'
val_csv = 'val.csv'
test_csv = 'test.csv'

class_names = os.listdir(all_image_dir)
class_names.sort()
classes_num = len(class_names)

train_dir_sz = int(class_sz*0.8*0.7)
val_dir_sz = int(class_sz*0.8*0.3)
test_dir_sz = int(class_sz*0.2)
def create_directory(dir_name):
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
    os.makedirs(dir_name)
    for class_name in class_names:
        os.makedirs(os.path.join(dir_name, class_name))
train_dir = '/kaggle/config/dataset/train/'
val_dir = '/kaggle/config/dataset/val/'
test_dir = '/kaggle/config/dataset/test/'

create_directory(train_dir)
create_directory(val_dir)
create_directory(test_dir)
indexes = np.random.permutation(class_sz)

def copy_images(start_idx, end_idx, source_dir, dest_dir, csv_file, class_num, size):
    file = open(csv_file, 'a+')
    writer = csv.writer(file)
    idx = 0
    
    for i in range(start_idx, end_idx):
        y = np.zeros(5)
        y[class_num] = 1
        src_file = os.path.join(source_dir, str(indexes[i]) + ".jpg")
        dst_file = os.path.join(dest_dir, str(indexes[i]) + ".jpg")
        writer.writerow([dst_file, y])
        shutil.copy(src_file, dst_file)
        idx+=1
        
    file.close()
# разделение набора данных на train, valid, test
def split_image_dataset():
    idx = 0
#     очитска csv-файлов
    open(train_csv, 'w').close()
    open(test_csv, 'w').close()
    open(val_csv, 'w').close()
    
    for class_name in class_names:
        class_data_dir = os.path.join(all_image_dir, class_name)
        copy_images(0, train_dir_sz,
                    class_data_dir, os.path.join(train_dir, class_name),
                   train_csv, idx, train_dir_sz)
        copy_images(train_dir_sz, train_dir_sz+val_dir_sz,
                    class_data_dir, os.path.join(val_dir, class_name),
                   val_csv, idx, val_dir_sz)
        copy_images(train_dir_sz+val_dir_sz, train_dir_sz+val_dir_sz+test_dir_sz,
                class_data_dir, os.path.join(test_dir, class_name),
                   test_csv, idx, test_dir_sz)
        idx+=1
input_img = Input(shape = (image_sz, image_sz, 3))

branch_1 = Conv2D(64, (1,1), padding='same', activation='relu')(input_img)
branch_1 = Conv2D(64, (3,3), padding='same', activation='relu')(branch_1)
branch_2 = Conv2D(64, (1,1), padding='same', activation='relu')(input_img)
branch_2 = Conv2D(64, (5,5), padding='same', activation='relu')(branch_2)
branch_3 = MaxPooling2D((3,3), strides=(1,1), padding='same')(input_img)
branch_3 = Conv2D(64, (1,1), padding='same', activation='relu')(branch_3)
output = concatenate([branch_1, branch_2, branch_3], axis = 3)

output = Flatten()(output)
out    = Dense(classes_num, activation='softmax')(output)

model = Model(inputs = input_img, outputs = out)
# print (model.summary())
if(split):
    split_image_dataset()
datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(image_sz, image_sz),
    batch_size=batch_sz,
    class_mode='categorical')
test_generator = datagen.flow_from_directory(
    test_dir,
    target_size=(image_sz, image_sz),
    batch_size=1,
    class_mode='categorical')
val_generator = datagen.flow_from_directory(
    val_dir,
    target_size=(image_sz, image_sz),
    batch_size=1,
    class_mode='categorical')
epochs = epochs_num
lrate = lr
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
nb_train_samples = train_dir_sz*classes_num
nb_validation_samples = val_dir_sz*classes_num
nb_test_samples = test_dir_sz*classes_num

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_sz,
    epochs=epochs,
    validation_data=val_generator,
    validation_steps=nb_validation_samples, 
    shuffle=True)
def per_class_accuracy():
    results = []
    
    for i in range(5):
        X = np.ndarray(shape=(test_dir_sz, image_sz, image_sz, 3),
                             dtype=np.float32)
        y = np.zeros((test_dir_sz,classes_num), dtype=np.int)
        y[:,i] = np.ones(test_dir_sz)
        
        idx = 0
        dirname = os.path.join(test_dir, class_names[i])
        
        for img in os.listdir(dirname):
            img = load_img(os.path.join(dirname,img))
            img = img.resize((image_sz,image_sz))
            x = img_to_array(img)
            x = x / 255
            X[idx] = x
            idx += 1
            
        scores = model.evaluate(np.array(X), y)
        results.append([class_names[i],scores[1]])
        
    return results
acc = per_class_accuracy()
all_acc= model.evaluate_generator(test_generator, nb_test_samples)
acc.append(["all classes", all_acc[1]])

print (tabulate(acc))