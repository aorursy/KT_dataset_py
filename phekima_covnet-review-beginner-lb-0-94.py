import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os, shutil

import keras

from keras import layers, Sequential

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ModelCheckpoint,EarlyStopping

from keras.layers import Conv2D,MaxPooling2D, Dropout, Dense, BatchNormalization, Flatten

from keras.callbacks import ReduceLROnPlateau

from keras import optimizers

from keras.optimizers import Nadam



import tensorflow as tf

tf.set_random_seed(0)



import matplotlib.pyplot as plt 

from glob import glob

%matplotlib inline



np.random.seed(101)



# Any results you write to the current directory are saved as output.
os.listdir('../input/')
train_dir = '../input/train/'

data = pd.DataFrame({'path': glob(os.path.join(train_dir,'*.tif'))                    

                    })

data.head()

data['id'] = data.path.apply(lambda x: str(x).split('/')[3].split('.')[0])

data.head()
df = pd.read_csv('../input/train_labels.csv')

data = data.merge(df, on='id')

data.head()
training_dir= '../training_dir'

validation_dir= '../validation_dir'

os.mkdir(training_dir)

os.mkdir(validation_dir)
df = data

df_0 = df[df.label==0]

df_1 = df[df.label==1]

df_1.head()
print(df_1.shape)

df_0.shape
categories = [0,1]

for category in categories:

    os.mkdir(os.path.join(training_dir,str(category))) #../training_dir/0 or 1

    os.mkdir(os.path.join(validation_dir,str(category)))

    

 # CREATING TRAINING DIRECTORY            

for category in categories:

    cdir= os.path.join(training_dir,str(category)) #creates '../1 or 0'

    for sample_count, path in enumerate(df[df.label ==category].path):

        id = path.split('/')[3] #generate destination id_name

        src = path

        dst = os.path.join(cdir,id) #destination

        shutil.copyfile(src,dst)

        if sample_count==70000: break

        

#CREATING VALIDATION DIRECTORY

for category in categories:

    cdir= os.path.join(validation_dir,str(category)) #creates '../1 or 0'

    for sample_count, path in enumerate(df[df.label ==category].path):

        if sample_count>70000:

            id = path.split('/')[3] #generate destination id_name

            src = path

            dst = os.path.join(cdir,id) #destination

            shutil.copyfile(src,dst)

            if sample_count==89000: break

        else: continue
len(os.listdir(os.path.join(validation_dir,'1')))
os.listdir(validation_dir)
data_gen = ImageDataGenerator(rotation_range=40,

                          rescale=1./255, width_shift_range=0.2, 

                              height_shift_range=0.2, 

                             shear_range=0.2,

                             zoom_range=0.2, horizontal_flip=True,

                             fill_mode='nearest')
batch_size=20

train_generator = data_gen.flow_from_directory(training_dir, 

                                               target_size=(96,96),

                                               class_mode='binary',

                                               batch_size=batch_size)

test_gen = ImageDataGenerator(rescale=1./255)



validation_generator_shuffled=test_gen.flow_from_directory(validation_dir,

                                                 target_size=(96,96),

                                                 class_mode='binary',

                                                 batch_size=batch_size)



validation_generator=test_gen.flow_from_directory(validation_dir,

                                                 target_size=(96,96),

                                                 class_mode='binary',

                                                 batch_size=batch_size,

                                                 shuffle=False)



pool_size= (2,2)



model = Sequential()

model.add(layers.Conv2D(32,3,input_shape=(96,96,3), activation='relu'))

model.add(layers.Conv2D(32,3,activation='relu'))

model.add(layers.Conv2D(32,3,activation='relu'))

model.add(layers.BatchNormalization())

model.add(layers.MaxPool2D(pool_size))

# model.add(Dropout(0.3))



model.add(layers.Conv2D(64,3,activation='relu'))

model.add(layers.Conv2D(64,3,activation='relu'))

model.add(layers.Conv2D(64,3,activation='relu'))

model.add(layers.BatchNormalization())

model.add(layers.MaxPool2D(pool_size))

# model.add(Dropout(0.3))



model.add(layers.Conv2D(128,3,activation='relu'))

model.add(layers.Conv2D(128,3,activation='relu'))

model.add(layers.Conv2D(128,3,activation='relu'))

model.add(layers.BatchNormalization())

model.add(layers.MaxPooling2D(pool_size))

# model.add(Dropout(0.3))



model.add(layers.Conv2D(256,3,activation='relu', padding='same'))

model.add(layers.Conv2D(256,3,activation='relu', padding='same'))

model.add(layers.Conv2D(256,3,activation='relu', padding='same'))

model.add(layers.BatchNormalization())

model.add(layers.MaxPooling2D(pool_size))

# model.add(Dropout(0.3))



model.add(layers.Flatten())

# model.add(layers.Dropout(0.5))

model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=optimizers.Nadam(), loss='binary_crossentropy', metrics=['accuracy'])

model.summary()
train_samples = 140000

checkpoint_name= '../working/model.h5'

checkpoint = ModelCheckpoint(checkpoint_name, save_best_only=True)

stop = EarlyStopping(patience=10, verbose=1)

LrReduce = ReduceLROnPlateau(verbose=1, patience=4)

model.fit_generator(train_generator, 

                    steps_per_epoch=train_samples//batch_size,

                   validation_data=validation_generator_shuffled,

                   epochs=20,

                   callbacks=[checkpoint,stop, LrReduce])
history = model.history.history
plt.plot(model.history.epoch, history['acc'],label='training_acc')

plt.plot(model.history.epoch, history['val_acc'],c='green', label='Validation Accuracy')

plt.xlabel('Epochs')

plt.legend()

plt.plot(model.history.epoch, history['loss'],label='training_loss')

plt.plot(model.history.epoch, history['val_loss'],c='green', label='Validation Loss')

plt.xlabel('Epochs')

plt.legend()
os.listdir('../working')
from keras.models import load_model

predictor = load_model(checkpoint_name)
validation_generator.filenames[:5]
test_dir = '../input/test/'

df_test = pd.DataFrame({'path': glob(os.path.join(test_dir,'*.tif'))                    

                    })

df_test['id'] = df_test.path.apply(lambda x: str(x).split('/')[3].strip())

# df_test.drop('path', axis=1, inplace=True)

df_test.head()

test_batch_size=2

test_generator = test_gen.flow_from_dataframe(df_test,directory='../input/test/',x_col='id',

                                              target_size=(96,96),class_mode=None,

                                             shuffle=False,

                                             batch_size=test_batch_size)
df_test.drop('path', inplace=True, axis=1)

df_test.id = test_generator.filenames

df_test.head()
for bb in test_generator:

    plt.imshow(bb[0])

    break
no_of_samples = 57458

predictions = predictor.predict_generator(test_generator,

                                         steps=no_of_samples//test_batch_size, verbose=1)
df_test.head()
predictions
results = pd.DataFrame({'label': predictions.reshape(-1,)}, index=range(0,no_of_samples))

results.head()
# def ro(x):

#     if x>=0.5: x=1

#     else: x=0

#     return x



dd = results

dd.label = dd.label.apply(round)

dd.label.value_counts()
results = pd.DataFrame({'label': predictions.reshape(-1,)}, index=range(0,no_of_samples))

results.head()
# df_test.drop('path', axis=1, inplace=True)

df_test.id = df_test.id.apply(lambda x: x.split('.')[0])

df_test.head()
submission = pd.concat([df_test,results],axis=1)

# submission2 = pd.concat([df_test,dd],axis=1)

submission.head(10)
submission.to_csv('submissions.csv', index=False)

# submission2.to_csv('submissionswhole.csv', index=False)
pd.read_csv('submissions.csv').head()
# def roundup(x):

#     if x>=0.5: x = 1

#     else: x=0

#     return x
# results.label.apply(roundup).value_counts()
# val_batch_labels=np.zeros(len(validation_generator.classes))
# i = 0

# for b, l in (validation_generator):

#     val_batch_labels[i*batch_size:batch_size*(i+1)] = l

#     i+=1

#     if i==3000:

#         break
# print('done')

# pd.DataFrame(val_batch_labels)[0].value_counts()


from sklearn.metrics import roc_curve,auc, confusion_matrix, classification_report

val_pred = predictor.predict_generator(validation_generator,

                                      steps=len(validation_generator.classes)//batch_size,

                                      verbose=1)
val_pred
false_positive_rate,true_positive_rate,threshold = roc_curve(validation_generator.classes,

                                                            val_pred)
val_pred_whole = np.where(val_pred>=0.5,1,0)

AUC = auc(false_positive_rate,true_positive_rate)

print(AUC)

print(classification_report(validation_generator.classes,val_pred_whole))
dg = pd.DataFrame(val_pred_whole, columns=['label'])

dg.label.value_counts()
validation_generator.classes
shutil.rmtree(validation_dir)

# os.mkdir('../test_dir')
# test_path= '../input/test'

# test_dir='../test_dir'

# # for image in os.listdir(test_path):

# for img in os.listdir(test_path):

#     src= os.path.join(test_path,img)

#     dst = os.path.join(test_dir,img)

#     shutil.copyfile(src,dst)

# print('done')

# len(os.listdir(test_dir))

    

    
# os.mkdir('../test_images')

# test_image_path = '../test_images'

# shutil.move(test_dir, test_image_path)
# test_generatorr = test_gen.flow_from_directory(test_image_path,target_size=(96,96),

#                                               shuffle=False, batch_size=test_batch_size,

#                                               class_mode='binary')
# test_image_path= os.path.join(test_image_path,'test_dir')

# os.listdir(test_image_path)[:5]
# data = pd.DataFrame({'path': glob(os.path.join(test_image_path,'*.tif'))                    

#                     })

# data.head()



# data['id'] = data.path.apply(lambda x: str(x).split('/')[3].split('.')[0])

# data.head()
# new_pred= predictor.predict_generator(test_generatorr, 

#                                       steps=len(test_generatorr.classes)//test_batch_size,

#                                     verbose=1)
# results = pd.DataFrame({'label': new_pred.reshape(-1,)}, index=range(0,no_of_samples))

# results.head()
# results.label.apply(roundup).value_counts()