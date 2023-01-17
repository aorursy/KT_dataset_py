

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

import shutil

print(os.listdir("../input"))



from glob import glob 

from skimage.io import imread

import gc



from sklearn.utils import shuffle

from sklearn.model_selection import train_test_split

from keras.utils import to_categorical
base_tile_dir = '../input/train/'

df = pd.DataFrame({'path': glob(os.path.join(base_tile_dir,'*.tif'))})

df['id'] = df.path.map(lambda x: x.split('/')[3].split(".")[0])

labels = pd.read_csv("../input/train_labels.csv")

df_data = df.merge(labels, on = "id")



# # removing this image because it caused a training error previously

# df_data = df_data[df_data['id'] != 'dd6dfed324f9fcb6f93f46f32fc800f2ec196be2']



# # removing this image because it's black

# df_data = df_data[df_data['id'] != '9369c7278ec8bcc6c880d99194de09fc2bd4efbe']

# df_data.head(3)
SAMPLE_SIZE = 20000 # load 20k negative examples



# take a random sample of class 0 with size equal to num samples in class 1

df0 = df_data[df_data['label'] == 0].sample(SAMPLE_SIZE, random_state = 82)

# filter out class 1

df1 = df_data[df_data['label'] == 1].sample(SAMPLE_SIZE, random_state = 82)



# concat the dataframes

df_data = shuffle(pd.concat([df0, df1], axis=0).reset_index(drop=True))



# train_test_split # stratify=y creates a balanced validation set.

y = df_data['label']

df_train, df_val = train_test_split(df_data, test_size=0.10, random_state=82, stratify=y)



# Create directories

train_path = 'base_dir/train'

valid_path = 'base_dir/valid'

test_path = '../input/test'

for fold in [train_path, valid_path]:

    for subf in ["0", "1"]:

        os.makedirs(os.path.join(fold, subf))
# Set the id as the index in df_data

df_data.set_index('id', inplace=True)

df_data.head()
for image in df_train['id'].values:

    # the id in the csv file does not have the .tif extension therefore we add it here

    fname = image + '.tif'

    label = str(df_data.loc[image,'label']) # get the label for a certain image

    src = os.path.join('../input/train', fname)

    dst = os.path.join(train_path, label, fname)

    shutil.copyfile(src, dst)



for image in df_val['id'].values:

    fname = image + '.tif'

    label = str(df_data.loc[image,'label']) # get the label for a certain image

    src = os.path.join('../input/train', fname)

    dst = os.path.join(valid_path, label, fname)

    shutil.copyfile(src, dst)

from keras.preprocessing.image import ImageDataGenerator



IMAGE_SIZE = 96

num_train_samples = len(df_train)

num_val_samples = len(df_val)

train_batch_size = 32

val_batch_size = 32



train_steps = np.ceil(num_train_samples / train_batch_size)

val_steps = np.ceil(num_val_samples / val_batch_size)



datagen = ImageDataGenerator(preprocessing_function=lambda x:(x - x.mean()) / x.std() if x.std() > 0 else x,

                            horizontal_flip=True,

                            vertical_flip=True)



train_gen = datagen.flow_from_directory(train_path,

                                        target_size=(IMAGE_SIZE,IMAGE_SIZE),

                                        batch_size=train_batch_size,

                                        class_mode='binary')



val_gen = datagen.flow_from_directory(valid_path,

                                        target_size=(IMAGE_SIZE,IMAGE_SIZE),

                                        batch_size=val_batch_size,

                                        class_mode='binary')



# Note: shuffle=False causes the test dataset to not be shuffled

test_gen = datagen.flow_from_directory(valid_path,

                                        target_size=(IMAGE_SIZE,IMAGE_SIZE),

                                        batch_size=1,

                                        class_mode='binary',

                                        shuffle=False)
from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation

from keras.layers import Conv2D, MaxPool2D

from keras.optimizers import RMSprop, Adam



kernel_size = (3,3)

pool_size= (2,2)

first_filters = 32

second_filters = 64

third_filters = 128



dropout_conv = 0.3

dropout_dense = 0.5



model = Sequential()

model.add(Conv2D(first_filters, kernel_size, activation = 'relu', input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)))

model.add(Conv2D(first_filters, kernel_size, use_bias=False))

model.add(BatchNormalization())

model.add(Activation("relu"))

model.add(MaxPool2D(pool_size = pool_size)) 

model.add(Dropout(dropout_conv))



model.add(Conv2D(second_filters, kernel_size, use_bias=False))

model.add(BatchNormalization())

model.add(Activation("relu"))

model.add(Conv2D(second_filters, kernel_size, use_bias=False))

model.add(BatchNormalization())

model.add(Activation("relu"))

model.add(MaxPool2D(pool_size = pool_size))

model.add(Dropout(dropout_conv))



model.add(Conv2D(third_filters, kernel_size, use_bias=False))

model.add(BatchNormalization())

model.add(Activation("relu"))

model.add(Conv2D(third_filters, kernel_size, use_bias=False))

model.add(BatchNormalization())

model.add(Activation("relu"))

model.add(MaxPool2D(pool_size = pool_size))

model.add(Dropout(dropout_conv))



#model.add(GlobalAveragePooling2D())

model.add(Flatten())

model.add(Dense(256, use_bias=False))

model.add(BatchNormalization())

model.add(Activation("relu"))

model.add(Dropout(dropout_dense))

model.add(Dense(1, activation = "sigmoid"))



# Compile the model

model.compile(Adam(0.01), loss = "binary_crossentropy", metrics=["accuracy"])
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

earlystopper = EarlyStopping(monitor='val_loss', patience=2, verbose=1, restore_best_weights=True)

reducel = ReduceLROnPlateau(monitor='val_loss', patience=1, verbose=1, factor=0.1)

history = model.fit_generator(train_gen, steps_per_epoch=train_steps, 

                    validation_data=val_gen,

                    validation_steps=val_steps,

                    epochs=13,

                   callbacks=[reducel, earlystopper])
from sklearn.metrics import roc_curve, auc, roc_auc_score

import matplotlib.pyplot as plt



# make a prediction

y_pred_keras = model.predict_generator(test_gen, steps=len(df_val), verbose=1)

fpr_keras, tpr_keras, thresholds_keras = roc_curve(test_gen.classes, y_pred_keras)

auc_keras = auc(fpr_keras, tpr_keras)

auc_keras
plt.figure(1)

plt.plot([0, 1], [0, 1], 'k--')

plt.plot(fpr_keras, tpr_keras, label='area = {:.3f}'.format(auc_keras))

plt.xlabel('False positive rate')

plt.ylabel('True positive rate')

plt.title('ROC curve')

plt.legend(loc='best')

plt.show()
base_test_dir = '../input/test/'

test_files = glob(os.path.join(base_test_dir,'*.tif'))

submission = pd.DataFrame()

file_batch = 5000

max_idx = len(test_files)

for idx in range(0, max_idx, file_batch):

    print("Indexes: %i - %i"%(idx, idx+file_batch))

    test_df = pd.DataFrame({'path': test_files[idx:idx+file_batch]})

    test_df['id'] = test_df.path.map(lambda x: x.split('/')[3].split(".")[0])

    test_df['image'] = test_df['path'].map(imread)

    K_test = np.stack(test_df["image"].values)

    K_test = (K_test - K_test.mean()) / K_test.std()

    predictions = model.predict(K_test)

    test_df['label'] = predictions

    submission = pd.concat([submission, test_df[["id", "label"]]])

submission.head()

#submission





shutil.rmtree(train_path)

shutil.rmtree(valid_path)

submission.to_csv("submission.csv", index = False, header = True)
pd.read_csv("submission.csv")