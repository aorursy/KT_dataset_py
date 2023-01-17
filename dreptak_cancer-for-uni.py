import gc
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time

from glob import glob
from os.path import join
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from tensorflow.python import keras
from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array

print(matplotlib.__version__)
# os.listdir('../input')
image_size = (224, 224)
kernel_size = (3, 3)

images_dirs = ['../input/ham10000_images_part_1',
                '../input/ham10000_images_part_2']

metadata_df = pd.read_csv('../input/HAM10000_metadata.csv')
# display(metadata_df)
classes = metadata_df.dx.value_counts()
print(classes.index)
classes_nums = dict(enumerate(classes.index))
nums_classes = {v: k for k, v in classes_nums.items()}

# print(classes_nums)
# print(nums_classes)
num_classes = classes.shape[0]
print("Number of classes = %d" % num_classes)
def preprocess(images_list):
    images = map(lambda i: load_img(i, target_size=(*image_size,)), images_list)
    images_nparray = np.array(list(map(img_to_array, images)))
    
    output = preprocess_input(images_nparray)
    del images
    del images_nparray
    gc.collect()
    
    return output
images_paths = [p for p in glob(join('../input', '*', '*.jpg'))]
print(len(images_paths))
start_time = time.time()

# trim to save memory :(
train_end_idx = int(len(images_paths)*0.2)
validation_end_idx = train_end_idx + int(len(images_paths)*0.1)
train_images_paths = images_paths[:train_end_idx]
validation_images_paths = images_paths[train_end_idx:validation_end_idx]

train_data = preprocess(train_images_paths)
validation_data = preprocess(validation_images_paths)

elapsed_time = time.time() - start_time
print("Took %ds" % elapsed_time)
# test_data.shape
file_names = [os.path.splitext(os.path.basename(f))[0] for f in images_paths]

ym = metadata_df[['image_id', 'dx']].set_index('image_id').reindex(file_names)
ym.replace({'dx': nums_classes}, inplace=True)
# ym.head()

y = keras.utils.to_categorical(ym['dx'], num_classes)
train_y = y[:train_end_idx]
validation_y = y[train_end_idx:validation_end_idx]
# display(y)
fig, axs = plt.subplots(4, 3, figsize=(20, 20))

for idx, ax in enumerate(axs.flat):
    ax.imshow(mpimg.imread(images_paths[idx]))
    ax.set_title(ym.iloc[idx].loc['dx'])
    ax.grid(True)

plt.show()
model = Sequential()

model.add(Conv2D(20, kernel_size=(*kernel_size,),
                 activation='relu',
                 input_shape=(*train_data.shape[1:],)))
model.add(Conv2D(20, kernel_size=(*kernel_size,), activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])
model.fit(train_data, train_y,
          batch_size=128,
          epochs=2,
          validation_split = 0.2)
print(train_data.shape)
nsamples, nx, ny, nz = train_data.shape
train_data_model2 = train_data.reshape((nsamples, nx*ny*nz))

print(validation_data.shape)
vnsamples, vnx, vny, vnz = validation_data.shape
validation_data_model2 = validation_data.reshape((vnsamples, vnx*vny*vnz))
try:
    del model2
except:
    pass
try:
    del model
except:
    pass
try:
    del train_data
except:
    pass
gc.collect()

model2 = SVC(kernel='linear')

model2.fit(train_data_model2, ym.dx.iloc[:train_end_idx].ravel())
model2_preds = model2.predict(validation_data_model2)
accuracy_score(model2_preds, ym.dx.iloc[train_end_idx:validation_end_idx])