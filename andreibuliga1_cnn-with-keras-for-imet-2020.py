# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import seaborn as sns
import matplotlib.pyplot as plt

labels = pd.read_csv('../input/imet-2020-fgvc7/labels.csv')
trainset = pd.read_csv('../input/imet-2020-fgvc7/train.csv',dtype='str')
testset = pd.read_csv('../input/imet-2020-fgvc7/sample_submission.csv',dtype='str')
labels.sample(10).head(10)
trainset.sample(10).head(10)
testset.sample(10).head(10)
import cv2
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras import optimizers
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, Activation, BatchNormalization
from PIL import Image
%matplotlib inline
sns.set(style="whitegrid")
warnings.filterwarnings("ignore")
from numpy.random import seed
from tensorflow import random
random.set_seed(0)
seed(0)
print('Number of training samples:',len(trainset))
print('Number of test samples:',len(testset))
print('Number of labels;',len(labels))
attribute_ids = trainset['attribute_ids'].values
# Get attributes for each images
attributes = []
for item_attributes in [x.split(' ') for x in attribute_ids]:
    for attribute in item_attributes:
        attributes.append(int(attribute))
attr_pd = pd.DataFrame(attributes, columns=['attribute_id'])
attr_pd = attr_pd.merge(labels)
attr_pd
top30 = attr_pd['attribute_name'].value_counts().to_frame()
top30=top30[:30]
unique_attr = attr_pd['attribute_id'].nunique()
print('Number of unique attributes:',unique_attr)
plt.subplots(figsize=(11,8))
ax = sns.barplot(y=top30.index,x='attribute_name',data=top30,order=reversed(top30.index),palette='rocket')
plt.ylabel('Surface type')
plt.xlabel('Count')
sns.despine()

attr_pd['tag'] = attr_pd['attribute_name'].apply(lambda x:x.split('::')[0])
group_attr = attr_pd.groupby('tag').count()
print('Number of attribute groups:',attr_pd['tag'].nunique())
plt.subplots(figsize=(12,8))
ax=sns.barplot(y=group_attr.index,x='attribute_name',data=group_attr,palette='rocket')
plt.ylabel('Attribute Group')
plt.xlabel('Count')
sns.despine()
trainset['Number of tags']=trainset['attribute_ids'].apply(lambda x:len(x.split(' ')))
trainset
sns.countplot(x='Number of tags',data=trainset,palette='rocket')
plt.ylabel('Surface type')
sns.despine()
c = 1
plt.figure(figsize=[16,16])
for img_name in os.listdir("../input/test/")[:16]:
    img = cv2.imread("../input/test/{}".format(img_name))[...,[2,1,0]]
    plt.subplot(4,4,c)
    plt.imshow(img)
    plt.title("test image {}".format(c))
    c += 1
plt.show();
sns.set_style('white')
plt.figure(figsize=[22,20])
count=1
for img_name in os.listdir('../input/imet-2020-fgvc7/train/')[:36]:
    img = cv2.imread('../input/imet-2020-fgvc7/train/%s'%img_name)
    plt.subplot(6,6,count)
    plt.imshow(img)
    plt.title('Item %s'%count)
    count+=1

def append_ext(fn):
    return fn+".png"
trainset["id"]=trainset["id"].apply(append_ext)
testset["id"]=testset["id"].apply(append_ext)

testset
BATCH_SIZE = 128
EPOCHS = 30
LEARNING_RATE = 0.0001
HEIGHT = 64
WIDTH = 64
CANAL = 3
N_CLASSES = unique_attr
classes = list(map(str,range(N_CLASSES)))
model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(5,5),padding='Same', input_shape=(HEIGHT, WIDTH, CANAL)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(filters=32, kernel_size=(5,5),padding='Same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.5))

model.add(Conv2D(filters=64, kernel_size=(4,4),padding='Same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(filters=64, kernel_size=(4,4),padding='Same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(1024))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(N_CLASSES, activation="sigmoid"))
model.summary()

optimizer = optimizers.adam(lr=LEARNING_RATE)
model.compile(optimizer=optimizer , loss="binary_crossentropy", metrics=["accuracy"])
datagen=ImageDataGenerator(rescale=1./255., validation_split=0.25)
test_datagen = ImageDataGenerator(rescale=1./255.)

train_generator=datagen.flow_from_dataframe(
    dataframe=trainset,
    directory='/kaggle/input/imet-2020-fgvc7/train/',
    x_col='id',
    y_col='attribute_ids',
    batch_size=BATCH_SIZE,
    shuffle=True,
    class_mode="categorical",    
    classes=classes,
    target_size=(HEIGHT, WIDTH),
    subset='training')

valid_generator=datagen.flow_from_dataframe(
    dataframe=trainset,
    directory='/kaggle/input/imet-2020-fgvc7/train/',
    x_col='id',
    y_col='attribute_ids',
    batch_size=BATCH_SIZE,
    shuffle=True,
    class_mode="categorical",    
    classes=classes,
    target_size=(HEIGHT, WIDTH),
    subset='validation')

test_generator = test_datagen.flow_from_dataframe(
    dataframe=testset,
    x_col="id",
    directory='/kaggle/input/imet-2020-fgvc7/test/',
    target_size = (HEIGHT, WIDTH),
    batch_size = 1,
    shuffle = False,
    class_mode = None)
STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
STEP_SIZE_VAL = valid_generator.n // valid_generator.batch_size

history = model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VAL,
                    epochs=EPOCHS,
                    verbose=2)
test_generator.reset()
n_steps = len(test_generator.filenames)
preds = model.predict_generator(test_generator, steps = n_steps)
predictions = []
for pred_ar in preds:
    valid = ''
    for idx, pred in enumerate(pred_ar):
        if pred > 0.3:  # Using 0.3 as threshold
            if len(valid) == 0:
                valid += str(idx)
            else:
                valid += (' %s' % idx)
    if len(valid) == 0:
        valid = str(np.argmax(pred_ar))
    predictions.append(valid)
history.history
sns.set_style("whitegrid")
fig, (ax1, ax2) = plt.subplots(1, 2, sharex='col', figsize=(20,7))

ax1.plot(history.history['accuracy'], label='Train Accuracy')
ax1.plot(history.history['val_accuracy'], label='Validation accuracy')
ax1.legend(loc='best')
ax1.set_title('Accuracy')

ax2.plot(history.history['loss'], label='Train loss')
ax2.plot(history.history['val_loss'], label='Validation loss')
ax2.legend(loc='best')
ax2.set_title('Loss')

plt.xlabel('Epochs')
sns.despine()
plt.show()
filenames=test_generator.filenames
results=pd.DataFrame({'id':filenames, 'attribute_ids':predictions})
results['id'] = results['id'].map(lambda x: str(x)[:-4])
results.to_csv('submission.csv',index=False)
results.sample(10).head(10)
