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
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import os
import scipy.misc
import shutil
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
import seaborn as sns
import random
labels = sorted(os.listdir('../input/flowers-recognition/flowers'))
labels
labels.remove('flowers')
labels
os.mkdir('/kaggle/working/place')


for label in labels:
    fns = os.listdir('../input/flowers-recognition/flowers/{}'.format(str(label)))
    
    for fn in fns:
        shutil.copyfile('../input/flowers-recognition/flowers/{}/{}'.format(str(label),str(fn)),'/kaggle/working/place/{}'.format(str(fn)))
print(len(os.listdir('/kaggle/working/place')))


model = tf.keras.models.Sequential()
vgg16 = tf.keras.applications.VGG16(include_top = False, input_shape = (150, 150, 3))
len(vgg16.layers)

vgg16.summary()
for layer in vgg16.layers:
    model.add(layer)
for layer in model.layers[:15]:
    layer.trainable = False
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(5, activation = 'softmax'))
model.summary()

model.compile(optimizer = 'adam',
             loss = 'categorical_crossentropy',
             metrics = ['accuracy'])

fns1 = []
df_list = []
for label in labels:
        fns1 = (os.listdir('../input/flowers-recognition/flowers/{}'.format(str(label))))
        df_list.append(pd.DataFrame({'filename': fns1, 'label': str(label)}, index = [j for j in range(len(fns1))]))
        

print((df_list))
for i in range(len(df_list)):
    if (i==0) or (i == 1):
        df = pd.concat([df_list[0], df_list[1]])
    else:
        df = pd.concat([df,df_list[i] ])
df_train, df_val = train_test_split(df, test_size = 0.2, random_state = 43)
df_train = df_train.reset_index(drop = True)
df_val = df_val.reset_index(drop = True)
df_train.drop_duplicates(keep=False, inplace = True, subset='filename')
df_val.drop_duplicates(keep=False, inplace = True, subset='filename')
df_train
df_val
train_datagen = ImageDataGenerator(
    rotation_range=15,
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)
train_generator = train_datagen.flow_from_dataframe(
    df_train, 
    '/kaggle/working/place', 
    x_col='filename',
    y_col='label',
    target_size=(150, 150),
    class_mode='categorical',
    batch_size= 16
)
valid_datagen = ImageDataGenerator(rescale = 1/255)
valid_generator = valid_datagen.flow_from_dataframe(
    df_val, 
    '/kaggle/working/place', 
    x_col='filename',
    y_col='label',
    target_size=(150, 150),
    class_mode='categorical',
    batch_size= 16,
    shuffle = False
)
history=model.fit_generator(train_generator, epochs = 10, validation_data = valid_generator)
predictions = model.predict_generator(valid_generator)
pred_digits=np.argmax(predictions,axis=1)
y_true = [labels[t] for t in valid_generator.classes]
y_pred = [labels[p] for p in pred_digits]
cm1 = confusion_matrix(y_pred = y_pred, y_true = y_true, labels = labels, normalize = 'true')
ax= plt.subplot()
sns.heatmap(cm1,xticklabels=labels,yticklabels=labels, annot=True)
model.evaluate(x = valid_generator)
plt.figure(figsize=(12, 12))
fns = os.listdir('/kaggle/working/place')
for i in range(9):
    filename = random.choice(fns)
    img = load_img('/kaggle/working/place/'+filename, target_size=(150, 150))
    x = np.expand_dims(img, axis=0)
    x = x/255
    category1 = model.predict(x)
    category1 = np.argmax(category1, axis=1)
    category1 = labels[int(category1)]
    plt.subplot(3, 3, i+1)
    plt.imshow(img)
    plt.xlabel(filename + '(' + "{}".format(category1)+")")
plt.tight_layout()
plt.show()
