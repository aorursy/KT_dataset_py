# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import cv2
sns.set(color_codes=True)
datadir = "/kaggle/input/siim-isic-melanoma-classification/"
df_train = pd.read_csv(datadir + "train.csv")
df_test = pd.read_csv(datadir + "test.csv")
df_train.head()
labels = ['benign', 'malignant']

sns.catplot(x="sex", y="age_approx", hue="benign_malignant", kind="bar", data=df_train);

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

import os
df_train['image_name'].head()

df_t_0 = df_train[df_train['target']==0].sample(3500)
df_t_0['target'] = "b"
df_t_1 = df_train[df_train['target']==1]
df_t_1['target'] = "m"
df_train = pd.concat([df_t_0, df_t_1])
df_train = train.reset_index()
del df_t_0
del df_t_1
print(len(df_train))
df_train['image_name'] = df_train['image_name'].apply(lambda x: os.path.join(datadir, "jpeg/" + "train/" + x + ".jpg"))
df_train.head()
img_arr = cv2.imread(df_train['image_name'][0])
def resize(arr):
    arr = cv2.resize(arr, (256,256))
    arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    return arr
plt.imshow(resize(img_arr))
plt.show()
X_train, X_val, y_train, y_val = train_test_split(df_train['image_name'], df_train['target'], test_size = 0.2, random_state = 1)
X_train.head()
X_train = pd.concat([X_train, y_train], axis = 1)
X_train.head()
X_val = pd.concat([X_val, y_val], axis = 1)
X_val.head()
train_image_generator = ImageDataGenerator(rescale=1./255, 
                                           rotation_range=10,
                                           width_shift_range=0.2,
                                           height_shift_range=0.2,
                                           shear_range=0.2,
                                           zoom_range=0.2,
                                           horizontal_flip=True,
                                           vertical_flip=True)
validation_image_generator = ImageDataGenerator(rescale=1./255)
batch_size = 128
img_size = 256
train_data_gen = train_image_generator.flow_from_dataframe(X_train,
                                                           x_col = "image_name",
                                                           y_col = "target",
                                                           batch_size=batch_size,
                                                           shuffle=True,
                                                           target_size=(img_size, img_size),
                                                           class_mode='raw')

val_data_gen = validation_image_generator.flow_from_dataframe(X_val,
                                                           x_col = "image_name",
                                                           y_col = "target",
                                                           batch_size=batch_size,
                                                           shuffle=True,
                                                           target_size=(img_size, img_size),
                                                           class_mode='raw')
from tensorflow.keras.layers import Input, Activation, BatchNormalization
from tensorflow.keras import Model

X_input = Input((img_size,img_size, 3))
X = Conv2D(16, (3, 3), padding = "same", name='conv1')(X_input)
X = BatchNormalization()(X)
X = Activation("relu")(X)
X = MaxPooling2D((2,2), padding="valid")(X)

X = Conv2D(32, (3, 3), padding = "same", name='conv2')(X)
X = BatchNormalization()(X)
X = Activation("relu")(X)
X = MaxPooling2D((2,2), padding="valid")(X)

X = Conv2D(80, (3, 3), padding = "same", name='conv3')(X)
X = BatchNormalization()(X)
X = Activation("relu")(X)
X = MaxPooling2D((2,2), padding="valid")(X)

X = Conv2D(256, (3, 3), padding = "same", name='conv4')(X)
X = BatchNormalization()(X)
X = Activation("relu")(X)
X = MaxPooling2D((2,2), padding="valid")(X)

X = Conv2D(512, (3, 3), padding = "same", name='conv5')(X)
X = BatchNormalization()(X)
X = Activation("relu")(X)
X = MaxPooling2D((2,2), padding="valid")(X)

X = Conv2D(1024, (3, 3), padding = "same", name='conv6')(X)
X = BatchNormalization()(X)
X = Activation("relu")(X)
X = MaxPooling2D((2,2), padding="valid")(X)

X = Flatten()(X)

X = Dense(4096)(X)
X = Activation("relu")(X)
X = Dense(2048)(X)
X = Activation("relu")(X)
X_out = Dense(1, activation="sigmoid")(X)

model = Model(inputs=X_input, outputs=X_out, name='nenet')
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
total_train = X_train.shape[0]
total_val = X_val.shape[0]
epochs = 3
history = model.fit(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=total_val // batch_size
)
df_test = pd.read_csv(datadir + "test.csv")
df_test['image_name'] = df_test['image_name'].apply(lambda x: os.path.join(datadir, "jpeg/" + "test/" + x + ".jpg"))
df_test.head()
test_image_generator = ImageDataGenerator(rescale=1./255)

test_data_gen = validation_image_generator.flow_from_dataframe(df_test,
                                                           x_col = "image_name",
                                                           y_col = None,
                                                           batch_size=batch_size,
                                                           shuffle=False,
                                                           target_size=(img_size, img_size),
                                                           class_mode=None)
predict = model.predict(test_data_gen, steps=df_test.shape[0]//batch_size)
answer = np.array(predict)
print(answer.shape)
for i in range(len(answer)):
    answer[i]=answer[i][0]
answer = list(answer)
df_test1 = pd.read_csv(datadir + "test.csv")
df_answer = pd.DataFrame(answer, columns=['target'])
df_image = df_test1['image_name']

df_final = pd.concat([df_image, df_answer], axis = 1)
df_final.head()
df_final.to_csv('submission.csv', header=True, index=False)
#model.save('nenet')
