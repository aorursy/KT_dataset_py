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
train = pd.read_csv("/kaggle/input/siim-isic-melanoma-classification/train.csv")
test = pd.read_csv("/kaggle/input/siim-isic-melanoma-classification/test.csv")
import seaborn as sns
import matplotlib.pyplot as plt
sns.distplot(train['age_approx'])
sns.barplot(train['sex'].value_counts().reset_index()['index'], train['sex'].value_counts().reset_index()['sex'] / np.sum(train['sex'].value_counts().reset_index()['sex']) * 100)
plt.title("Male and Female Melanoma Distribution")
df = train[train["benign_malignant"] == "benign"][["anatom_site_general_challenge", "diagnosis", "target"]]
g = sns.countplot(x='anatom_site_general_challenge', hue='diagnosis',data=train[train["benign_malignant"]=="benign"])
plt.xticks(rotation=270)

g = sns.countplot(x='anatom_site_general_challenge', hue='diagnosis',data=train[train["benign_malignant"]=="malignant"])
plt.xticks(rotation=270)

import cv2
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Flatten, Conv2D, MaxPool2D, Activation, Dense, Dropout
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.preprocessing import image
import tensorflow as tf
device_name = tf.test.gpu_device_name()
if "GPU" not in device_name:
    print("GPU device not found")
print('Found GPU at: {}'.format(device_name))
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True 
sess = tf.compat.v1.Session(config=config) 
tf.compat.v1.keras.backend.set_session(sess)


train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

img = cv2.imread('/kaggle/input/siim-isic-melanoma-classification/jpeg/train/ISIC_0077472.jpg')
plt.imshow(img)
train['image'] = [s + ".jpg" for s in train["image_name"]]

train_directory = '/kaggle/input/siim-isic-melanoma-classification/jpeg/train/'
test_directory = '/kaggle/input/siim-isic-melanoma-classification/jpeg/test'
train['target_string'] = train['target'].astype(str)
train_df, validate_df = train_test_split(train, test_size=0.20, random_state=42)

train_generator = train_datagen.flow_from_dataframe(
    train_df,
    directory = train_directory,
    x_col = 'image',
    y_col = 'target',
    target_size = (256,256),
    class_mode = 'raw'
)
validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_dataframe(
    validate_df, 
    directory = train_directory,
    x_col = 'image',
    y_col = 'target',
    target_size = (256,256),
    class_mode = 'raw'
)
classifier = tf.keras.models.Sequential()
classifier.add(tf.keras.layers.Conv2D(32, (3, 3), input_shape = (256, 256, 3), activation='relu'))
classifier.add(tf.keras.layers.MaxPool2D(pool_size=(4,4)))
classifier.add(tf.keras.layers.Flatten())
classifier.add(tf.keras.layers.Dense(10, activation = 'relu'))
classifier.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
classifier.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
classifier.fit_generator(train_generator, steps_per_epoch=5, epochs=2, validation_data=validation_generator, validation_steps=5)





dirname='../input/siim-isic-melanoma-classification/'
#prepare dataframe for test data
test_data = []
for i in range(len(test)):
    test_data.append(dirname + 'jpeg/test/' + test['image_name'].iloc[i] + '.jpg')
test_path = pd.DataFrame(test_data)
test_path.columns = ['images']
    #img = cv2.resize(img, (224,224))
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #img = img.astype(np.float32)/255.
    
    #img=np.reshape(img,(1,224,224,3))
#test data input pipeline
test_datagen=ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_dataframe(test_path, x_col='images', y_col=None, 
                                                   target_size = (256,256), shuffle=False, class_mode=None)
test_generator.reset()
with tf.device('/device:GPU:0'):
    preds = classifier.predict(test_generator, steps=test.shape[0]//10+1)
    ans = np.array(preds)
    print(ans.shape)
#prep recorded targets
ans=list(ans)
for i in range(len(ans)):
    ans[i]=ans[i][0]
final = {'image_name':list(test['image_name']), 'target':ans }

sub = pd.DataFrame(final, columns=['image_name', 'target'])
#save predictions
sub.to_csv('submission.csv', header=True, index=False)
