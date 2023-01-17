# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np
import matplotlib.pyplot as plt
import os 
import cv2
import random
import keras
from keras import preprocessing
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from tokenize import tokenize
from os import listdir
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn.model_selection import cross_validate, train_test_split
from tqdm import tqdm    #Helps in visualization
from random import shuffle #to shuffle the images 

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory


# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
training_set_path = "../input/dogs-vs-cats/train/train"
testing_set_path = "../input/dogs-vs-cats/test/test"
CATEGORIES = ['cat','dog']
IMG_SIZE = 224
ALL_DATA = os.listdir(training_set_path)[:10000]
print(len(ALL_DATA))

def label_img(img): 
    word_label = img.split('.')[-3]
    if word_label == 'cat': return [1,0]
    elif word_label == 'dog': return [0,1]
def create_train_data():
    training_data = []
    for img in tqdm(ALL_DATA):
        label = label_img(img)
        path = os.path.join(training_set_path,img)
        img = cv2.imread(path,cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        training_data.append([np.array(img),np.array(label)])
    shuffle(training_data)
    return training_data
labels = []
for i in ALL_DATA:
    if 'dog' in i:
        labels.append(1)
    else:
        labels.append(0)

sns.countplot(labels)
plt.title('Cats and Dogs')
train = create_train_data()
X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,3)
Y = np.array([i[1] for i in train])
def get_vgg_model():
  model = keras.applications.VGG16(include_top=False,weights="imagenet",input_shape=(IMG_SIZE,IMG_SIZE,3))
  for layer in model.layers:
    layer.trainable = False
  flat1 = keras.layers.Flatten()(model.layers[-1].output)
  class2 = keras.layers.Dropout(0.4)(flat1)
  class3 = keras.layers.Dense(128, activation='relu')(class2)
  output = keras.layers.Dense(len(CATEGORIES), activation='softmax')(class3)
  model = keras.Model(inputs=model.inputs, outputs=output)
  opt = keras.optimizers.SGD(lr=0.001, momentum=0.9)
  model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy , metrics=['accuracy'])
  return model
vgg_model = get_vgg_model()
vgg_model.summary()
vgg_model.fit(x=X, y=Y,epochs=5, batch_size=32, validation_split=0.2, verbose=1)
def load_image(filename):
	# load the image
	img = keras.preprocessing.image.load_img(filename, target_size=(IMG_SIZE,IMG_SIZE), color_mode='rgb')
	# convert to array
	img = keras.preprocessing.image.img_to_array(img)
	return img

def preprocess_image(img):
  img = img.reshape(1,224, 224, 3)
  return img


def predict_samples(model,from_n=0):
  plt.figure(figsize=(10,10))
  for i in range(1,25):
      filename = f'{testing_set_path}/{i+from_n}.jpg'
      img = load_image(filename)
      plt.subplot(5,5,i+1)
      plt.xticks([])
      plt.yticks([])
      plt.grid(False)
      plt.imshow(keras.preprocessing.image.load_img(filename, color_mode='rgb'))
      predict = model.predict(preprocess_image(img))
      prediction = np.argmax(predict, axis=1)
      plt.rcParams.update({'font.size': 20}) 
      plt.xlabel(CATEGORIES[prediction[0]])
  plt.show()
predict_samples(vgg_model)
# preparing test data 
test_filenames = os.listdir(testing_set_path)
test_df = pd.DataFrame({
    'filename': test_filenames
})
nb_samples = test_df.shape[0]
print(test_df.shape)
test_gen = preprocessing.image.ImageDataGenerator(featurewise_center=True)
test_gen.mean = [123.68, 116.779, 103.939]
test_generator = test_gen.flow_from_dataframe(
    test_df, 
    testing_set_path, 
    x_col='filename',
    y_col=None,
    class_mode=None,
    batch_size=32,
    target_size=(IMG_SIZE, IMG_SIZE),
    shuffle=False
)
predict = vgg_model.predict_generator(test_generator, steps=np.ceil(nb_samples/32))
test_df['category'] = np.argmax(predict, axis=1)
# see predicted results 
sample_test = test_df.sample(n=9).reset_index()
sample_test.head()
plt.figure(figsize=(12, 12))
for index, row in sample_test.iterrows():
    filename = row['filename']
    category = row['category']
    img = keras.preprocessing.image.load_img(testing_set_path+'/' +filename, target_size=(224, 224))
    plt.subplot(3, 3, index+1)
    plt.imshow(img)
    plt.xlabel(filename + '(' + "{}".format(category) + ')')
plt.tight_layout()
plt.show()
submission_df = test_df.copy()
submission_df['id'] = submission_df['filename'].str.split('.').str[0]
submission_df['label'] = submission_df['category']
submission_df.drop(['filename', 'category'], axis=1, inplace=True)
submission_df.to_csv('submission_13010030.csv', index=False)

plt.figure(figsize=(10,5))
sns.countplot(submission_df['label'])
plt.title("(Test data)")