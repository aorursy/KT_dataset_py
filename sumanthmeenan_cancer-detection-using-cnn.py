cd
import os

os.listdir('/tmp/../')
import numpy as np

import pandas as pd

import os

import cv2

import matplotlib.pyplot as plt

from IPython.display import display

print('Files in this directory:',os.listdir('../kaggle/input'))
def f(x):

    display(x)
images_list = os.listdir("../kaggle/input/train/")

print('Total number of training images:',len(images_list))
train_labels_df = pd.read_csv("../kaggle/input/train_labels.csv")

print("Total number of labels for training images: ",len(train_labels_df))
train_labels_df.columns.tolist()
print('First image id in training images:',images_list[0])

print("First image id in taining_labels csv:", train_labels_df.iloc[0,0])
img = cv2.imread('../kaggle/input/train/'+ images_list[0]) #opencv color order BGR

rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img)     #opencv BGR format   
print('shape of img:',img.shape)
plt.imshow(rgb_img) #matplotlib color order RGB
print(type(train_labels_df.iloc[0,1]))

train_labels_df.iloc[0,1]
print(type(train_labels_df.iloc[:,1]))

train_labels_df.iloc[:,1]
total_images = train_labels_df.iloc[:,0].tolist()

print('Total no of images: ', len(total_images))

non_tumor_images = train_labels_df[train_labels_df.iloc[:,1] == 0]['id'].tolist()

print('No. of non-tumor images:',len(non_tumor_images))

tumor_images = train_labels_df[train_labels_df.iloc[:,1] == 1]['id'].tolist()

print('No. of tumor images:',len(tumor_images))

train_labels_df['label'].value_counts()
# '.tif' is not there at the end of image ids in train_label.csv

tum_img = cv2.imread('../kaggle/input/train/' + tumor_images[550] + '.tif')

tum_img_grey = cv2.imread('../kaggle/input/train/' + tumor_images[550] + '.tif', cv2.IMREAD_GRAYSCALE)

tum_img = cv2.cvtColor(tum_img, cv2.COLOR_BGR2RGB)



non_tum_img = cv2.imread('../kaggle/input/train/' + non_tumor_images[250] + '.tif')

non_tum_img = cv2.cvtColor(non_tum_img, cv2.COLOR_BGR2RGB)



plt.imshow(tum_img)
plt.imshow(tum_img_grey)
tum_img_grey.shape
plt.imshow(tum_img_grey, cmap = 'gray')
non_tum_img.shape
plt.imshow(non_tum_img)
#AfterWork-  write functions given an 'img id' it shld return 0/1
print(images_list[0])

print(total_images[0])
train_labels_df.iloc[:,0] = [train_labels_df.iloc[:,0][i] + '.tif' for i in range(len(train_labels_df.iloc[:,0]))]
train_labels_df.head()
new_df = train_labels_df.copy()

new_df.head()
train_images = new_df.iloc[:,0].tolist()
# column_names = [f'p{i}' for i in range(1, 48*48 +1)]

# column_names[-1]

# df1 = pd.DataFrame(columns = column_names)

# df1.head()
# # df1['index'] = [i for i in range(0,220025)]

# # df1['index'] = list(range(0,120025))

# df1['index']

# f(df1.head())

# f(df1.tail())
p = cv2.imread('../kaggle/input/train/' + train_images[0], cv2.IMREAD_GRAYSCALE)

plt.imshow(p, cmap='gray')

p.shape
plt.imshow(p[31:79,31:79], cmap = 'gray')

p[31:79,31:79].shape
def crop(img):

    image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)

    crop_img = image[31:79,31:79]

    return crop_img

    
x = crop('../kaggle/input/train/'+train_images[0])

print('shape of x is:', x.shape)
from numpy import newaxis

def crop_image(img):

    image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)

    crop_img = image[31:79,31:79]

    crop_img = crop_img[:,:,newaxis]    

    return crop_img
eg = crop_image('../kaggle/input/train/' + train_images[0])

print('shape of this image is:', eg.shape)

eg
from keras.models import Sequential

from keras.layers import Convolution2D

from keras.layers import MaxPooling2D

from keras.layers import Flatten

from keras.layers import Dense

from keras.layers import Dropout


classifier = Sequential()

classifier.add(Convolution2D(32,3,3, input_shape = (48,48,1), activation = 'relu'))

classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Convolution2D(64,3,3, activation = 'relu'))

classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Convolution2D(128,3,3, activation = 'relu'))

classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Convolution2D(256,3,3, activation = 'relu'))

classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Flatten())

classifier.add(Dense(output_dim = 256, activation = 'relu'))

classifier.add(Dense(output_dim = 128, activation = 'relu'))

classifier.add(Dense(output_dim = 64, activation = 'relu'))

classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))



classifier.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=['accuracy'])

# def process(img):

#     img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)

#     img.resize((img.shape[0]*img.shape[1]))

# #     return img
# x = process('../kaggle//input/train/' + train_images[0])

# print('x from (96,96) shape to (96*96): ',x)
# for i in range(220025):

#     row = process('../input/train/' + train_images[i])

#     df1.loc[i] = row
from sklearn.utils import shuffle

shuffled_data =shuffle(new_df)
shuffled_data.head()
len(shuffled_data)
features = [crop_image('../kaggle/input/train/' + i) for i in  shuffled_data.iloc[:,0].tolist()]

print('Input features is a list containing ' + str(len(features)) + ' arrays')
print('shape of each example is:',features[0].shape)
labels = shuffled_data.iloc[:,1].tolist()

print("labels is a list of ouput labels containing 0's and 1's")
from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(features, labels, test_size=0.3)
print('Length of x_train', len(x_train))

print('Length of y_train', len(y_train))

print('Length of x_val', len(x_val))

print('Length of y_val', len(y_val))
print('No. of tumor examples in y_train =', sum(y_train))

print('No. of non-tumor examples in y_train =', len(y_train) - sum(y_train))

print('No. of tumor examples in y_val =', sum(y_val))

print('No. of non-tumor examples in y_val =', len(y_val) - sum(y_val))
print('type of x_train is:', type(x_train))

print('type of y_train is:', type(y_train))

print('type of x_val is:', type(x_val))

print('type of y_val is:', type(y_val))
x_train = np.array(x_train)/255

x_val = np.array(x_val)/255
x_train
print('shape of x_train is:', x_train.shape)

print('shape of x_val is:', x_val.shape)


from keras.preprocessing.image import ImageDataGenerator



datagen = ImageDataGenerator(shear_range = 0.2, zoom_range = 0.2,

                   horizontal_flip = True)

classifier.fit_generator(datagen.flow(x_train, y_train, batch_size=20),

                    steps_per_epoch=len(x_train) / 20, epochs = 70)
# c = np.reshape(y, y.shape + (1,))  -> (48,48) to (48,48,1)
weights = classifier.weights

weights
predictions = classifier.predict(x_val)

predictions
predictions.shape
predictions.resize(66008,)
predictions
y_val_pred = list(predictions)
y_val_predicted = [1  if i > 0.5 else 0 for i in y_val_pred]

y_val_predicted2 = [1  if i > 0.1 else 0 for i in y_val_pred]



sum(y_val_predicted)
sum(y_val_predicted2)
np.unique(y_val_pred)
from sklearn.metrics import confusion_matrix

confusion_matrix(y_val, y_val_predicted)
from sklearn.metrics import accuracy_score

accuracy_score(y_val, y_val_predicted)
x_val2 = x_val

datagen.fit(x_val2)
predictions2 = classifier.predict(x_val2)

predictions2
sum(predictions2)
np.unique(predictions2)
predictions2.shape
predictions2.resize(66008,)

predictions2.shape
y_val_pred3 = list(predictions2)
y_val_predicted3 = [1  if i > 0.5 else 0 for i in y_val_pred3]
sum(y_val_predicted3)
np.unique(y_val_predicted3)
from sklearn.metrics import confusion_matrix

confusion_matrix(y_val, y_val_predicted3)
from sklearn.metrics import accuracy_score

accuracy_score(y_val, y_val_predicted3)
classifier2 = Sequential()



classifier2.add(Convolution2D(32,3,3, input_shape = (48,48,1), activation = 'relu'))

classifier2.add(MaxPooling2D(pool_size=(2,2)))

classifier2.add(Dropout(0.2))



classifier2.add(Convolution2D(64,3,3, activation = 'relu'))

classifier2.add(MaxPooling2D(pool_size=(2,2)))

classifier2.add(Dropout(0.2))

classifier2.add(Convolution2D(128,3,3, activation = 'relu'))

classifier2.add(MaxPooling2D(pool_size=(2,2)))

classifier2.add(Dropout(0.2))



classifier2.add(Convolution2D(256,3,3, activation = 'relu'))

classifier2.add(MaxPooling2D(pool_size=(2,2)))

classifier2.add(Dropout(0.2))



classifier2.add(Flatten())

classifier2.add(Dense(output_dim = 256, activation = 'relu'))

classifier2.add(Dropout(0.2))

classifier2.add(Dense(output_dim = 128, activation = 'relu'))

classifier2.add(Dropout(0.2))

classifier2.add(Dense(output_dim = 64, activation = 'relu'))

classifier2.add(Dropout(0.2))

classifier2.add(Dense(output_dim = 1, activation = "sigmoid"))



classifier2.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=['accuracy'])



classifier2.summary()
from keras.preprocessing.image import ImageDataGenerator



datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2,

                   horizontal_flip = True)

datagen.fit(x_train)

classifier2.fit_generator(datagen.flow(x_train, y_train, batch_size=20),

                    steps_per_epoch=len(x_train) / 20, epochs = 20)
weights2 = classifier2.weights

weights2
predictions4 = classifier.predict(x_val)

predictions4
predictions4.resize(66008,)

predictions4
y_val_pred4 = list(predictions4)

y_val_predicted4 = [1  if i > 0.5 else 0 for i in y_val_pred4]

sum(y_val_predicted4)

np.unique(y_val_pred4)
from sklearn.metrics import confusion_matrix

confusion_matrix(y_val, y_val_predicted4)
from sklearn.metrics import accuracy_score

accuracy_score(y_val, y_val_predicted4)
test_images = os.listdir("../kaggle/input/test/")

print('Total number of training images:',len(test_images))

test_features = [crop_image('../kaggle/input/test/' + i) for i in  test_images]

test_features = np.array(test_features)

test_features = test_features/255

test_features[0]

test_features.shape
test_preds = classifier.predict(test_features)

test_preds.resize(57458,)

test_preds = list(test_preds)

test_values = [1  if i > 0.5 else 0 for i in test_preds]
sum(test_values)
df = {}

df['id'] = [i.split('.')[0] for i in test_images]

df['label'] = test_preds
test_images[0]
import pandas as pd
DataFrame = pd.DataFrame.from_dict(df)

DataFrame
submission_file = pd.read_csv('../kaggle/input/sample_submission.csv')
submission = DataFrame

submission.to_csv('predictions.csv', columns=['label']) 
submission.head()