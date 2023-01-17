import cv2 as cv

import glob

from keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import  train_test_split

import numpy as np

np.random.seed(2)

filesname = glob.glob("../input/proptit-aif-homework-1/final_train/final_train/*")

# filesname.sort()

names = []

print (filesname)

for t in filesname:

  names.append(int(t.split('/')[-1]))

names.sort()



marks = {}

id = 0

for name in names:

  Str = str(name)

  marks[Str] = id

  id+=1

print(marks)

# maps = ['0', '2', '6', '10', '14', '22', '33', '34']

train_set = []

labels = []

for files in filesname:

  # print(files)

  link = glob.glob(files + "/*.png")

  print(files)

  images = [cv.imread(img) for img in link]

  lab = files.split('/')[-1]

  for img in images:

      img_gray = np.array(cv.cvtColor(img,cv.COLOR_BGR2GRAY))

      resize = cv.resize(img_gray,(64,64))

      train_set.append(resize)

      labels.append(marks[lab])



train_set = np.array(train_set).reshape(-1,64,64,1)

print(train_set.shape)
from keras.utils import to_categorical



y = to_categorical(labels, num_classes=8)

print(y.shape)

X_train, X_val, Y_train, Y_val = train_test_split(train_set, y, test_size = 0.1,

                                                  random_state=2)


X_train, X_val, Y_train, Y_val = train_test_split(train_set, y, test_size = 0.1,

                                                  random_state=2)

# data_train

data_gen = ImageDataGenerator(rescale = 1.0/255,

                              rotation_range=10,width_shift_range=0.2,

                              height_shift_range = 0.2, zoom_range = 0.3,brightness_range = [0.2,1.3])

data_gen.fit(X_train)
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout,Flatten

from keras.models import Sequential



model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (3,3), padding = 'Same', activation = 'relu',

                 input_shape = [64,64,1]))



model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Dropout(0.25))

model.add(Conv2D(filters = 64 , kernel_size = (3,3), padding = 'Same', activation = 'relu'))

model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Dropout(0.25))

model.add(Conv2D(filters = 64 , kernel_size = (3,3), padding = 'Same', activation = 'relu'))

model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Dropout(0.25))





model.add(Flatten())

model.add(Dense(units = 1024, activation = 'relu'))

model.add(Dropout(0.5))

model.add(Dense(units = 512, activation = 'relu'))

model.add(Dropout(0.5))

# model.add(Dense(units = 512, activation = 'relu'))

# model.add(Dropout(0.5))

model.add(Dense(units = 8, activation = 'softmax'))



model.compile(optimizer='adam',metrics=['accuracy'],loss='categorical_crossentropy')

model.fit_generator(data_gen.flow(X_train,Y_train,batch_size=100),

                    validation_data = (X_val,Y_val),

                    steps_per_epoch = len(X_train)/100,

                    epochs = 10)


from keras.models import load_model



# Creates a HDF5 file 'my_model.h5'

model.save('my_model1.h5')







test = []

filenames = glob.glob("../input/proptit-aif-homework-1//final_test/final_test/*.png")

images = [cv.imread(img) for img in filenames]

id= []

name = [im for im in filenames]

ten = []

for t in name:

  ten.append(t.split('/')[-1])

pixel = {}

i = 0;

maps = [0,2,6,10,14,22,33,34,39]

for img in images:

    img_gray = np.array(cv.cvtColor(img,cv.COLOR_BGR2GRAY))

    resize = cv.resize(img_gray,(64,64))

    test.append(resize)

    pixel[ten[i]] = resize

    i+=1

    # break

test = np.array(test)/255.0

test = test.reshape(-1,64,64,1)




pred = model.predict(test)

pred = np.argmax(pred,axis = 1)

predict = {}

for i in range(1000):

  predict[ten[i]] = pred[i]



import pandas as pd

sub = pd.read_csv("../input/proptit-aif-homework-1/sampleSubmission.csv")



pre = sub['class']

label = []



for i in sub['path']:



  label.append(maps[predict[i]])



dict = { 'class': label,'path': sub['path']}  

df = pd.DataFrame(dict) 

df.to_csv('../output/kaggle/working1/mySubmission.csv',index=False) 

# s = pd.read_csv("mySubmission.csv")