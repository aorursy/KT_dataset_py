# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2 as cv # image lining

import matplotlib.pyplot as plt # image

import os

import ast # string to list

from sklearn import model_selection # Train : Test set

from sklearn.preprocessing import OneHotEncoder # OneHotEncoding



from keras.models import Sequential # Keras

from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, pooling

from keras.utils import np_utils



plt.gray() #graycolor setting
df = pd.read_csv("../input/quickdraw-doodle-recognition/train_simplified/cat.csv")
df
def vector_to_img(vector):

    image = np.zeros((256,256), np.uint8)

    for line in vector:

        for i in range(len(line[0])-1):

            cv.line(image,(line[0][i],line[1][i]),(line[0][i+1],line[1][i+1]),color=255)

    

    temp_image= cv.resize(image, dsize=(64, 64), interpolation=cv.INTER_AREA)

    del image

    return temp_image
df['drawing'][0]
vector_img = ast.literal_eval(df['drawing'][0])

a=vector_to_img(vector_img)

plt.imshow(a)
X = []

Y = []

Label = []

Label_num = 0
def File_Preprocessing(filename,X,Y,Label_num):

    df = pd.read_csv("../input/quickdraw-doodle-recognition/train_simplified/{}".format(filename))

    for i in range(500):

        img = vector_to_img(ast.literal_eval(df['drawing'][i])).reshape(64,64,1)

        X.append(img)

        Y.append([Label_num])

        del img

    del df

for dirname, _, filenames in os.walk('/kaggle/input/quickdraw-doodle-recognition/train_simplified'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        df = pd.read_csv("../input/quickdraw-doodle-recognition/train_simplified/{}".format(filename))

        Label.append(df['word'][0])

        for i in range(500):

            img = vector_to_img(ast.literal_eval(df['drawing'][i])).reshape(64,64,1)

            X.append(img)

            Y.append([Label_num])

        Label_num = Label_num + 1

        

enc = OneHotEncoder()

enc.fit(Y)

Y = enc.transform(Y).toarray()



print("Fin")
Label
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(np.array(X), np.array(Y), test_size = 0.2)
print(X_train.shape,X_test.shape,Y_train.shape,Y_test.shape)
#normalization

X_train = X_train.astype('float32')

X_test = X_test.astype('float32')

X_train /= 255

X_test /= 255
model = Sequential()

model.add(Conv2D(32, 3, 3, activation='relu', input_shape=(64,64, 1)))

print(model.output_shape)



model.add(Conv2D(32, 3, 3, activation='relu'))

model.add(pooling.MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))

print(model.output_shape)



model.add(Conv2D(32, 3, 3, activation='relu'))

model.add(pooling.MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))

print(model.output_shape)



model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(340, activation='softmax'))

print(model.output_shape)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=400, epochs=20, verbose=1)
loss,acc = model.evaluate(X_test, Y_test, verbose=0)

print("loss = {} , acc = {}".format(loss,acc))
model.save('kmj.hdf5')
test_simplified = pd.read_csv("../input/quickdraw-doodle-recognition/test_simplified.csv")
test_simplified
test_simplified.shape
X = []
for i in range(112199):

    img = vector_to_img(ast.literal_eval(df['drawing'][i])).reshape(64,64,1)

    X.append(img)

print("Fin")
X=np.array(X)

X.shape
plt.imshow(X[0].reshape(64,64))
pred = model.predict(X, verbose=1)
top_3 = np.argsort(-pred)[:, 0:3]
top_3
top_3_Label = []

for i in top_3:

    top_3_Label.append("{} {} {}".format(Label[i[0]],Label[i[1]],Label[i[2]]))

print("Fin")
sample_submission = pd.read_csv("../input/quickdraw-doodle-recognition/sample_submission.csv",index_col=['key_id'])

sample_submission
submission = sample_submission
submission['word'] = top_3_Label
submission
submission.to_csv('submission.csv')

submission.head()