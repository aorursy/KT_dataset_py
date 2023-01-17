import pandas as pd

import numpy as np

import matplotlib.image as mpimg

import time

from sklearn.model_selection import train_test_split



cactus_label = pd.read_csv('../input/train.csv')



#read in training set

train_img = []

train_lb = []

for i in range(len(cactus_label)):

    row = cactus_label.iloc[i]

    fileName = row['id']

    train_lb.append(row['has_cactus'])

    path = "../input/train/train/{}".format(fileName)

    im = mpimg.imread(path)

    train_img.append(im)

    

X_train, X_test, y_train, y_test = train_test_split(train_img, train_lb) 

X_train = np.array(X_train)

X_test = np.array(X_test)
cactus_label['has_cactus'].value_counts()
import pandas as pd

import os

test_img = []

sample = pd.read_csv('../input/sample_submission.csv')

folder = '../input/test/test/'

arr = []     

for i in range(len(sample)):

    row = sample.iloc[i]

    fileName = row['id']

    path = folder + fileName

    img = mpimg.imread(path)

    

    count0 =0

    count1 = 0

    count2 = 0

    for j in range(len(img)):

        for k in range(len(img[j])):

            for l in range(len(img[j][k])):

                if max(img[j][k]) == img[j][k][l]:

                    if l == 0:

                        count0 += 1

                    elif l == 1:

                        count1 += 1

                    else:

                        count2 += 1

                

#         print(max(0,img[0][i][0]), i)

    

    arr.append([count0, count1, count2])

    test_img.append(img)



print(arr)

    



# print(max(img[0][0]))

# print(len(img[0][0]))

    

#     print()

#     print(img[1])

                     

test_img = np.asarray(test_img)

# print(test_img[0])

# column = ['R','G','B']

# test_img2 = []

sampleTest = pd.read_csv('../input/train.csv')

folder1 = '../input/train/train/'

arr1 = []     

for i in range(len(sampleTest)):

    row1 = sampleTest.iloc[i]

    fileName1 = row1['id']

    path1 = folder1 + fileName1

    img1 = mpimg.imread(path1)

    

    count01 =0

    count11 = 0

    count21 = 0

    for j in range(len(img1)):

        for k in range(len(img1[j])):

            for l in range(len(img1[j][k])):

                if max(img1[j][k]) == img1[j][k][l]:

                    if l == 0:

                        count01 += 1

                    elif l == 1:

                        count11 += 1

                    else:

                        count21 += 1

                

#         print(max(0,img[0][i][0]), i)

    

    arr1.append([count01, count11, count21])

#     test_img2.append(img1)



print(arr1)

    



# print(max(img[0][0]))

# print(len(img[0][0]))

    

#     print()

#     print(img[1])

                     

# test_img2 = np.asarray(test_img2)
import numpy as np

import pandas as pd

# create an example array

# a = np.arange(12).reshape([3,4])

'''

# convert it to stacked format using Pandas

stacked = pd.Panel(arr)

stacked.columns = ['x', 'y']

# save to disk

stacked.to_csv('stacked.csv', index=False)

'''

# df = pd.read_csv("stacked.csv")

# print(df)

# print(arr[0])



import csv







with open("new_file.csv","w+") as my_csv:

    csvWriter = csv.writer(my_csv,delimiter=',')

    csvWriter.writerow(["Red", "Green", "Blue"])

    csvWriter.writerows(arr)

    

df = pd.read_csv("new_file.csv")

print(df)
with open("new_file1.csv","w+") as my_csv1:

    csvWriter = csv.writer(my_csv1,delimiter=',')

    csvWriter.writerow(["Red", "Green", "Blue"])

    csvWriter.writerows(arr1)

    

df1 = pd.read_csv("new_file1.csv")

print(df1)
# import the modules we'll need

from IPython.display import HTML

import pandas as pd

import numpy as np

import base64



# function that takes in a dataframe and creates a text link to  

# download it (will only work for files < 2MB or so)

def create_download_link(df, title = "Download CSV file", filename = "new_file1.csv"):  

    csv = df.to_csv()

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)



# create a random sample dataframe

df = pd.DataFrame(np.random.randn(50, 4), columns=list(''))



# create a link to download the dataframe

create_download_link(df)
v = open('../input/sample_submission.csv')

r = csv.reader(v)

row0 = next(r)

row0.append('red')





df2 = pd.read_csv('../input/sample_submission.csv')

print(row0)

     
# import numpy as np

# import pandas as pd

# # create an example array

# for i in range(len(arr)):

#     a = np.append(arr,arr[i])

# # convert it to stacked format using Pandas

# stacked = pd.Panel(a.swapaxes(1,2)).to_frame().stack().reset_index()

# stacked.columns = ['picture', 'red', 'green', 'blue']

# # save to disk

# stacked.to_csv('stacked.csv', index=False)

def iter_3D(matrix):

    for i in range(matrix.shape[0]):

        for j in range(matrix.shape[1]):

            for k in range(matrix.shape[2]):

                yield i, j, k



l = []



for i, j, k in iter_3D(arr):

    l.append('%d %d %d %d' %(str(indices_x(i, j, k)), str(indices_y(i, j, k)), str(indices_z(i, j, k)), str(matrix[i, j, k])))



# with open('file.csv', 'w') as f:

#     f.write("\n".join(l))

print(l)



# iter_3D(arr)
import matplotlib.pyplot as plt

# Data to plot

labels = 'Has Cactus', 'No Cacuts'

sizes = [13136, 4364]

colors = ['yellowgreen', 'lightskyblue']

 

# Plot

plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)

plt.show()
'''

Convert 3D ararys to 1D array

Paramter: a list of 3D images

Return: a list of 1D images

'''

def imageToFeatureVector(images):

    flatten_img = []

    for img in images:

        data = np.array(img)

        flattened = data.flatten()

        flatten_img.append(flattened)

    return flatten_img
from sklearn.neighbors import KNeighborsClassifier

# start = time.time()



# X_train_flatten = imageToFeatureVector(X_train)

# X_test_flatten = imageToFeatureVector(X_test)

u = imageToFeatureVector(arr)

# print(u)

column = ['Red','Green','Blue']

df2 = pd.DataFrame(u, columns=list('Red','Green','Blue'))





# import numpy as np

# import pandas as pd

# # create an example array

# a = arr

# # convert it to stacked format using Pandas

# stacked = pd.Panel(a.swapaxes(1,2)).to_frame().stack().reset_index()

# stacked.columns = ['picture', 'red', 'green', 'blue']

# # save to disk

# stacked.to_csv('stacked.csv', index=False)



knn = KNeighborsClassifier(n_neighbors = 5)

knn.fit(X_train_flatten, y_train) 

score = knn.score(X_test_flatten, y_test)



end = time.time()

# print("The run time of KNN is {:.3f} seconds".format(end-start))

# print("KNN alogirthm's test score is: {:.3f}".format(score))
import pandas as pd

import numpy as np

import matplotlib.image as mpimg

import time

from sklearn.model_selection import train_test_split

cactus_label = pd.read_csv('../input/train.csv')



#read in training set

train_img = []

train_lb = []

has_cactus = 0

no_cactus = 0

for i in range(len(cactus_label)):

    row = cactus_label.iloc[i]

    fileName = row['id'] 

    path = "../input/train/train/{}".format(fileName)

    im = mpimg.imread(path)

    if row['has_cactus'] == 1 and has_cactus < 4364:

        has_cactus+= 1

        train_lb.append(row['has_cactus'])

        train_img.append(im)

    elif row['has_cactus'] == 0 and no_cactus < 4364:

        no_cactus += 1

        train_lb.append(row['has_cactus'])

        train_img.append(im)





    

X_train, X_test, y_train, y_test = train_test_split(train_img, train_lb) 

X_train = np.array(X_train)

X_test = np.array(X_test)
import matplotlib.pyplot as plt

# Data to plot

labels = 'Has Cactus', 'No Cacuts'

sizes = [train_lb.count(1), train_lb.count(0)]

colors = ['yellowgreen', 'lightskyblue']

 

# Plot

plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)

plt.show()
from sklearn.neighbors import KNeighborsClassifier

start = time.time()



X_train_flatten = imageToFeatureVector(X_train)

X_test_flatten = imageToFeatureVector(X_test)



knn = KNeighborsClassifier(n_neighbors = 5)

knn.fit(X_train_flatten, y_train) 

score = knn.score(X_test_flatten, y_test)



end = time.time()

print("The run time of KNN is {:.3f} seconds".format(end-start))

print("KNN alogirthm's test score is: {:.3f}".format(score))
from sklearn.svm import LinearSVC



start = time.time()

linearKernel = LinearSVC().fit(X_train_flatten, y_train)

score = linearKernel.score(X_test_flatten,y_test)

end = time.time()



print("The run time of Linear SVC is {:.3f} seconds".format(end-start))

print("Linear SCV alogirthm's test score is: {:.3f}".format(score))
# try normalizing the features...

from sklearn import preprocessing

scaler = preprocessing.StandardScaler()

start = time.time()

scaler.fit(X_train_flatten)

X_test_normalized = scaler.transform(X_test_flatten)

X_train_normalized = scaler.transform(X_train_flatten)



linearKernel = LinearSVC().fit(X_train_normalized, y_train)

score = linearKernel.score(X_test_normalized,y_test)

end = time.time()

print("The run time of Linear SVC with normalized features is {:.3f} seconds".format(end-start))

print("Linear SCV with normalized features has test score of: {:.3f}".format(score))
import tensorflow as tf

start = time.time()

X_train_norm = tf.keras.utils.normalize(X_train, axis=1)

X_test_norm = tf.keras.utils.normalize(X_test, axis=1)



model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Flatten())



#add layers

model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))

model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))

model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))



#compile model

model.compile(optimizer='adam',

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])



#train 

model.fit(X_train_norm, np.array(y_train), epochs=10)



# Evaluate the model on test set

score = model.evaluate(X_test, np.array(y_test), verbose=0)

# Print test accuracy

print('\n', 'Test accuracy:', score[1])



end = time.time()

print("The run time of CNN is {:.3f} seconds".format(end-start))
from keras.models import Sequential

from keras.layers import Conv2D, Dense, Flatten, BatchNormalization, Dropout, LeakyReLU, DepthwiseConv2D, Flatten

from keras.layers.pooling import GlobalAveragePooling2D

from keras.optimizers import Adam

from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping



def create_model():

    model = Sequential()

        

    model.add(Conv2D(3, kernel_size = 3, activation = 'relu', input_shape = (32, 32, 3)))

    

    model.add(Conv2D(filters = 16, kernel_size = 3, activation = 'relu'))

    model.add(Conv2D(filters = 16, kernel_size = 3, activation = 'relu'))

    model.add(BatchNormalization())

    model.add(Dropout(0.2))

    

    model.add(DepthwiseConv2D(kernel_size = 3, strides = 1, padding = 'Same', use_bias = True))

    model.add(Conv2D(filters = 32, kernel_size = 1, activation = 'relu'))

    model.add(Conv2D(filters = 64, kernel_size = 1, activation = 'relu'))

    model.add(BatchNormalization())

    model.add(Dropout(0.2))

    

    model.add(DepthwiseConv2D(kernel_size = 3, strides = 2, padding = 'Same', use_bias = True))

    model.add(Conv2D(filters = 128, kernel_size = 1, activation = 'relu'))

    model.add(Conv2D(filters = 256, kernel_size = 1, activation = 'relu'))

    model.add(BatchNormalization())

    model.add(Dropout(0.2))

    

    model.add(DepthwiseConv2D(kernel_size = 3, strides = 1, padding = 'Same', use_bias = True))

    model.add(Conv2D(filters = 256, kernel_size = 1, activation = 'relu'))

    model.add(BatchNormalization())

    model.add(Conv2D(filters = 512, kernel_size = 1, activation = 'relu'))

    model.add(BatchNormalization())

    model.add(Dropout(0.2))

    

    model.add(DepthwiseConv2D(kernel_size = 3, strides = 2, padding = 'Same', use_bias = True))

    model.add(Conv2D(filters = 512, kernel_size = 1, activation = 'relu'))

    model.add(BatchNormalization())

    model.add(Conv2D(filters = 1024, kernel_size = 1, activation = 'relu'))

    model.add(BatchNormalization())

    model.add(Dropout(0.2))

    

    model.add(DepthwiseConv2D(kernel_size = 3, strides = 1, padding = 'Same', use_bias = True))

    model.add(Conv2D(filters = 1024, kernel_size = 1, activation = 'relu'))

    model.add(BatchNormalization())

    model.add(Conv2D(filters = 2048, kernel_size = 1, activation = 'relu'))

    model.add(BatchNormalization())

    model.add(Dropout(0.2))

    

    #model.add(GlobalAveragePooling2D())

    model.add(Flatten())

    

    model.add(Dense(470, activation = 'relu'))

    model.add(BatchNormalization())

    model.add(Dropout(0.5))

    

    model.add(Dense(256, activation = 'relu'))

    model.add(BatchNormalization())

    model.add(Dropout(0.5))

    

    model.add(Dense(128, activation = 'tanh'))



    model.add(Dense(1, activation = 'sigmoid'))



    model.compile(optimizer = 'rmsprop', loss = 'mean_squared_error', metrics = ['accuracy'])

    

    return model
model = create_model()



history = model.fit(X_train, 

            np.array(y_train), 

            batch_size = 128, 

            epochs = 8, 

            validation_data = (X_test, np.array(y_test)),

            verbose = 1)



predictions = model.predict(X_test, verbose = 1)



# Evaluate the model on test set

score = model.evaluate(X_test, np.array(y_test), verbose=0)

# Print test accuracy

print('Test accuracy:', score[1])
scaler = preprocessing.StandardScaler()

scaler.fit(X_train_flatten)

X_test_normalized = scaler.transform(X_test_flatten)

X_train_normalized = scaler.transform(X_train_flatten)

test_flatten = imageToFeatureVector(test_img)

test_normalized = scaler.transform(test_flatten)

linearKernel = LinearSVC().fit(X_train_normalized, y_train)

predictions = linearKernel.predict(test_normalized)

sample['has_cactus'] = predictions

sample.head()
sample.to_csv('sub.csv', index= False)