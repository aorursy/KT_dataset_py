import numpy as np 

import pandas as pd 

from keras.preprocessing.image import ImageDataGenerator # for data augmentation

from matplotlib import pyplot

import csv
# data preparation of training data

train = open("../input/train.csv").read()

train = train.split("\n")[1:-1]

train = [i.split(",") for i in train]

X_train = np.array([[int(i[j]) for j in range(1,len(i))] for i in train])

y_train = np.array([int(i[0]) for i in train])
# for the visualization, in this notebook, we only use the first 9 images

X_train = X_train[0:9]

y_train = y_train[0:9]
for i in range(0, 9):

        pyplot.subplot(3,3,i+1)

        pyplot.imshow(X_train[i].reshape((28, 28)))

pyplot.show()
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)

X_train = X_train.astype('float32')
filename = "new_image_data.csv"

new_data = [] #store new images

new_label = [] #store the lable of new images
# more kinds of augmentation can be found at https://keras.io/preprocessing/image/

datagen = ImageDataGenerator(zoom_range=0.1,

                             width_shift_range=0.3,

                             height_shift_range=0.3,

                             rotation_range=50)

datagen.fit(X_train)
#output new images into CSV

def write_to_csv(original_data, label, filename):

    for i in range(0, len(original_data)):

        pre_process = original_data[i].reshape((28*28,1))

        single_pic = []

        single_pic.append(label[0][i])

        for j in range(0,len(pre_process)):

            temp_pix = pre_process[j][0]

            single_pic.append(temp_pix)

        with open(filename,"a") as f:

            f_csv = csv.writer(f, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)

            f_csv.writerow(single_pic)
number_of_batches = 100

batches = 0



for X_batch, Y_batch in datagen.flow(X_train, y_train, batch_size=10):

    new_data.append(X_batch)

    new_label.append(Y_batch) 

#         loss = model.train(X_batch, Y_batch)

    batches += 1

    if batches >= number_of_batches:

        # we need to break the loop by hand because

        # the generator loops indefinitely

        break
#show the new images

for i in range(0, 4):

        pyplot.subplot(2,2,i+1)

        pyplot.imshow(new_data[1][i].reshape((28, 28)))

    # show the plot

pyplot.show()
#write new images into CSV

for i in range(0,len(new_data)):

    write_to_csv(new_data[i], new_label, filename)