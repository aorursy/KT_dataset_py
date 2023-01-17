import numpy as np

import pandas as pd

import os

print(os.listdir("../input"))



import matplotlib.pyplot as plt

import seaborn as sns

import keras

from keras.layers import Conv2D, MaxPool2D, Flatten, Dense,BatchNormalization, Dropout

from keras.models import Sequential

from keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split

from scipy import stats

import csv

import gc
# defining data directory paths

train_dir = "../input/train.csv"

test_dir = "../input/test.csv"



df = pd.read_csv(train_dir)

df.info()
labels = df["label"].values.tolist() # extracting labels from the database and converting it into a list

labels = np.array(labels)



n_classes = len(set(labels)) # defining number of classes



labels = keras.utils.to_categorical(labels) # converting the labels to one-hot format
df_train = df.drop(["label"], axis = 1) # extracting the image data

data = df_train.values.tolist() # converting image data to list

data = np.array(data)

data = data.astype('float32')/255.0 # converting data into range 0-1
dataframes_i = []

for i in range(10):

    tempdf = None

    tempdf = df[df["label"]==i].drop(["label"], axis = 1)

    temp = tempdf.values.tolist()

    dataframes_i.append(temp[0:5])

    

fig = plt.figure(figsize = (8,20)) #defining figure

def plot_images(image, index):

    fig.add_subplot(10,5, index)

    plt.axis("on")

    plt.tick_params(left = False, bottom=False, labelbottom=False, labelleft = False,)

    plt.imshow(image, cmap = 'Greys')

    return



index = 1

for i in dataframes_i:

    for j in i:

        x = np.array(j)

        x = x.reshape(28,28)

        plot_images(x, index)

        index += 1

plt.show()
print("Training data shape = " + str(data.shape))

print("Training labels shape = " + str(labels.shape))
gen_model = Sequential()

gen_model.add(Dense(784, activation = 'relu', input_shape = (784,)))

gen_model.add(Dense(512, activation = 'relu'))

gen_model.add(Dense(264, activation = 'relu'))

gen_model.add(Dense(10, activation = 'softmax'))

print("STANDARD NEURAL NETWORK MODEL :-")

gen_model.summary()
gen_model.compile(loss = 'categorical_crossentropy', optimizer = keras.optimizers.Adadelta(), metrics = ['accuracy'])
gen_model_hist = gen_model.fit(data, labels, batch_size = 32, epochs = 5, validation_split = 0.1)
plt.plot(gen_model_hist.history["acc"])

plt.plot(gen_model_hist.history["val_acc"])

plt.title("Training vs Validation Accuracy")

plt.legend(["Training","Validation"], loc = 'lower right')

plt.show()
del gen_model, gen_model_hist

gc.collect()
X_train_cnn = data.reshape(len(data), 28, 28, 1)
cnn_model = Sequential()

cnn_model.add(Conv2D(32, kernel_size = [3,3], activation = 'relu', input_shape = (28,28,1)))

cnn_model.add(Conv2D(64, kernel_size = [3,3], activation = 'relu'))

cnn_model.add(BatchNormalization())

cnn_model.add(MaxPool2D(pool_size = [2,2], strides = 2))

cnn_model.add(Conv2D(128, kernel_size = [3,3], activation = 'relu'))

cnn_model.add(MaxPool2D(pool_size = [2,2], strides = 2))

cnn_model.add(Flatten())

cnn_model.add(Dense(512, activation = 'relu'))

cnn_model.add(Dense(10, activation = 'softmax'))

print("CONVOLUTIONAL NEURAL NETWORK MODEL :-")

cnn_model.summary()
cnn_model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

cnn_model_hist = cnn_model.fit(X_train_cnn, labels, batch_size = 32, epochs = 6, validation_split = 0.1)
plt.plot(cnn_model_hist.history["acc"])

plt.plot(cnn_model_hist.history["val_acc"])

plt.title("Training vs Validation Accuracy (CNN Model)")

plt.legend(["Training","Validation"], loc = 'lower right')

plt.show()
del cnn_model, cnn_model_hist

gc.collect()
data_aug = ImageDataGenerator(featurewise_center = False,

                             samplewise_center = False,

                             featurewise_std_normalization = False,

                             samplewise_std_normalization = False,

                             zca_whitening = False,

                             rotation_range = 10,

                             zoom_range = 0.1,

                             width_shift_range = 0.1,

                             height_shift_range = 0.1,

                             horizontal_flip = False,

                             vertical_flip = False)
# defining several models

models_ensemble = []

for i in range(7):

    model = Sequential()

    model.add(Conv2D(32, kernel_size = [3,3], activation = 'relu', input_shape = (28,28,1)))

    model.add(Conv2D(64, kernel_size = [3,3], activation = 'relu'))

    model.add(BatchNormalization())

    model.add(MaxPool2D(pool_size = [2,2], strides = 2))

    model.add(Conv2D(128, kernel_size = [3,3], activation = 'relu'))

    model.add(MaxPool2D(pool_size = [2,2], strides = 2))

    model.add(Flatten())

    model.add(Dense(512, activation = 'relu'))

    model.add(Dense(10, activation = 'softmax'))

    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

    models_ensemble.append(model)
# defining training routine

model_histories = []

i = 1

for model in models_ensemble:

    xtrain, xtest, ytrain, ytest = train_test_split(X_train_cnn, labels, test_size = 0.07)

    print("Model " +str(i)+ " : ",end="")

    model_history = model.fit_generator(data_aug.flow(xtrain, ytrain, batch_size = 64), epochs = 1, verbose = 1, validation_data = (xtest, ytest), steps_per_epoch = xtrain.shape[0])

    model_histories.append(model_history)

    i += 1
# import and preprocess test data

testdata = pd.read_csv(test_dir)

testdata = testdata.values.tolist()

testdata = np.array(testdata)

testdata_reshaped = testdata.reshape(testdata.shape[0], 28, 28, 1)

testdata_reshaped = testdata_reshaped.astype('float')/255.0



def make_predictions_final_model(curr_model):

    prediction_array = curr_model.predict_on_batch(testdata_reshaped)

    predictions = [np.argmax(i) for i in prediction_array]

    return predictions
predictions_ensemble = [] 



# Make predictions using seperate models

for model in models_ensemble:

    curr_predictions = make_predictions_final_model(model)

    predictions_ensemble.append(curr_predictions)



prediction_per_image = []

# Make a list of predictions for a particular image 

for i in range(len(predictions_ensemble[0])):

    temppred = [predictions_ensemble[0][i], predictions_ensemble[1][i], predictions_ensemble[2][i], predictions_ensemble[3][i], predictions_ensemble[4][i], predictions_ensemble[5][i], predictions_ensemble[6][i]]

    prediction_per_image.append(temppred)

    

# Find the maximum occuring element in the array (list)

prediction_per_image = np.array(prediction_per_image)

modes = stats.mode(prediction_per_image, axis = 1)



# append the modes to the final prediction list

final_predictions = []      

for i in modes[0]:

    final_predictions.append(i[0])
final_csv = []

csv_title = ['ImageId', 'Label']

final_csv.append(csv_title)

for i in range(len(final_predictions)):

    image_id = i + 1

    label = final_predictions[i]

    temp = [image_id, label]

    final_csv.append(temp)



print(len(final_csv))



with open('submission_csv_aug.csv', 'w') as file:

    writer = csv.writer(file)

    writer.writerows(final_csv)

file.close()