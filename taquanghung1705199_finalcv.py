import cv2

import os

import glob, random

import numpy as np

import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier

from keras.preprocessing import image

from sklearn.externals.joblib import dump, load



def accuracy(confusion_matrix):

   diagonal_sum = confusion_matrix.trace()

   sum_of_all_elements = confusion_matrix.sum()

   return diagonal_sum / sum_of_all_elements



def fd_histogram(image):

    bins=16

    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    hist = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])

    cv2.normalize(hist, hist)

    return hist.flatten()



def fd_hu_moments(image):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    feature = cv2.HuMoments(cv2.moments(image)).flatten()

    return feature



def Get(link):

    data = []

    label = []

    for i in os.listdir(link):

        label_now=i

        for j in glob.glob(link + "/" + i + "/*.jpg"):

            image = cv2.imread(j)

            fv_hist=fd_histogram(image)

            fv_hu=fd_hu_moments(image)

            global_feature=np.hstack([fv_hist, fv_hu])

            data.append(global_feature)

            label.append(label_now)

    return data, label

pathtrain="/kaggle/input/data/data/train"

pathtest="/kaggle/input/data/data/test"
img_list = glob.glob(os.path.join(pathtrain, '*/*.jpg'))



for i, img_path in enumerate(random.sample(img_list, 6)):

    img = image.load_img(img_path, target_size=(300, 300))

    img = image.img_to_array(img, dtype=np.uint8)



    plt.subplot(2, 3, i+1),

    plt.imshow(img.squeeze()),

    plt.xticks([]),

    plt.yticks([])

plt.show()
x_train, y_train=Get(pathtrain)

print("Successfully retrieved train data")

y_train=np.array(y_train)

x_train=np.array(x_train)

print("Training size: ", len(x_train))

print(x_train.shape)

print(y_train.shape)



x_test, y_test=Get(pathtest)

print("Successfully retrieved test data")

y_test=np.array(y_test)

x_test=np.array(x_test)

print("Training size: ", len(x_test))

print(x_test.shape)

print(y_test.shape)
model = MLPClassifier(hidden_layer_sizes=(16, 8),

                       learning_rate_init=0.001,

                       max_iter=10,

                       activation = 'relu',

                       solver='adam',

                       random_state=42,

                       verbose=0)



N_TRAIN_SAMPLES = x_train.shape[0]

N_EPOCHS = 10

N_BATCH = 64

N_CLASSES = np.unique(y_train)



scores_train = []

scores_test = []



# EPOCH

epoch = 1

while epoch <= N_EPOCHS:

    print('epoch: ', epoch)

    # SHUFFLING

    random_perm = np.random.permutation(x_train.shape[0])

    mini_batch_index = 0

    while True:

        # MINI-BATCH

        indices = random_perm[mini_batch_index:mini_batch_index + N_BATCH]

        model.partial_fit(x_train[indices], y_train[indices], classes=N_CLASSES)

        mini_batch_index += N_BATCH



        if mini_batch_index >= N_TRAIN_SAMPLES:

            break



    # SCORE TRAIN

    scores_train.append(model.score(x_train, y_train))



    # SCORE TEST

    scores_test.append(model.score(x_test, y_test))



    epoch += 1



""" Plot """

fig, ax = plt.subplots(2, sharex=True, sharey=True)

ax[0].plot(scores_train)

ax[0].set_title('Train')

ax[1].plot(scores_test)

ax[1].set_title('Test')

fig.suptitle("Accuracy over epochs", fontsize=14)

plt.show()
import matplotlib.pyplot as plt



y_pred = model.predict(x_test)

print(y_pred.shape)
from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score 

from sklearn.metrics import classification_report 



cm = confusion_matrix(y_pred, y_test)



print("\nTraining Accuracy: ", model.score(x_train, y_train)*100)

print("\nValid Accuracy: ", model.score(x_test, y_test)*100)

print('Confusion Matrix:')

print (classification_report(y_test, y_pred))
dump(model, 'fruit_ANN.joblib')

print('Model saved!')
loaded_model = load('fruit_ANN.joblib')

print('model loaded!')
print(loaded_model)
link= "/kaggle/input/dataset/dataset"

for j in glob.glob(link + "/*.jpg"):

    image = cv2.imread(j)

    image = cv2.resize(image, (100, 100))

    fv_his = fd_histogram(image)

    fv_hu = fd_hu_moments(image)

    globals_feature = np.hstack([fv_his, fv_hu])

    globals_feature = globals_feature.reshape(1, -1)

    prediction = loaded_model.predict(globals_feature)

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    plt.xlabel(prediction)

    plt.xticks([])

    plt.yticks([])

    plt.show()

link= "/kaggle/input/test/test"

for j in glob.glob(link + "/*.jpg"):

    image = cv2.imread(j)

    image = cv2.resize(image, (100, 100))

    fv_his = fd_histogram(image)

    fv_hu = fd_hu_moments(image)

    globals_feature = np.hstack([fv_his, fv_hu])

    globals_feature = globals_feature.reshape(1, -1)

    prediction = loaded_model.predict(globals_feature)

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    plt.xlabel(prediction)

    plt.xticks([])

    plt.yticks([])

    plt.show()

# vậy đối với tập data từ fruit 360 của kaggle thì hầu như model predict chính xác hết 

# còn đối với tập data lấy từ gg thì 1 số trái model predict sai
