# !pip install tensorwatch
# !pip install regim
# # LOAD LIBRARIES

# import pandas as pd

# import numpy as np

# from sklearn.model_selection import train_test_split

# from keras.utils.np_utils import to_categorical

# from keras.models import Sequential

# from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization

# from keras.preprocessing.image import ImageDataGenerator

# from keras.callbacks import LearningRateScheduler



# # Tensorwatch

# import tensorwatch as tw

# import time

# from regim import DataUtils

# # LOAD THE DATA

# train = pd.read_csv("../input/digit-recognizer/train.csv")

# test = pd.read_csv("../input/digit-recognizer/test.csv")



# numOfImages = train.shape[0]

# print(numOfImages)
# # PREPARE DATA FOR NEURAL NETWORK

# Y_train = train["label"]

# Y_train = to_categorical(Y_train, num_classes = 10)

# Y_train_visualize = Y_train[50]



# # X_train = train.drop(labels = ["label"],axis = 1)

# X_train = X_train / 255.0



# X_train_visualize = X_train.iloc[:500,:]

# X_train = X_train.values.reshape(-1,28,28,1)



# X_test = test / 255.0

# X_test = X_test.values.reshape(-1,28,28,1)

# print(X_train_visualize.shape)

# print(X_train.shape)

# print(Y_train_visualize)
# # First we will get MNIST dataset

# # The regim package has DataUtils class that allows to get entire MNIST dataset without train/test split and reshaping each image as vector of 784 integers instead of 28x28 matrix.

# data = DataUtils.mnist_datasets(linearize=True, train_test=False)
# # The regim package has utility method that allows us to take k random samples for each class.

# # We also set as_np=True to convert images to numpy array from PyTorch tensor.

# # The no_test=True parameter instructs that we don't want to split our data as train and test.

# # The return value is a tuple of two numpy arrays, one containing input images and other labels.

# inputs, labels = DataUtils.sample_by_class(data, k=50, shuffle=True, as_np=True, no_test=True)

# # supply this dataset to TensorWatch and in just one line we can get lower dimensional components.

# # The get_tsne_components method takes a tuple of input and labels. 

# # The optional parameters features_col=0 and labels_col=1 tells which member of tuple is input features and truth labels.

# # Another optional parameter n_components=3 says that we should generate 3 components for each data point.

# components = tw.get_tsne_components((inputs, labels))

# # Now that we have 3D component for each data point in our dataset

# # we use ArrayStream class from TensorWatch that allows you to convert any iterables in to TensorWatch stream.

# # This stream then we supply to Visualizer class asking it to use tsne visualization type which is just fency 3D scatter plot.

# component_stream = tw.ArrayStream(components)

# vis = tw.Visualizer(component_stream, vis_type='tsne', 

#                     hover_images=inputs, hover_image_reshape=(28,28))

# vis.show()
# print(train.head(1))
# numofImagesTest = test.shape[0]

# print(numofImagesTest)
# print(X_test.shape)

# print(X_train.shape)

# # PREVIEW IMAGES

# plt.figure(figsize=(15,4.5))

# for i in range(30):  

#     plt.subplot(3, 10, i+1)

#     plt.imshow(X_train[i].reshape((28,28)),cmap=plt.cm.binary)

#     plt.axis('off')

# plt.subplots_adjust(wspace=-0.1, hspace=-0.1)

# plt.show()
# # CREATE MORE IMAGES VIA DATA AUGMENTATION

# datagen = ImageDataGenerator(

#         rotation_range=10,  

#         zoom_range = 0.10,  

#         width_shift_range=0.1, 

#         height_shift_range=0.1)
# # PREVIEW AUGMENTED IMAGES

# X_train3 = X_train[9,].reshape((1,28,28,1))

# Y_train3 = Y_train[9,].reshape((1,10))

# plt.figure(figsize=(15,4.5))

# for i in range(30):  

#     plt.subplot(3, 10, i+1)

#     X_train2, Y_train2 = datagen.flow(X_train3,Y_train3).next()

#     plt.imshow(X_train2[0].reshape((28,28)),cmap=plt.cm.binary)

#     plt.axis('off')

#     if i==9: X_train3 = X_train[11,].reshape((1,28,28,1))

#     if i==19: X_train3 = X_train[18,].reshape((1,28,28,1))

# plt.subplots_adjust(wspace=-0.1, hspace=-0.1)

# plt.show()
# # BUILD CONVOLUTIONAL NEURAL NETWORKS

# nets = 15

# model = [0] *nets

# for j in range(nets):

#     model[j] = Sequential()



#     model[j].add(Conv2D(32, kernel_size = 3, activation='relu', input_shape = (28, 28, 1)))

#     model[j].add(BatchNormalization())

#     model[j].add(Conv2D(32, kernel_size = 3, activation='relu'))

#     model[j].add(BatchNormalization())

#     model[j].add(Conv2D(32, kernel_size = 5, strides=2, padding='same', activation='relu'))

#     model[j].add(BatchNormalization())

#     model[j].add(Dropout(0.4))



#     model[j].add(Conv2D(64, kernel_size = 3, activation='relu'))

#     model[j].add(BatchNormalization())

#     model[j].add(Conv2D(64, kernel_size = 3, activation='relu'))

#     model[j].add(BatchNormalization())

#     model[j].add(Conv2D(64, kernel_size = 5, strides=2, padding='same', activation='relu'))

#     model[j].add(BatchNormalization())

#     model[j].add(Dropout(0.4))



#     model[j].add(Conv2D(128, kernel_size = 4, activation='relu'))

#     model[j].add(BatchNormalization())

#     model[j].add(Flatten())

#     model[j].add(Dropout(0.4))

#     model[j].add(Dense(10, activation='softmax'))



#     # COMPILE WITH ADAM OPTIMIZER AND CROSS ENTROPY COST

#     model[j].compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
# # DECREASE LEARNING RATE EACH EPOCH

# annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)

# # TRAIN NETWORKS

# history = [0] * nets

# epochs = 45

# for j in range(nets):

#     X_train2, X_val2, Y_train2, Y_val2 = train_test_split(X_train, Y_train, test_size = 0.1)

#     history[j] = model[j].fit_generator(datagen.flow(X_train2,Y_train2, batch_size=64),

#         epochs = epochs, steps_per_epoch = X_train2.shape[0]//64,  

#         validation_data = (X_val2,Y_val2), callbacks=[annealer], verbose=0)

#     print("CNN {0:d}: Epochs={1:d}, Train accuracy={2:.5f}, Validation accuracy={3:.5f}".format(

#         j+1,epochs,max(history[j].history['acc']),max(history[j].history['val_acc']) ))
# import cv2

# img_pred = cv2.imread("../input/picstotest/3.jpg", 0)



# plt.imshow(img_pred, cmap='gray')

# if img_pred.shape != [28,28]:

#     img = cv2.resize(img_pred, (28,28))

#     img_pred = img.reshape(28,28,1)

# else:

#     img_pred = img_pred.reshape(28,28,1)

# img_pred = img_pred/255.0

# img_pred = img_pred.reshape(1,28,28,1)

# img_pred.shape
# prediction = model[13].predict(img_pred)
# prediction
# prediction_probability = model[14].predict_proba(img_pred)

# prediction_probability
# model[14].save('../input/model.h5')

# model[14].save('model.h5')
# # ENSEMBLE PREDICTIONS AND SUBMIT

# results = np.zeros( (X_test.shape[0],10) ) 

# for j in range(nets):

#     results = results + model[j].predict(X_test)

# results = np.argmax(results,axis = 1)

# results = pd.Series(results,name="Label")

# submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

# submission.to_csv("MNIST-CNN-ENSEMBLE.csv",index=False)
# # PREVIEW PREDICTIONS

# plt.figure(figsize=(15,6))

# for i in range(40):  

#     plt.subplot(4, 10, i+1)

#     plt.imshow(X_test[i].reshape((28,28)),cmap=plt.cm.binary)

#     plt.title("predict=%d" % results[i],y=0.9)

#     plt.axis('off')

# plt.subplots_adjust(wspace=0.3, hspace=-0.1)

# plt.show()
# print('a')