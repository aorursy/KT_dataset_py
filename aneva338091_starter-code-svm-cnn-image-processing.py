# Load in Packages 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt, matplotlib.image as mpimg #plotting
from sklearn.model_selection import train_test_split #machine Learning
from sklearn import svm #support vector machines
%matplotlib inline
#Load CSV file
labeled_images = pd.read_csv("../input/train.csv")
#create labels from first column
labels = labeled_images.iloc[:, 0]
#create images from remaining columns
images = labeled_images.iloc[:, 1:]
#split data into train and testing 80% for training
train_images, test_images,train_labels, test_labels = train_test_split(images, labels, train_size=0.8, random_state=0)
#save copies of training and testing image data before image processing steps
train_images_original=train_images.copy()
test_images_original=test_images.copy()

#scale image matrices
max_value = images.values.max()
train_images_scaled = train_images.copy()/max_value
test_images_scaled= test_images.copy()/max_value

#turn image matrices of pixel values into 0 and 1s only
train_images.iloc[train_images>0] = 1
test_images.iloc[test_images>0] = 1

#plot histogram of 0's compared to 1's for sample image
plt.hist(train_images.iloc[5])
#function to plot images on grid of m by m with labels
def plot_as_grid(images, labels, m):
    n_pixels = len(images.columns)
    dimension = int(np.sqrt(n_pixels))

    # set up the figure
    fig = plt.figure(figsize=(6, 6))  # figure size in inches
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.5, wspace=0.05)

    # plot the digits: each image is max mxm pixels
    for i in range( min(m*m, len(images.index))):
        ax = fig.add_subplot(m, m, i + 1, xticks=[], yticks=[])    
    
        img=images.iloc[i].values.reshape((dimension,dimension))
    
        ax.imshow(img, cmap=plt.cm.binary, interpolation='nearest')
        plt.title(labels.iloc[i])
#compare by eye differences between raw data, processed and scaled images
plot_as_grid(test_images,test_labels,2)
plot_as_grid(test_images_scaled,test_labels,2)
plot_as_grid(test_images_original,test_labels,2)
#create support vector machine classifier, fit to processed data and check score on test data 
clf = svm.SVC()
clf.fit(train_images, train_labels)
#create support vector machine classifier, fit to raw data
clf_raw = svm.SVC()
clf_raw.fit(train_images_original, train_labels)
#create support vector machine classifier, fit to scaled data
clf_scaled = svm.SVC()
clf_scaled.fit(train_images_scaled, train_labels)
#check scores on training and testing data
trainingscore=clf.score(train_images,train_labels)
testingscore=clf.score(test_images,test_labels)
print("Training Score on 0&1 processed data:"+str(trainingscore))
print("Testinging Score on 0&1 processed data:"+str(testingscore))

testingscore_raw=clf_raw.score(test_images_original,test_labels)
print("Testing Score on raw data:"+str(testingscore_raw))

testingscore_scaled=clf_scaled.score(test_images_scaled,test_labels)
print("Testing Score on scaled data:"+str(testingscore_scaled))
trainingscore_raw=clf_raw.score(train_images_original,train_labels)
print("Training Score on raw data:"+str(trainingscore_raw))

trainingscore_scaled=clf_scaled.score(train_images_scaled,train_labels)
print("Training Score on scaled data:"+str(trainingscore_scaled))
from sklearn import metrics
#function to plot confusion matrix
def plot_confusion_matrix(labels, predictions):
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(labels, predictions))
#create results variable from test_images
results=clf.predict(test_images)
#create dataframe and plot images with predictions as titles
results_dataframe = pd.DataFrame(results)
results_dataframe.columns=['predictions']
plot_as_grid(test_images,results_dataframe['predictions'],10)
#create a confusion matrix to see which values have what issues
plot_confusion_matrix(test_labels,results)
test_data=pd.read_csv("../input/test.csv")
results=clf.predict(test_data)
df = pd.DataFrame(results)
df.index.name='ImageId'
df.index+=1
df.columns=['Label']
df.to_csv('results.csv', header=True)
def plot_incorrect_classifications(ypred, test_labels, test_images):
    "Plots incorrectly classified images and corresponding prediction"
    ypred = pd.DataFrame(ypred)
    ypred = ypred.set_index(test_labels.index.values)
    ypred.columns = ['prediction']
    predict_df = pd.concat([ypred, test_labels], axis=1)
    predict_df['Incorrect'] = predict_df.prediction != predict_df.label
    idx = predict_df.index[predict_df['Incorrect']]

    plot_as_grid(test_images.loc[idx], predict_df['prediction'].loc[idx], 5)
results_processed=clf.predict(test_images)
results_raw=clf_raw.predict(test_images_original)
results_scaled=clf_scaled.predict(test_images_scaled)
plot_incorrect_classifications(results_processed, test_labels, test_images)
plot_incorrect_classifications(results_raw, test_labels, test_images_original)
plot_incorrect_classifications(results_scaled, test_labels, test_images_scaled)
# Create first network with Keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
# create model
model_scaled = Sequential()
model_scaled.add(Dense(820, input_dim=784, activation='relu')) #dense type, 820 neurons, 784 inputs ie the number of pixels, activation function relu
model_scaled.add(Dense(410, activation='relu'))
model_scaled.add(Dense(10, activation='softmax'))
# Compile model
model_scaled.compile(optimizer='adam',  # Good default optimizer to start with
              loss='sparse_categorical_crossentropy',  # how will we calculate our "error." Neural network aims to minimize loss.
              metrics=['accuracy'])
# Fit the model
model_scaled.fit(train_images_scaled, train_labels, epochs=3)
# create model
model_raw = Sequential()
model_raw.add(Dense(820, input_dim=784, activation='relu'))
model_raw.add(Dense(410, activation='relu'))
model_raw.add(Dense(10, activation='softmax'))
# Compile model
model_raw.compile(optimizer='adam',  # Good default optimizer to start with
              loss='sparse_categorical_crossentropy',  # how will we calculate our "error." Neural network aims to minimize loss.
              metrics=['accuracy'])
# Fit the model
model_raw.fit(train_images_original, train_labels, epochs=3)
# create model
model_processed = Sequential()
model_processed.add(Dense(820, input_dim=784, activation='relu'))
model_processed.add(Dense(410, activation='relu'))
model_processed.add(Dense(10, activation='softmax'))
# Compile model
model_processed.compile(optimizer='adam',  # Good default optimizer to start with
              loss='sparse_categorical_crossentropy',  # how will we calculate our "error." Neural network aims to minimize loss.
              metrics=['accuracy'])
# Fit the model
model_processed.fit(train_images, train_labels, epochs=3)
predictions_scaled=model_scaled.predict(test_images_scaled)
predictions_raw=model_raw.predict(test_images_original)
predictions_processed=model_processed.predict(test_images)
def probability_matrix_to_classification(predictions):
    return np.argmax(predictions,axis = 1)
predictions_scaled_class=probability_matrix_to_classification(predictions_scaled)
predictions_raw_class=probability_matrix_to_classification(predictions_raw)
predictions_processed_class=probability_matrix_to_classification(predictions_processed)
def score_of_cnn(prediction_classes,testlabels):
    op=prediction_classes==testlabels
    return np.sum(op)/len(prediction_classes)
testingscore_scaled=score_of_cnn(predictions_scaled_class,test_labels)
print("Testing Score on scaled data:"+str(testingscore_scaled))

testingscore_raw=score_of_cnn(predictions_raw_class,test_labels)
print("Testing Score on raw data:"+str(testingscore_raw))

testingscore_processed=score_of_cnn(predictions_processed_class,test_labels)
print("Testing Score on processed data:"+str(testingscore_processed))