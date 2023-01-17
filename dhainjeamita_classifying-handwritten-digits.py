import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
from sklearn import datasets, svm, metrics
from sklearn.metrics.classification import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from numpy import argmax

dataset = pd.read_csv('../input/train.csv')
dataset.head(10)
testDataSet = dataset[:100]
TrainingDataSet = dataset

TestingDataSet = pd.read_csv('../input/test.csv')
TestDataValues = TestingDataSet.values
testData_Normalized = TestDataValues / 255
print (TrainingDataSet.shape)
valuesDataset = TrainingDataSet.values
targetValues = TrainingDataSet['label'].values
newDataset = valuesDataset[:, 1:785]
print (newDataset.shape)
print (targetValues.shape)

sampleImage1 = newDataset[3].reshape(28,28)
sampleImage2 = newDataset[1].reshape(28,28)
sampleImage3 = newDataset[2].reshape(28,28)
sampleImage4 = newDataset[10].reshape(28,28)
plt.subplot(221)
plt.imshow(sampleImage1, cmap=plt.get_cmap('gray'), interpolation='nearest')
plt.subplot(222)
plt.imshow(sampleImage2, cmap=plt.get_cmap('gray'), interpolation='nearest')

plt.subplot(223)
plt.imshow(sampleImage3, cmap=plt.get_cmap('gray'), interpolation='nearest')
plt.subplot(224)
plt.imshow(sampleImage4, cmap=plt.get_cmap('gray'), interpolation='nearest')


seed = 7
np.random.seed(seed)
#X_train, X_test, y_train, y_test = train_test_split(newDataset, targetValues, train_size = 32000 ,test_size=10000)
X_train, X_test, y_train, y_test = train_test_split(newDataset, targetValues, test_size = 0.33, random_state=seed)
print (X_train.shape)
#y_train = y_train.reshape(y_train.shape[0], 1)
#y_test = y_train.reshape(y_train.shape[0], 1)

# normalize inputs from 0-255 to 0-1
X_train_normalize = X_train / 255
X_test_normalize = X_test / 255
classifier = svm.SVC(gamma=0.001)
classifier.fit(X_train_normalize, y_train)
predicted = classifier.predict(X_test_normalize)
print("Classification report for classifier %s:\n%s\n" % (classifier, metrics.classification_report(y_test, predicted)))
print("Accuracy - %.3f\n" % (accuracy_score(y_test, predicted)))
images_and_predictions = list(zip(X_test, predicted))

for index, (image, prediction) in enumerate(images_and_predictions[:4]):
    plt.subplot(2, 4, index + 5)
    image = image.reshape(28,28)
    plt.axis('off')
    plt.imshow(image, cmap=plt.get_cmap('gray'), interpolation='nearest')
    plt.title('Prediction: %i' % prediction)
    
plt.show()

predicted = classifier.predict(testData_Normalized)
my_submission = pd.DataFrame({'ImageId': range(1,len(predicted) + 1), 'Label': predicted})
my_submission.to_csv('submission.csv', index=False)
my_submission.head()
num_pixels = 784
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]
print (num_classes)
seed = 7
np.random.seed(seed)
model = Sequential()
model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train_normalize, y_train, validation_data=(X_test_normalize, y_test), epochs=30, batch_size=10000, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_test_normalize, y_test, verbose=0)
print (scores)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))
NN_Predictions = model.predict_classes(testData_Normalized)
print (NN_Predictions[0])
#orgPredictions = argmax(NN_Predictions)
#print (orgPredictions)
my_submission = pd.DataFrame({'ImageId': range(1,len(NN_Predictions) + 1), 'Label': NN_Predictions})
my_submission.to_csv('submission_nn.csv', index=False)
my_submission.head()
