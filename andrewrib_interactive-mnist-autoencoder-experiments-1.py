import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import losses
from keras.callbacks import EarlyStopping
from tensorflow.examples.tutorials.mnist import input_data
from keras import backend as K
import tensorflow as tf
from ipywidgets import interact_manual,interact
import pandas as pd
from keras.models import load_model
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True,device_count = {'CPU' : 1, 'GPU' : 1}))
K.set_session(sess)
mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
print("Shape of the image data matrix: {0}".format(mnist.train.images.shape))
print("Shape of the label data: {0}".format(mnist.train.labels.shape))
exampleIndex = 3
exImage      = mnist.train.images[exampleIndex ,:]
exImageLabel = mnist.train.labels[exampleIndex]
plt.imshow(exImage.reshape(28,28),cmap='gray')
plt.show()
print("Image label: {0}".format(exImageLabel))
# First we will zip the training labels with the training images
dataWithLabels = zip(mnist.train.labels, mnist.train.images)

# Now let's turn this into a dictionary where subsets of the images in respect 
# to digit class are stored via the corresponding key.

# Init dataDict with keys [0,9] and empty lists.
digitDict = {}
for i in range(0,10):
    digitDict[i] = []

# Assign a list of image vectors to each corresponding digit class index. 
for i in dataWithLabels:
    digitDict[i[0]].append(i[1])

# Convert the lists into numpy matricies. (could be done above, but I claim ignorace)
for i in range(0,10):
    digitDict[i] = np.matrix(digitDict[i])
    print("Digit {0} matrix shape: {1}".format(i,digitDict[i].shape))
def simpleAE(encoding_dim = 32, input_dim = 784):

    # this is our input placeholder
    input_img = Input(shape=(input_dim,))
    # "encoded" is the encoded representation of the input
    encoded = Dense(encoding_dim, activation='relu')(input_img)
    # "decoded" is the lossy reconstruction of the input
    decoded = Dense(input_dim, activation='sigmoid')(encoded)

    # this model maps an input to its reconstruction
    return Model(input_img, decoded)
ae1 = simpleAE(64,784)
ae1.compile(optimizer='adadelta', loss='binary_crossentropy')
modelURI = "models/mnist_zero_autoencoder_1_64.h5"

print("Loading model...")

try:
    # Load model if found. 
    ae1 = load_model(modelURI)
    print("Model found and loaded.")
except:
    # Train model if model cannot be loaded. 
    print("Model not found. Training model..")
    
    history = ae1.fit(digitDict[0], digitDict[0],
                    epochs=100,
                    batch_size=20,
                    shuffle=True,verbose=1)
    
    # Save our Model
    print("Model Saved.")
    ae1.save(modelURI)
    
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
def dispRes1(digitClass=0,nthDigit=0):
    actual = digitDict[digitClass][nthDigit,:]
    pred = ae1.predict(actual)
    res = losses.mean_absolute_error(actual,pred)
    
    fig = plt.figure(figsize=(10, 10))
    fig.suptitle("Mean Absolute Reconstruction Error: {0:.5}%".format(sess.run(res)[0]*100), fontsize=16,y=0.73)
    
    ax1 = fig.add_subplot(1,2,1)
    ax1.axis('off')
    ax1.set_title("Original Image")
    ax1.imshow(actual.reshape(28,28),cmap="gray")
    
    ax2 = fig.add_subplot(1,2,2)
    ax2.axis('off')
    ax2.set_title("Reconstructed Image")
    ax2.imshow(pred.reshape(28,28),cmap="gray")
    
    plt.show()
    
interact_manual(dispRes1,digitClass=(0,9),nthDigit=(0,1000))
classLosses = []
for digitClass in range(0,10):
    lossTensor = losses.mean_absolute_error(digitDict[digitClass],ae1.predict(digitDict[digitClass]))
    res = sess.run(lossTensor)
    res.sort()
    classLosses.append(res)
def dispDists(digitClass=0):
    l = len(classLosses[digitClass])
    plt.figure(figsize=(6, 6))
    plt.hist(classLosses[digitClass],bins=20,range=(0, 0.125))
    plt.show()
    
interact(dispDists,digitClass=(0,9))
plt.figure(figsize=(10, 10))
for digitClass in range(0,10):
    l = len(classLosses[digitClass])
    plt.legend(['0','1','2','3','4','5','6','7','8','9'], loc='upper right')
    plt.hist(classLosses[digitClass],bins=20,range=(0, 0.125))
plt.show()
# Summary Statistics for Digit Reconstruction Loss
pd.DataFrame(classLosses).transpose().describe()
def simpleNN(encoding_dim = 10, input_dim = 1):

    # this is our input placeholder
    input_img = Input(shape=(input_dim,))
    # "encoded" is the encoded representation of the input
    encoded = Dense(encoding_dim, activation='relu')(input_img)
    # "decoded" is the lossy reconstruction of the input
    decoded = Dense(1, activation='sigmoid')(encoded)

    # this model maps an input to its reconstruction
    return Model(input_img, decoded)
# Preprocess Data for Classifier
zeroClass = classLosses[0]
otherClass = np.concatenate(classLosses[1:10])
labels = np.concatenate([np.zeros(len(zeroClass)),np.ones(len(otherClass))])
data = np.concatenate([zeroClass,otherClass])
modelURI = "models/reconstruction_loss_1_64.h5"

print("Loading model...")

try:
    # Load model if found. 
    snn = load_model(modelURI)
    print("Model found and loaded.")
except:
    # Train model if model cannot be loaded. 
    print("Model not found. Training model..")
    
    snn = simpleNN()
    snn.compile(optimizer='adadelta', loss='binary_crossentropy')
    history = snn.fit(data, labels,
                        epochs=25,
                        batch_size=20,
                        shuffle=True,verbose=1)
    
    # Save our Model
    print("Model Saved.")
    snn.save(modelURI)
    
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def dispRes1(digitClass=0,nthDigit=0):
    classificationThresh = 0.5
    actual = digitDict[digitClass][nthDigit,:]
    pred = ae1.predict(actual)
    res = losses.mean_absolute_error(actual,pred)
    res = sess.run(res)
    
    fig = plt.figure(figsize=(10, 10))
    fig.suptitle("Mean Absolute Reconstruction Error: {0:.5}%".format(res[0]*100), fontsize=16,y=0.73)
    
    ax1 = fig.add_subplot(1,2,1)
    ax1.axis('off')
    ax1.set_title("Original Image")
    ax1.imshow(actual.reshape(28,28),cmap="gray")
    
    ax2 = fig.add_subplot(1,2,2)
    ax2.axis('off')
    ax2.set_title("Reconstructed Image")
    ax2.imshow(pred.reshape(28,28),cmap="gray")
    
    plt.show()
    
    snnRes = snn.predict(res)[0][0]
    
    if snnRes < classificationThresh:
        print("Classification 0 : [{0}]".format( 1-snnRes ))
    else:
        print("Classification ANOMALY : [{0}]".format( snnRes ))
        
    
    
    
interact_manual(dispRes1,digitClass=(0,9),nthDigit=(0,1000))
classificationThresh = 0.5

# Get all the training images. 
actual = mnist.train.images

# Pass all the images through the trained zero digit autoencoder. 
pred = ae1.predict(actual)

# Calculuate the reconstruction loss for each image passed through the autoencoder. 
res = losses.mean_absolute_error(actual,pred)

# Use tensorflow to compute the value. 
res = sess.run(res)

# Pass the losses through the loss classifier. 
snnRes = snn.predict(res)
def tresholdFn( loss ):
    if loss < classificationThresh:
        return 0
    else:
        return 1

predictions = np.vectorize(tresholdFn)(snnRes)
nData = mnist.train.labels.shape[0]
nFalsePos = 0
falsePosIdxs = []
nFalseNeg = 0
falseNegIdxs = []

for index,prediction,label,in zip(range(nData),predictions,mnist.train.labels):
    # Compute the number of false negatives. 
    if label == 0 and prediction == 1:
        nFalseNeg += 1
        falseNegIdxs.append(index)
        
    # Compute the number of false positives. 
    if label != 0 and prediction == 0:
        nFalsePos += 1
        falsePosIdxs.append(index)
    
print("Number of false negatives: {0}".format(nFalseNeg))
print("Percentage of false negatives: {0:.3}%".format(nFalseNeg/digitDict[0].shape[0]*100))
print("---")
print("Number of false positives: {0}".format(nFalsePos))
print("Percentage of false positives: {0:.6}%".format(nFalsePos/otherClass.shape[0]*100))
autoResults = ae1.predict(mnist.train.images)
lossRes = losses.mean_absolute_error(mnist.train.images,autoResults)
lossRes = sess.run(lossRes)
def displayFalsePos(idx = 0):
    classificationThresh = 0.5
    
    indexVis = falsePosIdxs[idx]
    
    fig = plt.figure(figsize=(10, 10))
    fig.suptitle("False Positives - Mean Absolute Reconstruction Error: {0:.5}%".format(lossRes[indexVis]*100), fontsize=16,y=0.73)
    
    ax1 = fig.add_subplot(1,2,1)
    ax1.axis('off')
    ax1.set_title("Original Image")
    ax1.imshow(mnist.train.images[indexVis,:].reshape(28,28),cmap="gray")
    
    ax2 = fig.add_subplot(1,2,2)
    ax2.axis('off')
    ax2.set_title("Reconstructed Image")
    ax2.imshow(autoResults[indexVis,:].reshape(28,28),cmap="gray")
    
    plt.show()
    
    print("Absolute index value: {0}.".format(indexVis))
    print("Image label: {0}.".format(mnist.train.labels[indexVis]))
        
interact(displayFalsePos,idx = (0,len(falsePosIdxs)-1))
def displayFalseNeg(idx = 0):
    classificationThresh = 0.5
    
    indexVis = falseNegIdxs[idx]
    
    fig = plt.figure(figsize=(10, 10))
    fig.suptitle("False Negatives - Mean Absolute Reconstruction Error: {0:.5}%".format(lossRes[indexVis]*100), fontsize=16,y=0.73)
    
    ax1 = fig.add_subplot(1,2,1)
    ax1.axis('off')
    ax1.set_title("Original Image")
    ax1.imshow(mnist.train.images[indexVis,:].reshape(28,28),cmap="gray")
    
    ax2 = fig.add_subplot(1,2,2)
    ax2.axis('off')
    ax2.set_title("Reconstructed Image")
    ax2.imshow(autoResults[indexVis,:].reshape(28,28),cmap="gray")
    
    plt.show()
    
    print("Absolute index value: {0}.".format(indexVis))
    print("Image label: {0}.".format(mnist.train.labels[indexVis]))
        
interact(displayFalseNeg,idx = (0,len(falseNegIdxs)-1))
