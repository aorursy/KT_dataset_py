# Pay no mind to this section.  There are a lot of images in this notebook.  This script
# downloads all the images so I can reference them locally on the Kaggle server.  
# Have to do this since the images push the notebook over Kaggle's 1MB notebook limit.

import os.path
import requests

images = ["average_digit.png", "average_intensities.png", "confmat_jitter0.5.png", "confmat_jitter0.png", "confmat_jitter1.png", "confmat_jitter2.png", "confmat_jitter3.png", "confmat_jitter4.png", "confmat_jitter5.png", "confmat1.png", "counts.png", "digits.png", "epochs.png", "misclassifications.png", "mnist_6.png", "MNIST_Keras_epoch_time.png", "Synthetic0.png", "Synthetic0_jitter0.5.png", "Synthetic0_jitter0.png", "Synthetic0_jitter1.png", "Synthetic0_jitter2.png", "Synthetic0_jitter3.png", "Synthetic0_jitter4.png", "Synthetic0_jitter5.png", "Synthetic1.png", "Synthetic1_jitter0.5.png", "Synthetic1_jitter0.png", "Synthetic1_jitter1.png", "Synthetic1_jitter2.png", "Synthetic1_jitter3.png", "Synthetic1_jitter4.png", "Synthetic1_jitter5.png", "Synthetic2.png", "Synthetic2_jitter0.5.png", "Synthetic2_jitter0.png", "Synthetic2_jitter1.png", "Synthetic2_jitter2.png", "Synthetic2_jitter3.png", "Synthetic2_jitter4.png", "Synthetic2_jitter5.png", "Synthetic3.png", "Synthetic3_jitter0.5.png", "Synthetic3_jitter0.png", "Synthetic3_jitter1.png", "Synthetic3_jitter2.png", "Synthetic3_jitter3.png", "Synthetic3_jitter4.png", "Synthetic3_jitter5.png", "Synthetic4.png", "Synthetic4_jitter0.5.png", "Synthetic4_jitter0.png", "Synthetic4_jitter1.png", "Synthetic4_jitter2.png", "Synthetic4_jitter3.png", "Synthetic4_jitter4.png", "Synthetic4_jitter5.png", "Synthetic5.png", "Synthetic5_jitter0.5.png", "Synthetic5_jitter0.png", "Synthetic5_jitter1.png", "Synthetic5_jitter2.png", "Synthetic5_jitter3.png", "Synthetic5_jitter4.png", "Synthetic5_jitter5.png", "Synthetic6.png", "Synthetic6_jitter0.5.png", "Synthetic6_jitter0.png", "Synthetic6_jitter1.png", "Synthetic6_jitter2.png", "Synthetic6_jitter3.png", "Synthetic6_jitter4.png", "Synthetic6_jitter5.png", "Synthetic7.png", "Synthetic7_jitter0.5.png", "Synthetic7_jitter0.png", "Synthetic7_jitter1.png", "Synthetic7_jitter2.png", "Synthetic7_jitter3.png", "Synthetic7_jitter4.png", "Synthetic7_jitter5.png", "Synthetic8.png", "Synthetic8_jitter0.5.png", "Synthetic8_jitter0.png", "Synthetic8_jitter1.png", "Synthetic8_jitter2.png", "Synthetic8_jitter3.png", "Synthetic8_jitter4.png", "Synthetic8_jitter5.png", "Synthetic9.png", "Synthetic9_jitter0.5.png", "Synthetic9_jitter0.png", "Synthetic9_jitter1.png", "Synthetic9_jitter2.png", "Synthetic9_jitter3.png", "Synthetic9_jitter4.png", "Synthetic9_jitter5.png", "zdigits.png" ]

for image in images:
    if ~ os.path.isfile( image ):
        url = 'http://neillwhite.dynu.net/MachineLearning/' + image
        image_data = requests.get(url).content
        with open(image, 'wb') as fid:
            fid.write(image_data)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import cv2

%matplotlib inline

np.random.seed(42)

from pylab import gcf

from scipy.misc import imresize
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from keras.utils.np_utils import to_categorical 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, SpatialDropout2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

sns.set(style='white', context='notebook', palette='deep')
trainRows = pd.read_csv("../input/train.csv")
testRows = pd.read_csv("../input/test.csv")
# train dimensions
trainRows.shape
# test dimensions
testRows.shape
trainRows.head()
testRows.head()
# split train into trainX and trainY (label)
trainY = trainRows["label"]
# by removing the label from trainX, dataset is same as test
trainX = trainRows.drop(labels=["label"],axis=1)
trainXRows = trainX
# these are 28x28 captures
trainX = trainX.values.reshape(-1,28,28,1)
test = testRows.values.reshape(-1,28,28,1)
# greyscale values are encoded 2**8 = 0-255 different values.  Scale to 0-1 range:
trainX = trainX / 255.0
trainXRows = trainXRows / 255.0
test = test / 255.0
trainY.value_counts().plot(kind='bar')
trainY.value_counts()
import matplotlib

cmap = matplotlib.cm.get_cmap('Spectral')
fig = plt.figure(figsize=(16,16))
# i'm sure there's a much more "pythonic" way to do this. Â¯\_(ãƒ„)_/Â¯
for i in range(10):
    idxDigit = [j for j in range(len(trainX)) if trainY[j] == i]  # group by label
    meanPixels = trainXRows.iloc[idxDigit].mean()  # take mean of all '7s', for instance
    digitMeans = meanPixels.values.reshape(-1,28,28,1)
    plt.subplot(4,3,i+1)
    plt.imshow(digitMeans[0][:,:,0],cmap=cmap)
    plt.colorbar()
    
plt.show()
import matplotlib
cmap = matplotlib.cm.get_cmap('Spectral')
fig = plt.figure(figsize=(10,10))
digitMeans = trainXRows.mean().values.reshape(-1,28,28,1)
plt.imshow(digitMeans[0][:,:,0],cmap=cmap)
plt.colorbar()
plt.show()

# next, take a look at the intensity values to see if we can apply parametric significance tests for individual pixels
fig = plt.figure(figsize=(16,8))
plt.subplot(1,2,1)
plt.hist(trainXRows.mean(),30)
# on the right hand side, omit first peak of zeros so get a better look at distribution
plt.subplot(1,2,2)
plt.hist(trainXRows.mean(),30)
plt.axis([0.01,0.6,0,50])
import matplotlib
import math
cmap = matplotlib.cm.get_cmap('Spectral')
fig = plt.figure(figsize=(16,16))
for i in range(10):
    idxDigit = [j for j in range(len(trainX)) if trainY[j] == i]  # group by label
    idxNonDigit = set( range(len(trainX)) ) - set( idxDigit )
    idxNonDigit = list( idxNonDigit )
    meanDigitPixels = trainXRows.iloc[idxDigit].mean()  # take mean of all '7s', for instance
    stdDigitPixels = trainXRows.iloc[idxDigit].std()
    numDigit = len(idxDigit)
    numNonDigit = len(idxNonDigit)
    meanNonDigitPixels = trainXRows.iloc[idxNonDigit].mean()
    stdNonDigitPixels = trainXRows.iloc[idxNonDigit].std()
    pooledVariance = (stdDigitPixels**2)/numDigit + (stdNonDigitPixels**2)/numNonDigit
    zscores = ( meanDigitPixels - meanNonDigitPixels ) / pooledVariance.apply(np.sqrt)
    zImage = zscores.values.reshape(-1,28,28,1)
    plt.subplot(4,3,i+1)
    plt.imshow(zImage[0][:,:,0],cmap=cmap)
    plt.colorbar()
    
plt.show()
random_seed = 1984
trainY = to_categorical( trainY, num_classes = 10 )
subTrainX, valX, subTrainY, valY = train_test_split(trainX, trainY, test_size = 0.1, random_state=random_seed)

model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (3,3), padding = 'valid', 
                 activation ='relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 32, kernel_size = (3,3), padding = 'same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(SpatialDropout2D(0.25))


model.add(Conv2D(filters = 32, kernel_size = (3,3), padding = 'same', 
                 activation ='relu'))
model.add(Conv2D(filters = 32, kernel_size = (3,3), padding = 'same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(SpatialDropout2D(0.25))


model.add(Flatten())
model.add(Dense(128, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))

model.compile(optimizer = Adam(), loss = "categorical_crossentropy", metrics=["accuracy"])
model.summary()
epochs = 10 
batch_size = 128
imageDataGenerator = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

imageDataGenerator.fit(subTrainX)
# Uncomment this line if you have the GPU to train this network with augmented data

#fit = model.fit_generator(datagen.flow(subTrainX,subTrainY, batch_size=32), epochs = epochs, validation_data = (valX,valY), verbose = 2, steps_per_epoch=subTrainX.shape[0]/32 )

#otherwise, use this fit
fit = model.fit(subTrainX, subTrainY, batch_size = batch_size, epochs = epochs, verbose = 2, validation_data = (valX,valY)  )
# Plot the loss and accuracy curves for training and validation 
fig, ax = plt.subplots(2,1)
ax[0].plot(fit.history['loss'], color='b', label="Training loss")
ax[0].plot(fit.history['val_loss'], color='r', label="Validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(fit.history['acc'], color='b', label="Training accuracy")
ax[1].plot(fit.history['val_acc'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)
# predict the digits from the 4200 record validation set
valPredY = model.predict(valX)
# valPredY is a 4200 x 10 array of confidences for each digit over the 4200 records
# take the maximum value as the network prediction
valYCall = np.argmax(valPredY,axis=1)
# now take the true, annotated values from the one-hot valY vector and similarly make calls
valYTrue = np.argmax(valY,axis=1)
confusion_mat = confusion_matrix(valYTrue, valYCall)
df_cm = pd.DataFrame(confusion_mat, index = [i for i in range(0,10)], columns = [i for i in range(0,10)])
plt.figure(figsize = (6,5))
conf_mat = sns.heatmap(df_cm, annot=True, cmap='Blues', fmt='g', cbar = False)
conf_mat.set(xlabel='Predictions', ylabel='Truth')
errors = (valYCall != valYTrue)  # True/False array
callErrors = valYCall[errors]    # bad predictions
predErrors = valPredY[errors]    # probs of all classes for errors
trueErrors = valYTrue[errors]    # the true values of all errors
valXErrors = valX[errors]        # the pixel data for each error

maxCols = 5
for i in range(10):
    idxErrors = ( trueErrors == i )
    numErrors = sum( idxErrors )
    if numErrors == 0:
        continue
    # now sort by probability
    digitCallErrors = callErrors[idxErrors]
    digitPredErrors = predErrors[idxErrors]
    digitXErrors = valXErrors[idxErrors]
    numRows = math.ceil( numErrors / maxCols )
    numCols = maxCols
    fig, ax = plt.subplots(numRows,numCols,sharex=True,sharey=True,figsize=(14,3 * numRows))
    fig.suptitle( "Misclassifications of " + str(i), fontsize=18)
    for j in range(numErrors):
        row = math.floor( j / maxCols )
        col = j % maxCols
        maxProb = np.max(digitPredErrors[j])
        truthProb = digitPredErrors[j][i]
        plotTitle = "Predicted: {:1d} ({:0.2f})\nTruth: {:1d} ({:0.2f})".format(digitCallErrors[j],maxProb,i,truthProb)
        if ax.ndim == 1:
            ax[col].imshow(digitXErrors[j][:,:,0])
            ax[col].set_title(plotTitle)
        else:
            ax[row,col].imshow(digitXErrors[j][:,:,0])
            ax[row,col].set_title(plotTitle)
import random
from scipy.misc import imresize
from skimage.transform import resize
import cv2

def getRandomPoint(x,y,jitterRadius):
    startX = x
    startY = y
    radius = random.uniform(0,jitterRadius)
    angle = random.uniform(0,2*math.pi)
    endX = startX + radius * math.cos( angle )
    endY = startY + radius * math.sin( angle )
    return(endX,endY)

def drawLine(image, points, brushWidth):
    # if points is length 2, draw line
    # if points is length 3, draw quadratic Bezier curve
    # if points is length 4, draw cubic Bezier curve
    
    # make a darkness gradient so that cells a distance a way from the pen point
    # are less darkly colored.  Make so up to 1/ff brush width, the color is 1.0
    # then linearly ramp to 0
    ff = 5
    numPoints = len( points )
    totalDistance = 0
    if numPoints > 1:
        # calcululate the length of all the lines
        # dist = sqrt( ( x2 - x1 )^2 + (y2 - y1 )^2 )
        squares = [(points[i+1][0]-points[i][0])*(points[i+1][0]-points[i][0])+(points[i+1][1]-points[i][1])*(points[i+1][1]-points[i][1]) for i in range(len(points)-1)]
        totalDistance = math.sqrt( sum( squares ) )
    if numPoints == 2:
        # just a simple line
        aX   = points[0][0]
        aY   = points[0][1]
        bX   = points[1][0]
        bY   = points[1][1]
        numSteps = round( totalDistance )
        stepSize = 1/numSteps
        for t in [s/numSteps for s in range(0,numSteps+1)]:
            x = (1-t)*aX + t*bX
            y = (1-t)*aY + t*bY
            # now color brush width and every square that is within brush width/2;
            # make sure we don't paint off the end of the image
            xStart = max( round( x - brushWidth/2.0), 0 )
            xEnd = min( round( x + brushWidth/2.0 + 1), image.shape[0] )
            yStart = max( round( y - brushWidth/2.0 ), 0 )
            yEnd = min( round( y + brushWidth/2.0 + 1), image.shape[1] )
            for xcoord in range( xStart, xEnd ):
                for ycoord in range( yStart, yEnd ):
                    distToTip = math.sqrt( ( x - xcoord ) * ( x - xcoord ) + ( y - ycoord ) * ( y - ycoord ) )
                    if distToTip <= brushWidth/2:
                        if distToTip <= brushWidth/ff:
                            pixelDarkness = 1.0
                        else:
                            pctTraveled = distToTip/(brushWidth/2)
                            pixelDarkness = 1.0-pctTraveled
                        # take the max between this calc and what was already there in case there was
                        # a darker line there already
                        image[xcoord,ycoord,0] = max(image[xcoord,ycoord,0],pixelDarkness)
    elif numPoints == 3:
        aX   = points[0][0]
        aY   = points[0][1]
        bezX = points[1][0]
        bezY = points[1][1]
        bX   = points[2][0]
        bY   = points[2][1]
        # the distance between starting and ending points is totalDist.  Want to make sure all the pixels
        # in the line are colored so have to walk line slowly (stepSize) since it is not a straight line but a
        # curve. Figure the curve is at most a half circle so arc length is roughly distance * pi/2 or simply 1.5
        arcLength = totalDistance * 1.5
        numSteps = round( arcLength )
        stepSize = 1/numSteps
        for t in [s/numSteps for s in range(0,numSteps+1)]:
            x = (1-t)*(1-t)*aX + 2*(1-t)*t*bezX + t*t*bX
            y = (1-t)*(1-t)*aY + 2*(1-t)*t*bezY + t*t*bY
            # now color brush width and every square that is within brush width/2;
            # make sure we don't paint off the end of the image
            xStart = max( round( x - brushWidth/2.0), 0 )
            xEnd = min( round( x + brushWidth/2.0 + 1), image.shape[0] )
            yStart = max( round( y - brushWidth/2.0 ), 0 )
            yEnd = min( round( y + brushWidth/2.0 + 1), image.shape[1] )
            for xcoord in range( xStart, xEnd ):
                for ycoord in range( yStart, yEnd ):
                    distToTip = math.sqrt( ( x - xcoord ) * ( x - xcoord ) + ( y - ycoord ) * ( y - ycoord ) )
                    if distToTip < brushWidth/2:
                        if distToTip <= brushWidth/ff:
                            pixelDarkness = 1.0
                        else:
                            pctTraveled = distToTip/(brushWidth/2)
                            pixelDarkness = 1.0-pctTraveled
                        # take the max between this calc and what was already there in case there was
                        # a darker line there already
                        image[xcoord,ycoord,0] = max(image[xcoord,ycoord,0],pixelDarkness)
    elif numPoints == 4:
        aX    = points[0][0]
        aY    = points[0][1]
        bez1X = points[1][0]
        bez1Y = points[1][1]
        bez2X = points[2][0]
        bez2Y = points[2][1]
        cX    = points[3][0]
        cY    = points[3][1]
        # the distance between starting and ending points is totalDist.  Want to make sure all the pixels
        # in the line are colored so have to walk line slowly (stepSize) since it is not a straight line but a
        # curve. Figure the curve is more than a half circle so arc length is roughly > distance * pi/2 or simply 1.7
        arcLength = totalDistance * 1.7
        numSteps = round( arcLength )
        stepSize = 1/numSteps
        for t in [s/numSteps for s in range(0,numSteps+1)]:
            x = (1-t)*(1-t)*(1-t)*aX + 3*(1-t)*(1-t)*t*bez1X + 3*(1-t)*t*t*bez2X + t*t*t*cX
            y = (1-t)*(1-t)*(1-t)*aY + 3*(1-t)*(1-t)*t*bez1Y + 3*(1-t)*t*t*bez2Y + t*t*t*cY
            # now color brush width and every square that is within brush width/2;
            # make sure we don't paint off the end of the image
            xStart = max( round( x - brushWidth/2.0), 0 )
            xEnd = min( round( x + brushWidth/2.0 + 1), image.shape[0] )
            yStart = max( round( y - brushWidth/2.0 ), 0 )
            yEnd = min( round( y + brushWidth/2.0 + 1), image.shape[1] )
            for xcoord in range( xStart, xEnd ):
                for ycoord in range( yStart, yEnd ):
                    distToTip = math.sqrt( ( x - xcoord ) * ( x - xcoord ) + ( y - ycoord ) * ( y - ycoord ) )
                    if distToTip < brushWidth/2:
                        if distToTip <= brushWidth/ff:
                            pixelDarkness = 1.0
                        else:
                            pctTraveled = distToTip/(brushWidth/2)
                            pixelDarkness = 1.0-pctTraveled
                        # take the max between this calc and what was already there in case there was
                        # a darker line there already
                        image[xcoord,ycoord,0] = max(image[xcoord,ycoord,0],pixelDarkness)
            
    return image

def get0(zoomFactor,destinationRadius): 
    syntheticDigit = np.zeros((28 * zoomFactor,28 * zoomFactor,1))
    brushWidth = random.uniform(2.8,5.0) * zoomFactor
    # this is for digit '0'!
    pointA = [7,14]
    pointB = [7,7]
    pointC = [21,7]
    pointD = [21,14]
    pointE = [21,21]
    pointF = [7,21]
    pointG = [7,13]

    destinationRadius *= 2
    
    # aX,aY are the coordinates for the top of the '0', d is the bottom; b and c are Bezier supports
    (aX,aY) = getRandomPoint(pointA[0]*zoomFactor,pointA[1]*zoomFactor,destinationRadius*zoomFactor)
    (bX,bY) = getRandomPoint(pointB[0]*zoomFactor,pointB[1]*zoomFactor,destinationRadius*zoomFactor)
    (cX,cY) = getRandomPoint(pointC[0]*zoomFactor,pointC[1]*zoomFactor,destinationRadius*zoomFactor)
    (dX,dY) = getRandomPoint(pointD[0]*zoomFactor,pointD[1]*zoomFactor,destinationRadius*zoomFactor)
    (eX,eY) = getRandomPoint(pointE[0]*zoomFactor,pointE[1]*zoomFactor,destinationRadius*zoomFactor)
    (fX,fY) = getRandomPoint(pointF[0]*zoomFactor,pointF[1]*zoomFactor,destinationRadius*zoomFactor)
    (gX,gY) = getRandomPoint(pointG[0]*zoomFactor,pointG[1]*zoomFactor,destinationRadius*zoomFactor)
    # draw left hand arc
    syntheticDigit = drawLine( syntheticDigit, [(aX,aY), (bX,bY), (cX,cY), (dX,dY)], brushWidth)                         
    # draw right hand arc
    syntheticDigit = drawLine( syntheticDigit, [(dX,dY), (eX,eY), (fX,fY), (gX,gY)], brushWidth)

    # now, need to downsample
    #syntheticDigit = resize(syntheticDigit,(28,28))
    #syntheticDigit = imresize(syntheticDigit,1/zoomFactor)
    syntheticDigit = cv2.resize(syntheticDigit,dsize=(28,28))
    syntheticDigit = syntheticDigit.reshape(28,28,1)
    
    return syntheticDigit

def get1(zoomFactor,destinationRadius): 
    syntheticDigit = np.zeros((28 * zoomFactor,28 * zoomFactor,1))
    brushWidth = random.uniform(2.8,5.0) * zoomFactor
    # this is for digit '1'!
    pointA = [6,11]
    pointB = [4,15]
    pointC = [21,14]

    # bX,bY are the coordinates for the top of the '1'
    (bX,bY) = getRandomPoint(pointB[0]*zoomFactor,pointB[1]*zoomFactor,destinationRadius*zoomFactor)
    # now, randomly include that thingy that people occasionally put at the top of ones
    putThingy = bool(random.getrandbits(1))
    if putThingy:
        # get point A, get radius and angle
        (aX,aY) = getRandomPoint(pointA[0]*zoomFactor,pointA[1]*zoomFactor,destinationRadius*zoomFactor)
        # now need to calculate the Bezier point P1
        bezStartX = ( aX + bX )/2
        bezStartY = ( aY + bY )/2
        distAB = math.sqrt( ( bX-aX )*(bX-aX) + (bY-aY)*(bY-aY) )
        bezRadius = random.uniform(0,distAB/2.1)
        (bezX,bezY) = getRandomPoint(bezStartX,bezStartY,bezRadius)
        syntheticDigit = drawLine( syntheticDigit, [(aX,aY), (bezX,bezY), (bX,bY)], brushWidth)
    # okay, now draw the long line from the top of the one to the bottom
    # destination 
    (cX,cY) = getRandomPoint(pointC[0]*zoomFactor,pointC[1]*zoomFactor,destinationRadius*3*zoomFactor)
    # now get Bezier point between B and C
    bezStartX = ( bX + cX )/2
    bezStartY = ( bY + cY )/2
    distBC = math.sqrt( ( cX-bX )*(cX-bX) + (cY-bY)*(cY-bY) )
    bezRadius = random.uniform(0,distBC/3.5)  # want this to be closer to the middle
    (bezX,bezY) = getRandomPoint(bezStartX,bezStartY,bezRadius)
    syntheticDigit = drawLine( syntheticDigit, [(bX,bY), (bezX,bezY), (cX,cY)], brushWidth)
    # now, figure out if we need to draw a base.  You tend to only see a base when there's a thingy.
    if putThingy:
        putBase = bool(random.getrandbits(1))
        if putBase:
            baseHalfLength = zoomFactor * 4
            (dX,dY) = getRandomPoint(cX,cY-baseHalfLength,destinationRadius*zoomFactor)
            (eX,eY) = getRandomPoint(cX,cY+baseHalfLength,destinationRadius*zoomFactor)
            syntheticDigit = drawLine( syntheticDigit, [(dX,dY), (cX,cY), (eX,eY)], brushWidth)
            
    # now, need to downsample
    #syntheticDigit = resize(syntheticDigit,(28,28))
    #syntheticDigit = imresize(syntheticDigit,1/zoomFactor)
    syntheticDigit = cv2.resize(syntheticDigit,dsize=(28,28))
    syntheticDigit = syntheticDigit.reshape(28,28,1)
    
    return syntheticDigit

def get2(zoomFactor,destinationRadius): 
    syntheticDigit = np.zeros((28 * zoomFactor,28 * zoomFactor,1))
    brushWidth = random.uniform(2.8,5.0) * zoomFactor
    # this is for digit '2'!
    pointA = [7,7]
    pointB = [5,21]
    pointC = [16,16]
    pointD = [21,7]
    pointE = [18,7]
    pointF = [19,14]
    pointG = [21,21]

    destinationRadius *= 2
    
    # aX,aY are the coordinates for the top of the '0', d is the bottom; b and c are Bezier supports
    (aX,aY) = getRandomPoint(pointA[0]*zoomFactor,pointA[1]*zoomFactor,destinationRadius*zoomFactor)
    (bX,bY) = getRandomPoint(pointB[0]*zoomFactor,pointB[1]*zoomFactor,destinationRadius*zoomFactor)
    (cX,cY) = getRandomPoint(pointC[0]*zoomFactor,pointC[1]*zoomFactor,destinationRadius*zoomFactor)
    (dX,dY) = getRandomPoint(pointD[0]*zoomFactor,pointD[1]*zoomFactor,destinationRadius*zoomFactor)
    (eX,eY) = getRandomPoint(pointE[0]*zoomFactor,pointE[1]*zoomFactor,destinationRadius*zoomFactor)
    (fX,fY) = getRandomPoint(pointF[0]*zoomFactor,pointF[1]*zoomFactor,destinationRadius*zoomFactor)
    (gX,gY) = getRandomPoint(pointG[0]*zoomFactor,pointG[1]*zoomFactor,destinationRadius*zoomFactor)
    # draw left hand arc
    syntheticDigit = drawLine( syntheticDigit, [(aX,aY), (bX,bY), (cX,cY), (dX,dY)], brushWidth)                         
    # draw right hand arc
    syntheticDigit = drawLine( syntheticDigit, [(dX,dY), (eX,eY), (fX,fY), (gX,gY)], brushWidth)

    # now, need to downsample
    #syntheticDigit = resize(syntheticDigit,(28,28))
    #syntheticDigit = imresize(syntheticDigit,1/zoomFactor)
    syntheticDigit = cv2.resize(syntheticDigit,dsize=(28,28))
    syntheticDigit = syntheticDigit.reshape(28,28,1)
    
    return syntheticDigit

def get3(zoomFactor,destinationRadius): 
    syntheticDigit = np.zeros((28 * zoomFactor,28 * zoomFactor,1))
    brushWidth = random.uniform(2.8,5.0) * zoomFactor
    # this is for digit '2'!
    pointA = [7,9]
    pointB = [7,24]
    pointC = [14,24]
    pointD = [14,11]
    pointE = [14,24]
    pointF = [21,24]
    pointG = [21,9]

    destinationRadius *= 2
    
    # aX,aY are the coordinates for the top of the '0', d is the bottom; b and c are Bezier supports
    (aX,aY) = getRandomPoint(pointA[0]*zoomFactor,pointA[1]*zoomFactor,destinationRadius*zoomFactor)
    (bX,bY) = getRandomPoint(pointB[0]*zoomFactor,pointB[1]*zoomFactor,destinationRadius*zoomFactor)
    (cX,cY) = getRandomPoint(pointC[0]*zoomFactor,pointC[1]*zoomFactor,destinationRadius*zoomFactor)
    (dX,dY) = getRandomPoint(pointD[0]*zoomFactor,pointD[1]*zoomFactor,destinationRadius*zoomFactor)
    (eX,eY) = getRandomPoint(pointE[0]*zoomFactor,pointE[1]*zoomFactor,destinationRadius*zoomFactor)
    (fX,fY) = getRandomPoint(pointF[0]*zoomFactor,pointF[1]*zoomFactor,destinationRadius*zoomFactor)
    (gX,gY) = getRandomPoint(pointG[0]*zoomFactor,pointG[1]*zoomFactor,destinationRadius*zoomFactor)
    # draw left hand arc
    syntheticDigit = drawLine( syntheticDigit, [(aX,aY), (bX,bY), (cX,cY), (dX,dY)], brushWidth)                         
    # draw right hand arc
    syntheticDigit = drawLine( syntheticDigit, [(dX,dY), (eX,eY), (fX,fY), (gX,gY)], brushWidth)

    # now, need to downsample
    #syntheticDigit = resize(syntheticDigit,(28,28))
    #syntheticDigit = imresize(syntheticDigit,1/zoomFactor)
    syntheticDigit = cv2.resize(syntheticDigit,dsize=(28,28))
    syntheticDigit = syntheticDigit.reshape(28,28,1)
    
    return syntheticDigit

def get4(zoomFactor,destinationRadius):
    # there are two main ways to make a number 4: one with the pen leaving the paper once
    # (like a checkmark and a cross through it).  The other way is in one motion, with the
    # pen never leaving the paper.  I figure the 2nd way of doing it isn't as prevalent, so
    # make the probability of that one happening something like 15%.  Hey, I just noticed
    # that this four -> 4 <- is of the 2nd variety!
    probabilityOfWeird4 = 0.15
    pickANumber = random.random()  # between 0 and 1
    if pickANumber > probabilityOfWeird4:
        return get4a(zoomFactor,destinationRadius)
    else:
        return get4b(zoomFactor,destinationRadius)
    
def get4a(zoomFactor,destinationRadius): 
    syntheticDigit = np.zeros((28 * zoomFactor,28 * zoomFactor,1))
    brushWidth = random.uniform(2.8,5.0) * zoomFactor
    # this is for digit '4', the common one
    pointA = [7,10]
    pointB = [14,9]
    pointC = [14,19]
    pointD = [7,16]
    pointE = [21,15]
    
    (aX,aY) = getRandomPoint(pointA[0]*zoomFactor,pointA[1]*zoomFactor,destinationRadius*zoomFactor)
    (bX,bY) = getRandomPoint(pointB[0]*zoomFactor,pointB[1]*zoomFactor,destinationRadius*zoomFactor)
    (cX,cY) = getRandomPoint(pointC[0]*zoomFactor,pointC[1]*zoomFactor,destinationRadius*zoomFactor)
    (dX,dY) = getRandomPoint(pointD[0]*zoomFactor,pointD[1]*zoomFactor,destinationRadius*zoomFactor)
    (eX,eY) = getRandomPoint(pointE[0]*zoomFactor,pointE[1]*zoomFactor,destinationRadius*zoomFactor)

    # get Bezier support between A and B.  Don't want this to be a big arc though because generally
    # that first line of a 4 is pretty straight.  People don't start messing up until the 2nd line
    bezStartX = ( aX + bX )/2
    bezStartY = ( aY + bY )/2
    distAB = math.sqrt( ( bX-aX )*(bX-aX) + (bY-aY)*(bY-aY) )
    bezRadius = random.uniform(0,distAB/4)
    (bezX,bezY) = getRandomPoint(bezStartX,bezStartY,bezRadius)
    syntheticDigit = drawLine( syntheticDigit, [(aX,aY), (bezX,bezY), (bX,bY)], brushWidth)
    # okay, now draw the 2nd line from left to right
    # first get Bezier point between B and C
    bezStartX = ( bX + cX )/2
    bezStartY = ( bY + cY )/2
    distBC = math.sqrt( ( cX-bX )*(cX-bX) + (cY-bY)*(cY-bY) )
    bezRadius = random.uniform(0,distBC/2.1)  # this can have a little bit of arc to it
    (bezX,bezY) = getRandomPoint(bezStartX,bezStartY,bezRadius)
    syntheticDigit = drawLine( syntheticDigit, [(bX,bY), (bezX,bezY), (cX,cY)], brushWidth)
    # now, draw the big line down the middle
    # first get Bezier support
    bezStartX = ( dX + eX )/2
    bezStartY = ( dY + eY )/2
    distDE = math.sqrt( ( eX-dX )*(eX-dX) + (eY-dY)*(eY-dY) )
    bezRadius = random.uniform(0,distDE/4)  # this can have a little bit of arc to it
    (bezX,bezY) = getRandomPoint(bezStartX,bezStartY,bezRadius)
    syntheticDigit = drawLine( syntheticDigit, [(dX,dY), (bezX,bezY), (eX,eY)], brushWidth)
    
    # now, need to downsample
    #syntheticDigit = resize(syntheticDigit,(28,28))
    #syntheticDigit = imresize(syntheticDigit,1/zoomFactor)
    syntheticDigit = cv2.resize(syntheticDigit,dsize=(28,28))
    syntheticDigit = syntheticDigit.reshape(28,28,1)
    
    return syntheticDigit

def get4b(zoomFactor,destinationRadius): 
    syntheticDigit = np.zeros((28 * zoomFactor,28 * zoomFactor,1))
    brushWidth = random.uniform(2.8,5.0) * zoomFactor
    # this is for digit '4', the secondary one.  Since I don't write these, I'm not sure
    # which end to start and while it seems arbitrary, I want to follow what humans do as
    # much as possible.  Guess we'll start at the right hand side and work towards the bottom
    pointA = [14,18]
    pointB = [14,9]
    pointC = [7,15]
    pointD = [21,14]
    
    (aX,aY) = getRandomPoint(pointA[0]*zoomFactor,pointA[1]*zoomFactor,destinationRadius*zoomFactor)
    (bX,bY) = getRandomPoint(pointB[0]*zoomFactor,pointB[1]*zoomFactor,destinationRadius*zoomFactor)
    (cX,cY) = getRandomPoint(pointC[0]*zoomFactor,pointC[1]*zoomFactor,destinationRadius*zoomFactor)
    (dX,dY) = getRandomPoint(pointD[0]*zoomFactor,pointD[1]*zoomFactor,destinationRadius*zoomFactor)

    # get Bezier support between A and B.  Don't want this to be a big arc though because generally
    # that first line of a 4 is pretty straight.  People don't start messing up until the 2nd line
    bezStartX = ( aX + bX )/2
    bezStartY = ( aY + bY )/2
    distAB = math.sqrt( ( bX-aX )*(bX-aX) + (bY-aY)*(bY-aY) )
    bezRadius = random.uniform(0,distAB/4)
    (bezX,bezY) = getRandomPoint(bezStartX,bezStartY,bezRadius)
    syntheticDigit = drawLine( syntheticDigit, [(aX,aY), (bezX,bezY), (bX,bY)], brushWidth)
    # okay, now draw the 2nd line from left to right
    # first get Bezier point between B and C
    bezStartX = ( bX + cX )/2
    bezStartY = ( bY + cY )/2
    distBC = math.sqrt( ( cX-bX )*(cX-bX) + (cY-bY)*(cY-bY) )
    bezRadius = random.uniform(0,distBC/3.1)  # this can have a little bit of arc to it
    (bezX,bezY) = getRandomPoint(bezStartX,bezStartY,bezRadius)
    syntheticDigit = drawLine( syntheticDigit, [(bX,bY), (bezX,bezY), (cX,cY)], brushWidth)
    # now, draw the big line down the middle
    # first get Bezier support
    bezStartX = ( cX + dX )/2
    bezStartY = ( cY + dY )/2
    distCD = math.sqrt( ( dX-cX )*(dX-cX) + (dY-cY)*(dY-cY) )
    bezRadius = random.uniform(0,distCD/4)  # this can have a little bit of arc to it
    (bezX,bezY) = getRandomPoint(bezStartX,bezStartY,bezRadius)
    syntheticDigit = drawLine( syntheticDigit, [(cX,cY), (bezX,bezY), (dX,dY)], brushWidth)
    
    # now, need to downsample
    #syntheticDigit = resize(syntheticDigit,(28,28))
    #syntheticDigit = imresize(syntheticDigit,1/zoomFactor)
    syntheticDigit = cv2.resize(syntheticDigit,dsize=(28,28))
    syntheticDigit = syntheticDigit.reshape(28,28,1)
    
    return syntheticDigit

def get5(zoomFactor,destinationRadius): 
    syntheticDigit = np.zeros((28 * zoomFactor,28 * zoomFactor,1))
    brushWidth = random.uniform(2.8,5.0) * zoomFactor
    # this is for digit '4', the common one
    pointA = [8,8]
    pointB = [7,21]
    pointC = [8,8]  # not used
    pointD = [14,7]
    pointE = [14,23]
    pointF = [21,21]
    pointG = [21,7]
    
    (aX,aY) = getRandomPoint(pointA[0]*zoomFactor,pointA[1]*zoomFactor,destinationRadius*zoomFactor)
    (bX,bY) = getRandomPoint(pointB[0]*zoomFactor,pointB[1]*zoomFactor,destinationRadius*zoomFactor)
    (cX,cY) = getRandomPoint(aX,aY,destinationRadius*0.5*zoomFactor)  # base off of aX,aY
    (dX,dY) = getRandomPoint(pointD[0]*zoomFactor,pointD[1]*zoomFactor,2*destinationRadius*zoomFactor)
    (eX,eY) = getRandomPoint(pointE[0]*zoomFactor,pointE[1]*zoomFactor,destinationRadius*zoomFactor)
    (fX,fY) = getRandomPoint(pointF[0]*zoomFactor,pointF[1]*zoomFactor,destinationRadius*zoomFactor)
    (gX,gY) = getRandomPoint(pointG[0]*zoomFactor,pointG[1]*zoomFactor,destinationRadius*zoomFactor)

    # get Bezier support between A and B.  Don't want this to be a big arc though because generally
    # that first line of a 5 is pretty straight.  
    bezStartX = ( aX + bX )/2
    bezStartY = ( aY + bY )/2
    distAB = math.sqrt( ( bX-aX )*(bX-aX) + (bY-aY)*(bY-aY) )
    bezRadius = random.uniform(0,distAB/4)
    (bezX,bezY) = getRandomPoint(bezStartX,bezStartY,bezRadius)
    syntheticDigit = drawLine( syntheticDigit, [(aX,aY), (bezX,bezY), (bX,bY)], brushWidth)
    # okay, now draw the 2nd line from left to right
    # first get Bezier point between C and D
    bezStartX = ( cX + dX )/2
    bezStartY = ( cY + dY )/2
    distCD = math.sqrt( ( dX-cX )*(dX-cX) + (dY-cY)*(dY-cY) )
    bezRadius = random.uniform(0,distCD/3.1)  # this can have a little bit of arc to it
    (bezX,bezY) = getRandomPoint(bezStartX,bezStartY,bezRadius)
    syntheticDigit = drawLine( syntheticDigit, [(cX,cY), (bezX,bezY), (dX,dY)], brushWidth)
    # now, draw the big arc of the 5 w/ cubic Bezier curve
    # first get Bezier support
    syntheticDigit = drawLine( syntheticDigit, [(dX,dY), (eX,eY), (fX,fY), (gX,gY)], brushWidth)

    # now, need to downsample
    #syntheticDigit = resize(syntheticDigit,(28,28))
    #syntheticDigit = imresize(syntheticDigit,1/zoomFactor)
    syntheticDigit = cv2.resize(syntheticDigit,dsize=(28,28))
    syntheticDigit = syntheticDigit.reshape(28,28,1)
    
    return syntheticDigit
    
def get6(zoomFactor,destinationRadius):
    # not super pleased by the lack of diversity of the loops that I'm seeing
    # would like to make a model which has an oblong loop aligned with initial large sweeping arc
    syntheticDigit = np.zeros((28 * zoomFactor,28 * zoomFactor,1))
    brushWidth = random.uniform(2.8,5.0) * zoomFactor
    # this is for digit '6'
    pointA = [7,21]
    pointB = [8,5]
    pointC = [21,9] 
    pointD = [21,21]
    pointE = [9,23]
    pointF = [14,12]
    pointG = [17,10]
    
    (aX,aY) = getRandomPoint(pointA[0]*zoomFactor,pointA[1]*zoomFactor,destinationRadius*zoomFactor)
    (bX,bY) = getRandomPoint(pointB[0]*zoomFactor,pointB[1]*zoomFactor,destinationRadius*zoomFactor)
    (cX,cY) = getRandomPoint(pointC[0]*zoomFactor,pointC[1]*zoomFactor,destinationRadius*zoomFactor)
    (dX,dY) = getRandomPoint(pointD[0]*zoomFactor,pointD[1]*zoomFactor,destinationRadius*zoomFactor)
    (eX,eY) = getRandomPoint(pointE[0]*zoomFactor,pointE[1]*zoomFactor,destinationRadius*zoomFactor)
    (fX,fY) = getRandomPoint(pointF[0]*zoomFactor,pointF[1]*zoomFactor,destinationRadius*zoomFactor)
    (gX,gY) = getRandomPoint(pointG[0]*zoomFactor,pointG[1]*zoomFactor,destinationRadius*zoomFactor)

    syntheticDigit = drawLine( syntheticDigit, [(aX,aY), (bX,bY), (cX,cY), (dX,dY)], brushWidth)
    syntheticDigit = drawLine( syntheticDigit, [(dX,dY), (eX,eY), (fX,fY), (gX,gY)], brushWidth)

    # now, need to downsample
    #syntheticDigit = resize(syntheticDigit,(28,28))
    #syntheticDigit = imresize(syntheticDigit,1/zoomFactor)
    syntheticDigit = cv2.resize(syntheticDigit,dsize=(28,28))
    syntheticDigit = syntheticDigit.reshape(28,28,1)    
    
    return syntheticDigit

def get7(zoomFactor,destinationRadius):
    syntheticDigit = np.zeros((28 * zoomFactor,28 * zoomFactor,1))
    brushWidth = random.uniform(3.2,5.0) * zoomFactor
    # this is for digit '7'!
    pointA = [7,9]
    pointB = [7,21]
    pointC = [21,14]

    (aX,aY) = getRandomPoint(pointA[0]*zoomFactor,pointA[1]*zoomFactor,destinationRadius*zoomFactor)
    (bX,bY) = getRandomPoint(pointB[0]*zoomFactor,pointB[1]*zoomFactor,destinationRadius*zoomFactor)
    (cX,cY) = getRandomPoint(pointC[0]*zoomFactor,pointC[1]*zoomFactor,destinationRadius*zoomFactor)

    # now, randomly include that thingy that people occasionally hang off the front of their sevens.  I'm guessing
    # it shows up 15% of the time.
    putThingy = False
    probabilityOfThingy = 0.15
    pickANumber = random.random()  # between 0 and 1
    if pickANumber < probabilityOfThingy:
        putThingy = True
    if putThingy:
        thingyLength = 1.6
        thingyX = aX + thingyLength * zoomFactor
        thingyY = aY 
        syntheticDigit = drawLine( syntheticDigit, [(aX,aY), (thingyX,thingyY)], brushWidth)
    
    # get Bezier support between A and B (top line of 7)
    bezStartX = ( aX + bX )/2
    bezStartY = ( aY + bY )/2
    distAB = math.sqrt( ( bX-aX )*(bX-aX) + (bY-aY)*(bY-aY) )
    bezRadius = random.uniform(0,distAB/3)
    (bezX,bezY) = getRandomPoint(bezStartX,bezStartY,bezRadius)
    syntheticDigit = drawLine( syntheticDigit, [(aX,aY), (bezX,bezY), (bX,bY)], brushWidth)
    
     # okay, now draw the line to the bottom
    # first get Bezier point between B and C
    bezStartX = ( bX + cX )/2
    bezStartY = ( bY + cY )/2
    distBC = math.sqrt( ( cX-bX )*(cX-bX) + (cY-bY)*(cY-bY) )
    bezRadius = random.uniform(0,distBC/2.1)  # this can have a lot of arc to it
    (bezX,bezY) = getRandomPoint(bezStartX,bezStartY,bezRadius)
    syntheticDigit = drawLine( syntheticDigit, [(bX,bY), (bezX,bezY), (cX,cY)], brushWidth)
    
    # now, randomly include that dash that people occasionally put through their sevens.  I'm guessing
    # it shows up 10% of the time.
    putDash = False
    probabilityOfDash = 0.10
    pickANumber = random.random()  # between 0 and 1
    if pickANumber < probabilityOfDash:
        putDash = True
    if putDash:
        # now need to calculate the Bezier point P1
        bezX = ( bX + cX )/2
        bezY = ( bY + cY )/2
        distBC = math.sqrt( ( cX-bX )*(cX-bX) + (cY-bY)*(cY-bY) )
        bezRadius = random.uniform(0,distBC/7.1)
        dashLength = 5
        (dX,dY) = getRandomPoint(bezX,bezY - (dashLength/2)*zoomFactor,bezRadius)
        (eX,eY) = getRandomPoint(bezX,bezY + (dashLength/2)*zoomFactor,bezRadius)
        syntheticDigit = drawLine( syntheticDigit, [(dX,dY), (bezX,bezY), (eX,eY)], brushWidth)
    
    # now, need to downsample
    #syntheticDigit = resize(syntheticDigit,(28,28))
    #syntheticDigit = imresize(syntheticDigit,1/zoomFactor)
    syntheticDigit = cv2.resize(syntheticDigit,dsize=(28,28))
    syntheticDigit = syntheticDigit.reshape(28,28,1)    
    
    return syntheticDigit

def get8(zoomFactor,destinationRadius):
    destinationRadius *= 2
    syntheticDigit = np.zeros((28 * zoomFactor,28 * zoomFactor,1))
    brushWidth = random.uniform(3.2,5.0) * zoomFactor
    # this is for digit '8'
    pointA = [7,17]
    pointB = [7,8]
    pointC = [14,8] 
    pointD = [14,13]
    pointE = [14,20]
    pointF = [21,20]
    pointG = [21,13]
    pointH = [21,8]
    pointI = [14,8]
    pointJ = [14,14]
    pointK = [14,20]
    pointL = [7,20]
    pointM = [7,17]
    
    (aX,aY) = getRandomPoint(pointA[0]*zoomFactor,pointA[1]*zoomFactor,destinationRadius*zoomFactor)
    (bX,bY) = getRandomPoint(pointB[0]*zoomFactor,pointB[1]*zoomFactor,destinationRadius*zoomFactor)
    (cX,cY) = getRandomPoint(pointC[0]*zoomFactor,pointC[1]*zoomFactor,destinationRadius*zoomFactor)
    (dX,dY) = getRandomPoint(pointD[0]*zoomFactor,pointD[1]*zoomFactor,destinationRadius*zoomFactor)
    (eX,eY) = getRandomPoint(pointE[0]*zoomFactor,pointE[1]*zoomFactor,destinationRadius*zoomFactor)
    (fX,fY) = getRandomPoint(pointF[0]*zoomFactor,pointF[1]*zoomFactor,destinationRadius*zoomFactor)
    (gX,gY) = getRandomPoint(pointG[0]*zoomFactor,pointG[1]*zoomFactor,destinationRadius*0.5*zoomFactor)
    (hX,hY) = getRandomPoint(pointH[0]*zoomFactor,pointH[1]*zoomFactor,destinationRadius*zoomFactor)
    (iX,iY) = getRandomPoint(pointI[0]*zoomFactor,pointI[1]*zoomFactor,destinationRadius*zoomFactor)
    (jX,jY) = getRandomPoint(pointJ[0]*zoomFactor,pointJ[1]*zoomFactor,destinationRadius*zoomFactor)
    (kX,kY) = getRandomPoint(pointK[0]*zoomFactor,pointK[1]*zoomFactor,destinationRadius*zoomFactor)
    (lX,lY) = getRandomPoint(pointL[0]*zoomFactor,pointL[1]*zoomFactor,destinationRadius*zoomFactor)
    (mX,mY) = getRandomPoint(pointM[0]*zoomFactor,pointM[1]*zoomFactor,destinationRadius*zoomFactor)
    
    syntheticDigit = drawLine( syntheticDigit, [(aX,aY), (bX,bY), (cX,cY), (dX,dY)], brushWidth)
    syntheticDigit = drawLine( syntheticDigit, [(dX,dY), (eX,eY), (fX,fY), (gX,gY)], brushWidth)
    syntheticDigit = drawLine( syntheticDigit, [(gX,gY), (hX,hY), (iX,iY), (jX,jY)], brushWidth)
    syntheticDigit = drawLine( syntheticDigit, [(jX,jY), (kX,kY), (lX,lY), (mX,mY)], brushWidth)

    # now, need to downsample
    #syntheticDigit = resize(syntheticDigit,(28,28))
    #syntheticDigit = imresize(syntheticDigit,1/zoomFactor)
    syntheticDigit = cv2.resize(syntheticDigit,dsize=(28,28))
    syntheticDigit = syntheticDigit.reshape(28,28,1)    
    
    return syntheticDigit

def get9(zoomFactor,destinationRadius):
    syntheticDigit = np.zeros((28 * zoomFactor,28 * zoomFactor,1))
    brushWidth = random.uniform(3.2,5.0) * zoomFactor
    # this is for digit '9'!
    pointA = [7,15]
    pointB = [6,3]
    pointC = [21,11]
    pointD = [7,16]
    pointE = [21,14]

    (aX,aY) = getRandomPoint(pointA[0]*zoomFactor,pointA[1]*zoomFactor,destinationRadius*zoomFactor)
    (bX,bY) = getRandomPoint(pointB[0]*zoomFactor,pointB[1]*zoomFactor,destinationRadius*zoomFactor)
    (cX,cY) = getRandomPoint(pointC[0]*zoomFactor,pointC[1]*zoomFactor,destinationRadius*zoomFactor)
    (dX,dY) = getRandomPoint(pointD[0]*zoomFactor,pointD[1]*zoomFactor,destinationRadius*zoomFactor)
    (eX,eY) = getRandomPoint(pointE[0]*zoomFactor,pointE[1]*zoomFactor,destinationRadius*zoomFactor)
    
    # start with loop
    syntheticDigit = drawLine( syntheticDigit, [(aX,aY), (bX,bY), (cX,cY), (dX,dY)], brushWidth)
    
    # get Bezier support between D and E (vertical part of the 9)
    bezStartX = ( dX + eX )/2
    bezStartY = ( dY + eY )/2
    distDE = math.sqrt( ( eX-dX )*(eX-dX) + (eY-dY)*(eY-dY) )
    bezRadius = random.uniform(0,distDE/3)
    (bezX,bezY) = getRandomPoint(bezStartX,bezStartY,bezRadius)
    syntheticDigit = drawLine( syntheticDigit, [(dX,dY), (bezX,bezY), (eX,eY)], brushWidth)
    
    # now, need to downsample
    #syntheticDigit = resize(syntheticDigit,(28,28))
    #syntheticDigit = imresize(syntheticDigit,1/zoomFactor)
    syntheticDigit = cv2.resize(syntheticDigit,dsize=(28,28))
    syntheticDigit = syntheticDigit.reshape(28,28,1)    
    
    return syntheticDigit

def getDigit(number,zoomFactor,radius):
    if number == 0:
        return get0(zoomFactor,radius)
    elif number == 1:
        return get1(zoomFactor,radius)
    elif number == 2:
        return get2(zoomFactor,radius)
    elif number == 3:
        return get3(zoomFactor,radius)
    elif number == 4:
        return get4(zoomFactor,radius)
    elif number == 5:
        return get5(zoomFactor,radius)
    elif number == 6:
        return get6(zoomFactor,radius)
    elif number == 7:
        return get7(zoomFactor,radius)
    elif number == 8:
        return get8(zoomFactor,radius)
    elif number == 9:
        return get9(zoomFactor,radius)
    
zoomFactor = 10
jitterRadii = [0, 0.5, 1, 2, 3, 4, 5]

numDigitsToSample = 100   # was 100, reduced for Kaggle kernel

truth = []
call = []
maxCols = 5

for jitterRadius in jitterRadii:
    for i in range(10):
        numRows = 2
        numCols = maxCols
        numExamples = numRows * numCols
        fig, ax = plt.subplots(numRows,numCols,sharex=True,sharey=True,figsize=(14,3 * numRows))
        fig.suptitle( "Synthetic " + str(i) + " with jitter " + str(jitterRadius), fontsize=18)
        for j in range(numDigitsToSample):
            syntheticDigit = getDigit(i,zoomFactor,jitterRadius)
            d = syntheticDigit.reshape(-1,28,28,1)
            pred = model.predict(d)
            maxProb = pred.max(axis=1)[0]
            truthProb = pred[0][i]
            predCall = np.argmax(pred,axis=1)[0]
            call.append( predCall )
            truth.append( i )
            if j < numExamples:
                row = math.floor( j / maxCols )
                col = j % maxCols 
                plotTitle = "Predicted: {:1d} ({:0.2f})\nTruth: {:1d} ({:0.2f})".format(predCall,maxProb,i,truthProb)
                if ax.ndim == 1:
                    ax[col].imshow(syntheticDigit[:,:,0])
                    ax[col].set_title(plotTitle) 
                else:
                    ax[row,col].imshow(syntheticDigit[:,:,0])
                    ax[row,col].set_title(plotTitle)
            
    confusion_mat = confusion_matrix(truth, call)
    df_cm = pd.DataFrame(confusion_mat, index = [i for i in range(0,10)], columns = [i for i in range(0,10)])
    plt.figure(figsize = (6,5))
    conf_mat = sns.heatmap(df_cm, annot=True, cmap='Blues', fmt='g', cbar = False)
    conf_mat.set(xlabel='Predictions', ylabel='Truth')
    ax = plt.axes()
    ax.set_title('Synthetic Data with jitter ' + str(jitterRadius))
    truth = []
    call = []
# predict the test set
test_pred = model.predict(test)
# select the indix with the maximum probability
test_calls = np.argmax(test_pred,axis = 1)
results = pd.Series(test_calls,name="Label")

dataFrame = pd.concat([pd.Series(range(1,testRows.shape[0]+1),name = "ImageId"),results],axis = 1)
dataFrame.to_csv("MNIST_CNN.csv",index=False)