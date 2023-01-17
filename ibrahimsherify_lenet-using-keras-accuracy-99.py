import os
import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Activation, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
PATH = "../input/shapes/"
IMG_SIZE = 64
Shapes = ["circle", "square", "triangle", "star"]
Labels = []
Dataset = []

# From kernel: https://www.kaggle.com/smeschke/load-data
for shape in Shapes:
    print("Getting data for: ", shape)
    #iterate through each file in the folder
    for path in os.listdir(PATH + shape):
        #add the image to the list of images
        image = cv2.imread(PATH + shape + '/' + path)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        Dataset.append(image)
        #add an integer to the labels list 
        Labels.append(Shapes.index(shape))

print("\nDataset Images size:", len(Dataset))
print("Image Shape:", Dataset[0].shape)
print("Labels size:", len(Labels))
sns.countplot(x= Labels)
print("Count of Star images:", Labels.count(Shapes.index("star")))
print("Count of Circles images:", Labels.count(Shapes.index("circle")))
print("Count of Squares images:", Labels.count(Shapes.index("square")))
print("Count of Triangle images:", Labels.count(Shapes.index("triangle")))
index = np.random.randint(0, len(Dataset) - 1, size= 20)
plt.figure(figsize=(15,10))

for i, ind in enumerate(index, 1):
    img = Dataset[ind]
    lab = Labels[ind]
    lab = Shapes[lab]
    plt.subplot(4, 5, i)
    plt.title(lab)
    plt.axis('off')
    plt.imshow(img)
# Normalize images
Dataset = np.array(Dataset)
Dataset = Dataset.astype("float32") / 255.0

# One hot encode labels
Labels = np.array(Labels)
Labels = to_categorical(Labels)

# Split Dataset to train\test
(trainX, testX, trainY, testY) = train_test_split(Dataset, Labels, test_size=0.2, random_state=42)

print("X Train shape:", trainX.shape)
print("X Test shape:", testX.shape)
print("Y Train shape:", trainY.shape)
print("Y Test shape:", testY.shape)
class LeNet():
    @staticmethod
    def build(numChannels, imgRows, imgCols, numClasses,  pooling= "max", activation= "relu"):
        # initialize the model
        model = Sequential()
        inputShape = (imgRows, imgCols, numChannels)

        # add first set of layers: Conv -> Activation -> Pool
        model.add(Conv2D(filters= 6, kernel_size= 5, input_shape= inputShape))
        model.add(Activation(activation))

        if pooling == "max":
            model.add(MaxPooling2D(pool_size= (2, 2), strides= (2, 2)))
        else:
            model.add(AveragePooling2D(pool_size= (2, 2), strides= (2, 2)))

        # add second set of layers: Conv -> Activation -> Pool
        model.add(Conv2D(filters= 16, kernel_size= 5))
        model.add(Activation(activation))

        if pooling == "avg":
            model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))
        else:
            model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # Flatten -> FC 120 -> Dropout -> Activation
        model.add(Flatten())
        model.add(Dense(120))
        model.add(Dropout(0.5))
        model.add(Activation(activation))

        # FC 84 -> Dropout -> Activation
        model.add(Dense(84))
        model.add(Dropout(0.5))
        model.add(Activation(activation))

        # FC 4-> Softmax
        model.add(Dense(numClasses))
        model.add(Activation("softmax"))

        return model
BS = 120
LR = 0.01
EPOCHS = 10
opt = SGD(lr= LR)
# First model with max pooling
model = LeNet.build(3, IMG_SIZE, IMG_SIZE, 4, pooling= "max")
model.compile(loss= "categorical_crossentropy", optimizer= opt, metrics= ["accuracy"])
model.summary()
# Train model
H1 = model.fit(trainX, trainY, validation_data= (testX, testY), batch_size= BS,
              epochs= EPOCHS, verbose=1)

# Evaluate the train and test data
scores_train = model.evaluate(trainX, trainY, verbose= 1)
scores_test = model.evaluate(testX, testY, verbose= 1)

print("\nModel with Max Pool Accuracy on Train Data: %.2f%%" % (scores_train[1]*100))
print("Model with Max Pool Accuracy on Test Data: %.2f%%" % (scores_test[1]*100))
# Second model with average pooling
model = LeNet.build(3, IMG_SIZE, IMG_SIZE, 4, pooling= "average")
model.compile(loss= "categorical_crossentropy", optimizer= opt, metrics= ["accuracy"])

model.summary()
# Train model
H2 = model.fit(trainX, trainY, validation_data= (testX, testY), batch_size= BS,
              epochs= EPOCHS, verbose= 1)

# Evaluate the train and test data
scores_train = model.evaluate(trainX, trainY, verbose= 1)
scores_test = model.evaluate(testX, testY, verbose= 1)

print("\nModel with Average Pool Accuracy on Train Data: %.2f%%" % (scores_train[1]*100))
print("Model with Average Pool Accuracy on Test Data: %.2f%%" % (scores_test[1]*100))
plt.figure(figsize=(15,5))
plt.plot(np.arange(0, EPOCHS), H1.history["acc"], label="Max Pool Train Acc")
plt.plot(np.arange(0, EPOCHS), H1.history["val_acc"], label="Max Pool Test Acc")
plt.plot(np.arange(0, EPOCHS), H2.history["acc"], label="Avg Pool Train Acc")
plt.plot(np.arange(0, EPOCHS), H2.history["val_acc"], label="Avg Pool Test Acc")
plt.title("Comparing Models Train\Test Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend(loc="upper left")
plt.figure(figsize=(15,5))
plt.plot(np.arange(0, EPOCHS), H1.history["loss"], label="Max Pool Train Loss")
plt.plot(np.arange(0, EPOCHS), H1.history["val_loss"], label="Max Pool Test Loss")
plt.plot(np.arange(0, EPOCHS), H2.history["loss"], label="Avg Pool Train Loss")
plt.plot(np.arange(0, EPOCHS), H2.history["val_loss"], label="Avg Pool Test Loss")
plt.title("Comparing Models Train\Test Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="upper left")