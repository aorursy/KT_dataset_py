import numpy as np
import matplotlib.pyplot as plt
import random
import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.callbacks import TensorBoard,EarlyStopping
# Let us the check the version
print("Tensorflow version : ", tf.__version__)

# if you set the GPU, it will show here
print("Number of GPUs available are :",tf.config.experimental.list_logical_devices("GPU")) 

# Set the seed to replicate the result
from numpy.random import seed
seed(27)

# load the data set
dataset = keras.datasets.fashion_mnist
(train_X, train_Y),(test_x,test_y)=dataset.load_data()

# We have 10 labels, so let us create an array to access the class names based on the labler number
classNames = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Check the size of the datasets
print("Train dataset size :", len(train_X))
print("Test  dataset size :", len(test_x))

# Let us check the shape of training image
print("shape  :", train_X[0].shape)
# let us plot some image randomly and it's label
plt.figure(figsize=(10,10))
for i in range(16):
  plt.subplot(4,4,i+1)
  plt.xticks([])
  plt.yticks([])
  randomNumber = random.randint(0,len(train_X))     
  plt.imshow(train_X[randomNumber], cmap='gray')
  plt.xlabel(classNames[train_Y[randomNumber]]) 
# We are working on classification problem, so let us normalize the pixel values before passing them to the model
train_X = train_X/255.0
test_x = test_x/255.0

# Clear the logs from the previous runs if any
!rm -rf ./logs

# create the model and assign the leyers
model = Sequential()
model.add(Flatten(input_shape=(28,28))) # input layer 
model.add(Dense(512,activation='relu')) # hidden layer
model.add(Dense(10,activation='softmax')) # output layer
model.summary()
opt = Adam(learning_rate=0.001) # default value is 0.001
model.compile(optimizer=opt,
              loss=SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# we are setting time to directory name to distinguish
log_dir = "logs/fit/"+datetime.datetime.now().strftime("%Y%M%D-%H%M%S") 

tensorboard_cb = TensorBoard(log_dir=log_dir, histogram_freq=1)

# We will add "EarlyStopping" to the callbacks list for enabling the early stopping of the model
earlystopping_cb = EarlyStopping(     
    monitor='val_loss',  # we are monitoring the validation loss, so our model stops training if the validation loss is started increasing
    min_delta=0, # minimum acceptable threshold for monitor value
    patience=3, # Number of epochs with no improvement after which training will be stopped
    verbose=0, 
    mode='auto', # if we set to auto, it will set to minimum in case of loss, it will set to maximum in case of accuracy 
    baseline=None, 
    restore_best_weights=True  # restores the best weight
)

# We will add the callbacks to change the default behaviour of the model.
model.fit(x=train_X, 
          y=train_Y, 
          epochs=25, 
          validation_split=0.2, 
          callbacks=[tensorboard_cb,earlystopping_cb])
# we need to call this to load the tensorboard
%load_ext tensorboard
%tensorboard --logdir logs/fit
testLoss, testAccuracy = model.evaluate(test_x, test_y, verbose=2)
print("Test loss :", testLoss)
print("Test accuracy :", testAccuracy)
predicted = model.predict(test_x)
predicted.shape
classNames[np.argmax(predicted[177])]
plt.imshow(test_x[177])
def plotImage(i, predictedArray, trueLabel, img):
  trueLabel, img = trueLabel[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predictedLabel = np.argmax(predictedArray)
  if(predictedLabel == trueLabel):
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(classNames[predictedLabel],
                                100*np.max(predictedArray),
                                classNames[trueLabel]),
                                color=color)
# Plot the first X test images, their predicted labels, and the true labels.
# Color correct prediction in blue and incorrect prediction in red.
numRows = 4
numCols = 4
num_images = numRows*numCols
plt.figure(figsize=(2*2*numCols, 2*numRows))
for i in range(num_images):
  plt.subplot(numRows, 2*numCols, 2*i+1)
  plotImage(i, predicted[i], test_y, test_x)
plt.tight_layout()
plt.show()
# Save the entire model as a SavedModel.
!mkdir -p savedModel
model.save('savedModel/myModel')
#Loading the model from saved location
loadedModel = tf.keras.models.load_model('savedModel/myModel')

# Check its architecture
loadedModel.summary()