# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
%matplotlib inline

np.random.seed(2)

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
import cv2
import tensorflow as tf

sns.set(style='white', context='notebook', palette='deep')
# Load the data
train = pd.read_csv("/kaggle/input/i2a2-brasil-pneumonia-classification/train.csv")
test = pd.read_csv("/kaggle/input/i2a2-brasil-pneumonia-classification/test.csv")
df_train = pd.read_csv("/kaggle/input/i2a2-brasil-pneumonia-classification/train.csv")
df_train.head()
Y_train = train["pneumonia"]

# Drop 'label' column
X_train = train.drop(labels = ["pneumonia"],axis = 1) 

# free some space
del train 

g = sns.countplot(Y_train)

Y_train.value_counts()
#https://towardsdatascience.com/complete-image-augmentation-in-opencv-31a6b02694f5

#https://machinelearningmastery.com/how-to-configure-image-data-augmentation-when-training-deep-learning-neural-networks/
# example of horizontal shift image augmentation
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
# load the image
img = load_img('/kaggle/input/i2a2-brasil-pneumonia-classification/images/1a4b8ecb-4eb9-49dd-af42-2c5801333ede.jpeg')
# convert to numpy array
data = img_to_array(img)
# expand dimension to one sample
samples = expand_dims(data, 0)
# create image data augmentation generator
datagen = ImageDataGenerator(width_shift_range=[-200,200])
# prepare iterator
it = datagen.flow(samples, batch_size=1)
# generate samples and plot
for i in range(9):
	# define subplot
	pyplot.subplot(330 + 1 + i)
	# generate batch of images
	batch = it.next()
	# convert to unsigned integers for viewing
	image = batch[0].astype('uint8')
	# plot raw pixel data
	pyplot.imshow(image)
# show the figure
pyplot.show()
#Imagem com pneumonia
#img = cv2.imread('/kaggle/input/i2a2-brasil-pneumonia-classification/images/7c05f8d3-57d4-4e4d-8dcd-7ae57699708c.jpeg',0) ## classe =2

img = cv2.imread('/kaggle/input/i2a2-brasil-pneumonia-classification/images/1a4b8ecb-4eb9-49dd-af42-2c5801333ede.jpeg',0) ## classe =2

#(2136, 3216)

#img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img = cv2.resize(img, (224, 224))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #im = im.reshape(desired_size, desired_size , 1)
    #im = cv2.addWeighted(im, 4, cv2.blur(im, ksize=(10,10)), -4, 128)
img = cv2.addWeighted ( img,4, cv2.GaussianBlur( img , (0,0) , 300/10) ,-4 ,128) # the trick is to add this line
#img = cv2.equalizeHist(img) 

print(img.shape)

plt.imshow(img)
# example of vertical shift image augmentation
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
# load the image
img = load_img('/kaggle/input/i2a2-brasil-pneumonia-classification/images/1a4b8ecb-4eb9-49dd-af42-2c5801333ede.jpeg')
# convert to numpy array
data = img_to_array(img)
# expand dimension to one sample
samples = expand_dims(data, 0)
# create image data augmentation generator
datagen = ImageDataGenerator(height_shift_range=0.15)
# prepare iterator
it = datagen.flow(samples, batch_size=1)
# generate samples and plot
for i in range(9):
	# define subplot
	pyplot.subplot(330 + 1 + i)
	# generate batch of images
	batch = it.next()
	# convert to unsigned integers for viewing
	image = batch[0].astype('uint8')
	# plot raw pixel data
	pyplot.imshow(image)
# show the figure
pyplot.show()
# example of rotation image augmentation
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
# load the image
img = load_img('/kaggle/input/i2a2-brasil-pneumonia-classification/images/1a4b8ecb-4eb9-49dd-af42-2c5801333ede.jpeg')
# convert to numpy array
data = img_to_array(img)
# expand dimension to one sample
samples = expand_dims(data, 0)
# create image data augmentation generator
datagen = ImageDataGenerator(rotation_range=45)
# prepare iterator
it = datagen.flow(samples, batch_size=1)
# generate samples and plot
for i in range(9):
	# define subplot
	pyplot.subplot(330 + 1 + i)
	# generate batch of images
	batch = it.next()
	# convert to unsigned integers for viewing
	image = batch[0].astype('uint8')
	# plot raw pixel data
	pyplot.imshow(image)
# show the figure
pyplot.show()
#Shear shear_range
# example of Shear shear_range image augmentation
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
# load the image
img = load_img('/kaggle/input/i2a2-brasil-pneumonia-classification/images/1a4b8ecb-4eb9-49dd-af42-2c5801333ede.jpeg')
# convert to numpy array
data = img_to_array(img)
# expand dimension to one sample
samples = expand_dims(data, 0)
# create image data augmentation generator
datagen = ImageDataGenerator(shear_range=20)
# prepare iterator
it = datagen.flow(samples, batch_size=1)
# generate samples and plot
for i in range(9):
	# define subplot
	pyplot.subplot(330 + 1 + i)
	# generate batch of images
	batch = it.next()
	# convert to unsigned integers for viewing
	image = batch[0].astype('uint8')
	# plot raw pixel data
	pyplot.imshow(image)
# show the figure
pyplot.show()
#https://minerandodados.com.br/lidando-com-classes-desbalanceadas-machine-learning/
# Check the data
X_train.isnull().any().describe()
test.isnull().any().describe()
print(X_train)
os.mkdir('/kaggle/working/pneum-preproc/')
os.mkdir('/kaggle/working/pneum-preproc/test')
os.mkdir('/kaggle/working/pneum-preproc/train')
os.listdir('/kaggle/working/pneum-preproc/')
def preprocess_image(image_path):
      image = cv2.imread(image_path)
      image = cv2.resize(image, (224,224))
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      image = cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , 300/10) ,-4 ,128) # the trick is to add this line
      #image = cv2.equalizeHist(image) 
      # Normalization
      image = image.astype(np.float32)/255.
      return image
from tqdm import tqdm

desired_size=224
depth = 3

N = df_train.shape[0]
X_train = np.empty((N, desired_size, desired_size, depth), dtype=np.uint8)
#X_train = np.empty((N, desired_size, desired_size), dtype=np.uint8)

for i, image_id in enumerate(tqdm(df_train['fileName'])):
    X_train[i, :, :, :] = preprocess_image(
        f'/kaggle/input/i2a2-brasil-pneumonia-classification/images/{image_id}'
    )
    #Save images preprocessadas
    cv2.imwrite('/kaggle/working/pneum-preproc/train/' + image_id ,X_train[i])
# pre processando dados teste
from tqdm import tqdm

desired_size=224
depth = 3

N = test.shape[0]
X_test = np.empty((N, desired_size, desired_size, depth), dtype=np.uint8)

for i, image_id in enumerate(tqdm(test['fileName'])):
    X_test[i, :, :, :] = preprocess_image(
        f'/kaggle/input/i2a2-brasil-pneumonia-classification/images/{image_id}'
    )
    #Save images preprocessadas
    cv2.imwrite('/kaggle/working/pneum-preproc/test/' + image_id ,X_test[i])
y_train = pd.get_dummies(df_train['pneumonia']).values
print(y_train)
# Split the train and the validation set for the fitting
# stratify = para classes desbalanceadas
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train,test_size=0.3,random_state = 1, stratify=Y_train)
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras.layers import GlobalMaxPooling2D, Dense,Flatten,MaxPooling2D,GlobalAveragePooling2D
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
from tensorflow.keras import Sequential
from keras.applications.vgg16 import VGG16
from keras.applications.xception import preprocess_input
from keras.applications.resnet50 import ResNet50
from tensorflow.python.keras.layers import Flatten, Dense, Conv2D, Dropout, BatchNormalization
import datetime, os

# importando a Xception pré-treinada
model_0 = tf.keras.applications.resnet50.ResNet50(input_shape = (224, 224, 3),
                                           include_top = False,
                                           weights = 'imagenet')
#https://www.kaggle.com/madz2000/pneumonia-detection-using-cnn-92-6-accuracy
model_0.trainable = True
model_2 = Sequential()
model_2.add(model_0)
model_2.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model_2.add(MaxPooling2D((2, 2)))


model_2.add(Flatten())
model_2.add(Dense(100))
model_2.add(Dense(1, activation = 'sigmoid'))

model_2.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

# visualizando o sumario da rede
model_2.summary()
logs_dir = '.\logs'
# early stopping - efetuada a parada do treinamento quando o value_loss não obtem mais melhorias
early_stopping = EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience= 10,
                              verbose=0, mode='auto')

# model checkpoint - armazena o melhor modelo ou peso treinado para ser usado no teste final
# https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint
mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True)

#tensorboard callback -->> não entendo bem de como fucniona o tensor board mas foi necessario manter aqui no código para mantero o callback
# 
logdir = os.path.join(logs_dir,datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
tensorboard_callback = TensorBoard(logdir, histogram_freq = 1)

#reduce lr on plateau - aplica a redução da taxa de aprendizado quando a metrica para de ser melhorada, é aplicado a cada 10 epocas
# https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ReduceLROnPlateau
red_lr_plat = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)

# https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint
callbacks = [tensorboard_callback,early_stopping,mc, red_lr_plat]
# With data augmentation

datagen = ImageDataGenerator(
            width_shift_range=[-200,200],
            height_shift_range=0.15,
            rotation_range=45,
            shear_range=20)  # randomly flip images


datagen.fit(X_train)
# prepare an iterators to scale images
train_iterator = datagen.flow(X_train,Y_train, batch_size=32)
val_iterator = datagen.flow(X_val,Y_val,batch_size=32)
print('Batches train=%d, test=%d' % (len(train_iterator), len(val_iterator)))
X_val.shape
X_test.shape
test.shape
#fit model
history = model_2.fit_generator(train_iterator,
                            steps_per_epoch = 20,
                            validation_data = val_iterator,
                            validation_steps = 1,
                            epochs = 10,
                            callbacks=callbacks)

# mostrando o treinamento
history
# Plot the loss and accuracy curves for training and validation 
fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_accuracy'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)
# Look at confusion matrix 

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Predict the values from the validation dataset
Y_pred = model_2.predict(X_val)
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
# Convert validation observations to one hot vectors
Y_true = np.argmax(Y_val,axis = 1) 
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
# plot the confusion matrix
plot_confusion_matrix(confusion_mtx, classes = range(2)) 
# Display some error results 

# Errors are difference between predicted labels and true labels
errors = (Y_pred_classes - Y_true != 0)

Y_pred_classes_errors = Y_pred_classes[errors]
Y_pred_errors = Y_pred[errors]
Y_true_errors = Y_true[errors]
X_val_errors = X_val[errors]

def display_errors(errors_index,img_errors,pred_errors, obs_errors):
    """ This function shows 6 images with their predicted and real labels"""
    n = 0
    nrows = 2
    ncols = 3
    fig, ax = plt.subplots(nrows,ncols,sharex=True,sharey=True)
    for row in range(nrows):
        for col in range(ncols):
            error = errors_index[n]
            ax[row,col].imshow((img_errors[error]).reshape((28,28)))
            ax[row,col].set_title("Predicted label :{}\nTrue label :{}".format(pred_errors[error],obs_errors[error]))
            n += 1

# Probabilities of the wrong predicted numbers
Y_pred_errors_prob = np.max(Y_pred_errors,axis = 1)

# Predicted probabilities of the true values in the error set
true_prob_errors = np.diagonal(np.take(Y_pred_errors, Y_true_errors, axis=1))

# Difference between the probability of the predicted label and the true label
delta_pred_true_errors = Y_pred_errors_prob - true_prob_errors

# Sorted list of the delta prob errors
sorted_dela_errors = np.argsort(delta_pred_true_errors)

# Top 6 errors 
most_important_errors = sorted_dela_errors[-6:]

# Show the top 6 errors
display_errors(most_important_errors, X_val_errors, Y_pred_classes_errors, Y_true_errors)
test.shape
print(X_test)
test.head()
X_test.shape
# predict results
results = model_2.predict(X_test)

# select the indix with the maximum probability
results = np.argmax(results,axis = 1)

results = pd.Series(results,name="pneumonia")
print(results)
filenames=test['fileName']
filenames.head()
#submission=pd.concat({"fileName":filenames,
#                      "pneumonia": results})
submission=pd.DataFrame({"fileName":filenames,
                      "pneumonia": results})
results.to_csv("submission_v7.csv",index=False)