import os,cv2

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

from pylab import rcParams

rcParams['figure.figsize'] = 20, 10



from sklearn.utils import shuffle

from  sklearn.model_selection import train_test_split



import keras



from keras.utils import np_utils



from keras import backend as K



from keras.models import Sequential

from keras.layers.core import Dense, Dropout, Activation, Flatten

from keras.layers.convolutional import Convolution2D, MaxPooling2D

from keras.optimizers import SGD,RMSprop,adam

from keras.preprocessing.image import ImageDataGenerator

import pandas as pd
# get the data

filname = '../input/facial-expression/fer2013/fer2013.csv'

label_map = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

names=['emotion','pixels','usage']

df=pd.read_csv('../input/facial-expression/fer2013/fer2013.csv',names=names, na_filter=False)

im=df['pixels']

df.head(10)
from keras.models import Sequential

from keras.layers import Dense , Activation , Dropout ,Flatten

from keras.layers.convolutional import Conv2D

from keras.layers.convolutional import MaxPooling2D

from keras.metrics import categorical_accuracy

from keras.models import model_from_json

from keras.callbacks import ModelCheckpoint

from keras.optimizers import *

from keras.layers.normalization import BatchNormalization
def getData(filname):

    # images are 48x48

    # N = 35887

    Y = []

    X = []

    first = True

    for line in open(filname):

        if first:

            first = False

        else:

            row = line.split(',')

            Y.append(int(row[0]))

            X.append([int(p) for p in row[1].split()])



    X, Y = np.array(X) / 255.0, np.array(Y)

    return X, Y
X, Y = getData(filname)

num_class = len(set(Y))

print(num_class)
# keras with tensorflow backend

N, D = X.shape

X = X.reshape(N, 48, 48, 1)
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

y_train = (np.arange(num_class) == y_train[:, None]).astype(np.float32)

y_test = (np.arange(num_class) == y_test[:, None]).astype(np.float32)
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
from keras.models import load_model

model = load_model('/kaggle/input/model-visualize/menu_visualize/model_keras.h5')

model.load_weights('/kaggle/input/model-visualize/menu_visualize/model_weights.h5')
model.summary()
from keras.utils import plot_model

plot_model(model, to_file='model.png')
def predict_emotion(image):

    x_test = np.expand_dims(image,axis=0)

    y_predict = np.argmax(model.predict(x_test))

    emotion_dict = {0:'Anger',1:'Disgust',2:'Fear',3:'Happy',4:'Sad',5:'Surprise',6:'Neutral'}

    return(emotion_dict[y_predict])

    
from pylab import rcParams

rcParams['figure.figsize'] = 5, 10

def plot_image(image):

    img = image.reshape(48,48)

    plt.imshow(img, interpolation='nearest')

    plt.show()

image_tes1 = X_test[4]

plot_image(image_tes1)

predict_emotion(image_tes1)
image_tes2 = X_test[800]

plot_image(image_tes2)

predict_emotion(image_tes2)
image_tes3 = X_test[90]

plot_image(image_tes3)

predict_emotion(image_tes3)
image_tes4 = X_test[60]

plot_image(image_tes4)

predict_emotion(image_tes4)
import cv2
img_5 = cv2.imread('/kaggle/input/test-images-gray/test_images_small/g.png')

img_gray_5 = cv2.cvtColor(img_5, cv2.COLOR_BGR2GRAY)



img_5_resize = cv2.resize(img_gray_5, (48, 48))

img_5 = img_5_resize.reshape(48, 48, 1)



plot_image(img_5)

predict_emotion(img_5)
img_5 = cv2.imread('/kaggle/input/test-images-gray/test_images_small/c.png')

img_gray_5 = cv2.cvtColor(img_5, cv2.COLOR_BGR2GRAY)



img_5_resize = cv2.resize(img_gray_5, (48, 48))

img_5 = img_5_resize.reshape(48, 48, 1)



plot_image(img_5)

predict_emotion(img_5)
img_5 = cv2.imread('/kaggle/input/test-images-gray/test_images_small/e.png')

img_gray_5 = cv2.cvtColor(img_5, cv2.COLOR_BGR2GRAY)



img_5_resize = cv2.resize(img_gray_5, (48, 48))

img_5 = img_5_resize.reshape(48, 48, 1)



plot_image(img_5)

predict_emotion(img_5)
from keras.models import Model



def layer_image(input_image, col_size, row_size, act_index):



    image_array = input_image

    x_enpanded = np.expand_dims(image_array, axis=0)

    y_pred = np.argmax(model.predict(x_enpanded))

    emotion_dict = {0:'Anger',1:'Disgust',2:'Fear',3:'Happy',4:'Sad',5:'Surprise',6:'Neutral'}

    label = emotion_dict[y_pred]



    layer_outputs = [layer.output for layer in model.layers][1:]

    activation_model = Model(inputs=model.input, outputs=layer_outputs)

    activations = activation_model.predict(x_enpanded)



    activation = activations[act_index]

    activation_index=0

    fig, ax = plt.subplots(row_size, col_size, figsize=(row_size*5,col_size*5))

    for row in range(0,row_size):

        for col in range(0,col_size):

            ax[row][col].imshow(activation[0, :, :, activation_index])

            activation_index += 1

    return fig
image_gain1 = layer_image(image_tes1, 8,8,1)
#image_gain2 = layer_image(image_tes2, 8,8,1)
#image_gain3 = layer_image(image_tes3, 8,8,1)
#image_gain4 = layer_image(image_tes4, 8,8,0)
best_model = model
from sklearn.metrics import confusion_matrix

results = best_model.predict_classes(X_test)

cm = confusion_matrix(np.where(y_test == 1)[1], results)

#cm = cm.astype(np.float) / cm.sum(axis=1)[:, np.newaxis]
import seaborn as sns

import pandas as pd
label_mapdisgust = ['anger','contempt','disgust','fear','happy','sadness','surprise']
#Transform to df for easier plotting

cm_df = pd.DataFrame(cm, index = label_mapdisgust,

                     columns = label_mapdisgust

                    )
final_cm = cm_df
plt.figure(figsize = (5,5))

sns.heatmap(final_cm, annot = True,cmap='Greys',cbar=False,linewidth=2,fmt='d')

plt.title('CNN Emotion Classify')

plt.ylabel('True class')

plt.xlabel('Prediction class')

plt.show()
from sklearn.metrics import roc_curve,auc

from itertools import cycle
new_label = ['anger','contempt','disgust','fear','happy','sadness','surprise']

final_label = new_label

new_class = 7
#predict

y_pred = best_model.predict(X_test)
#ravel flatten the array into single vector

y_pred_ravel = y_pred.ravel()

lw = 2
fpr = dict()

tpr = dict()

roc_auc = dict()



for i in range(new_class):

    fpr[i], tpr[i], _ = roc_curve(y_test[:,i], y_pred[:,i])

    roc_auc[i] = auc(fpr[i], tpr[i])

    

#colors = cycle(['red', 'green','black'])

colors = cycle(['red', 'green','black','blue', 'yellow','purple','orange'])

for i, color in zip(range(new_class), colors):

    plt.plot(fpr[i], tpr[i], color=color, lw=lw,

             label='ROC curve of class {0}'''.format(final_label[i]))

    



plt.plot([0, 1], [0, 1], 'k--', lw=lw)

plt.xlim([0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver Operating Characteristic')

plt.legend(loc="lower right")

plt.show()