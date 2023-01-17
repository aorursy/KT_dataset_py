import tensorflow.keras.layers as Layers

import tensorflow.keras.activations as Actications

import tensorflow.keras.models as Models

import tensorflow.keras.optimizers as Optimizer

import tensorflow.keras.metrics as Metrics

import tensorflow.keras.utils as Utils

from keras.utils.vis_utils import model_to_dot

import os

import matplotlib.pyplot as plot

import cv2

import numpy as np

from sklearn.utils import shuffle

from sklearn.metrics import confusion_matrix as CM

from random import randint

from IPython.display import SVG

import matplotlib.gridspec as gridspec

import scikitplot as skplt

from sklearn.manifold import TSNE

import time

from matplotlib import pyplot

import keras.backend as K

import tensorflow as tf

import seaborn as sns

import matplotlib.patheffects as PathEffects
def get_images(directory):

    Images = []

    Labels = []  

    label = 0

    

    for labels in os.listdir(directory): 

        if labels == '1_Table': 

            label = 0

        elif labels == '2_Armchair':

            label = 1

        elif labels == '3_Sofa':

            label = 2

        elif labels == '4_Chair':

            label = 3

        

        for image_file in os.listdir(directory+labels): #Extracting the file name of the image from Class Label folder

            image = cv2.imread(directory+labels+r'/'+image_file) #Reading the image (OpenCV)

            image = cv2.resize(image,(150,150)) #Resize the image, Some images are different sizes. (Resizing is very Important)

            Images.append(image)

            Labels.append(label)

    

    return shuffle(Images,Labels,random_state=817328462) #Shuffle the dataset you just prepared.



def get_classlabel(class_code):

    labels = {0:'1_Table', 1:'2_Armchair', 2:'3_Sofa', 3:'4_Chair'}

    

    return labels[class_code]
Images, Labels = get_images('/kaggle/input/exam-minor-2019/img_kagl_train/img_kagl_train/') #Extract the training images from the folders.



Images = np.array(Images) #converting the list of images to numpy array.

Labels = np.array(Labels)
print("Shape of Images:",Images.shape)

print("Shape of Labels:",Labels.shape)
def f1(y_true, y_pred):

    y_pred = K.round(y_pred)

    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)

    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)

    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)

    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)



    p = tp / (tp + fp + K.epsilon())

    r = tp / (tp + fn + K.epsilon())



    f1 = 2*p*r / (p+r+K.epsilon())

    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)

    return K.mean(f1)
model = Models.Sequential([

    Layers.Conv2D(200,kernel_size=(3,3),activation='relu',input_shape=(150,150,3)),

    Layers.Conv2D(180,kernel_size=(3,3),activation='relu'),

    Layers.MaxPool2D(5,5),

    Layers.Conv2D(180,kernel_size=(3,3),activation='relu'),

    Layers.Conv2D(140,kernel_size=(3,3),activation='relu'),

    Layers.Conv2D(100,kernel_size=(3,3),activation='relu'),

    Layers.Conv2D(50,kernel_size=(3,3),activation='relu'),

    Layers.MaxPool2D(5,5),

    Layers.Flatten(),

    Layers.Dense(180,activation='relu'),

    Layers.Dense(100,activation='relu'),

    Layers.Dense(50,activation='relu'),

    Layers.Dropout(rate=0.5),

    Layers.Dense(4,activation='softmax')

])



model.compile(optimizer=Optimizer.Adam(lr=0.0001),loss='sparse_categorical_crossentropy',metrics=[f1,'accuracy'])



model.summary()
resultTrain = model.fit(Images,Labels,epochs=45,validation_split=0.30)
plot.plot(resultTrain.history['accuracy'])

plot.plot(resultTrain.history['val_accuracy'])

plot.title('Model accuracy')

plot.ylabel('Accuracy')

plot.xlabel('Epoch')

plot.legend(['Train', 'Test'], loc='upper left')

plot.show()



plot.plot(resultTrain.history['loss'])

plot.plot(resultTrain.history['val_loss'])

plot.title('Model loss')

plot.ylabel('Loss')

plot.xlabel('Epoch')

plot.legend(['Train', 'Test'], loc='upper left')

plot.show()
pred_images,no_labels = get_images('/kaggle/input/exam-minor-2019/img_kagl_test/')

pred_images = np.array(pred_images)

pred_images.shape

prediction = model.predict(Images)
fig = plot.figure(figsize=(30, 30))

outer = gridspec.GridSpec(5, 5, wspace=0.2, hspace=0.2)



for i in range(25):

    inner = gridspec.GridSpecFromSubplotSpec(2, 1,subplot_spec=outer[i], wspace=0.1, hspace=0.1)

    rnd_number = randint(0,len(pred_images))

    pred_image = np.array([pred_images[rnd_number]])

    pred_class = get_classlabel(model.predict_classes(pred_image)[0])

    pred_prob = model.predict(pred_image).reshape(4)

    for j in range(2):

        if (j%2) == 0:

            ax = plot.Subplot(fig, inner[j])

            ax.imshow(pred_image[0])

            ax.set_title(pred_class)

            ax.set_xticks([])

            ax.set_yticks([])

            fig.add_subplot(ax)

        else:

            ax = plot.Subplot(fig, inner[j])

            ax.bar([0,1,2,3],pred_prob)

            fig.add_subplot(ax)





fig.show()
skplt.metrics.plot_confusion_matrix(

    Labels, 

    prediction.argmax(axis=1))
nsamples, nx, ny, nz = Images.shape

D2 = Images.reshape((nsamples,nx*ny*nz))
# T-SNE Implementation



t0 = time.time()

X_reduced_tsne = TSNE(n_components=2, random_state=42).fit_transform(D2)

t1 = time.time()

print("T-SNE took {:.2} s".format(t1 - t0))
def scatter(x, colors):

    # choose a color palette with seaborn.

    num_classes = len(np.unique(colors))

    palette = np.array(sns.color_palette("hls", num_classes))



    # create a scatter plot.

    f = pyplot.figure(figsize=(8, 8))

    ax = pyplot.subplot(aspect='equal')

    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=palette[colors.astype(np.int)])

    pyplot.xlim(-25, 25)

    pyplot.ylim(-25, 25)

    ax.axis('off')

    ax.axis('tight')



    # add the labels for each digit corresponding to the label

    txts = []



    for i in range(num_classes):



        # Position of each label at median of data points.



        xtext, ytext = np.median(x[colors == i, :], axis=0)

        txt = ax.text(xtext, ytext, str(i), fontsize=24)

        txt.set_path_effects([

            PathEffects.Stroke(linewidth=5, foreground="w"),

            PathEffects.Normal()])

        txts.append(txt)



    return f, ax, sc, txts
scatter(X_reduced_tsne,Labels)