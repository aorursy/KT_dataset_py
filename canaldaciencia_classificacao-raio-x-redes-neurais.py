# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load

import matplotlib.pyplot as plt

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        pass



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from PIL import Image

pil_im = Image.open('../input/logocanal/LOGO PNG.png')

pil_im
from os import listdir

from os.path import isfile, join



mypath='/kaggle/input/chest-xray-pneumonia/chest_xray/train/PNEUMONIA/'



pneumonia_files = [mypath+f for f in listdir(mypath) if isfile(join(mypath, f))]



pneumo_class=[1 for f in listdir(mypath) if isfile(join(mypath, f))]
from os import listdir

from os.path import isfile, join



mypath='/kaggle/input/chest-xray-pneumonia/chest_xray/val/PNEUMONIA/'



pneumonia_files_val = [mypath+f for f in listdir(mypath) if isfile(join(mypath, f))]



pneumo_class_val=[1 for f in listdir(mypath) if isfile(join(mypath, f))]
mypath='/kaggle/input/chest-xray-pneumonia/chest_xray/train/NORMAL/'



normal_files = [mypath+f for f in listdir(mypath) if isfile(join(mypath, f))]



normal_class=[0 for f in listdir(mypath) if isfile(join(mypath, f))]
mypath='/kaggle/input/chest-xray-pneumonia/chest_xray/val/NORMAL/'



normal_files_val = [mypath+f for f in listdir(mypath) if isfile(join(mypath, f))]



normal_class_val=[0 for f in listdir(mypath) if isfile(join(mypath, f))]
len(normal_files)
len(pneumonia_files)
pneumonia_files=pneumonia_files[:len(normal_files)]

pneumo_class=pneumo_class[:len(normal_files)]
len(pneumonia_files)
len(normal_files)
import pandas as pd

df_pneumo=pd.DataFrame(pneumonia_files,columns=['filename'])

df_pneumo['class']=pneumo_class



df_normal=pd.DataFrame(normal_files,columns=['filename'])

df_normal['class']=normal_class



df=pd.concat([df_pneumo,df_normal],axis=0)
import pandas as pd

df_pneumo_val=pd.DataFrame(pneumonia_files_val,columns=['filename'])

df_pneumo_val['class']=pneumo_class_val



df_normal_val=pd.DataFrame(normal_files_val,columns=['filename'])

df_normal_val['class']=normal_class_val



df_val=pd.concat([df_pneumo_val,df_normal_val],axis=0)
df=df.sample(frac=1)
from tensorflow.keras.preprocessing.image import ImageDataGenerator



datagen = ImageDataGenerator( featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range = 30,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.2, # Randomly zoom image 

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

        horizontal_flip = True,  # randomly flip images

        vertical_flip=False)

it = datagen.flow_from_dataframe(dataframe=df,batch_size=10,image_size=(256, 256),class_mode='raw')
images,labels=it.next()
plt.figure(figsize=(100,100))

plt.subplots_adjust(left=0.1, right=0.2, top=0.2, bottom=0.1)



# generate samples and plot

for i in range(9):

    plt.subplot(330 + 1 + i)

    images,labels = it.next()

    image = images[0].astype('uint8')

    plt.imshow(image)

    plt.axis('off')

# show the figure

plt.show()

datagen = ImageDataGenerator(featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range = 30,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.2, # Randomly zoom image 

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

        horizontal_flip = True,  # randomly flip images

        vertical_flip=False,rescale=1.0/255.0)

it = datagen.flow_from_dataframe(dataframe=df,batch_size=10,image_size=(256, 256),class_mode='raw')
datagen = ImageDataGenerator(rescale=1.0/255.0)

it_val = datagen.flow_from_dataframe(dataframe=df_val,image_size=(256, 256),class_mode='raw',batch_size=10)
import keras,os

from keras.models import Sequential

from keras.layers import Dense, Conv2D, MaxPool2D , Flatten, Dropout, Activation

from keras.preprocessing.image import ImageDataGenerator

from keras import layers

import numpy as np

import tensorflow as tf



model = tf.keras.models.Sequential([

  

    # Criando a entrada da rede neural com input de 256 por 256 colorido

    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(256, 256, 3)),

    tf.keras.layers.MaxPooling2D(2, 2),

  

    # Segunda camada de convolução

    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

  

    # Terceira camada de convolução

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

  

    # Quarta camada de convolução

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

  

    # Quinta camada de convolução

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),



  

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(512, activation='relu'), # 512 neuron hidden layer

    # Saida de rede neural com função de ativação sigmoid

    tf.keras.layers.Dense(1, activation='sigmoid')

])



# to get the summary of the model

model.summary()
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

checkpoint = ModelCheckpoint("/kaggle/working/model.h5", monitor='val_acc', verbose=1, save_best_only=True, 

                             save_weights_only=False, mode='auto', period=1)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,

                              patience=1, min_lr=0.001)

early = EarlyStopping(monitor='val_loss',  patience=2, 

                      verbose=2, mode='auto',

                      restore_best_weights=True)
import tensorflow as tf

from tensorflow.keras.optimizers import RMSprop

import tensorflow.keras.metrics as metrics



model.compile(optimizer=RMSprop(lr=0.001), loss=keras.losses.binary_crossentropy, metrics=['accuracy', metrics.AUC(name='auc')])
history = model.fit_generator(steps_per_epoch=269,generator=it,validation_data=it_val,epochs=40,callbacks=[checkpoint,early,reduce_lr])

mypath='/kaggle/input/chest-xray-pneumonia/chest_xray/test/NORMAL/'



normal_files = [mypath+f for f in listdir(mypath) if isfile(join(mypath, f))]



normal_class=[0 for f in listdir(mypath) if isfile(join(mypath, f))]
mypath='/kaggle/input/chest-xray-pneumonia/chest_xray/test/PNEUMONIA/'



pneumonia_files = [mypath+f for f in listdir(mypath) if isfile(join(mypath, f))]



pneumo_class=[1 for f in listdir(mypath) if isfile(join(mypath, f))]
df_pneumo=pd.DataFrame(pneumonia_files,columns=['filename'])

df_pneumo['class']=pneumo_class



df_normal=pd.DataFrame(normal_files,columns=['filename'])

df_normal['class']=normal_class



df=pd.concat([df_pneumo,df_normal],axis=0)
from tensorflow.keras.preprocessing.image import ImageDataGenerator



datagen = ImageDataGenerator(rescale=1.0/255.0)

ittest = datagen.flow_from_dataframe(dataframe=df,batch_size=624,shuffle=False,image_size=(256, 256),class_mode='raw')
images,labels=ittest.next()
y_test=labels
predict = model.predict_generator(ittest)
threshold=0.5
y_pred=[1 if p>threshold else 0 for p in predict]
def plot_confusion_matrix(cm,

                          target_names,

                          title='Confusion matrix',

                          cmap=None,

                          normalize=True):



    import itertools



    accuracy = np.trace(cm) / float(np.sum(cm))

    misclass = 1 - accuracy



    if cmap is None:

        cmap = plt.get_cmap('Blues')



    plt.figure(figsize=(8, 6))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()



    if target_names is not None:

        tick_marks = np.arange(len(target_names))

        plt.xticks(tick_marks, target_names, rotation=45)

        plt.yticks(tick_marks, target_names)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]





    thresh = cm.max() / 1.5 if normalize else cm.max() / 2

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        if normalize:

            plt.text(j, i, "{:0.4f}".format(cm[i, j]),

                     horizontalalignment="center",

                     color="white" if cm[i, j] > thresh else "black")

        else:

            plt.text(j, i, "{:,}".format(cm[i, j]),

                     horizontalalignment="center",

                     color="white" if cm[i, j] > thresh else "black")





    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))

    plt.show()
from sklearn.metrics import confusion_matrix

c=confusion_matrix(y_test, y_pred)

plot_confusion_matrix(cm           = c, 

                      normalize    = False,

                      target_names = ['Normal', 'Pneumonia'],

                      title        = "Confusion Matrix")
model.save_weights('/kaggle/working/model.h5')
from keras.utils.vis_utils import plot_model

plot_model(model, to_file='/kaggle/working/model_plot.png', show_shapes=True, show_layer_names=True)
images,labels=it.next()
# Read Image

import numpy as np

import skimage

Xi =images[0]



plt.imshow(Xi) 







preds = model.predict(np.array([Xi]))

print(preds)

print(labels[0])
import skimage.segmentation

superpixels = skimage.segmentation.quickshift(Xi, kernel_size=4,max_dist=200, ratio=0.2)

num_superpixels = np.unique(superpixels).shape[0]

plt.imshow(skimage.segmentation.mark_boundaries(Xi/2+0.5, superpixels))



#Generate perturbations

num_perturb = 150

perturbations = np.random.binomial(1, 0.5, size=(num_perturb, num_superpixels))





import copy

def perturb_image(img,perturbation,segments): 

  active_pixels = np.where(perturbation == 1)[0]

  mask = np.zeros(segments.shape)

  for active in active_pixels:

      mask[segments == active] = 1 

  perturbed_image = copy.deepcopy(img)

  perturbed_image = perturbed_image*mask[:,:,np.newaxis]

  return perturbed_image





print(perturbations[0]) 

plt.imshow(perturb_image(Xi/2+0.5,perturbations[0],superpixels))
predictions = []

for pert in perturbations:

  perturbed_img = perturb_image(Xi,pert,superpixels)

  pred =model.predict(perturbed_img[np.newaxis,:,:,:])

  predictions.append(pred)



predictions = np.array(predictions)

print(predictions.shape)
#Compute distances to original image

import sklearn.metrics

original_image = np.ones(num_superpixels)[np.newaxis,:] #Perturbation with all superpixels enabled 

distances = sklearn.metrics.pairwise_distances(perturbations,original_image, metric='cosine').ravel()

print(distances.shape)





kernel_width = 0.25

weights = np.sqrt(np.exp(-(distances**2)/kernel_width**2)) 

print(weights.shape)


#Estimate linear model

from sklearn.linear_model import LinearRegression

class_to_explain = 0 #Labrador class

simpler_model = LinearRegression()

simpler_model.fit(X=perturbations, y=predictions[:,:,0], sample_weight=weights)

coeff = simpler_model.coef_[0]



#Use coefficients from linear model to extract top features

num_top_features = 4

top_features = np.argsort(coeff)[-num_top_features:] 



#Show only the superpixels corresponding to the top features

mask = np.zeros(num_superpixels) 

mask[top_features]= True #Activate top superpixels

plt.imshow(perturb_image(Xi/2+0.5,mask,superpixels))