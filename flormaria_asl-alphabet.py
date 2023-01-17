import pandas as pd 

import numpy as np

import os

import cv2

import skimage



imageSize=50

train_dir = "../input/asl-alphabet/asl_alphabet_train/asl_alphabet_train/"

test_dir =  "../input/asl-alphabet/asl_alphabet_test/asl_alphabet_test/"

from tqdm import tqdm

def get_data(folder):

    X = []

    y = []

    i = 0

    for folderName in os.listdir(folder):

        

        if not folderName.startswith('.'):

            if folderName in ['A']:label = 0

            elif folderName in ['B']:label = 1

            elif folderName in ['C']:label = 2

            elif folderName in ['D']:label = 3

            elif folderName in ['E']:label = 4

            elif folderName in ['F']:label = 5

            elif folderName in ['G']:label = 6

            elif folderName in ['H']:label = 7

            elif folderName in ['I']:label = 8

            elif folderName in ['J']:label = 9

            elif folderName in ['K']:label = 10

            elif folderName in ['L']:label = 11

            elif folderName in ['M']:label = 12

            elif folderName in ['N']:label = 13

            elif folderName in ['O']:label = 14

            elif folderName in ['P']:label = 15

            elif folderName in ['Q']:label = 16

            elif folderName in ['R']:label = 17

            elif folderName in ['S']:label = 18

            elif folderName in ['T']:label = 19

            elif folderName in ['U']:label = 20

            elif folderName in ['V']:label = 21

            elif folderName in ['W']:label = 22

            elif folderName in ['X']:label = 23

            elif folderName in ['Y']:label = 24

            elif folderName in ['Z']:label = 25

            elif folderName in ['del']:label = 26

            elif folderName in ['nothing']:label = 27

            elif folderName in ['space']:label = 28           

            else:label = 29

            i=0

            for image_filename in tqdm(os.listdir(folder + folderName)):

                if i > 1000:

                    break

                i =  i+1

                

                img_file = cv2.imread(folder + folderName + '/' + image_filename)

                if img_file is not None:

                    img_file = skimage.transform.resize(img_file, (imageSize, imageSize, 3))

                    img_arr = np.asarray(img_file)

                    X.append(img_arr)

                    y.append(label)

    X = np.asarray(X)

    y = np.asarray(y)

    return X,y

X_train_load, y_train_load = get_data(train_dir)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_train_load, y_train_load, test_size=0.2, random_state=30) 
# Encode labels to hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])

from keras.utils.np_utils import to_categorical

y_trainHot = to_categorical(y_train, num_classes = 30)

y_testHot = to_categorical(y_test, num_classes = 30)
# Shuffle data to permit further subsampling

from sklearn.utils import shuffle

X_train, y_trainHot = shuffle(X_train, y_trainHot, random_state=13)

X_test, y_testHot = shuffle(X_test, y_testHot, random_state=13)
import matplotlib.pyplot as plt

def plotHistogram(a):

    """

    #Plot histogram of RGB Pixel Intensities

    """

    

    plt.figure(figsize=(10,5))

    plt.subplot(1,2,1)

    plt.imshow(a)

    plt.axis('off')

    histo = plt.subplot(1,2,2)

    histo.set_ylabel('Count')

    histo.set_xlabel('Pixel Intensity')

    n_bins = 30

    plt.hist(a[:,:,0].flatten(), bins= n_bins, lw = 0, color='r', alpha=0.5);

    plt.hist(a[:,:,1].flatten(), bins= n_bins, lw = 0, color='g', alpha=0.5);

    plt.hist(a[:,:,2].flatten(), bins= n_bins, lw = 0, color='b', alpha=0.5);\

    #print(title)

plotHistogram(X_train[0])

plotHistogram(X_train[2])

plotHistogram(X_train[21])
import glob

import os

import random



mylist = os.listdir('../input/asl-alphabet/asl_alphabet_train/asl_alphabet_train/A')

def plotThreeImages(path,images):

    r = random.sample(images, 3)

    plt.figure(figsize=(16,16))

    plt.subplot(131)

    plt.imshow(cv2.imread(path+r[0]))

    plt.subplot(132)

    plt.imshow(cv2.imread(path+r[1]))

    plt.subplot(133)

    plt.imshow(cv2.imread(path+r[2]))

    

plotThreeImages('../input/asl-alphabet/asl_alphabet_train/asl_alphabet_train/A/', mylist)
import seaborn as sns

map_characters = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'del', 27: 'nothing', 28: 'space', 29: 'other'}

dict_characters=map_characters



df = pd.DataFrame()

df["labels"]=y_train

lab = df['labels']

dist = lab.value_counts()

sns.countplot(lab)

print(dict_characters)
import keras

from keras import optimizers

from sklearn.utils import class_weight

from keras.applications.vgg16 import VGG16



map_characters1 = map_characters

class_weight1 = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)

weight_path1 = '../input/keras-pretrained-models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

weight_path2 = '../input/keras-pretrained-models/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

pretrained_model_1 = VGG16(weights = weight_path1, include_top=False, input_shape=(imageSize, imageSize, 3))

#optimizer1 = optimizers.Adam()

#optimizer2 = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

#optimizer3 = optimizers.RMSprop(learning_rate)

#optimizer4 = optimizers.SGD(lr=0.01, clipnorm=1.)
optimizer_RMSprop = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
optimizer_Adagrad = keras.optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
optimizer_adadelta = keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
optimizer_Adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
optimizer_adax = keras.optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
optimizer_Nadam = keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
def showChartLearningRate(history, epochs):

    # show a nicely formatted classification report

    print("[INFO] evaluating network...")

    #loss: 0.1149 - mean_absolute_error: 0.2270 - val_loss: 0.1080 - val_mean_absolute_error

    # plot the training loss and accuracy

    N = epochs

    plt.style.use("ggplot")

    plt.figure()

    

    plt.plot(np.arange(0, N), history.history["loss"], label="train_loss")

    plt.plot(np.arange(0, N), history.history["val_loss"], label="val_loss")

    plt.plot(np.arange(0, N), history.history["acc"], label="train_acc")

    plt.plot(np.arange(0, N), history.history["val_acc"], label="val_acc")

    plt.title("Training Loss and validation loss on Dataset")

    plt.xlabel("Epochs #")

    plt.ylabel("Loss/Mean_absolute_error")

    #plt.xticks(sd)

    #plt.xticklabels(learningValue)

    plt.legend(loc="lower left")

    plt.show()
import itertools 

def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    plt.figure(figsize = (8,8))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=90)

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
from keras.layers import Dense, Flatten

from keras.models import Model

import keras.callbacks

import keras

from keras.optimizers import SGD

from keras.callbacks import LearningRateScheduler



#from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

#from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, learning_curve, GridSearchCV

#from tensorflow.keras.callbacks import keras

def pretrainedNetwork(xtrain,ytrain,xtest,ytest,pretrainedmodel,pretrainedweights,classweight,numclasses,numepochs,optimizer,labels):

    base_model = pretrained_model_1 # Topless

    # Add top layer

    x = base_model.output

    x = Flatten()(x)

    predictions = Dense(numclasses, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # Train top layer

    for layer in base_model.layers:

        layer.trainable = False

    

    model.compile(loss='categorical_crossentropy', 

                  optimizer=optimizer, 

                  metrics=['mean_absolute_error','accuracy'])

    

    callbacks_list = [keras.callbacks.EarlyStopping(monitor='val_acc', patience=3, verbose=1)]

    #model.summary()

    

    history = model.fit(xtrain,ytrain, epochs=numepochs, class_weight=classweight, validation_data=(xtest,ytest), verbose=1)

    print('history loss')

    print(history.history["loss"])

    print('history val_loss')

    print(history.history["val_loss"])

    print('history mean_absolute_error')

    print(history.history["mean_absolute_error"])

    print('history val_mean_absolute_error')

    print(history.history["val_mean_absolute_error"])

    print('history val_mean_absolute_error')

    print(history.history["val_mean_absolute_error"])

    

     # Evaluate model

    score = model.evaluate(xtest,ytest, verbose=0)

    print('\nKeras CNN - accuracy:', score[1], '\n')

    y_pred = model.predict(xtest)

    

    import sklearn

    Y_pred_classes = np.argmax(y_pred,axis = 1) 

    Y_true = np.argmax(ytest,axis = 1) 

    

    from sklearn.metrics import confusion_matrix

    confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 

    print(confusion_mtx)

    

    showChartLearningRate(history, numepochs)

    

    plot_confusion_matrix(confusion_mtx, classes = list(labels.values()))

    plt.show()

    

    return model

#epochs = 3

#mymodel = pretrainedNetwork(X_train, y_trainHot, X_test, y_testHot,pretrained_model_1,weight_path1,class_weight1,30,epochs,optimizer1,map_characters1)
epochs = 5

mymodel_RMSprop = pretrainedNetwork(X_train, y_trainHot, X_test, y_testHot,pretrained_model_1,weight_path1,class_weight1,30,epochs,optimizer_RMSprop,map_characters1)
epochs = 5

mymodel_Adagrad = pretrainedNetwork(X_train, y_trainHot, X_test, y_testHot,pretrained_model_1,weight_path1,class_weight1,30,epochs,optimizer_Adagrad,map_characters1)
epochs = 5

mymodel_adadelta = pretrainedNetwork(X_train, y_trainHot, X_test, y_testHot,pretrained_model_1,weight_path1,class_weight1,30,epochs,optimizer_adadelta,map_characters1) 
epochs = 5

mymodel_Adam = pretrainedNetwork(X_train, y_trainHot, X_test, y_testHot,pretrained_model_1,weight_path1,class_weight1,30,epochs,optimizer_Adam,map_characters1) 
epochs = 5

mymodel_adax = pretrainedNetwork(X_train, y_trainHot, X_test, y_testHot,pretrained_model_1,weight_path1,class_weight1,30,epochs,optimizer_adax,map_characters1) 
epochs = 5

mymodel_Nadam = pretrainedNetwork(X_train, y_trainHot, X_test, y_testHot,pretrained_model_1,weight_path1,class_weight1,30,epochs,optimizer_Nadam,map_characters1) 
def get_data2(_img):

    X = []

    img_file = cv2.imread(_img)

    img_file = skimage.transform.resize(img_file, (imageSize, imageSize, 3))

    img_arr = np.asarray(img_file)

    X.append(img_arr)               

    X = np.asarray(X)

    return X



img_p='../input/asl-alphabet/asl_alphabet_test/asl_alphabet_test/B_test.jpg'

img_preds= get_data2(img_p) 
import numpy as geek

pre = mymodel_Nadam.predict(img_preds)

pred_idx = geek.argmax(pre, axis=1)



print(pred_idx)

print(pre)
mymodel_Nadam.summary()
def get_output_layer(model, layer_name):

    # get the symbolic outputs of each "key" layer (we gave them unique names).

    layer_dict = dict([(layer.name, layer) for layer in model.layers])

    layer = layer_dict[layer_name]

    return layer
from keras.preprocessing import image

from keras.applications.vgg16 import preprocess_input

def _load_image(img_path):    

    img = image.load_img(img_path, target_size=(50,50))

    img = image.img_to_array(img)

    img = np.expand_dims(img, axis=0)

    img = preprocess_input(img)

    return img 
def cam(img_path):

  

    import numpy as np

    from keras.applications.vgg16 import decode_predictions

    import matplotlib.image as mpimg

    from keras import backend as K

    import pandas as pd

    import matplotlib.pyplot as plt

    %matplotlib inline

    #K.clear_session()

    

    img=mpimg.imread(img_path)

    plt.imshow(img)

    x = _load_image(img_path)

    preds = mymodel_Nadam.predict(img_preds)

    

    #predictions = pd.DataFrame(decode_predictions(preds, top=3)[0],columns=['col1','category','probability']).iloc[:,1:]

    argmax = np.argmax(preds[0])

    output = mymodel_Nadam.output[:, argmax]

    last_conv_layer = mymodel_Nadam.get_layer('block5_conv3')

    grads = K.gradients(output, last_conv_layer.output)[0]

    pooled_grads = K.mean(grads, axis=(0, 1, 2))

    iterate = K.function([mymodel_Nadam.input], [pooled_grads, last_conv_layer.output[0]])

    pooled_grads_value, conv_layer_output_value = iterate([x])

    for i in range(125):

        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

    heatmap = np.mean(conv_layer_output_value, axis=-1)

    heatmap = np.maximum(heatmap, 0)

    heatmap /= np.max(heatmap)

    import cv2

    img = cv2.imread(img_path)

    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    hif = .8

    superimposed_img = heatmap * hif + img

    output = 'output.jpeg'

    cv2.imwrite(output, superimposed_img)

    img=mpimg.imread(output)

    plt.imshow(img)

    plt.axis('off')

   # plt.title(predictions.loc[0,'category'].upper())

    return None
cam(img_p)