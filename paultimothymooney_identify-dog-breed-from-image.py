from tqdm import tqdm

import seaborn as sns

from keras.preprocessing import image

import cv2

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import os

import itertools

from PIL import Image

import sklearn as sklearn

from sklearn.metrics import confusion_matrix

from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from sklearn import model_selection

from sklearn.model_selection import train_test_split, learning_curve

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.neural_network import MLPClassifier as MLPC

from sklearn import model_selection

import tensorflow as tf

import keras

from keras import backend as K

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

from keras import initializers, layers, models

from keras.utils import to_categorical

from keras.preprocessing.image import ImageDataGenerator

from keras import callbacks

from keras.utils.vis_utils import plot_model

from mpl_toolkits.axes_grid1 import ImageGrid

from os import listdir, makedirs

from os.path import join, exists, expanduser

from sklearn.metrics import log_loss, accuracy_score

from keras.preprocessing import image

from keras.applications.vgg16 import VGG16

from keras.applications.resnet50 import ResNet50

from keras.applications import xception

from keras.applications import inception_v3

from keras.applications.vgg16 import preprocess_input, decode_predictions

import datetime as dt

start = dt.datetime.now()

%matplotlib inline
df_train = pd.read_csv('../input/dog-breed-identification/labels.csv')

df_test = pd.read_csv('../input/dog-breed-identification/sample_submission.csv')

df_train.head(10)
yy = pd.value_counts(df_train['breed'])



fig, ax = plt.subplots()

fig.set_size_inches(15, 9)

sns.set_style("whitegrid")



ax = sns.barplot(x = yy.index, y = yy, data = df_train)

ax.set_xticklabels(ax.get_xticklabels(), rotation = 90, fontsize = 8)

ax.set(xlabel='Dog Breed', ylabel='Count')

ax.set_title('Distribution of Dog breeds')
labels = df_train

top_breeds = sorted(list(labels['breed'].value_counts().head(16).index))

labels = labels[labels['breed'].isin(top_breeds)]

labels.breed.value_counts().plot(kind='bar')

#df_train = labels ### remove this line to go back to 120 different breeds instead of 16
targets_series = pd.Series(df_train['breed'])

one_hot = pd.get_dummies(targets_series, sparse = True)

one_hot_labels = np.asarray(one_hot)



im_size = 128

x_train1 = []

y_train1 = []

x_test1 = []

i = 0 



for f, breed in tqdm(df_train.values):

    img = cv2.imread('../input/dog-breed-identification/train/{}.jpg'.format(f))

    label = one_hot_labels[i]

    x_train1.append(cv2.resize(img, (im_size, im_size)))

    y_train1.append(label)

    i += 1



for f in tqdm(df_test['id'].values):

    img = cv2.imread('../input/dog-breed-identification/test/{}.jpg'.format(f))

    x_test1.append(cv2.resize(img, (im_size, im_size)))



y_train_raw = np.array(y_train1, np.uint8)

x_train_raw = np.array(x_train1, np.float32) / 255.

x_testContest  = np.array(x_test1, np.float32) / 255.



num_class = y_train_raw.shape[1]



print(x_train_raw.shape)

print(y_train_raw.shape)

# print(x_testContest.shape)    
x_train,x_test,y_train,y_test = train_test_split(x_train_raw, y_train_raw, test_size=0.3, random_state=1)
fig, ax = plt.subplots()

img = image.load_img('../input/dog-breed-identification/train/fff43b07992508bc822f33d8ffd902ae.jpg')

img = image.img_to_array(img)

ax.imshow(img / 255.) 

ax.axis('off')

plt.show()
df_train2 = pd.read_csv('../input/dog-breed-identification/labels.csv')

top_breeds = df_train2['breed']



plt.subplot(1, 2, 1)

plt.title(top_breeds[np.where(y_train[5]==1)[0][0]])

plt.axis('off')

plt.imshow(x_train[5])

plt.subplot(1, 2, 2)

plt.title(top_breeds[np.where(y_train[7]==1)[0][0]])

plt.axis('off')

plt.imshow(x_train[7])

plt.show()
import random

df = df_train

n = len(df)

breed = set(df['breed'])

n_class = len(breed)

class_to_num = dict(zip(breed, range(n_class)))

num_to_class = dict(zip(range(n_class), breed))

%config InlineBackend.figure_format = 'retina'

n=100

plt.figure(figsize=(12, 6))

for i in range(8):

    random_index = random.randint(0, n-1)

    plt.subplot(2, 4, i+1)

    plt.imshow(x_train[random_index][:,:,::-1])

    plt.title(num_to_class[y_train[random_index].argmax()])

    plt.axis('off')

# Plot confusion matrix

def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    plt.figure(figsize = (30,30))

    #plt.figure(figsize = (15,15))

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



# Special callback to see learning curves

class MetricsCheckpoint(Callback):

    """Callback that saves metrics after each epoch"""

    def __init__(self, savepath):

        super(MetricsCheckpoint, self).__init__()

        self.savepath = savepath

        self.history = {}

    def on_epoch_end(self, epoch, logs=None):

        for k, v in logs.items():

            self.history.setdefault(k, []).append(v)

        np.save(self.savepath, self.history)



def plotKerasLearningCurve():

    plt.figure(figsize=(10,5))

    metrics = np.load('logs.npy')[()]

    filt = ['acc'] # try to add 'loss' to see the loss learning curve

    for k in filter(lambda x : np.any([kk in x for kk in filt]), metrics.keys()):

        l = np.array(metrics[k])

        plt.plot(l, c= 'r' if 'val' not in k else 'b', label='val' if 'val' in k else 'train')

        x = np.argmin(l) if 'loss' in k else np.argmax(l)

        y = l[x]

        plt.scatter(x,y, lw=0, alpha=0.25, s=100, c='r' if 'val' not in k else 'b')

        plt.text(x, y, '{} = {:.4f}'.format(x,y), size='15', color= 'r' if 'val' not in k else 'b')   

    plt.legend(loc=4)

    plt.axis([0, None, None, None]);

    plt.grid()

    plt.xlabel('Number of epochs')

    

map_characters = {0:'none',1:'affenpinscher',2:'afghan_hound',3:'african_hunting_dog',

4:'airedale',5:'american_staffordshire_terrier',6:'appenzeller',7:'australian_terrier',

8:'basenji',9:'basset',10:'beagle',11:'bedlington_terrier',12:'bernese_mountain_dog',

13:'black-and-tan_coonhound',14:'blenheim_spaniel',15:'bloodhound',16:'bluetick',

17:'border_collie',18:'border_terrier',19:'borzoi',20:'boston_bull',21:'bouvier_des_flandres',

22:'boxer',23:'brabancon_griffon',24:'briard',25:'brittany_spaniel',26:'bull_mastiff',

27:'cairn',28:'cardigan',29:'chesapeake_bay_retriever',30:'chihuahua',31:'chow',

32:'clumber',33:'cocker_spaniel',34:'collie',35:'curly-coated_retriever',36:'dandie_dinmont',

37:'dhole',38:'dingo',39:'doberman',40:'english_foxhound',41:'english_setter',

42:'english_springer',43:'entlebucher',44:'eskimo_dog',45:'flat-coated_retriever',

46:'french_bulldog',47:'german_shepherd',48:'german_short-haired_pointer',49:'giant_schnauzer',

50:'golden_retriever',51:'gordon_setter',52:'great_dane',53:'great_pyrenees',

54:'greater_swiss_mountain_dog',55:'groenendael',56:'ibizan_hound',57:'irish_setter',

58:'irish_terrier',59:'irish_water_spaniel',60:'irish_wolfhound',61:'italian_greyhound',

62:'japanese_spaniel',63:'keeshond',64:'kelpie',65:'kerry_blue_terrier',66:'komondor',

67:'kuvasz',68:'labrador_retriever',69:'lakeland_terrier',70:'leonberg',71:'lhasa',72:'malamute',

73:'malinois',74:'maltese_dog',75:'mexican_hairless',76:'miniature_pinscher',77:'miniature_poodle',

78:'miniature_schnauzer',79:'newfoundland',80:'norfolk_terrier',81:'norwegian_elkhound',

82:'norwich_terrier',83:'old_english_sheepdog',84:'otterhound',85:'papillon',86:'pekinese',

87:'pembroke',88:'pomeranian',89:'pug',90:'redbone',91:'rhodesian_ridgeback',92:'rottweiler',

93:'saint_bernard',94:'saluki',95:'samoyed',96:'schipperke',97:'scotch_terrier',98:'scottish_deerhound',

99:'sealyham_terrier',100:'shetland_sheepdog',101:'shih-tzu',102:'siberian_husky',103:'silky_terrier',

104:'soft-coated_wheaten_terrier',105:'staffordshire_bullterrier',106:'standard_poodle',

107:'standard_schnauzer',108:'sussex_spaniel',109:'tibetan_mastiff',110:'tibetan_terrier',111:'toy_poodle',

112:'toy_terrier',113:'vizsla',114:'walker_hound',115:'weimaraner',116:'welsh_springer_spaniel',

117:'west_highland_white_terrier',118:'whippet',119:'wire-haired_fox_terrier',120:'yorkshire_terrier'}
#map_characters = {0:'afghan_hound', 1:'airedale', 2:'basenji', 3:'beagle', 4:'bernese_mountain_dog', 5:'cairn', 6:'entlebucher', 7:'great_pyrenees', 8:'japanese_spaniel', 9:'leonberg', 10:'maltese_dog', 11:'pomeranian', 12:'samoyed', 13:'scottish_deerhound', 14:'shih-tzu', 15:'tibetan_terrier'}
num_classes = 120

#num_classes = 16

def runKerasCNNAugment(a,b,c,d):

    #global model

    batch_size = 128

    epochs = 10

    im_size = 128

    #img_rows, img_cols = X_train.shape[1],X_train.shape[2]

    #input_shape = (img_rows, img_cols, 3)

    input_shape = (im_size,im_size,3)

    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3),

                     activation='relu',

                     input_shape=input_shape))

    model.add(Conv2D(64, (3, 3), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))

    model.add(Dropout(0.5))

    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,

                  optimizer=keras.optimizers.RMSprop(lr=0.0001),

                  metrics=['accuracy'])

    datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=True,  # randomly flip images

        vertical_flip=False)  # randomly flip images

    model.fit_generator(datagen.flow(a,b, batch_size=32),

                        steps_per_epoch=len(a) / 32, epochs=epochs, validation_data = [c, d],callbacks = [MetricsCheckpoint('logs')])

    score = model.evaluate(c,d, verbose=0)

    print('\nKeras CNN #1C - accuracy:', score[1],'\n')

    y_pred = model.predict(c)

    #map_characters = {0: 'No Ship', 1: 'Ship'}

    print('\n', sklearn.metrics.classification_report(np.where(d > 0)[1], np.argmax(y_pred, axis=1), target_names=list(map_characters.values())), sep='')    

    score = model.evaluate(c,d, verbose=0)

    Y_pred_classes = np.argmax(y_pred,axis = 1) 

    Y_true = np.argmax(d,axis = 1) 

    confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 

    plotKerasLearningCurve()

    plt.show()

    plot_confusion_matrix(confusion_mtx, classes = list(map_characters.values())) 

    plt.show()

    return model

runKerasCNNAugment(x_train,y_train,x_test,y_test)
from keras.applications.vgg16 import VGG16

from keras.models import Model

weight_path = '../input/keras-pretrained-models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

epochs = 10

num_class = 120

#num_class = 16

def vgg16network(a,b,c,d):

    global model

    base_model = VGG16(#weights='imagenet',

        weights = weight_path, include_top=False, input_shape=(im_size, im_size, 3))

    # Add a new top layer

    x = base_model.output

    x = Flatten()(x)

    predictions = Dense(num_class, activation='softmax')(x)

    # This is the model we will train

    model = Model(inputs=base_model.input, outputs=predictions)

    # First: train only the top layers (which were randomly initialized)

    for layer in base_model.layers:

        layer.trainable = False

    model.compile(loss='categorical_crossentropy', 

                  optimizer=keras.optimizers.RMSprop(lr=0.0001), 

                  metrics=['accuracy'])

    callbacks_list = [keras.callbacks.EarlyStopping(monitor='val_acc', patience=3, verbose=1)]

    model.summary()

    model.fit(a,b, epochs=epochs, validation_data=(c,d), verbose=1,callbacks = [MetricsCheckpoint('logs')])

    score = model.evaluate(c,d, verbose=0)

    print('\nKeras CNN #2 - accuracy:', score[1], '\n')

    y_pred = model.predict(c)

    print('\n', sklearn.metrics.classification_report(np.where(d > 0)[1], np.argmax(y_pred, axis=1), target_names=list(map_characters.values())), sep='') 

    Y_pred_classes = np.argmax(y_pred,axis = 1) 

    Y_true = np.argmax(d,axis = 1) 

    confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 

    plotKerasLearningCurve()

    plt.show()

    plot_confusion_matrix(confusion_mtx, classes = list(map_characters.values()))

    plt.show()

    return model

vgg16network(x_train,y_train,x_test,y_test)
# preds = model.predict(x_testContest, verbose=1)

# sub = pd.DataFrame(preds)

# # Set column names to those generated by the one-hot encoding earlier

# col_names = one_hot.columns.values

# sub.columns = col_names

# # Insert the column id from the sample_submission at the start of the data frame

# sub.insert(0, 'id', df_test['id'])

# sub.head(5)



# submission = sub

# submission.to_csv('new_submission.csv', index=False)
end = dt.datetime.now()

print('Total time {} s.'.format((end - start).seconds))