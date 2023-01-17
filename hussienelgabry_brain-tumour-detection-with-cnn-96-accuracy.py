!pip install imutils

import numpy as np 

import pandas as pd 

from plotly.offline import init_notebook_mode, iplot

import os

from os import listdir

import plotly.graph_objs as go

import tensorflow as tf

from keras.preprocessing.image import ImageDataGenerator

import cv2

import matplotlib.pyplot as plt

%matplotlib inline

import imutils    

import itertools

from sklearn.metrics import accuracy_score, confusion_matrix

from tensorflow.keras.models import Model,load_model

from tensorflow.keras.layers import Conv2D,Input,ZeroPadding2D,BatchNormalization,Flatten,Activation,Dense,MaxPooling2D

from sklearn.model_selection import train_test_split

from sklearn.utils import shuffle #shuffling the data improves the model

image_dir="../input/brain-mri-images-for-brain-tumor-detection/"
os.makedirs('../output/kaggle/working/augmented-images', exist_ok = True)

os.makedirs('../output/kaggle/working/augmented-images/yes', exist_ok = True)

os.makedirs('../output/kaggle/working/augmented-images/no', exist_ok = True)



augmented_data_path ='../output/kaggle/working/augmented-images/'



augmented_yes =augmented_data_path+'yes'

augmented_no = augmented_data_path+'no'



IMG_SIZE = (224,224)
def augment_data(file_dir, n_generated_samples, save_to_dir):

    data_gen = ImageDataGenerator(rotation_range=10, 

                                  width_shift_range=0.1, 

                                  height_shift_range=0.1, 

                                  shear_range=0.1, 

                                  brightness_range=(0.3, 1.0),

                                  horizontal_flip=True, 

                                  vertical_flip=True, 

                                  fill_mode='nearest'

                                 )



    for filename in listdir(file_dir):

        image = cv2.imread(file_dir + '/' + filename)

        # reshape the image

        image = image.reshape((1,)+image.shape)

        save_prefix = 'aug_' + filename[:-4]

        i=0

        for batch in data_gen.flow(x=image, batch_size=1, save_to_dir=save_to_dir,save_prefix=save_prefix, save_format='jpg'):

                i += 1

                if i > n_generated_samples:

                    break




dir = os.listdir('../output/kaggle/working/augmented-images/yes') 



if len(dir) == 0: 



    # augment data for the examples with label equal to 'yes' representing tumurous examples

    augment_data(file_dir=image_dir+'yes',n_generated_samples=6, save_to_dir=augmented_data_path+'yes')

    # augment data for the examples with label equal to 'no' representing non-tumurous examples

    augment_data(file_dir=image_dir+'no', n_generated_samples=9, save_to_dir=augmented_data_path+'no')

else:

    print(len(dir))
def load_data(dir_list):



    # load all images in a directory

    X = []

    y = []

#     image_width, image_height = image_size

    

    for directory in dir_list:

        for filename in listdir(directory):

            image = cv2.imread(directory+'/'+filename)

#             image = crop_brain_contour(image, plot=False)

#             image = cv2.resize(image, dsize=(image_width, image_height), interpolation=cv2.INTER_CUBIC)

#             # normalize values

#             image = image / 255.

#             # convert image to numpy array and append it to X

            X.append(image)

            # append a value of 1 to the target array if the image

            # is in the folder named 'yes', otherwise append 0.

            if directory[-3:] == 'yes':

                y.append([1])

            else:

                y.append([0])

                

    X = np.array(X)

    y = np.array(y)

    

    # Shuffle the data

    X, y = shuffle(X, y)

    

    print(f'Number of examples is: {len(X)}')

    print(f'X shape is: {X.shape}')

    print(f'y shape is: {y.shape}')

    

    return X, y
def plot_samples(X, y, labels_dict, n=50):

    """

    Creates a gridplot for desired number of images (n) from the specified set

    """

    for index in range(len(labels_dict)):

        imgs = X[np.argwhere(y == index)][:n]

        j = 10

        i = int(n/j)



        plt.figure(figsize=(15,6))

        c = 1

        for img in imgs:

            plt.subplot(i,j,c)

            plt.imshow(img[0])



            plt.xticks([])

            plt.yticks([])

            c += 1

        plt.suptitle('Tumor: {}'.format(labels_dict[index]))

        plt.show()


X_train, y_train = load_data([augmented_yes, augmented_no])
plot_samples(X_train, y_train, ['yes','no'], 20)
def crop_brain_contour(image, plot=False):

    

    # Convert the image to grayscale, and blur it slightly

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    

    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]

    thresh = cv2.erode(thresh, None, iterations=2)

    thresh = cv2.dilate(thresh, None, iterations=2)



    # Find contours in thresholded image, then grab the largest one

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cnts = imutils.grab_contours(cnts)

    c = max(cnts, key=cv2.contourArea)

    # extreme points

    extLeft = tuple(c[c[:, :, 0].argmin()][0])

    extRight = tuple(c[c[:, :, 0].argmax()][0])

    extTop = tuple(c[c[:, :, 1].argmin()][0])

    extBot = tuple(c[c[:, :, 1].argmax()][0])

    

    # crop new image out of the original image using the four extreme points (left, right, top, bottom)

    new_image = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]            



    if plot:

        plt.figure()

        plt.subplot(1, 2, 1)

        plt.imshow(image)

        plt.tick_params(axis='both', which='both', top=False, bottom=False, left=False, right=False,labelbottom=False, labeltop=False, labelleft=False, labelright=False)

        plt.title('Original Image')

        plt.subplot(1, 2, 2)

        plt.imshow(new_image)

        plt.tick_params(axis='both', which='both',top=False, bottom=False, left=False, right=False,labelbottom=False, labeltop=False, labelleft=False, labelright=False)

        plt.title('Cropped Image')

        plt.show()

    

    return new_image
img = cv2.imread('../input/brain-mri-images-for-brain-tumor-detection/brain_tumor_dataset/yes/Y108.jpg')

img = cv2.resize(

            img,

            dsize=IMG_SIZE,

            interpolation=cv2.INTER_CUBIC

        )

gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

gray = cv2.GaussianBlur(gray, (5, 5), 0)



# threshold the image, then perform a series of erosions +

# dilations to remove any small regions of noise

thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]

thresh = cv2.erode(thresh, None, iterations=2)

thresh = cv2.dilate(thresh, None, iterations=2)



# find contours in thresholded image, then grab the largest one

cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cnts = imutils.grab_contours(cnts)

c = max(cnts, key=cv2.contourArea)



# find the extreme points

extLeft = tuple(c[c[:, :, 0].argmin()][0])

extRight = tuple(c[c[:, :, 0].argmax()][0])

extTop = tuple(c[c[:, :, 1].argmin()][0])

extBot = tuple(c[c[:, :, 1].argmax()][0])



# add contour on the image

img_cnt = cv2.drawContours(img.copy(), [c], -1, (0, 255, 255), 4)



# add extreme points

img_pnt = cv2.circle(img_cnt.copy(), extLeft, 8, (0, 0, 255), -1)

img_pnt = cv2.circle(img_pnt, extRight, 8, (0, 255, 0), -1)

img_pnt = cv2.circle(img_pnt, extTop, 8, (255, 0, 0), -1)

img_pnt = cv2.circle(img_pnt, extBot, 8, (255, 255, 0), -1)



# crop

ADD_PIXELS = 0

new_img = img[extTop[1]-ADD_PIXELS:extBot[1]+ADD_PIXELS, extLeft[0]-ADD_PIXELS:extRight[0]+ADD_PIXELS].copy()
plt.figure(figsize=(15,6))

plt.subplot(141)

plt.imshow(img)

plt.xticks([])

plt.yticks([])

plt.title('Step 1. Get the original image')

plt.subplot(142)

plt.imshow(img_cnt)

plt.xticks([])

plt.yticks([])

plt.title('Step 2. Find the biggest contour')

plt.subplot(143)

plt.imshow(img_pnt)

plt.xticks([])

plt.yticks([])

plt.title('Step 3. Find the extreme points')

plt.subplot(144)

plt.imshow(new_img)

plt.xticks([])

plt.yticks([])

plt.title('Step 4. Crop the image')

plt.show()
def Croping_Data(train):



    # load all images in a directory

    X = []

    y = []

    

    for img in train:

        image = crop_brain_contour(img, plot=False)

        X.append(image)

                

    X = np.array(X)

    

    return X
X = Croping_Data(X_train)
plot_samples(X, y_train, ['yes','no'], 20)
def Resize_Data(train):



    # load all images in a directory

    X = []

    y = []

    

    IMG_WIDTH, IMG_HEIGHT = (240, 240)

    

    for img in train:

        image = cv2.resize(img, dsize=(IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_CUBIC)

        # normalize values

        image = image / 255.

        # convert image to numpy array and append it to X

        X.append(image)

                

    X = np.array(X)

    

    return X
augmented_yes =augmented_data_path+'yes'

augmented_no = augmented_data_path+'no'



IMG_WIDTH, IMG_HEIGHT = (240, 240)



X = Resize_Data(X)

y = y_train



plot_samples(X, y_train, ['yes','no'],10)
def split_data(X, y, test_size=0.2):

       

    X_train, X_test_val, y_train, y_test_val = train_test_split(X, y, test_size=test_size)

    X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, test_size=0.5)

    

    return X_train, y_train, X_val, y_val, X_test, y_test
X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y, test_size=0.3)
print ("number of training examples = " + str(X_train.shape[0]))

print ("number of validation examples = " + str(X_val.shape[0]))

print ("number of test examples = " + str(X_test.shape[0]))
y = dict()

y[0] = []

y[1] = []

for set_name in (y_train, y_val, y_test):

    y[0].append(np.sum(set_name == 0))

    y[1].append(np.sum(set_name == 1))



trace0 = go.Bar(

    x=['Train Set', 'Validation Set', 'Test Set'],

    y=y[0],

    name='No',

    marker=dict(color='#33cc33'),

    opacity=0.7

)

trace1 = go.Bar(

    x=['Train Set', 'Validation Set', 'Test Set'],

    y=y[1],

    name='Yes',

    marker=dict(color='#ff3300'),

    opacity=0.7

)

data = [trace0, trace1]

layout = go.Layout(

    title='Count of classes in each set',

    xaxis={'title': 'Set'},

    yaxis={'title': 'Count'}

)

fig = go.Figure(data, layout)

iplot(fig)
def build_model(input_shape):

    X_input = Input(input_shape) 

    X = ZeroPadding2D((2, 2))(X_input) 

    

    X = Conv2D(32, (7, 7), strides = (1, 1))(X)

    X = BatchNormalization(axis = 3, name = 'bn0')(X)

    X = Activation('relu')(X) 

    

    X = MaxPooling2D((4, 4))(X) 

    X = MaxPooling2D((4, 4))(X) 

    X = Flatten()(X) 

    X = Dense(1, activation='sigmoid')(X) 

    model = Model(inputs = X_input, outputs = X)

    

    return model
IMG_SHAPE = (IMG_WIDTH, IMG_HEIGHT, 3)

model=build_model(IMG_SHAPE)

model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(x=X_train, y=y_train, batch_size=32, epochs=22, validation_data=(X_val, y_val))
history = model.history.history
def plot_metrics(history):

    

    train_loss = history['loss']

    val_loss = history['val_loss']

    train_acc = history['accuracy']

    val_acc = history['val_accuracy']

    

    # Loss

    plt.figure()

    plt.plot(train_loss, label='Training Loss')

    plt.plot(val_loss, label='Validation Loss')

    plt.title('Loss')

    plt.legend()

    plt.show()

    

    # Accuracy

    plt.figure()

    plt.plot(train_acc, label='Training Accuracy')

    plt.plot(val_acc, label='Validation Accuracy')

    plt.title('Accuracy')

    plt.legend()

    plt.show()
plot_metrics(history)


def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    plt.figure(figsize = (6,6))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=90)

    plt.yticks(tick_marks, classes)

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]



    thresh = cm.max() / 2.

    cm = np.round(cm,2)

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    plt.show()
labels = ['yes','no']

# validate on val set

predictions = model.predict(X_val)

predictions = [1 if x>0.5 else 0 for x in predictions]



accuracy = accuracy_score(y_val, predictions)

print('Val Accuracy = %.2f' % accuracy)



confusion_mtx = confusion_matrix(y_val, predictions) 

cm = plot_confusion_matrix(confusion_mtx, classes = labels, normalize=False)
# validate on val set

predictions = model.predict(X_test)

predictions = [1 if x>0.5 else 0 for x in predictions]



accuracy = accuracy_score(y_test, predictions)

print('Val Accuracy = %.2f' % accuracy)



confusion_mtx = confusion_matrix(y_test, predictions) 

cm = plot_confusion_matrix(confusion_mtx, classes = labels, normalize=False)
for i in range(5):

    plt.figure()

    plt.imshow(X_test[i])

    plt.xticks([])

    plt.yticks([])

    plt.title(f'Actual class: {y_test[i]}\nPredicted class: {predictions[i]}')

    plt.show()