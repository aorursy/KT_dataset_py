import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math
root = "../input/flowers-recognition/flowers/flowers"
os.listdir(root)
# delete extra files
classes = os.listdir(root)
extra_ext = ".ini"
special_ext =  ".db"

for el in classes:
    if el.endswith(extra_ext):
        classes.remove(el)
        os.remove(el)
nclasses = len(classes)
classes = [x.lower() for x in classes]

# check duplicates in labels
if len(classes) != len(list(set(classes))):
    print("Directory pre-processing error. There are duplicates in the labels.\n")
    exit(0)

classes = sorted(classes)
classes
for cl in classes:
    clsdir = os.path.join(root, cl)
    images = os.listdir(clsdir)
    for img in images:
        if img.endswith(extra_ext):
            images.remove(img)
            os.remove(os.path.join(clsdir, img))
def trim_margin(img, lim):
    return img[lim:-lim, lim:-lim]
def trim_whitespace(img):
    
    gray = cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = 255*(gray < 128).astype(np.uint8) # To invert the img to white
    coords = cv2.findNonZero(gray) # Find all non-zero points (text)
    x, y, w, h = cv2.boundingRect(coords) # Find minimum spanning bounding box
    _img = img[y:y+h, x:x+w] # Crop the original image depending on thresholds
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return _img
supported_dims = [16, 32, 64, 128]
def img_resize(img, dims):
    if dims in supported_dims:
        return cv2.resize(img, (dims, dims))
    else:
        print("Incorrect image dimensions.\n")
        return None
import cv2
from cv2 import imread, cvtColor, resize, threshold, calcHist, equalizeHist
def equalize_hist(img):
    img_yuv = cvtColor(img, cv2.COLOR_RGB2YUV)

    # equalize the histogram of the Y channel
    img_yuv[:,:,0] = equalizeHist(img_yuv[:,:,0])

    # convert the YUV image back to RGB format
    imgo = cvtColor(img_yuv, cv2.COLOR_YUV2RGB)

    return imgo
color = ('r','g','b')
def check_grayscale(img):
    '''
    This function returns True if the image is grayscale, False if not.
    The way to calculate that is to see if the 3 colors have the same pixel distribution.
    '''
    if img.shape[2] is 1:
        return True
    hists = []
    for i,col in enumerate(color):
        histr = calcHist([img],[i],None,[256],[0,256])
        hists.append(histr.tolist())

    return hists[1:] == hists[:-1] # https://stackoverflow.com/a/3844832/4569908
def binarize(img):
    '''
    This function decides if the image will be binarized or not
    '''
    hist = cv2.calcHist([img],[0],None,[256],[0,256])
    
    top_limit = 5
    binarize_limit = 32
    black = sum(hist[top_limit:binarize_limit])
    white = sum(hist[-binarize_limit:-top_limit])
    total_pix = img.shape[0] * img.shape[1]
    
    binarize_thres = 0.4
    if (black/total_pix > binarize_thres and white/total_pix > binarize_thres):
        return True
    
    return False
def decide_color_format(img):
    if img.shape[2] is 3 and not check_grayscale(img): # img is in RGB mode
        return 0
    if binarize(img): # img can be binarized
        return 2
    return 1 # img is not RGB and cannot be binaized --> grayscale
import random

def color_code_dataset(path, dim=32):
    img_per_class = 5
    colors = []
    for cl in classes:
        clsdir = os.path.join(root, cl)
        i = 0
        img_in_class = os.listdir(clsdir)
        random_imgs = random.sample(range(len(img_in_class)), img_per_class)
        i = 0
        for i in range(len(random_imgs)):
            imgpath = img_in_class[random_imgs[i]]
            if imgpath.endswith(".db") or imgpath.endswith(".pyc"):
                continue
            totalimgpath = os.path.join(clsdir, imgpath)

            img = imread(totalimgpath, cv2.IMREAD_COLOR)
            img = trim_margin(img, int(img.shape[0] * 0.05))
            img = trim_whitespace(img)
            img = img_resize(img, dim)
            img = equalize_hist(img)
            colors.append(decide_color_format(img))

    total_sample_img = nclasses * img_per_class
    if colors.count(0)/total_sample_img > 0.8:
        return 0 #rgb
    if colors.count(2)/total_sample_img > 0.8:
        return 2 #binary
    return 1 #grayscale
def transform_color(img, color_code, dim):
    '''
    This function transforms the image into RGB, grayscale or binary 
    depending on the choice of the color format decided earlier.
    '''
    
    if color_code is 0:
        return cvtColor(img, cv2.COLOR_BGR2RGB) # return rgb image
    
    if color_code is 1:
        return cvtColor(img, cv2.COLOR_BGR2GRAY).reshape(dim, dim, 1)
    
    if color_code is 2:
        img = cvtColor(img, cv2.COLOR_BGR2GRAY).reshape(dim, dim, 1)
        (_, _img) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        _img = 255*(_img < 128).astype(np.uint8) # To invert the text to white
        return _img # returns binarized image
    
    return img
other_extensions = [".db", ".pyc", ".py"]
def open_images(path, classes, color_code, dim=32):
    
    xall = []
    yall = []
    label = 0
    j = 0

    for cl in classes:
        clsdir = os.path.join(path, cl)
        for imgname in os.listdir(clsdir):
            bad_ext_found = 0
            for other_ext in other_extensions:
                if imgname.endswith(other_ext):
                    bad_ext_found = 1
                    break
            if not bad_ext_found:
                print("Opening files in {}: {}".format(cl, str(j + 1)), end="\r")
                imgpath = os.path.join(clsdir, imgname)

                #open and pre-process images
                img = imread(imgpath, cv2.IMREAD_COLOR)
                img = trim_margin(img, int(img.shape[0] * 0.05))
                img_no_trim = img
                img = trim_whitespace(img)
                if img.shape[0] < dim or img.shape[1] < dim:
                    img = img_no_trim
                img = img_resize(img, dim)
                img = equalize_hist(img)
                img = transform_color(img, color_code, dim)

                xall.append(img)  # Get image 
                yall.append(label)  # Get image label (folder name)
                j += 1

        j = 0
        label += 1
        print()

    n = len(xall)
    print("{} images in set".format(n))
    return xall, yall
# set parameters
dim = 32
color_code = color_code_dataset(root, dim)
nchannels = 3
if color_code > 0:
    nchannels = 1
print("Opening images:\n")
xall, yall = open_images(root, classes, color_code, dim)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
%matplotlib inline
# https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplot.html
# subplot(2,3,3) = subplot(233)
# a grid of 3x3 is created, then plots are inserted in some of these slots

_xall = np.asarray(xall)

if color_code > 0: # 1 channel
    _xall = _xall.reshape(-1, dim, dim)

for i in range(0,9): # how many imgs will show from the 3x3 grid
    plt.subplot(330 + (i+1)) # open next subplot
    plt.imshow(_xall[i + 155], cmap=plt.get_cmap('gray'))
    plt.title(yall[i + 155]);
import seaborn as sns
sns.set(style='white', context='notebook', palette='deep')

# plot how many images there are in each class
sns.countplot(yall)

_yall = pd.Series(yall)

# array with each class and its number of images
vals_class = _yall.value_counts()
print(vals_class)

# mean and std
cls_mean = np.mean(vals_class)
cls_std = np.std(vals_class,ddof=1)

print("The mean amount of elements per class is", cls_mean)
print("The standard deviation in the element per class distribution is", cls_std)

# 68% - 95% - 99% rule, the 68% of the data should be cls_std away from the mean and so on
# https://en.wikipedia.org/wiki/68%E2%80%9395%E2%80%9399.7_rule
metric_opt = None
if cls_std > cls_mean * (0.6827 / 2):
    print("The standard deviation is high")
    metric_opt = -1
    
# if the data is skewed then we won't be able to use accurace as its results will be misleading and we may use F-beta score instead.
vals_class.sort_index(inplace=True)
vals_lst = list(vals_class.values)
minclass = vals_lst.index(min(vals_lst))
vals_lst = sorted(vals_lst)
interesting_class = -1 # there is no interesting class

# if there is one class with much less images than the others
if (vals_lst[1] - vals_lst[0])/(vals_lst[-1] - vals_lst[0]) > 0.3:
    metric_opt = 1
    interesting_class = minclass
    
# which class is mostly imbalanced, -1 if none
interesting_class
xall = np.asarray(xall)
yall = np.asarray(yall)
xall.shape, yall.shape
xall = xall / 255
from keras.utils.np_utils import to_categorical

print("Shape of labels before: ", yall.shape) 
yall = to_categorical(yall, num_classes = nclasses)
print("Shape of labels after: ", yall.shape) 
from sklearn.model_selection import train_test_split

# fix random seed for reproducibility
seed = 35
np.random.seed(seed)

# split xtrain/xval from testset
split_pct = 0.1
_xall, xtest, _yall, ytest = train_test_split(xall, yall, test_size=split_pct, random_state=seed, shuffle=True, stratify=yall)

# split xtrain from xval
split_pct = 0.2
xtrain, xval, ytrain, yval = train_test_split(_xall, _yall, test_size=split_pct, random_state=seed, shuffle=True, stratify=_yall)

print("trainset:", xtrain.shape, ytrain.shape, "\tvalset:", xval.shape, yval.shape, "\ttestset:", xtest.shape, ytest.shape)
import functools
import tensorflow as tf
from keras import backend as K

from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Dropout, Lambda, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPool2D, AvgPool2D
from keras.regularizers import l1, l2
def create_model(nchannels):

    model = Sequential()

    model.add(Conv2D(filters=32, kernel_size=(3,3), padding='valid', activation='relu', input_shape=(dim,dim,nchannels)))
    model.add(Conv2D(filters=64, kernel_size=(3,3), padding='valid', activation='relu', kernel_regularizer=l2(0.0001)))
    model.add(MaxPool2D(pool_size=(3,3), strides=(2,2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=64, kernel_size=(3,3), padding='valid', activation='relu', kernel_regularizer=l2(0.0001)))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding='valid', activation='relu', kernel_regularizer=l2(0.0001)))
    model.add(MaxPool2D(pool_size=(3,3), strides=(2,2)))
    model.add(Dropout(0.2))    

    model.add(Flatten())
    model.add(Dense(120, activation='relu'))
    model.add(Dense(120, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(nclasses, activation='softmax'))
    
    return model
def single_class_accuracy(interesting_class_id):
    def fn(y_true, y_pred):
        class_id_true = K.argmax(y_true, axis=-1)
        class_id_preds = K.argmax(y_pred, axis=-1)
        # Replace class_id_preds with class_id_true for recall here
        accuracy_mask = K.cast(K.equal(class_id_preds, interesting_class_id), 'int32')
        class_acc_tensor = K.cast(K.equal(class_id_true, class_id_preds), 'int32') * accuracy_mask
        class_acc = K.sum(class_acc_tensor) / K.maximum(K.sum(accuracy_mask), 1)
        return class_acc
    return fn
def mcor(y_true, y_pred):
    #matthews_correlation
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    
    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.epsilon())

def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.
    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.
    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def f1(y_true, y_pred):
    
    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.
        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.
        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall
    
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
def as_keras_metric(method):
    @functools.wraps(method)
    def wrapper(self, args, **kwargs):
        """ Wrapper for turning tensorflow metrics into keras metrics """
        value, update_op = method(self, args, **kwargs)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([update_op]):
            value = tf.identity(value)
        return value
    return wrapper
def compile_model(model, optimizer="adam", loss="categorical_crossentropy", metric_opt=0, interesting_class=0):
    
    if metric_opt == 0:
        metric = ["accuracy"]
    if metric_opt == 1:
        metric = [single_class_accuracy(interesting_class)]
    elif metric_opt == 2:
        metric = [mcor, f1, recall]
    elif metric_opt == 3:
        precision2 = as_keras_metric(tf.metrics.precision)
        recall2 = as_keras_metric(tf.metrics.recall)
        metric = [precision2, recall2]
        
    model.compile(optimizer=optimizer, loss=loss, metrics=metric)
model = create_model(nchannels)
if not metric_opt:
    metric_opt = 0
elif metric_opt == -1: # there is class imbalance
    metric_opt = 2
elif metric_opt != 1: # no class imbalance, no "interesting" class found
    metric_opt = 0
    
compile_model(model, "adam", "categorical_crossentropy", metric_opt)
model.summary()
learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', patience=3, verbose=1, factor=0.5, min_lr=0.00001)
def add_data_augmn(xtrain):

    datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=30,  # randomly rotate images in the range (degrees, 0 to 180)
            zoom_range=0.1, # Randomly zoom image 
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=False,  # randomly flip images
            vertical_flip=False)  # randomly flip images

    datagen.fit(xtrain)
    return xtrain, datagen
size_thres = [1000, 5000, 10000, 50000, 100000]
batch_sizes = [16, 32, 64, 128, 256]
epochs_thres = [50, 40, 35, 20, 15]
def set_batch_epochs(nimages):
    for i in range(len(size_thres)):
        if nimages < size_thres[i]:
            return epochs_thres[i], batch_sizes[i]
    return epochs_thres[-1], batch_sizes[-1]
epochs, batch_size = set_batch_epochs(xtrain.shape[0])
data_aug = 1

if data_aug > 0:
    xtrain, datagen = add_data_augmn(xtrain)
    history_train = model.fit_generator(datagen.flow(xtrain,ytrain, batch_size=batch_size),
                                  epochs=epochs, 
                                  validation_data=(xval,yval),
                                  verbose=1, 
                                  shuffle=True, # always shuffle, in small datasets one epoch could contain just 1 label --> it doesn't learn!
                                  callbacks=[learning_rate_reduction])

else:
    history_train = model.fit(x=xtrain, 
                        y=ytrain, 
                        batch_size=batch_size, 
                        epochs=epochs, 
                        verbose=1, 
                        callbacks=[learning_rate_reduction],#, early_stop], 
                        validation_split=0.0, 
                        validation_data=(xval,yval), 
                        shuffle=True, 
                        class_weight=None, 
                        sample_weight=None, 
                        initial_epoch=0, 
                        steps_per_epoch=None, 
                        validation_steps=None)

metrics = list(history_train.history.keys())

# Plot the loss and accuracy curves for training and validation 
iterations = int((len(metrics) - 1)/2)
fig, ax = plt.subplots(iterations, 1)

for i in range(iterations):
    ax[i].plot(history_train.history[metrics[i + iterations]], color='b', label="train_" + metrics[i + iterations])
    ax[i].plot(history_train.history[metrics[i]], color='r', label=metrics[i], axes=ax[i])
    ax[i].grid(color='black', linestyle='-', linewidth=0.25)
    legend = ax[i].legend(loc='best', shadow=True)
from sklearn.metrics import confusion_matrix
import itertools

# Confusion matrix
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
ypred = model.predict(xval)
# Convert predictions classes from one hot vectors to labels: [0 0 1 0 0 ...] --> 2
ypred_classes = np.argmax(ypred,axis=1)
# Convert validation observations from one hot vectors to labels
ytrue = np.argmax(yval,axis=1)
# compute the confusion matrix
confusion_mtx = confusion_matrix(ytrue, ypred_classes) 
# plot the confusion matrix
plot_confusion_matrix(confusion_mtx, range(nclasses), False, "Confusion matrix of the val set")
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

xtest = xtest.reshape(-1, dim, dim, nchannels)
ypredtest = model.predict_classes(xtest)
# Convert validation observations from one hot vectors to labels
ytrue = np.argmax(ytest,axis=1)
print("Test accuracy score:", accuracy_score(ytrue, ypredtest))
print("Test F1 score:", f1_score(ytrue, ypredtest, average="macro"))
print("Test precision score:", precision_score(ytrue, ypredtest, average="macro"))
print("Test recall score:", recall_score(ytrue, ypredtest, average="macro"))
# Predict the values from the validation dataset
ypred = model.predict(xtest)
# Convert predictions classes from one hot vectors to labels: [0 0 1 0 0 ...] --> 2
ypred_classes = np.argmax(ypred,axis=1)
# Convert validation observations from one hot vectors to labels
ytrue = np.argmax(ytest,axis=1)
# compute the confusion matrix
confusion_mtx = confusion_matrix(ytrue, ypred_classes)
# plot the confusion matrix
plot_confusion_matrix(confusion_mtx, range(nclasses), False, "Confusion matrix of the test set")
model = create_model(nchannels)
compile_model(model, "adam", "categorical_crossentropy", metric_opt)
learning_rate_reduction = ReduceLROnPlateau(monitor='loss', patience=3, verbose=1, factor=0.5, min_lr=0.00001)
epochs, batch_size = set_batch_epochs(xall.shape[0])
xall, datagen = add_data_augmn(xall)
history = model.fit_generator(datagen.flow(xall,yall, batch_size=batch_size),
                              epochs=epochs, 
                              verbose=1, 
                              callbacks=[learning_rate_reduction])
metrics = list(history.history.keys())

fig, ax = plt.subplots(len(metrics)-1, 1)

for i in range(len(metrics)-1):
    ax[i].plot(history.history[metrics[i]], color='b', label=metrics[i])
    ax[i].grid(color='black', linestyle='-', linewidth=0.25)
    legend = ax[i].legend(loc='best', shadow=True)

xtest = xtest.reshape(-1, dim, dim, nchannels)
ypredtest = model.predict_classes(xtest)
# Convert validation observations from one hot vectors to labels
ytrue = np.argmax(ytest,axis=1)
print("test_acc", accuracy_score(ytrue, ypredtest))
# Predict the values from the validation dataset
ypred = model.predict(xtest)
# Convert predictions classes from one hot vectors to labels: [0 0 1 0 0 ...] --> 2
ypred_classes = np.argmax(ypred,axis=1)
# Convert validation observations from one hot vectors to labels
ytrue = np.argmax(ytest,axis=1)
# compute the confusion matrix
confusion_mtx = confusion_matrix(ytrue, ypred_classes) 
# plot the confusion matrix
plot_confusion_matrix(confusion_mtx, classes=range(nclasses))
