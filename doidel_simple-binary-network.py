import os
#print(os.listdir("../lib"))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
%matplotlib inline
from PIL import Image
import math

np.random.seed(2)

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
import itertools

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, Lambda
from keras.optimizers import RMSprop, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
import keras.backend as K


sns.set(style='white', context='notebook', palette='deep')
goes_classes = ['quiet','A','B','C','M','X']

def flux_to_class(f: float, only_main=False):
    'maps the peak_flux of a flare to one of the following descriptors: \
    *quiet* = 1e-9, *B* >= 1e-7, *C* >= 1e-6, *M* >= 1e-5, and *X* >= 1e-4\
    See also: https://en.wikipedia.org/wiki/Solar_flare#Classification'
    decade = int(min(math.floor(math.log10(f)), -4))
    main_class = goes_classes[decade + 9] if decade >= -8 else 'quiet'
    sub_class = str(round(10 ** -decade * f)) if main_class != 'quiet' and only_main != True else ''
    return main_class + sub_class
def create_simple_image_set(phase):
    df = pd.read_csv('../input/sdobenchmark_full/' + phase + '/meta_data.csv', sep=",", parse_dates=["start", "end"], index_col="id")
    new_df = {'id': [], 'label': [], 'img': []}
    for row in df.iterrows():
        ar_nr, p = row[0].split("_", 1)
        img_path = os.path.join('../input/sdobenchmark_full/', phase, ar_nr, p)
        
        if not os.path.isdir(img_path):
            print(img_path + ' does not exist!')
            continue
        
        for img_name in os.listdir(img_path):
            if img_name.endswith('_magnetogram.jpg'):
                new_df['id'].append(row[0] + '-' + img_name.split('__')[0])
                new_df['label'].append(flux_to_class(row[1]['peak_flux'], only_main=True))
                
                # load the image and preprocess it
                im = Image.open(os.path.join(img_path, img_name))
                im = im.crop((44, 44, 212, 212))
                im = im.resize((28,28), Image.ANTIALIAS)
                im = np.array(im) / 255.0
                im = im.reshape(28,28,1)
                new_df['img'].append(im)
    
    return pd.DataFrame(data=new_df)

train = create_simple_image_set('training')
test = create_simple_image_set('test')
print(train["label"].value_counts())

# join the labels
Y_train = ((train["label"] == 'X') | (train["label"] == 'M')).astype(int)
print('')
print('New: ')
print(Y_train.value_counts())

sns.countplot(Y_train)

# Drop 'label' column
X_train = np.asanyarray(list(train['img']))

# free some space
del train

# and do the same for validation data
# Here we could also split the training data into test and validation. But we'd have to make sure to split by Active Region numbers (top-level folder)
Y_val = ((test["label"] == 'X') | (test["label"] == 'M')).astype(int)
X_val = np.asanyarray(list(test['img']))
del test
g = plt.imshow(X_train[0][:,:,0])
# Define the weighted binary crossentropy
# While keras has a 'class_weight' parameter in fit_generator, it doesn't support the use on two binary classes like this.
# An alternative is to use sample_weights, or to add a weight in a customized binary_crossentropy loss
true_weight = Y_train.value_counts()[0] / Y_train.value_counts()[1]
print('positive class weight: ' + str(true_weight))
def weighted_binary_crossentropy(y_true, y_pred):
    y_true = K.clip(y_true, K.epsilon(), 1)
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    logloss = -(y_true * K.log(y_pred) * true_weight + (1 - y_true) * K.log(1 - y_pred))
    return K.mean( logloss, axis=-1)
# Set the CNN model 
# my CNN architechture is In -> [[Conv2D->relu]*2 -> MaxPool2D -> Dropout]*2 -> Flatten -> Dense -> Dropout -> Out

model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))

model.add(Dense(1, activation = "sigmoid"))

# Define the optimizer
optimizer = Adam()

# Compile the model
model.compile(optimizer = optimizer , loss = weighted_binary_crossentropy)

# Set a learning rate annealer
learning_rate_reduction = ReduceLROnPlateau(monitor='loss', #val_
                                            patience=4,
                                            min_delta=1e-8,
                                            verbose=1, 
                                            factor=0.5,
                                            min_lr=0.00001)

batch_size = 128

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True)  # randomly flip images

datagen.fit(X_train)
# Fit the model
history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),
                              epochs = 10,
                              validation_data = (X_val,Y_val),
                              verbose = 2,
                              steps_per_epoch=X_train.shape[0] // batch_size, 
                              callbacks=[learning_rate_reduction])
# Plot the loss and accuracy curves for training and validation 
plt.plot(history.history['loss'], color='b', label="Training loss")
plt.plot(history.history['val_loss'], color='r', label="validation loss")
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()
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
Y_pred = model.predict(X_val)
Y_pred = np.round(Y_pred)
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_val, Y_pred) 
# plot the confusion matrix
plot_confusion_matrix(confusion_mtx, classes = range(2)) 
# copied from https://github.com/i4Ds/SDOBenchmark/blob/master/notebooks/utils/statistics.py
def true_skill_statistic(y_true, y_pred):
    'Calculates the True Skill Statistic (TSS) on binarized predictions\
    It is not sensitive to the balance of the samples\
    This statistic is often used in weather forecasts (including solar weather)\
    1 = perfect prediction, 0 = random prediction, -1 = worst prediction'
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print(tn, fp, fn, tp)
    return tp / (tp + fn) - fp / (fp + tn)

print(f'Predicted {np.sum(np.array(Y_pred))} M+, {len(Y_pred)-np.sum(np.array(Y_pred))} < M')
print('TSS: ' + str(true_skill_statistic(Y_val, Y_pred)))
fpr, tpr, _ = roc_curve(Y_val, Y_pred)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % auc(fpr, tpr))
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc="lower right")
plt.show()