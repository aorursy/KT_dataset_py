import numpy as np

import pandas as pd

import cv2

import os

from glob import glob 

import matplotlib.pyplot as plt



import random

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_curve, auc, roc_auc_score



from keras_preprocessing.image import ImageDataGenerator



from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation

from keras.layers import Conv2D, MaxPool2D



from keras.callbacks import EarlyStopping, ReduceLROnPlateau



from IPython.display import clear_output

path = "../input/" 

labels = pd.read_csv(path + 'train_labels.csv')

train_path = path + 'train/'

test_path = path + 'test/'
df = pd.DataFrame({'path': glob(os.path.join(train_path,'*.tif'))})

df['id'] = df.path.map(lambda x: ((x.split("n")[2].split('.')[0])[1:]))

df = df.merge(labels, on = "id")

df.head(3)
def readImage(path):

    # OpenCV reads the image in bgr format by default

    bgr_img = cv2.imread(path)

    # We flip it to rgb for visualization purposes

    b,g,r = cv2.split(bgr_img)

    rgb_img = cv2.merge([r,g,b])

    return rgb_img
positive_indices = list(np.where(df["label"] == True)[0])

negative_indices = list(np.where(df["label"] == False)[0])

rand_pos_inds = random.sample(positive_indices, 4)

rand_neg_inds = random.sample(negative_indices, 4)



fig, ax = plt.subplots(2,4, figsize=(20,8))

fig.suptitle('Histopathologic scans of lymph node sections',fontsize=20, fontweight='bold')



for i in range(0, 4):

    ax[0,i].imshow(readImage(df.iloc[rand_pos_inds[i],0]))

    ax[0,i].set_title("Positive Example", fontweight='bold')

    

    ax[1,i].imshow(readImage(df.iloc[rand_neg_inds[i],0]))

    ax[1,i].set_title("Negative Example", fontweight='bold')
IMG_SIZE = 196

BATCH_SIZE = 128
test_list = os.listdir(test_path)

train_list = os.listdir(train_path)

print("There are " + str(len(train_list)) + " training examples.")

print("There are " + str(len(test_list)) + " test examples.")
df['label'] = df['label'].astype(str)

train, valid = train_test_split(df, test_size=0.2, stratify = df['label'])

def crop_centre(image, crop_length):

    original_size = image.shape[0]

    centre = original_size // 2

    lower_bound = centre - crop_length // 2 

    upper_bound = centre + crop_length // 2

    image = image[(lower_bound):(upper_bound),(lower_bound):(upper_bound)]

    return image


train_datagen = ImageDataGenerator(rescale=1./255,

                                  vertical_flip = True,

                                  horizontal_flip = True,

                                  rotation_range=90,

                                  zoom_range=0.2, 

                                  width_shift_range=0.1,

                                  height_shift_range=0.1,

                                  shear_range=0.05,

                                  channel_shift_range=0.1)



test_datagen = ImageDataGenerator(rescale = 1./255) 


train_generator = train_datagen.flow_from_dataframe(dataframe = train, 

                                                    directory = None,

                                                    x_col = 'path', 

                                                    y_col = 'label',

                                                    target_size = (IMG_SIZE,IMG_SIZE),

                                                    class_mode = "binary",

                                                    batch_size=BATCH_SIZE,

                                                    seed = 110318,

                                                    shuffle = True)
valid_generator = test_datagen.flow_from_dataframe(dataframe = valid,

                                                   directory = None,

                                                   x_col = 'path',

                                                   y_col = 'label',

                                                   target_size = (IMG_SIZE,IMG_SIZE),

                                                   class_mode = 'binary',

                                                   batch_size = BATCH_SIZE,

                                                   shuffle = False)
from keras.applications.resnet50 import ResNet50



dropout_fc = 0.5



conv_base = ResNet50(weights = 'imagenet', include_top = False, input_shape = (IMG_SIZE,IMG_SIZE,3))



my_model = Sequential()



my_model.add(conv_base)

my_model.add(Flatten())

my_model.add(Dense(256, use_bias=False))

my_model.add(BatchNormalization())

my_model.add(Activation("relu"))

my_model.add(Dropout(dropout_fc))

my_model.add(Dense(1, activation = "sigmoid"))

my_model.summary()
conv_base.Trainable=True



set_trainable=False

for layer in conv_base.layers:

    if layer.name == 'res5a_branch2a':

        set_trainable = True

    if set_trainable:

        layer.trainable = True

    else:

        layer.trainable = False
from keras import optimizers

my_model.compile(optimizers.Adam(0.001), loss = "binary_crossentropy", metrics = ["accuracy"])
train_step_size = train_generator.n // train_generator.batch_size

valid_step_size = valid_generator.n // valid_generator.batch_size
earlystopper = EarlyStopping(monitor='val_loss', patience=3, verbose=2, restore_best_weights=True)

reduce = ReduceLROnPlateau(monitor='val_loss', patience=1, verbose=1, factor=0.1)
history = my_model.fit_generator(train_generator,

                                     steps_per_epoch = train_step_size,

                                     epochs = 10,

                                     validation_data = valid_generator,

                                     validation_steps = valid_step_size,

                                     callbacks = [reduce, earlystopper],

                                     verbose = 2)



    
epochs = [i for i in range(1, len(history.history['loss'])+1)]



plt.plot(epochs, history.history['loss'], color='blue', label="training_loss")

plt.plot(epochs, history.history['val_loss'], color='red', label="validation_loss")

plt.legend(loc='best')

plt.title('training')

plt.xlabel('epoch')

plt.savefig("training.png", bbox_inches='tight')

plt.show()



plt.plot(epochs, history.history['acc'], color='blue', label="training_accuracy")

plt.plot(epochs, history.history['val_acc'], color='red',label="validation_accuracy")

plt.legend(loc='best')

plt.title('validation')

plt.xlabel('epoch')

plt.savefig("validation.png", bbox_inches='tight')

plt.show()
roc_validation_generator = ImageDataGenerator(rescale=1./255).flow_from_dataframe(valid,

                                                                                  x_col = 'path',

                                                                                  y_col = 'label',

                                                                                  target_size = (IMG_SIZE,IMG_SIZE),

                                                                                  class_mode = 'binary',

                                                                                  batch_size = BATCH_SIZE,

                                                                                  shuffle = False)

predictions = my_model.predict_generator(roc_validation_generator, steps=len(roc_validation_generator), verbose=2)

false_positive_rate, true_positive_rate, threshold = roc_curve(roc_validation_generator.classes, predictions)

area_under_curve = auc(false_positive_rate, true_positive_rate)



plt.plot([0, 1], [0, 1], 'k--')

plt.plot(false_positive_rate, true_positive_rate, label='AUC = {:.3f}'.format(area_under_curve))

plt.xlabel('False positive rate')

plt.ylabel('True positive rate')

plt.title('ROC curve')

plt.legend(loc='best')

plt.savefig('ROC_PLOT.png', bbox_inches='tight')

plt.show()
testdf = pd.DataFrame({'path': glob(os.path.join(test_path, '*.tif'))})

testdf['id'] = testdf.path.map(lambda x: (x.split("/")[3].split('.')[0]))

testdf.head(3)
tta_datagen = ImageDataGenerator(rescale=1./255, #Normalise

                                 vertical_flip = True,

                                 horizontal_flip = True,

                                 rotation_range=90,

                                 zoom_range=0.2, 

                                 width_shift_range=0.1,

                                 height_shift_range=0.1,

                                 shear_range=0.05,

                                 channel_shift_range=0.1)
tta_steps = 5

submission = pd.DataFrame()

for index in range(0, len(testdf)):

    data_frame = pd.DataFrame({'path': testdf.iloc[index,0]}, index=[index])

    data_frame['id'] = data_frame.path.map(lambda x: x.split('/')[3].split('.')[0])

    img_path = data_frame.iloc[0,0]

    test_img = cv2.imread(img_path)

    test_img = cv2.resize(test_img,(IMG_SIZE,IMG_SIZE))

    test_img = np.expand_dims(test_img, axis = 0)  

    predictionsTTA = []

    for i in range(0, tta_steps):

        preds = my_model.predict_generator(tta_datagen.flow_from_dataframe(dataframe = data_frame,

                                                                           directory = None,

                                                                           x_col = 'path',

                                                                           target_size = (IMG_SIZE, IMG_SIZE),

                                                                           class_mode = None,

                                                                           batch_size = 1,

                                                                           shuffle = False), steps = 1)

        predictionsTTA.append(preds)

    clear_output()

    prediction_entry = np.array(np.mean(predictionsTTA, axis=0))

    data_frame['label'] = prediction_entry

    submission = pd.concat([submission, data_frame[['id', 'label']]])

    
submission.set_index('id')

submission.head(3)
submission.to_csv('submission.csv', index=False, header=True)