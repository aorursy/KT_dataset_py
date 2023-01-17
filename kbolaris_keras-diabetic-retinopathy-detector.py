# Helper libraries

import tensorflow

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import cv2

import os

%matplotlib inline

print(tensorflow.__version__)
train_df = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')

train_df['id_code'] = train_df['id_code'].apply(lambda x:x+'.png')

train_df['diagnosis'] = train_df['diagnosis'].astype(str)

test_df = pd.read_csv('../input/aptos2019-blindness-detection/test.csv')

test_df['id_code'] = test_df['id_code'].apply(lambda x:x+'.png')



num_classes = train_df['diagnosis'].nunique()

diag_text = ['Normal', 'Mild', 'Moderate', 'Severe', 'Proliferative']
def display_raw_images(df, columns = 4, rows = 3):

    fig=plt.figure(figsize = (5 * columns, 4 * rows))

    for i in range(columns * rows):

        image_name = df.loc[i,'id_code']

        image_id = df.loc[i,'diagnosis']

        img = cv2.imread(f'../input/aptos2019-blindness-detection/train_images/{image_name}')[...,[2, 1, 0]]

        fig.add_subplot(rows, columns, i + 1)

        plt.title(diag_text[int(image_id)])

        plt.imshow(img)

    plt.tight_layout()



display_raw_images(train_df)
unique, counts = np.unique(train_df['diagnosis'], return_counts=True)

plt.bar(unique, counts)

plt.title('Class Frequency')

plt.xlabel('Class')

plt.ylabel('Frequency')

plt.show()
from sklearn.utils import class_weight



sklearn_class_weights = class_weight.compute_class_weight(

               'balanced',

                np.unique(train_df['diagnosis']), 

                train_df['diagnosis'])



print(sklearn_class_weights)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Callback

from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D

from tensorflow.keras.applications import DenseNet121, ResNet50, InceptionV3, Xception

from tensorflow.keras.models import Model, Sequential

from tensorflow.keras.optimizers import Adam 



def create_resnet50_model(input_shape, n_out):

    base_model = ResNet50(weights = None,

                          include_top = False,

                          input_shape = input_shape)

    

    base_model.load_weights('../input/keras-pretrained-models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')

    model = Sequential()

    model.add(base_model)

    model.add(GlobalAveragePooling2D())

    model.add(Dropout(0.5))

    model.add(Dense(2048, activation = 'relu'))

    model.add(Dropout(0.5))    

    model.add(Dense(n_out, activation = 'sigmoid'))

    return model



def create_inception_v3_model(input_shape, n_out):

    base_model = InceptionV3(weights = None,

                             include_top = False,

                             input_shape = input_shape)

    base_model.load_weights('../input/keras-pretrained-models/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')

    model = Sequential()

    model.add(base_model)

    model.add(GlobalAveragePooling2D())

    model.add(Dropout(0.5))

    model.add(Dense(2048, activation = 'relu'))

    model.add(Dropout(0.5))    

    model.add(Dense(n_out, activation = 'sigmoid'))

    return model



def create_xception_model(input_shape, n_out):

    base_model = Xception(weights = None,

                             include_top = False,

                             input_shape = input_shape)

    base_model.load_weights('../input/keras-pretrained-models/xception_weights_tf_dim_ordering_tf_kernels_notop.h5')

    model = Sequential()

    model.add(base_model)

    model.add(GlobalAveragePooling2D())

    model.add(Dropout(0.5))

    model.add(Dense(2048, activation = 'relu'))

    model.add(Dropout(0.5))    

    model.add(Dense(n_out, activation = 'sigmoid'))

    return model



def create_densenet121_model(input_shape, n_out):

    base_model = DenseNet121(weights = None,

                             include_top = False,

                             input_shape = input_shape)

    base_model.load_weights('../input/densenet-keras/DenseNet-BC-121-32-no-top.h5')

    model = Sequential()

    model.add(base_model)

    model.add(GlobalAveragePooling2D())

    model.add(Dropout(0.5))

    model.add(Dense(2048, activation = 'relu'))

    model.add(Dropout(0.5))    

    model.add(Dense(n_out, activation = 'sigmoid'))

    return model
#IMAGE_HEIGHT = 224

#IMAGE_WIDTH = 224

#model = create_resnet50_model(input_shape = (IMAGE_HEIGHT, IMAGE_WIDTH, 3), n_out = num_classes)

#model = create_densenet121_model(input_shape = (IMAGE_HEIGHT, IMAGE_WIDTH, 3), n_out = num_classes)



IMAGE_HEIGHT = 299

IMAGE_WIDTH = 299

#model = create_inception_v3_model(input_shape = (IMAGE_HEIGHT, IMAGE_WIDTH, 3), n_out = num_classes)

model = create_xception_model(input_shape = (IMAGE_HEIGHT, IMAGE_WIDTH, 3), n_out = num_classes)



model.summary()
PRETRAINED_MODEL = '../input/pretrained_blindness_detector/blindness_detector.h5'



if (os.path.exists(PRETRAINED_MODEL)):

  print('Restoring model from ' + PRETRAINED_MODEL)

  model.load_weights(PRETRAINED_MODEL)

else:

  print('No pretrained model found. Using fresh model.')



current_epoch = 0
from tqdm import tqdm_notebook as tqdm



def crop_image_from_gray(img, tol = 7):

    if img.ndim == 2:

        mask = img > tol

        return img[np.ix_(mask.any(1),mask.any(0))]

    elif img.ndim == 3:

        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        mask = gray_img > tol        

        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]

        if (check_shape == 0): # image is too dark so that we crop out everything,

            return img # return original image

        else:

            img1 = img[:,:,0][np.ix_(mask.any(1),mask.any(0))]

            img2 = img[:,:,1][np.ix_(mask.any(1),mask.any(0))]

            img3 = img[:,:,2][np.ix_(mask.any(1),mask.any(0))]

            img = np.stack([img1,img2,img3],axis = -1)

        return img



def preprocess_image(image_path, sigmaX = 10):

    image = cv2.imread(image_path)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = crop_image_from_gray(image)

    image = cv2.resize(image, (IMAGE_HEIGHT, IMAGE_WIDTH))

    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0,0), sigmaX), -4, 128)        

    return image



print("Preprocessing training images...")

x_train = np.empty((train_df.shape[0], IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype = np.uint8)

for i, image_id in enumerate(tqdm(train_df['id_code'])):

    x_train[i, :, :, :] = preprocess_image(f'../input/aptos2019-blindness-detection/train_images/{image_id}')
def display_preprocessed_images(df, columns = 4, rows = 3):

    fig=plt.figure(figsize = (5 * columns, 4 * rows))

    for i in range(columns * rows):

        image_name = df.loc[i,'id_code']

        image_id = df.loc[i,'diagnosis']

        img = x_train[i]

        fig.add_subplot(rows, columns, i + 1)

        plt.title(diag_text[int(image_id)])

        plt.imshow(img)

    plt.tight_layout()



display_preprocessed_images(train_df)
y_train = pd.get_dummies(train_df['diagnosis']).values

y_train_multi = np.empty(y_train.shape, dtype = y_train.dtype)

y_train_multi[:, 4] = y_train[:, 4]



for i in range(3, -1, -1):

    y_train_multi[:, i] = np.logical_or(y_train[:, i], y_train_multi[:, i + 1])
from sklearn.model_selection import train_test_split



x_train, x_val, y_train, y_val = train_test_split(

    x_train, y_train_multi, 

    test_size = 0.20, 

    random_state = 2006

)
from tensorflow.keras.preprocessing.image import ImageDataGenerator



TRAIN_DATA_ROOT = '../input/aptos2019-blindness-detection/train_images'

TEST_DATA_ROOT  = '../input/aptos2019-blindness-detection/test_images'



BATCH_SIZE = 16



train_datagen = ImageDataGenerator(

    rotation_range = 360, 

    horizontal_flip = True, 

#    vertical_flip = True,

    zoom_range = [0.98, 1.02], 

    width_shift_range = 0.01,

    height_shift_range = 0.01)



train_generator = train_datagen.flow(

    x_train, 

    y_train,

    batch_size = BATCH_SIZE, 

    shuffle = True)
WARMUP_EPOCHS = 2

WARMUP_LEARNING_RATE = 1e-3



for layer in model.layers:

    layer.trainable = False



for i in range(-5, 0):

    model.layers[i].trainable = True



model.compile(optimizer = Adam(lr = WARMUP_LEARNING_RATE),

              loss = 'binary_crossentropy',  

              metrics = ['accuracy'])



warmup_history = model.fit_generator(generator = train_generator,

#                              class_weight = sklearn_class_weights,

                              steps_per_epoch = train_generator.n // train_generator.batch_size,

                              validation_data = (x_val, y_val),

                              epochs = WARMUP_EPOCHS,

                              use_multiprocessing = True,

                              workers = 4,                                     

                              verbose = 1).history
FINETUNING_EPOCHS = 20

FINETUNING_LEARNING_RATE = 1e-4



for layer in model.layers:

    layer.trainable = True



model.compile(optimizer = Adam(lr = FINETUNING_LEARNING_RATE), 

              loss = 'binary_crossentropy',

              metrics = ['accuracy'])



checkpoint = ModelCheckpoint(

    'blindness_detector_best.h5', 

    monitor = 'val_acc',  

    save_best_only = True, 

    save_weights_only = True,

    verbose = 1)



rlrop = ReduceLROnPlateau(

    monitor = 'val_loss', 

    patience = 3, 

    factor = 0.5, 

    min_lr = 1e-6, 

    verbose = 1)



stopping = EarlyStopping(

    monitor = 'val_acc', 

    patience = 8, 

    restore_best_weights = True, 

    verbose = 1)



finetune_history = model.fit_generator(generator = train_generator,

#                              class_weight = sklearn_class_weights,

                              steps_per_epoch = train_generator.n // train_generator.batch_size,

                              validation_data = (x_val, y_val),

                              epochs = FINETUNING_EPOCHS,

                              callbacks = [checkpoint, rlrop, stopping],         

                              use_multiprocessing = True,

                              workers = 4,

                              verbose = 1).history
training_accuracy = warmup_history['acc'] + finetune_history['acc']

validation_accuracy = warmup_history['val_acc'] + finetune_history['val_acc']

training_loss = warmup_history['loss'] + finetune_history['loss']

validation_loss = warmup_history['val_loss'] + finetune_history['val_loss']



plt.figure(figsize = (8, 8))

plt.subplot(2, 1, 1)

plt.plot(training_accuracy, label = 'Training Accuracy')

plt.plot(validation_accuracy, label = 'Validation Accuracy')

plt.legend(loc = 'lower right')

plt.ylabel('Accuracy')

plt.ylim([min(plt.ylim()), 1])

plt.title('Training and Validation Accuracy')



plt.subplot(2, 1, 2)

plt.plot(training_loss, label = 'Training Loss')

plt.plot(validation_loss, label = 'Validation Loss')

plt.legend(loc = 'upper right')

plt.ylabel('Cross Entropy')

plt.ylim([0, 1.0])

plt.title('Training and Validation Loss')

plt.xlabel('epoch')

plt.show()
validation_predictions_raw = model.predict(x_val)

validation_predictions = validation_predictions_raw > 0.5

validation_predictions = validation_predictions.astype(int).sum(axis=1) - 1

validation_truth = y_val.sum(axis=1) - 1
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score



def plot_confusion_matrix(cm, target_names, title = 'Confusion matrix', cmap = plt.cm.Blues):

    plt.grid(False)

    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(target_names))

    plt.xticks(tick_marks, target_names, rotation = 90)

    plt.yticks(tick_marks, target_names)

    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')



np.set_printoptions(precision = 2)

cm = confusion_matrix(validation_truth, validation_predictions)

cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]



plot_confusion_matrix(cm = cm, target_names = diag_text)

plt.show()



print('Confusion Matrix')

print(cm)



print('Classification Report')

print(classification_report(validation_truth, validation_predictions, target_names = diag_text))



print("Validation Cohen Kappa score: %.3f" % cohen_kappa_score(validation_predictions, validation_truth, weights = 'quadratic'))
def plot_image(prediction_array, predicted_label, true_label, img):

    plt.grid(False)

    plt.xticks([])

    plt.yticks([])

    plt.imshow(img, cmap = plt.cm.binary)

    if predicted_label == true_label:

        color = 'blue'

    else:

        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(diag_text[predicted_label], 100 * np.max(prediction_array), diag_text[true_label]), color = color)



def plot_prediction(prediction_array, predicted_label, true_label):

    plt.grid(False)

    plt.xticks([])

    plt.yticks([])

    thisplot = plt.bar(range(5), prediction_array, color = "#777777")

    plt.ylim([0, 1]) 

    thisplot[predicted_label].set_color('red')

    thisplot[true_label].set_color('blue')

  

# Plot some validation images, their predicted label, and the true label

# Color correct predictions in blue, incorrect predictions in red

plt.figure(figsize=(24, 6))

num_cols = 4

num_rows = 4

for i in range(num_rows * num_cols):

    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)

    plot_image(validation_predictions_raw[i], validation_predictions[i], validation_truth[i], x_val[i])

    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)

    plot_prediction(validation_predictions_raw[i], validation_predictions[i], validation_truth[i])

plt.show() 
x_train = None

x_val = None

print("Preprocessing test images...")

!mkdir 'test_images_preprocessed/'

for i, image_id in enumerate(tqdm(test_df['id_code'])):

    image = preprocess_image(f'../input/aptos2019-blindness-detection/test_images/{image_id}')    

    cv2.imwrite(f'./test_images_preprocessed/{image_id}', image)

    
test_datagen = ImageDataGenerator()

test_generator = test_datagen.flow_from_dataframe(

    dataframe = test_df,

    directory = "./test_images_preprocessed/",

    x_col = "id_code",

    target_size = (IMAGE_HEIGHT, IMAGE_WIDTH),

    batch_size = 1,

    shuffle = False,

    class_mode = None)



y_test = model.predict_generator(test_generator) > 0.5

y_test = y_test.astype(int).sum(axis = 1) - 1
unique, counts = np.unique(y_test, return_counts = True)

plt.bar(unique, counts)



unique, counts = np.unique(validation_truth, return_counts = True)

plt.bar(unique, counts)



plt.title('Class Frequency Training and Predictions')

plt.xlabel('Class')

plt.ylabel('Frequency')

plt.show()
submission_df = pd.read_csv('../input/aptos2019-blindness-detection/test.csv')

submission_df['diagnosis'] = y_test

submission_df.to_csv('submission.csv', index = False)
!rm -rf /kaggle/working/test_images_preprocessed/