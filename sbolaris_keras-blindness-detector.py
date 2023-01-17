# Helper libraries

import tensorflow

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import cv2

import os
train_df = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')

train_df['id_code'] = train_df['id_code'].apply(lambda x:x+'.png')

train_df['diagnosis'] = train_df['diagnosis'].astype(str)

test_df = pd.read_csv('../input/aptos2019-blindness-detection/test.csv')

test_df['id_code'] = test_df['id_code'].apply(lambda x:x+'.png')
diag_text = ['Normal', 'Mild', 'Moderate', 'Severe', 'Proliferative']



def display_samples(df, columns = 4, rows = 3):

    fig=plt.figure(figsize = (5 * columns, 4 * rows))

    for i in range(columns * rows):

        image_name = df.loc[i,'id_code']

        image_id = df.loc[i,'diagnosis']

        img = cv2.imread(f'../input/aptos2019-blindness-detection/train_images/{image_name}')[...,[2, 1, 0]]

        fig.add_subplot(rows, columns, i + 1)

        plt.title(diag_text[int(image_id)])

        plt.imshow(img)

    plt.tight_layout()



display_samples(train_df)
from tqdm import tqdm_notebook as tqdm



IMAGE_HEIGHT = 224

IMAGE_WIDTH = 224



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

!mkdir -p 'train_images_preprocessed'

for i, image_id in enumerate(tqdm(train_df['id_code'])):

    image = preprocess_image(f'../input/aptos2019-blindness-detection/train_images/{image_id}')

    cv2.imwrite(f'./train_images_preprocessed/{image_id}', image)    

    

   
from tensorflow.keras.preprocessing.image import ImageDataGenerator



num_classes = train_df['diagnosis'].nunique()



TRAIN_DATA_ROOT = './train_images_preprocessed/'

TEST_DATA_ROOT  = './test_images_preprocessed/'



BATCH_SIZE = 16



train_datagen = ImageDataGenerator(

    rescale = 1/255, 

    rotation_range = 360, 

    horizontal_flip = True, 

    vertical_flip = True,

    zoom_range = [0.98, 1.02], 

    width_shift_range = 0.01,

    height_shift_range = 0.01,

    validation_split = 0.20)



validation_datagen = ImageDataGenerator(

    rescale = 1/255, 

    validation_split = 0.20)



train_generator = train_datagen.flow_from_dataframe(

    dataframe = train_df,

    directory = TRAIN_DATA_ROOT,

    x_col = 'id_code',

    y_col = 'diagnosis',

    batch_size = BATCH_SIZE, 

    shuffle = True,

    class_mode = 'categorical',

    target_size = (IMAGE_HEIGHT, IMAGE_WIDTH),

    subset = 'training')



validation_generator = validation_datagen.flow_from_dataframe(

    dataframe = train_df,

    directory = TRAIN_DATA_ROOT,

    x_col = 'id_code',

    y_col = 'diagnosis',

    batch_size = BATCH_SIZE, 

    shuffle = True,

    class_mode = 'categorical', 

    target_size = (IMAGE_HEIGHT, IMAGE_WIDTH),

    subset = 'validation')
# Use the line below to show inline in a notebook

%matplotlib inline



unique, counts = np.unique(train_generator.classes, return_counts=True)

plt.bar(unique, counts)



unique, counts = np.unique(validation_generator.classes, return_counts=True)

plt.bar(unique, counts)



plt.title('Class Frequency')

plt.xlabel('Class')

plt.ylabel('Frequency')

plt.show()
from sklearn.utils import class_weight



sklearn_class_weights = class_weight.compute_class_weight(

               'balanced',

                np.unique(train_generator.classes), 

                train_generator.classes)

print(sklearn_class_weights)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input

from tensorflow.keras.applications import DenseNet121, ResNet50

from tensorflow.keras.models import Model, Sequential

from tensorflow.keras.optimizers import Adam 



def create_resnet50_model(input_shape, n_out):

    base_model = ResNet50(weights = None,

                          include_top = False,

                          input_shape = input_shape)

    base_model.load_weights('../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')

    model = Sequential()

    model.add(base_model)

    model.add(GlobalAveragePooling2D())

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

    model.add(Dense(n_out, activation = 'sigmoid'))

    return model
#model = create_resnet50_model(input_shape = (IMAGE_HEIGHT, IMAGE_WIDTH, 3), n_out = num_classes)

model = create_densenet121_model(input_shape = (IMAGE_HEIGHT, IMAGE_WIDTH, 3), n_out = num_classes)

model.summary()
PRETRAINED_MODEL = '../input/pretrained_blindness_detector/blindness_detector.h5'

if (os.path.exists(PRETRAINED_MODEL)):

  print('Restoring model from ' + PRETRAINED_MODEL)

  model.load_weights(PRETRAINED_MODEL)

else:

  print('No pretrained model found. Using fresh model.')



current_epoch = 0
WARMUP_EPOCHS = 2

WARMUP_LEARNING_RATE = 1e-3



for layer in model.layers:

    layer.trainable = False



for i in range(-3, 0):

    model.layers[i].trainable = True



model.compile(optimizer = Adam(lr = WARMUP_LEARNING_RATE),

              loss = 'binary_crossentropy',  

              metrics = ['accuracy'])



warmup_history = model.fit_generator(generator = train_generator,

                              class_weight = sklearn_class_weights,

                              steps_per_epoch = train_generator.n // train_generator.batch_size,

                              validation_data = validation_generator,

                              validation_steps = validation_generator.n // validation_generator.batch_size,

                              epochs = WARMUP_EPOCHS,

                              use_multiprocessing = True,

                              workers = 4,                                     

                              verbose = 1).history
FINETUNING_EPOCHS = 20

FINETUNING_LEARNING_RATE = 1e-4



for layer in model.layers:

    layer.trainable = True



model.compile(optimizer = Adam(lr = FINETUNING_LEARNING_RATE), 

              loss = 'categorical_crossentropy',

              metrics = ['accuracy'])



stopping = EarlyStopping(

    monitor = 'val_loss', 

    mode = 'min', 

    patience = 5, 

    restore_best_weights = True, 

    verbose = 1)



rlrop = ReduceLROnPlateau(

    monitor = 'val_loss', 

    mode = 'min', 

    patience =  8, 

    factor = 0.5, 

    min_lr = 1e-6, 

    verbose = 1)



checkpoint = ModelCheckpoint(

    'blindness_detector_best.h5', 

    monitor = 'val_acc',  

    save_best_only = True, 

    save_weights_only = False,

    mode = 'max',

    verbose = 1)



finetune_history = model.fit_generator(generator = train_generator,

                              class_weight = sklearn_class_weights,

                              steps_per_epoch = train_generator.n // train_generator.batch_size,

                              validation_data = validation_generator,

                              validation_steps = validation_generator.n // validation_generator.batch_size,

                              epochs = FINETUNING_EPOCHS,

                              callbacks = [stopping, rlrop, checkpoint],

                              use_multiprocessing = True,

                              workers = 4,

                              verbose = 1).history
training_accuracy = warmup_history['acc'] + finetune_history['acc']

validation_accuracy = warmup_history['val_acc'] + finetune_history['val_acc']

training_loss = warmup_history['loss'] + finetune_history['loss']

validation_loss = warmup_history['val_loss'] + finetune_history['val_loss']



plt.figure(figsize=(8, 8))

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



validation_generator.reset()

    

Y_pred = model.predict_generator(validation_generator, 

                                 steps = validation_generator.n // validation_generator.batch_size + 1)

y_pred = np.argmax(Y_pred, axis = 1)



np.set_printoptions(precision = 2)

cm = confusion_matrix(validation_generator.classes, y_pred)

cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]



plot_confusion_matrix(cm = cm, target_names = diag_text)

plt.show()



print('Confusion Matrix')

print(cm)



print('Classification Report')

print(classification_report(validation_generator.classes, y_pred, target_names = diag_text))



print("Train Cohen Kappa score: %.3f" % cohen_kappa_score(y_pred, validation_generator.classes, weights = 'quadratic'))
def plot_image(prediction_array, true_label, img):

    predicted_label = np.argmax(prediction_array) 

    plt.grid(False)

    plt.xticks([])

    plt.yticks([])

    plt.imshow(img, cmap = plt.cm.binary)

    if predicted_label == true_label:

        color = 'blue'

    else:

        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(diag_text[predicted_label], 100 * np.max(prediction_array), diag_text[true_label]), color = color)



def plot_prediction(prediction_array, true_label):

    plt.grid(False)

    plt.xticks([])

    plt.yticks([])

    thisplot = plt.bar(range(5), prediction_array, color = "#777777")

    plt.ylim([0, 1]) 

    predicted_label = np.argmax(prediction_array)

    thisplot[predicted_label].set_color('red')

    thisplot[true_label].set_color('blue')

  

label_names = sorted(validation_generator.class_indices.items(), key = lambda pair:pair[1])

label_names = np.array([key.title() for key, value in label_names])



for image_batch, label_batch in validation_generator:

  break



predictions = model.predict(image_batch)

prediction_labels = label_names[np.argmax(predictions, axis = -1)]



# Plot the first X test images, their predicted label, and the true label

# Color correct predictions in blue, incorrect predictions in red

plt.figure(figsize=(24, 6))

num_cols = 4

num_rows = validation_generator.batch_size // num_cols

for i in range(num_rows * num_cols):

    true_label = np.argmax(label_batch[i]);

    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)

    plot_image(predictions[i], true_label, image_batch[i])

    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)

    plot_prediction(predictions[i], true_label)

plt.show() 
unique, counts = np.unique(train_generator.classes, return_counts=True)

plt.bar(unique, counts)



unique, counts = np.unique(y_pred, return_counts=True)

plt.bar(unique, counts)



plt.title('Class Frequency Training and Predictions')

plt.xlabel('Class')

plt.ylabel('Frequency')

plt.show()
!rm -rf /kaggle/working/train_images_preprocessed/



print("Preprocessing test images...")

!mkdir -p 'test_images_preprocessed'

for i, image_id in enumerate(tqdm(test_df['id_code'])):

    image = preprocess_image(f'../input/aptos2019-blindness-detection/test_images/{image_id}')    

    cv2.imwrite(f'./test_images_preprocessed/{image_id}', image) 

    

test_datagen = ImageDataGenerator(rescale = 1./255)



test_generator = test_datagen.flow_from_dataframe(  

        dataframe = test_df,

        directory = TEST_DATA_ROOT,

        x_col = 'id_code',

        target_size = (IMAGE_HEIGHT, IMAGE_WIDTH),

        batch_size = 1,

        shuffle = False,

        class_mode = None)    

    

predict = model.predict_generator(test_generator, steps = len(test_generator.filenames))

results=pd.DataFrame({'id_code':test_generator.filenames,

                      'diagnosis':np.argmax(predict, axis = 1)})

results['id_code'] = results['id_code'].map(lambda x: str(x)[:-4])

results.to_csv('submission.csv', index = False)
!rm -rf /kaggle/working/test_images_preprocessed/