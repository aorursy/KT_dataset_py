!cd ..
!mkdir /kaggle/tmp/
%cd /kaggle/tmp/
!pwd
# to deal with file system
import os
# for reading images
import cv2
# to read and process xml files
from bs4 import BeautifulSoup


# for various operations (you know why these 3 are being used)
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
%matplotlib inline



# preprocessing images
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

# for train and validation split
from sklearn.model_selection import train_test_split



# modelling with VGG19
from tensorflow.keras.applications import  VGG19
model_name = 'vgg_19'
from tensorflow.keras.applications.vgg19  import preprocess_input as vgg_preprocess_input
from tensorflow.keras import Model
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D, Dropout, BatchNormalization, Flatten, Dense, Input
from tensorflow.keras.optimizers import Adam


# for callbacks
import time
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
from datetime import datetime


# model evaluation
from tensorflow.keras.metrics import Recall, Precision
from tensorflow_addons.metrics import F1Score
from sklearn.metrics import classification_report
# setting some parameters
image_size = 224 # this is the size that gave me better results than default size of 224
validation_split_size = 0.20 # 20% will be used for validation

# define the hyperparamets for training the neural network
batch_size = 32
init_lr = 0.0_001
num_epochs = 100

# directories
labels_path = '../input/face-mask-detection/annotations/'
images_path = '../input/face-mask-detection/images/'

!mkdir './VGG19'
save_vgg19 = './VGG19/'
images = sorted(os.listdir("../input/face-mask-detection/images/"))
labels = sorted(os.listdir("../input/face-mask-detection/annotations/"))



len(images) == len(labels), len(images), len(labels)
def generate_label_dictionary(xml_loc): 
    """
    takes location to image and xml files on file system and return the image as numpy array and extracted bounding boxes
    """
    with open(xml_loc) as xml_file:
        # read the input file
        soup = BeautifulSoup(xml_file.read(), 'xml')
        objects = soup.find_all('object')

        # extract the number of persons in an image
        num_persons = len(objects)

        # to store all the points for boundary boxes and target labels
        boxes = []
        labels = []
        # doing it now
        for obj in objects:
            # extract output class and append it to 'boxes' list
            if obj.find('name').text == "without_mask":
                labels.append(0)
            elif obj.find('name').text == "mask_weared_incorrect":
                labels.append(1)
            elif obj.find('name').text == "with_mask":
                labels.append(2)
            else:
                break
            
            # extract bounding box and append it to 'labels' list
            xmin = int(obj.find('xmin').text)
            ymin = int(obj.find('ymin').text)
            xmax = int(obj.find('xmax').text)
            ymax = int(obj.find('ymax').text)
            boxes.append([xmin, ymin, xmax, ymax])
        

        # converting them to numpy arrays
        boxes = np.array(boxes)
        labels = np.array(labels)

        # save them to dictionary
        target = {}
        target["labels"] = labels
        target["boxes"] = boxes

        return target, num_persons
targets=[] # store coordinates of bounding boxes
num_persons=[] # stores number of faces in each image

#run the loop for number of images we have
for label_path in labels:
    # generate label
    target_image, num_persons_image = generate_label_dictionary(labels_path+label_path)
    targets.append(target_image)
    num_persons.append(num_persons_image)
print(len(targets))
print()
print(targets[0: 100: 10])
print()
print(num_persons[0: 100: 10])
face_images = []
face_labels = []

# read each image from the file system and extract only the faces using the boundaries extracted in previous step
for i, image_path in enumerate(images):
    image_read = cv2.imread(images_path+image_path, cv2.IMREAD_COLOR)
    # get co-ordinates of the image
    for j in range(0, num_persons[i]):
        # get the locations of boundary box now
        face_locs = targets[i]['boxes'][j]
        # extract the face now using those co-ordinates
        temp_face = image_read[face_locs[1]:face_locs[3], face_locs[0]:face_locs[2]]
        temp_face = cv2.resize(temp_face, (image_size, image_size))
        temp_face = vgg_preprocess_input(temp_face)
        
        # store this processed image to list now
        face_images.append(temp_face)
        # store it's respective label too
        face_labels.append(targets[i]['labels'][j])

# convert them to numpy arrays
face_images = np.array(face_images, dtype=np.float32)
face_labels = np.array(face_labels)
print(face_images.shape, face_labels.shape)
np.unique(face_labels, return_counts=True)
def show_face_and_label(index):
    plt.imshow(face_images[index])
    plt.show()

    face_label_num = face_labels[index]

    if face_label_num == 0:
        face_label_text = "doesn't have a mask on."
    elif face_label_num == 1:
        face_label_text = "wore mask improperly."
    elif face_label_num == 2:
        face_label_text = "has a mask on."
    else:
        face_label_text = "error"
    return 'person {}'.format(face_label_text)
show_face_and_label(2)
show_face_and_label(46)
show_face_and_label(47)
# since one-hot encoding need to be done for 
face_labels_enc = to_categorical(face_labels)
face_labels_enc
pd.DataFrame(face_labels_enc).apply(pd.Series.value_counts, normalize=False).to_dict()
pd.DataFrame(face_labels_enc).apply(pd.Series.value_counts, normalize=True).to_dict()
train_imgs, val_imgs, train_targets, val_targets = train_test_split(face_images, face_labels_enc,
                                                                    stratify=face_labels_enc,
                                                                    test_size=validation_split_size, random_state=100, shuffle=True)

train_imgs.shape, val_imgs.shape, train_targets.shape, val_targets.shape
# ensuring that the samples are stratified between train and test splits to validate the model right way
print(pd.DataFrame(train_targets).apply(pd.Series.value_counts, normalize=True))
print()
print(pd.DataFrame(val_targets).apply(pd.Series.value_counts, normalize=True))
face_images, face_labels, face_labels_enc, face_locs, num_persons, targets, images, labels = None, None, None, None, None, None, None, None
del face_images, face_labels, face_labels_enc, face_locs, num_persons, targets, images, labels
# RAM usage after this = ~3.7GB (reduction of more than 3.5 GB)
train_image_generator = ImageDataGenerator(zoom_range=0.1, width_shift_range=0.1, height_shift_range=0.1,
                                           shear_range=0.15,fill_mode="nearest")
vgg19_base = VGG19(include_top=False, pooling=None,
                   input_shape=(image_size, image_size, 3)) # with max pooling (None, 2048)


inner = vgg19_base.output


## only the followeing layers will be trained or weights updated will only be of below layers
inner = AveragePooling2D(pool_size=(7, 7))(inner)
inner = Flatten()(inner)
inner = Dense(units=256, activation='relu')(inner)
inner = Dropout(rate=0.25)(inner)
inner = Dense(units=3, activation='softmax')(inner)


model_1 = Model(inputs=vgg19_base.input, outputs=inner)


model_1.summary()
model_1.compile(loss = 'categorical_crossentropy',                             # "multi log-loss"  as loss
                optimizer = Adam(lr=init_lr, decay=init_lr / num_epochs),      # "adam"            as optimiser
                metrics = [Recall(name='recall'), 'accuracy',
                           F1Score(average='macro', name='macro_f1', num_classes=3), # weighted_f1,
                           F1Score(average='weighted', name='weighted_f1', num_classes=3),
                           Precision(name='precision')])
model_save_cb = ModelCheckpoint(filepath= save_vgg19+model_name+'-epoch{epoch:03d}-recall-{val_recall:.5f}-acc-{val_accuracy:.5f}.h5',
                                monitor='val_recall', mode='max', 
                                verbose=1, save_best_only=False, save_weights_only=True)
# I will be storing the complete model to be able to resume training should something happens and also to load the model with best fbeta score on validation set for evaluation


# since recall is my primary metric of choice, i want the training to be stopped, when recall doesn't increase even after 15 epochs.
early_stop_cb = EarlyStopping(monitor='val_recall', min_delta=0, patience=15, verbose=1, mode='max')
history_vgg19 = model_1.fit(train_image_generator.flow(x=train_imgs, y=train_targets, batch_size=batch_size, seed=100),
                            steps_per_epoch=len(train_imgs) // batch_size,
                            
                            validation_data = (val_imgs, val_targets),
                            validation_steps=len(val_imgs) // batch_size,
                            
                            epochs=num_epochs,
                            
                            class_weight={0:5, 1:13, 2:1}, # got these weights after a lot of eperimenting
                            
                            callbacks=[model_save_cb, early_stop_cb],
                            
                            verbose=2
                            )
# printing all the maximum scores
max(history_vgg19.history['val_recall']), max(history_vgg19.history['val_macro_f1']), max(history_vgg19.history['val_weighted_f1']), max(history_vgg19.history['val_accuracy'])
train_stats = pd.DataFrame(history_vgg19.history)

# looking at the epochs that had best recall and macro-f1 scores for validaiton set
train_stats.sort_values(by=['val_recall'], inplace=False, ascending=False).head()
train_stats.plot(y=['val_recall', 'recall'], kind="line")
train_stats.plot(y=['val_macro_f1', 'macro_f1'], kind="line")
train_stats.plot(y=['val_weighted_f1', 'weighted_f1'], kind="line")
train_stats.plot(y=['val_accuracy', 'accuracy'], kind="line")
train_stats.plot(y=['val_loss', 'loss'], kind="line")
train_stats.plot(y=['val_precision', 'precision'], kind="line")
very_good_epochs = []
for col in ['val_recall', 'val_accuracy','val_macro_f1', 'val_precision', 'val_weighted_f1']:
    epoch = train_stats.loc[:,col].argmax()
    very_good_epochs.append(epoch)
    print(train_stats.loc[epoch, ['val_recall', 'val_accuracy','val_macro_f1', 'val_weighted_f1']])
    print()
# looking at all the rows with highest results for respective metric
good_results = train_stats.loc[set(very_good_epochs),
                               ['val_recall', 'val_accuracy','val_macro_f1', 'val_weighted_f1', 'val_precision']]

# since recall is my primary metric
good_results.sort_values(by=['val_recall', 'val_accuracy', 'val_macro_f1'], ascending=False, inplace=True)
good_results
models_not_to_delete = []
for epoch in list(np.array(good_results.index)):
    good_vals = good_results.loc[epoch, ['val_recall', 'val_accuracy']].values
    best_model_loc = f'{save_vgg19}vgg_19-epoch{epoch+1:03d}-recall-{good_vals[0]:.5f}-acc-{good_vals[1]:.5f}.h5'
    print(best_model_loc)
    models_not_to_delete.append(best_model_loc)
    model_2 = None
    del model_2
    model_2 = None
    try:
        model_2 = Model(inputs=vgg19_base.input, outputs=inner)
        model_2.load_weights(filepath=best_model_loc)
        val_preds = model_2.predict(val_imgs, batch_size=32)
        val_preds = np.argmax(val_preds, axis=1)
        print(classification_report(y_true=val_targets.argmax(axis=1), y_pred=val_preds, target_names=['without mask', 'incorrectly worn', 'with mask']))
    except OSError:
        print('file not found')
models_not_to_delete
!cp ./VGG19/vgg_19-epoch039-recall-0.95706-acc-0.95828.h5 ../working/
!cp ./VGG19/vgg_19-epoch046-recall-0.95337-acc-0.95460.h5 ../working/