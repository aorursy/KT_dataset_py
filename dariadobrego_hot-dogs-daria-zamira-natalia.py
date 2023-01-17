import os

train = os.listdir("../input/hotdogs-spbu/train/train")

test = os.listdir("../input/hotdogs-spbu/test/test")
from tensorflow.keras.datasets import mnist

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization

from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D,GlobalAveragePooling2D

from tensorflow.keras.utils import to_categorical

from tensorflow.keras import backend as K

from keras.utils import np_utils

from keras.utils.np_utils import to_categorical

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras import optimizers

from tensorflow.keras.preprocessing import image

from tensorflow.keras.applications import ResNet50



import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import os

from matplotlib.image import imread



from sklearn.metrics import roc_auc_score

from sklearn.model_selection import train_test_split
SEED = 257



TRAIN_DIR = ("../input/hotdogs-spbu/train/train")

TEST_DIR = ("../input/hotdogs-spbu/test/test")
categories = ['hot dog', 'not hot dog']
X, y = [], []



for category in categories:

    category_dir = os.path.join(TRAIN_DIR, category)

    for image_path in os.listdir(category_dir):

        X.append((os.path.join(category_dir, image_path)))

        y.append(category)



df = pd.DataFrame({

    'filename': X,

    'category': y

})
len(X), len(y)
#training dataframe



df
#we actually have class imbalance



df['category'].value_counts().plot.bar()
#I will use VGG16 application as a pre-trained model



from tensorflow.keras.applications import VGG16

#Load the VGG model

image_size = 100

vgg_conv = VGG16(weights='../input/vgg-weights/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top=False, input_shape=(image_size, image_size, 3))
# Freeze the layers except the last 4 layers

for layers in (vgg_conv.layers)[:-4]:

    print(layers)

    layers.trainable = False
from tensorflow.keras import optimizers

from tensorflow.keras import models

from tensorflow.keras import layers

# Create the model

model = models.Sequential()



# Add the vgg convolutional base model



model.add(vgg_conv)

# Add new layers

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(512, activation='relu'))

model.add(Dense(512, activation='relu'))

model.add(Dense(2, activation='softmax'))





# Show a summary of the model. Check the number of trainable parameters

model.summary()
#compile model



model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#reduce learning rate when a metric has stopped improving



from tensorflow.keras.callbacks import ReduceLROnPlateau



learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 

                                            patience=2, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)



callbacks = [learning_rate_reduction]
#create training and testing dataset



train_df, validate_df = train_test_split(df, test_size=0.20, random_state=42)

train_df = train_df.reset_index(drop=True)

validate_df = validate_df.reset_index(drop=True)
train_df['category'].value_counts().plot.bar()
validate_df['category'].value_counts().plot.bar()
total_train = train_df.shape[0]

total_validate = validate_df.shape[0]

batch_size=15

total_validate
train_datagen = ImageDataGenerator(

    rotation_range=15,

    rescale=1./255,

    shear_range=0.1,

    zoom_range=0.2,

    horizontal_flip=True,

    width_shift_range=0.1,

    height_shift_range=0.1

)



train_generator = train_datagen.flow_from_dataframe(

    train_df, 

    x_col='filename',

    y_col='category',

    target_size=(100,100),

    class_mode='categorical',

    batch_size=15

)



validation_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = validation_datagen.flow_from_dataframe(

    validate_df, 

    x_col='filename',

    y_col='category',

    target_size=(100,100),

    class_mode='categorical',

    batch_size=batch_size

)
epochs=3 if False else 30

history = model.fit_generator(

    train_generator, 

    epochs=epochs,

    validation_data=validation_generator,

    validation_steps=total_validate//batch_size,

    steps_per_epoch=total_train//batch_size,

    callbacks=callbacks

)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

ax1.plot(history.history['loss'], color='b', label="Training loss")

ax1.plot(history.history['val_loss'], color='r', label="validation loss")

ax1.set_xticks(np.arange(1, epochs, 1))

ax1.set_yticks(np.arange(0, 1, 0.1))



ax2.plot(history.history['accuracy'], color='b', label="Training accuracy")

ax2.plot(history.history['val_accuracy'], color='r',label="Validation accuracy")

ax2.set_xticks(np.arange(1, epochs, 1))



legend = plt.legend(loc='best', shadow=True)

plt.tight_layout()

plt.show()

from tensorflow.keras.utils import plot_model

plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
#predict validation data



predict_for_validation = model.predict(validation_generator,steps=np.ceil(total_validate/batch_size))

predict_for_validation
#create copy of validate_df



validate_df2 = pd.DataFrame({

    'filename': validate_df['filename'],

    'category': validate_df['category']

})

validate_df2[:5]
#create array of validation images



leaderboard_X2 = []

validate_df22 = np.asarray(validate_df2['filename'])

for x in range(0,len(validate_df22)):

  i = imread(validate_df22[x])

  leaderboard_X2.append(i)

len(leaderboard_X2)
# predict validation probabilities



leaderboard_X2 = np.asarray(leaderboard_X2)



validation_probabilities = model.predict_proba(leaderboard_X2)

validation_probabilities[:10]
# choose probabilities of class 1



not_hot_dog_validation_probabilities = []

for i in validation_probabilities:

  class1 = i[1]

  not_hot_dog_validation_probabilities.append(class1)
not_hot_dog_validation_probabilities[:10]
validate_df2['category'] = validate_df2['category'].replace({ 'hot dog': 0, 'not hot dog': 1 })

np_validate_df2 = validate_df2['category'].to_numpy() #create array of classes of the validation set
len(np_validate_df2)
# ROC-AUC score



score = roc_auc_score(np_validate_df2,not_hot_dog_validation_probabilities)

score
from sklearn.metrics import roc_curve, accuracy_score

fpr , tpr , thresholds = roc_curve(np_validate_df2,not_hot_dog_validation_probabilities)
import matplotlib.pyplot as plt

plt.title('Receiver Operating Characteristic')

plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % score)

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0, 1])

plt.ylim([0, 1])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()
#create test data



test_filenames = os.listdir("../input/hotdogs-spbu/test/test/")

test_filenames = [f for f in test_filenames if str(f).strip().endswith('.png')]

test_df = pd.DataFrame({

    'filename': test_filenames

})

nb_samples = test_df.shape[0]
test_df['filename']
test_gen = ImageDataGenerator(rescale=1./255)

test_generator = test_gen.flow_from_dataframe(

    test_df, 

    "../input/hotdogs-spbu/test/test/",

    x_col='filename',

    y_col=None,

    class_mode=None,

    target_size=(100,100),

    batch_size=batch_size,

    shuffle=False

)
# predict test data

predict = model.predict(test_generator, steps=np.ceil(nb_samples/batch_size))

predict
test_df['category'] = ""
test_df['category'] = np.argmax(predict, axis=-1)

test_df
label_map = dict((v,k) for k,v in train_generator.class_indices.items())

test_df['category'] = test_df['category'].replace(label_map)



label_map
test_df['category'].value_counts().plot.bar()
test_df['category'][:15]
from keras.preprocessing.image import load_img

sample_test = test_df.head(18)

sample_test.head()

plt.figure(figsize=(12, 24))

for index, row in sample_test.iterrows():

    filename = row['filename']

    category = row['category']

    img = load_img("../input/hotdogs-spbu/test/test/"+filename, target_size=(100,100))

    plt.subplot(6, 3, index+1)

    plt.imshow(img)

    plt.xlabel('(' + "{}".format(category) + ')' )

plt.tight_layout()

plt.show()
model.save_weights("modelvgg.h5")
#create test data filenames and probabilities dataframe



leaderboard_X = []

leaderboard_filenames = []
#create first column of filenames



for image_path in os.listdir("../input/hotdogs-spbu/test/test/"):

  if image_path=='.DS_Store':

    continue

  else:

    x = imread(os.path.join("../input/hotdogs-spbu/test/test/", image_path))

    leaderboard_X.append(x)

leaderboard_filenames = test_df['filename'].tolist()
# read images and predict probabilities



leaderboard_X = np.asarray(leaderboard_X)



probabilities = model.predict_proba(leaderboard_X)

probabilities[20],leaderboard_filenames[24]
# create array of probabilities for class 0



hot_dog_probability = []

for i in probabilities:

  class0 = i[0]

  hot_dog_probability.append(class0)

hot_dog_probability[:10]
submission = pd.DataFrame(

    {

        'image_id': leaderboard_filenames, 

        'image_hot_dog_probability': hot_dog_probability

    }

)
submission.to_csv('submitfinal.csv', index=False)
plt.axis("off");

plt.imshow(leaderboard_X[0]);