import os

import numpy as np

# set the seed for to make results reproducable

np.random.seed(2)

import pandas as pd





import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set_style('darkgrid')





from tensorflow.random import set_seed

# set the seed for to make results reproducable

set_seed(2)

from tensorflow.keras.preprocessing.image import ImageDataGenerator
!find '../input/fruit-recognition/' -type d -print
def load_images_from_folder(folder, only_path = False, label = ""):

# Load the paths to the images in a directory

# or load the images

    if only_path == False:

        images = []

        for filename in os.listdir(folder):

            img = plt.imread(os.path.join(folder,filename))

            if img is not None:

                images.append(img)

        return images

    else:

        path = []

        for filename in os.listdir(folder):

            img_path = os.path.join(folder,filename)

            if img_path is not None:

                path.append([label,img_path])

        return path
# Load the paths on the images

images = []



dirp = "../input/fruit-recognition/"

# Load the paths on the images

images = []



for f in os.listdir(dirp):

    if "png" in os.listdir(dirp+f)[0]:

        images += load_images_from_folder(dirp+f,True,label = f)

    else: 

        for d in os.listdir(dirp+f):

            images += load_images_from_folder(dirp+f+"/"+d,True,label = f)

            

# Create a dataframe with the paths and the label for each fruit

df = pd.DataFrame(images, columns = ["fruit", "path"])

df.head()
df.loc[:,'fruit'].nunique()
# Assign to each fruit a specific number

fruit_names = sorted(df.fruit.unique())

mapped_fruit_names = dict(zip([t for t in range(len(fruit_names))], fruit_names))

df["label"] = df["fruit"].map(mapped_fruit_names)

print(mapped_fruit_names)
fruit_values = df.loc[:,'fruit'].value_counts
sns.set(rc={'figure.figsize':(10, 5)})

plt.xticks(rotation=45)

sns.barplot(data=pd.DataFrame(fruit_values(normalize=False)).reset_index(),

            x='index', y='fruit', dodge=False)
sns.set(rc={'figure.figsize':(10, 5)})

plt.xticks(rotation=45)

sns.barplot(data=pd.DataFrame(fruit_values(normalize=True)).reset_index(), x='index', y='fruit')
fruit_values(normalize=True)
df.shape
0.05*70_549
from sklearn.model_selection import StratifiedShuffleSplit
strat_split = StratifiedShuffleSplit(n_splits=5, test_size=0.05, random_state=2)
for tr_index, val_index in strat_split.split(df, df.loc[:,'fruit']):

    strat_train_df = df.loc[tr_index]

    strat_valid_df = df.loc[val_index]
strat_train_df.shape,strat_valid_df.shape
strat_train_df.loc[:,'fruit'].value_counts()
strat_valid_df.loc[:,'fruit'].value_counts()
img_width=300

img_height=300





train_data_IDG = ImageDataGenerator(rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2,

                                    zoom_range=0.2, fill_mode='nearest', horizontal_flip=True, rescale=1.0/255)



valid_data_IDG = ImageDataGenerator(rescale=1.0/255, validation_split=0.05)





training_data = train_data_IDG.flow_from_dataframe(dataframe=strat_train_df,

                                                   y_col='fruit', x_col='path',

                                                   target_size=(img_width,img_height), batch_size=8, shuffle=True,

                                                   class_mode='categorical')



validation_data = valid_data_IDG.flow_from_dataframe(dataframe=strat_valid_df,

                                                     y_col='fruit', x_col='path',

                                                     target_size=(img_width,img_height), shuffle=True,

                                                     class_mode='categorical')



print()

print(set(training_data.classes))

print()

print(set(validation_data.classes))
training_data.target_size, training_data.n, validation_data.target_size, validation_data.n
training_data.image_shape, training_data.batch_size, training_data.class_mode, validation_data.image_shape, validation_data.batch_size, validation_data.class_mode
tr_set_images, tr_set_labels = next(training_data)

val_set_images, val_set_labels = next(validation_data)

tr_set_images.shape, tr_set_labels.shape, val_set_images.shape, val_set_labels.shape
i = 2

print(mapped_fruit_names[np.where(tr_set_labels[i] == 1)[0][0]])

plt.imshow(X=tr_set_images[i])
i = 6

print(mapped_fruit_names[np.where(tr_set_labels[i] == 1)[0][0]])

plt.imshow(X=tr_set_images[i])
i = 7

print(mapped_fruit_names[np.where(val_set_labels[i] == 1)[0][0]])

plt.imshow(X=val_set_images[i])
from tensorflow.keras.applications.xception import Xception

from tensorflow.keras import layers

from tensorflow.keras import Model
base_xception = Xception(include_top=False, weights='imagenet', input_shape=(299,299,3), pooling='max')



inner = base_xception.output

inner = layers.Dense(units=512, activation='relu')(inner)

inner = layers.Dropout(rate=0.5)(inner)

inner = layers.Dense(units=15, activation='softmax')(inner)



model_1 = Model(inputs=base_xception.input, outputs=inner)

model_1.summary()
from tensorflow_addons.metrics import FBetaScore

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.metrics import CategoricalAccuracy, Precision, Recall

from tensorflow.keras.losses import CategoricalCrossentropy



model_1.compile(loss = CategoricalCrossentropy(),

                optimizer = Adam(),

                metrics = [FBetaScore(num_classes=15, average='macro', name='fbeta_score'),

                           CategoricalAccuracy(name='cat_acc'),

                           # GeometricMeanScore(average='weighted'),

                           Precision(name='precision'), Recall(name='recall')])
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau



early_stop_cb = EarlyStopping(monitor='val_fbeta_score', min_delta=0, patience=4, verbose=1, mode='max')

reduce_learning_rate_cb = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=2, cooldown=2, min_lr=0.00001, verbose=1)
new_maps_2 = pd.Series(training_data.classes).value_counts()

new_maps_2 = new_maps_2.rename("fruit")

new_maps_2
new_maps = new_maps_2.to_dict()

new_maps
sum(new_maps.values())
set(training_data.classes)
# Scaling by total/15 helps keep the loss to a similar magnitude.

# The sum of the weights of all examples stays the same.

fruit_weights = {i:(1/new_maps[i])*(67_021)/15.0 for i in range(15)}

fruit_weights
history_model_1 = model_1.fit(training_data,

                              validation_data=validation_data,

                              class_weight=fruit_weights,

                              verbose=1, epochs=30,

                              callbacks=[early_stop_cb, reduce_learning_rate_cb])
# save the final model with all the weights

model_1.save(filepath='/kaggle/working', include_optimizer=True, save_format='.tf')
plt.plot(history_model_1.history['loss'])

plt.plot(history_model_1.history['val_loss'])

plt.title('loss by epochs')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'val'], loc='best')

plt.show()
plt.plot(history_model_1.history['fbeta_score'])

plt.plot(history_model_1.history['val_fbeta_score'])

plt.title('f-beta score by epochs(beta=15)')

plt.ylabel('F-Score')

plt.xlabel('epoch')

plt.legend(['train', 'val'], loc='best')

plt.show()
plt.plot(history_model_1.history['cat_acc'])

plt.plot(history_model_1.history['val_cat_acc'])

plt.title('Categorical Accuracy by epochs')

plt.ylabel('Categorical Accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'val'], loc='best')

plt.show()
plt.plot(history_model_1.history['precision'])

plt.plot(history_model_1.history['val_precision'])

plt.plot(history_model_1.history['recall'])

plt.plot(history_model_1.history['val_recall'])

plt.title('Precision and Recall by epochs')

plt.ylabel('Precision and Recall')

plt.xlabel('epoch')

plt.legend(['train_precision', 'val_precision','train_recall', 'val_recall'], loc='best')

plt.show()