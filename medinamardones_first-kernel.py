import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

import os
original_dataset_dir = "../input"
print(os.listdir(original_dataset_dir))
target_names = {
    0:  "Nucleoplasm",  
    1:  "Nuclear membrane",   
    2:  "Nucleoli",   
    3:  "Nucleoli fibrillar center",   
    4:  "Nuclear speckles",
    5:  "Nuclear bodies",   
    6:  "Endoplasmic reticulum",   
    7:  "Golgi apparatus",   
    8:  "Peroxisomes",   
    9:  "Endosomes",   
    10:  "Lysosomes",   
    11:  "Intermediate filaments",   
    12:  "Actin filaments",   
    13:  "Focal adhesion sites",   
    14:  "Microtubules",   
    15:  "Microtubule ends",   
    16:  "Cytokinetic bridge",   
    17:  "Mitotic spindle",   
    18:  "Microtubule organizing center",   
    19:  "Centrosome",   
    20:  "Lipid droplets",   
    21:  "Plasma membrane",   
    22:  "Cell junctions",   
    23:  "Mitochondria",   
    24:  "Aggresome",   
    25:  "Cytosol",   
    26:  "Cytoplasmic bodies",   
    27:  "Rods & rings"
}

target_numbers = dict((v,int(k)) for k,v in target_names.items())

target_names
train_data = pd.read_csv('../input/train.csv')
train_data.head()
num_samples = train_data.shape[0]
num_samples
for target in target_names.keys() :
    train_data[target_names[target]] = 0

def fill_rows(row) :
    row.Target = np.array(row.Target.split(" ")).astype(np.int)
    for num in row.Target:
        name = target_names[int(num)]
        row.loc[name] = int(1)
    return row

train_data = train_data.apply(fill_rows, axis=1)
train_data.head()
from imageio import imread

def load_image(basepath, image_id):
    images = np.zeros(shape=(4,512,512))
    images[0,:,:] = imread(basepath + image_id + "_green" + ".png")
    images[1,:,:] = imread(basepath + image_id + "_red" + ".png")
    images[2,:,:] = imread(basepath + image_id + "_blue" + ".png")
    images[3,:,:] = imread(basepath + image_id + "_yellow" + ".png")
    return images

def make_image_row(image, subax, title="Title"):
    subax[0].imshow(image[0], cmap="Greens")
    subax[1].imshow(image[1], cmap="Reds")
    subax[1].set_title("stained microtubules")
    subax[2].imshow(image[2], cmap="Blues")
    subax[2].set_title("stained nucleus")
    subax[3].imshow(image[3], cmap="Oranges")
    subax[3].set_title("stained endoplasmatic reticulum")
    subax[0].set_title(title)
    return subax

def make_title(file_id):
    file_targets = train_data.loc[train_data.Id==file_id, "Target"].values[0]
    title = " | "
    for n in file_targets:
        title += target_names[n] + " | "
    return title
for row in train_data.iloc[4:7].itertuples() :
    fig, ax = plt.subplots(1, 4, figsize=(24,24))
    make_image_row(load_image("../input/train/", row.Id), ax, make_title(row.Id))
target_prob = train_data.drop(["Id", "Target"],axis=1).sum(axis=0).sort_values(ascending=False)
plt.figure(figsize=(20,10))
sns.barplot(y=target_prob.index.values, x=(target_prob.values)/num_samples, palette="Blues_d")

train_data["Quantity"] = train_data.drop(["Id", "Target"],axis=1).sum(axis=1)
count = train_data["Quantity"].value_counts()
plt.figure(figsize=(15,5))
sns.barplot(y=(count.values)/num_samples, x=count.index.values, palette="ch:2.5,-.2,dark=.3")
train_data["Id_green"] = train_data['Id'].apply(lambda row: row + '_green')
from keras.preprocessing.image import ImageDataGenerator

datagen=ImageDataGenerator(rescale=1./255.,validation_split=0.25)

train_generator=datagen.flow_from_dataframe(
                dataframe=train_data,
                directory="../input/train/",
                x_col="Id_green",
                y_col="Quantity",
                has_ext=False,                                      
                subset="training",
                batch_size=64,
                seed=42,
                shuffle=True,
                class_mode="categorical",
                target_size=(512,512),
                color_mode = 'grayscale')

validation_generator=datagen.flow_from_dataframe(
                dataframe=train_data,
                directory="../input/train/",
                x_col="Id_green",
                y_col="Quantity",
                has_ext=False,
                subset="validation",
                batch_size=64,
                seed=42,
                shuffle=True,
                class_mode="categorical",
                target_size=(512,512),
                color_mode = 'grayscale')
for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break
from keras import Sequential
from keras import layers, models
from keras import optimizers

model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(512, 512, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(5, activation='sigmoid'))

print(model.summary())

model.compile(loss='categorical_crossentropy',
          optimizer=optimizers.RMSprop(lr=1e-4),
          metrics=['acc'])

history = model.fit_generator(
      train_generator,
      steps_per_epoch=300,
      epochs=8,
      validation_data=validation_generator,
      validation_steps=80,
      use_multiprocessing=True,
      workers=8)

model.save('quantity_1.h5')
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()