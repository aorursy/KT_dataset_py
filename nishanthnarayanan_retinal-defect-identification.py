import os
for filenames in os.listdir('/kaggle/input/'):
        print(os.path.join("/kaggle/input", filenames))
import numpy as np 
import pandas as pd

import random

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img

import matplotlib.pyplot as plt
base_dir = os.path.join("/kaggle/input/kermany2018/oct2017/OCT2017 /")
print("Base directory --> ", os.listdir(base_dir))
# Train set
train_dir = os.path.join(base_dir + "train/")
print("Train --> ", os.listdir(train_dir))

# Test set
test_dir = os.path.join(base_dir +"test/")
print("Test --> ", os.listdir(test_dir))

# Validation set
validation_dir = os.path.join(base_dir +"val/")
print("Validation --> ", os.listdir(validation_dir)[:5])
# Displaying random image from the dataset

fig, ax = plt.subplots(1, 4, figsize=(15, 10))

sample_paper = random.choice(os.listdir(train_dir + "DRUSEN"))
image = load_img(train_dir + "DRUSEN/" + sample_paper)
ax[0].imshow(image)
ax[0].set_title("DRUSEN")
ax[0].axis("Off")

sample_rock = random.choice(os.listdir(train_dir + "DME"))
image = load_img(train_dir + "DME/" + sample_rock)
ax[1].imshow(image)
ax[1].set_title("DME")
ax[1].axis("Off")

sample_scissor = random.choice(os.listdir(train_dir + "CNV"))
image = load_img(train_dir + "CNV/" + sample_scissor)
ax[2].imshow(image)
ax[2].set_title("CNV")
ax[2].axis("Off")

sample_scissor = random.choice(os.listdir(train_dir + "NORMAL"))
image = load_img(train_dir + "NORMAL/" + sample_scissor)
ax[3].imshow(image)
ax[3].set_title("NORMAL")
ax[3].axis("Off")

plt.show()
model = tf.keras.models.Sequential([
    
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    
    tf.keras.layers.Dense(4, activation='softmax')
])

model.summary()
model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])

train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2, # Shifting image width by 20%
      height_shift_range=0.2,# Shifting image height by 20%
      shear_range=0.2,       # Rotation across X-axis by 20%
      zoom_range=0.2,        # Image zooming by 20%
      horizontal_flip=True,
      fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size = (150, 150),
    class_mode = 'categorical',
    batch_size = 100
)
validation_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = validation_datagen.flow_from_directory(
    test_dir,
    target_size = (150, 150),
    class_mode = 'categorical',
    batch_size = 20
)
history = model.fit_generator(
      train_generator,
      steps_per_epoch = np.ceil(83484/100),  # 83484 images = batch_size * steps
      epochs = 10,
      validation_data=validation_generator,
      validation_steps = np.ceil(968/20),  # 968 images = batch_size * steps
      verbose = 1)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.figure(figsize=(7,7))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure(figsize=(7,7))

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
model.save('/kaggle/working/model.h5')
model.save_weights('/kaggle/working/model_weights.h5')
import os
import shutil
dest = "/kaggle/working/test_image"
os.mkdir(dest) 
for i in ['NORMAL', 'DME', 'DRUSEN', 'CNV']:
    src = os.path.join(validation_dir, i)
    src_files = os.listdir(src)
    for file_name in src_files:
        if(file_name != ".DS_Store"):
            full_file_name = os.path.join(src, file_name)
            if os.path.isfile(full_file_name):
                shutil.copy(full_file_name, dest)
len(os.listdir(dest))
test_img = os.listdir(dest)

test_df = pd.DataFrame({'Image': test_img})
test_df.head()
test_gen = ImageDataGenerator(rescale=1./255)

test_generator = test_gen.flow_from_dataframe(
    test_df, 
    dest, 
    x_col = 'Image',
    y_col = None,
    class_mode = None,
    target_size = (150, 150),
    batch_size = 20,
    shuffle = False
)
predict = model.predict_generator(test_generator, steps = int(np.ceil(32/20)))
# Identifying the classes

label_map = dict((v,k) for k,v in train_generator.class_indices.items())
label_map
test_df['Label'] = np.argmax(predict, axis = -1)

test_df['Label'] = test_df['Label'].replace(label_map)
test_df.head()
test_df.Label.value_counts().plot.bar(color = ['red','blue','green'])
plt.xticks(rotation = 0)
plt.show()
v = random.randint(0, 24)

sample_test = test_df.iloc[v:(v+18)].reset_index(drop = True)
sample_test.head()

plt.figure(figsize=(12, 24))
for index, row in sample_test.iterrows():
    filename = row['Image']
    category = row['Label']
    img = load_img(dest +"/" + filename, target_size = (150, 150))
    plt.subplot(6, 3, index + 1)
    plt.imshow(img)
    plt.xlabel(filename + ' ( ' + "{}".format(category) + ' )' )
plt.tight_layout()
plt.show()
lis = []
for ind in test_df.index: 
    if(test_df['Label'][ind] in test_df['Image'][ind]):
        lis.append(1)
    else:
        lis.append(0)
print("Accuracy of the model on test data is {:.2f}%".format((sum(lis)/len(lis))*100))