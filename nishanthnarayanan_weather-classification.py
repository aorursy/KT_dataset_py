import os
for dirname, _, filenames in os.walk('/kaggle/input'):
        print(dirname)
import pandas as pd
import numpy as np

import random
import shutil

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img

import matplotlib.pyplot as plt
base_dir = os.path.join("/kaggle/input/multiclass-weather-dataset/dataset/")
os.listdir(base_dir)
train_dir = "/train/"
test_dir = base_dir + "alien_test/"
# Code to replicate the whole directory

def copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)
            
copytree(base_dir, train_dir) # Define source directory and destination directory

# Here we remove the unwanted folders by condition
            
for i in os.listdir(train_dir):
    if i not in ['sunrise', 'shine', 'cloudy', 'rainy', 'foggy']:
        try:
            os.remove(train_dir + i)
        except:
            shutil.rmtree(train_dir + i)
print("Train directory -->", os.listdir(train_dir))
print("Test directory -->", os.listdir(test_dir)[:5])
# Displaying random image from the dataset

fig, ax = plt.subplots(1, 5, figsize=(15, 10))

sample_paper = random.choice(os.listdir(train_dir + "rainy"))
image = load_img(train_dir + "rainy/" + sample_paper)
ax[0].imshow(image)
ax[0].set_title("Rainy")
ax[0].axis("Off")

sample_rock = random.choice(os.listdir(train_dir + "foggy"))
image = load_img(train_dir + "foggy/" + sample_rock)
ax[1].imshow(image)
ax[1].set_title("Foggy")
ax[1].axis("Off")

sample_scissor = random.choice(os.listdir(train_dir + "shine"))
image = load_img(train_dir + "shine/" + sample_scissor)
ax[2].imshow(image)
ax[2].set_title("Shine")
ax[2].axis("Off")

sample_scissor = random.choice(os.listdir(train_dir + "sunrise"))
image = load_img(train_dir + "sunrise/" + sample_scissor)
ax[3].imshow(image)
ax[3].set_title("Sunrise")
ax[3].axis("Off")

sample_scissor = random.choice(os.listdir(train_dir + "cloudy"))
image = load_img(train_dir + "cloudy/" + sample_scissor)
ax[4].imshow(image)
ax[4].set_title("Cloudy")
ax[4].axis("Off")

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
    
    tf.keras.layers.Dense(256, activation='relu'),
    
    tf.keras.layers.Dense(5, activation='softmax')
])
model.summary()
model.compile(loss = 'categorical_crossentropy',
              optimizer = 'SGD',
              metrics = ['accuracy'])
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy')>0.85):
            print("\nReached >85% accuracy so cancelling training!")
            self.model.stop_training = True
        
callbacks = myCallback()
train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.4, # Shifting image width by 40%
      height_shift_range=0.4,# Shifting image height by 40%
      shear_range=0.2,       # Rotation across X-axis by 20%
      zoom_range=0.3,        # Image zooming by 30%
      horizontal_flip=True,
      fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size = (150, 150),
    class_mode = 'categorical',
    batch_size = 20
)
history = model.fit_generator(
      train_generator,
      steps_per_epoch = np.ceil(1500/20),  # 1500 images = batch_size * steps
      epochs = 50,
      callbacks=[callbacks],
      verbose = 2)
print("Accuracy of the model on train data is {:.2f}%".format(history.history["accuracy"][-1]*100))
test_img = os.listdir(os.path.join(test_dir))

test_df = pd.DataFrame({'Image': test_img})
test_df.head()
len(test_df)
test_gen = ImageDataGenerator(rescale=1./255)

test_generator = test_gen.flow_from_dataframe(
    test_df, 
    test_dir, 
    x_col = 'Image',
    y_col = None,
    class_mode = None,
    target_size = (150, 150),
    batch_size = 20,
    shuffle = False
)
predict = model.predict_generator(test_generator, steps = int(np.ceil(30/20)))
# Identifying the classes

label_map = dict((v,k) for k,v in train_generator.class_indices.items())
label_map
test_df['Label'] = np.argmax(predict, axis = -1) # axis = -1 --> To compute the max element index within list of lists

test_df['Label'] = test_df['Label'].replace(label_map)
test_df.Label.value_counts().plot.bar(color = ['red','blue','green','yellow','orange'])
plt.xticks(rotation = 0)
plt.show()
v = random.randint(0, 12)

sample_test = test_df.iloc[v:(v+18)].reset_index(drop = True)
sample_test.head()

plt.figure(figsize=(12, 24))
for index, row in sample_test.iterrows():
    filename = row['Image']
    category = row['Label']
    img = load_img(test_dir + filename, target_size = (150, 150))
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