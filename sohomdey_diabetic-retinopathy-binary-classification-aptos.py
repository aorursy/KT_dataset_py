import os
import cv2
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from albumentations import CLAHE
from albumentations import Compose, OneOf, Flip, Transpose, Rotate, RandomRotate90, ShiftScaleRotate, RandomBrightnessContrast, RandomGamma, HueSaturationValue, RGBShift

from sklearn.metrics import classification_report
df_labels = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')

print(df_labels.shape)
df_labels.head()
seed = 1
df_labels = df_labels.sample(frac=1, random_state=seed)
# Separate each classes
df_labels_0 = df_labels[df_labels['diagnosis']==0]
df_labels_1 = df_labels[df_labels['diagnosis']==1]
df_labels_2 = df_labels[df_labels['diagnosis']==2]
df_labels_3 = df_labels[df_labels['diagnosis']==3]
df_labels_4 = df_labels[df_labels['diagnosis']==4]

# Train test split
df_labels_0_train = df_labels_0[:-200]
df_labels_1_train = df_labels_1[:-50]
df_labels_2_train = df_labels_2[:-50]
df_labels_3_train = df_labels_3[:-50]
df_labels_4_train = df_labels_4[:-50]

df_labels_0_valid = df_labels_0[-200:]
df_labels_1_valid = df_labels_1[-50:]
df_labels_2_valid = df_labels_2[-50:]
df_labels_3_valid = df_labels_3[-50:]
df_labels_4_valid = df_labels_4[-50:]

print(f'Number of train samples with label 0 = {len(df_labels_0_train)}')
print(f'Number of train samples with label 1 = {len(df_labels_1_train)}')
print(f'Number of train samples with label 2 = {len(df_labels_2_train)}')
print(f'Number of train samples with label 3 = {len(df_labels_3_train)}')
print(f'Number of train samples with label 4 = {len(df_labels_4_train)}')


# Class balancing by oversampling
df_labels_0_train = df_labels_0_train.sample(n=2000, replace=True, random_state=seed)
df_labels_1_train = df_labels_1_train.sample(n=500, replace=True, random_state=seed)
df_labels_2_train = df_labels_2_train.sample(n=500, replace=True, random_state=seed)
df_labels_3_train = df_labels_3_train.sample(n=500, replace=True, random_state=seed)
df_labels_4_train = df_labels_4_train.sample(n=500, replace=True, random_state=seed)


df_labels_train = pd.concat([df_labels_0_train, df_labels_1_train, df_labels_2_train, df_labels_3_train, df_labels_4_train])
df_labels_valid = pd.concat([df_labels_0_valid, df_labels_1_valid, df_labels_2_valid, df_labels_3_valid, df_labels_4_valid])

print(f'Number of train samples = {len(df_labels_train)}')
print(f'Number of valid samples = {len(df_labels_valid)}')
df_labels_train = df_labels_train.sample(frac=1, random_state=seed)
df_labels_valid = df_labels_valid.sample(frac=1, random_state=seed)
def read_image(file_path):
    image = cv2.imread(file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (512,512))
    return image
def crop_resize(image, shape=(256,256), thresh=7):

    if image.ndim ==2:
        # If the image is already grayscale
        mask = image>thresh
        cropped_image = image[np.ix_(mask.any(1),mask.any(0))]
        cropped_image = cv2.resize(cropped_image, shape)
        return cropped_image

    elif image.ndim==3:
        # If the image is a colour image
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        mask = gray_image>thresh

        # Sanity check the shape of the cropped image
        check_shape = image[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]

        if (check_shape == 0): 
            # The image is too dark, cropping not possible.
            image = cv2.resize(image, shape)
            return image

        else:
            # Crop each channel separately and stack them back
            channel_1 = image[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            channel_2 = image[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            channel_3 = image[:,:,2][np.ix_(mask.any(1),mask.any(0))]
            cropped_image = np.stack([channel_1, channel_2, channel_3], axis=-1)
            cropped_image = cv2.resize(cropped_image, shape)
            return cropped_image
def enhance(image):
    #image = CLAHE(always_apply=True, p=1.0)(image=image)['image']
    image = cv2.addWeighted(image,4,cv2.GaussianBlur(image,(0,0),10),-4,128)
    return image
IMG_DIM = 224

folder_path = '../input/aptos2019-blindness-detection/train_images/'
train_images = [] 
train_labels = []
 
c = 0
for i in range(len(df_labels_train)):
    
    id_code = df_labels_train.iloc[i]['id_code']
    diagnosis = df_labels_train.iloc[i]['diagnosis']
    
    file_path = folder_path + id_code + '.png'
    
    image = read_image(file_path)
    image = crop_resize(image, (IMG_DIM, IMG_DIM))
    image = enhance(image)
    
    train_images.append(image)
    if diagnosis == 0:
        train_labels.append([0])
    else:
        train_labels.append([1])
    
    c+=1
    
    if c%1000 == 0:
        print(f'{c} files processed')
    

train_images = np.array(train_images)
train_labels = np.array(train_labels)

print(train_images.shape)
print(train_labels.shape)
valid_images = [] 
valid_labels = []

for i in range(len(df_labels_valid)):
    
    id_code = df_labels_valid.iloc[i]['id_code']
    diagnosis = df_labels_valid.iloc[i]['diagnosis']
    
    file_path = folder_path + id_code + '.png'
    
    image = read_image(file_path)
    image = crop_resize(image, (IMG_DIM, IMG_DIM))
    image = enhance(image)
    
    valid_images.append(image)
    if diagnosis == 0:
        valid_labels.append([0])
    else:
        valid_labels.append([1])

valid_images = np.array(valid_images)
valid_labels = np.array(valid_labels)

print(valid_images.shape)
print(valid_labels.shape)
print(np.unique(train_labels, return_counts=True))
print(np.unique(valid_labels, return_counts=True))
class Generator(tf.keras.utils.Sequence):

    def __init__(self, x_set, y_set, batch_size=1, augment=True):
        self.x = x_set
        self.y = y_set
        self.batch_size = batch_size
        self.augment = augment
        self.augmentations = Compose(
        [
            OneOf([
                    Flip(p=0.7),
                    Transpose(p=0.3),
                    ], p=0.8),
            
            OneOf([
                    Rotate(limit=90, border_mode=0, p=0.7),
                    RandomRotate90(p=0.3),
                    ], p=0.6),
                        
            OneOf([
                    RandomBrightnessContrast(p=0.7),
                    RandomGamma(p=0.3),
                    ], p=0.4),            
        ])
         

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        if self.augment is True:
            batch_x = np.array([self.augmentations(image=i)['image'] for i in batch_x])


        return batch_x/255, batch_y/1
batch_size = 64

def train_generator_func():
    generator = Generator(train_images, train_labels, batch_size, True)
    return generator

def valid_generator_func():
    generator = Generator(valid_images, valid_labels, batch_size, False)
    return generator


train_generator = tf.data.Dataset.from_generator(
    train_generator_func,
    output_types=(tf.float32, tf.float32),
    output_shapes=((None, IMG_DIM, IMG_DIM, 3), (None, 1))
).repeat().prefetch(tf.data.experimental.AUTOTUNE)

valid_generator = tf.data.Dataset.from_generator(
    valid_generator_func,
    output_types=(tf.float32, tf.float32),
    output_shapes=((None, IMG_DIM, IMG_DIM, 3), (None, 1))
).prefetch(tf.data.experimental.AUTOTUNE)
for i, j in train_generator:
    break

fig, axes = plt.subplots(1, 5, figsize=(12, 4))
fig.suptitle('Train Images', fontsize=15)
axes = axes.flatten()
for img, lbl, ax in zip(i, j, axes):
    ax.imshow(img)
    ax.title.set_text(str(lbl.numpy()))
    ax.axis('off')
plt.tight_layout()
plt.show()
base_model = tf.keras.applications.DenseNet121(input_shape=(IMG_DIM, IMG_DIM, 3),
                                               include_top=False,
                                               weights='imagenet')
model = tf.keras.Sequential([
                                base_model,
                                tf.keras.layers.GlobalAveragePooling2D(),
                                tf.keras.layers.BatchNormalization(),
                                tf.keras.layers.Dropout(0.2),
                                tf.keras.layers.Dense(512, activation='relu'),
                                tf.keras.layers.BatchNormalization(),
                                tf.keras.layers.Dense(1, activation='sigmoid'),
                            ])
model.summary()
base_learning_rate = 0.001
model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=0.2),
              metrics=['accuracy'])
steps_per_epoch = len(train_images)//batch_size
validation_steps = len(valid_images)//batch_size
epochs = 10
initial_epoch = 0

history = model.fit(
    x=train_generator,
    steps_per_epoch=steps_per_epoch,
    initial_epoch=initial_epoch,
    epochs=epochs,
    validation_data=valid_generator, 
    validation_steps=validation_steps,
    verbose=1,
)
y_true = valid_labels
y_pred = model.predict(valid_images/255)
y_pred = y_pred>0.5

print('\n\n')
print(classification_report(y_true, y_pred))
fine_tune_at = 0
for layer in base_model.layers[fine_tune_at:]:
    layer.trainable =  True
base_learning_rate = 0.000001
model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=0.2),
              metrics=['accuracy'])
steps_per_epoch = (2*len(train_images))//batch_size
validation_steps = len(valid_images)//batch_size
epochs = 10
initial_epoch = 0

history = model.fit(
    x=train_generator,
    steps_per_epoch=steps_per_epoch,
    initial_epoch=initial_epoch,
    epochs=epochs,
    validation_data=valid_generator, 
    validation_steps=validation_steps,
    verbose=1,
)
y_true = valid_labels
y_pred = model.predict(valid_images/255)
y_pred = y_pred>0.5

print('\n\n')
print(classification_report(y_true, y_pred))
