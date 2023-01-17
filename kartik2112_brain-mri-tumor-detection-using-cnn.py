import numpy as np # linear algebra
import tensorflow as tf
import keras
import os
import shutil
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

%matplotlib inline
!apt-get install tree --quiet
# !rm -rf ./train
# !rm -rf ./test
# !rm -rf ./val
!mkdir train test val train/yes train/no test/yes test/no val/yes val/no
!tree -d
img_path = '../input/brain-mri-images-for-brain-tumor-detection/brain_tumor_dataset/'

for class1 in os.listdir(img_path):
    num_images = len(os.listdir(os.path.join(img_path,class1)))
    for (n,filename) in enumerate(os.listdir(os.path.join(img_path,class1))):
        img = os.path.join(img_path,class1,filename)
        if n < int(0.1 * num_images):
            shutil.copy(img,'test/'+class1+'/'+filename)
        elif n < int(0.8 * num_images):
            shutil.copy(img,'train/'+class1+'/'+filename)
        else:
            shutil.copy(img,'val/'+class1+'/'+filename)
def load_data(image_dir):
    images = []
    y = []
    classNum = 0
    for class1 in tqdm(os.listdir(image_dir)):
        for file_name in os.listdir(os.path.join(image_dir,class1)):
            images.append(cv2.imread(os.path.join(image_dir,class1,file_name)))
            y.append(classNum)
        classNum += 1
    print(f'Loaded {len(images)} images from {image_dir} directory')
    images = np.array(images)
    y = np.array(y)
    return images,y
def show_samples(X,y,label_dict={0:'no',1:'yes'},n=30):
    for class1 in label_dict.keys():
        imgs = X[y == class1][:n]
        j = 10
        i = n // 10
        plt.figure(figsize=(15,6))
        for (c,img) in enumerate(imgs,1):
            plt.subplot(i,j,c)
            plt.imshow(img)
            plt.xticks([])
            plt.yticks([])
        plt.suptitle(f'Tumor: {label_dict[class1]}')
train_images, train_labels = load_data('train/')
val_images, val_labels = load_data('val/')
test_images, test_labels = load_data('test/')

train_images[0].shape
show_samples(train_images,train_labels)
data = [[(train_labels == 0).sum(),(val_labels == 0).sum(),(test_labels == 0).sum()],
        [(train_labels == 1).sum(),(val_labels == 1).sum(),(test_labels == 1).sum()]]

labels = ['Train', 'Test', 'Validation']
X = np.arange(3)

# fig = plt.figure()
width = 0.35
plt.bar(X - width/2,data[0],width,label='no')
plt.bar(X + width/2,data[1],width,label='yes')
plt.legend(loc='best')
plt.xticks(X,labels=labels)

demo_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.05,
    height_shift_range=0.05,
    rescale=1./255,
    shear_range=0.05,
    brightness_range=[0.5, 1.5],
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='constant',
    cval=0
)
batch1 = demo_datagen.flow_from_directory('train/').next()
show_samples(batch1[0], batch1[1].argmax(axis=1))
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from tensorflow.keras.callbacks import EarlyStopping
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.05,
    height_shift_range=0.05,
    rescale=1./255,
    shear_range=0.05,
    brightness_range=[0.5, 1.5],
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='constant',
    cval=0
)
RANDOM_SEED = 0
IMG_SIZE = (224,224)
train_gen = datagen.flow_from_directory(
        'train/',
        color_mode='rgb',
        target_size=IMG_SIZE,
        batch_size=32,
        class_mode='binary',
        seed=RANDOM_SEED
)

valid_gen = datagen.flow_from_directory(
        'val/',
        color_mode='rgb',
        target_size=IMG_SIZE,
        batch_size=16,
        class_mode='binary',
        seed=RANDOM_SEED
)
base_model = VGG16(include_top=False,weights='imagenet',input_shape=(224,224,3))
base_model.summary()
model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(300,activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.005)))
model.add(Dense(1,activation='sigmoid'))
model.layers[0].trainable = False

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.01,
    decay_steps=500,
    decay_rate=0.0001)
opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

model.compile(
    loss='binary_crossentropy',
    optimizer=opt,
    metrics=['accuracy']
)

model.summary()
train_step_size = train_gen.n//train_gen.batch_size
val_step_size = valid_gen.n//valid_gen.batch_size
es = EarlyStopping(
    monitor='val_accuracy', 
    mode='max',
    patience=6
)

history = model.fit_generator(
    train_gen,
    steps_per_epoch=train_step_size,
    epochs=10,
    validation_data=valid_gen,
    validation_steps=val_step_size,
    callbacks=[es]
)
model.layers[0].trainable = True
model.compile(
    loss='binary_crossentropy',
    optimizer=opt,
    metrics=['accuracy']
)

history = model.fit_generator(
    train_gen,
    steps_per_epoch=train_step_size,
    epochs=30,
    validation_data=valid_gen,
    validation_steps=val_step_size,
    callbacks=[es]
)
# plot model performance
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(1, len(history.epoch) + 1)

plt.figure(figsize=(15,5))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Train Set')
plt.plot(epochs_range, val_acc, label='Val Set')
plt.legend(loc="best")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Model Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Train Set')
plt.plot(epochs_range, val_loss, label='Val Set')
plt.legend(loc="best")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Model Loss')

plt.tight_layout()
plt.show()
test_gen = datagen.flow_from_directory(
        'test/',
        color_mode='rgb',
        target_size=IMG_SIZE,
        batch_size=16,
        class_mode='binary',
        seed=RANDOM_SEED
)
model.evaluate_generator(valid_gen)
[1 if x > 0.5 else 0 for x in model.predict_generator(test_gen)]
from sklearn.metrics import confusion_matrix
confusion_matrix(preds)
model.save('brain-mri-vgg16-27 May 20.h5')
