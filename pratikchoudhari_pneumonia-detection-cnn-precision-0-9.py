import gc
import cv2
import numpy as np 
import pandas as pd 
import seaborn as sns
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from sklearn.metrics import confusion_matrix,precision_score,recall_score

sns.set(font_scale=1.4)
#base image path in kaggle env
BASE_IMAGE_PATH = Path('/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/')
#Reshape input images to 200x200, free to be changed
IMG_SIZE=200
#returns image path of a set of data, train, test and val
def get_imgs_path(cat):
    PATH = BASE_IMAGE_PATH / cat
    normal_path = PATH / 'NORMAL'
    pneumonia_path = PATH / 'PNEUMONIA'
    normal_imgs_path = normal_path.glob('*.jpeg')
    pneumonia_imgs_path = pneumonia_path.glob('*.jpeg')
    normal_imgs = [path for path in normal_imgs_path]
    pneumonia_imgs = [path for path in pneumonia_imgs_path]
    print('\n{} normal images in {} directory'.format(len(normal_imgs),cat))
    print('{} pneumonia images in {} directory\n'.format(len(pneumonia_imgs),cat))
    return normal_imgs,pneumonia_imgs

#return images and labels from given path
def get_images(normal_imgs,pneumonia_imgs):
    normal_images=[]
    normal_labels=[]
    for image_path in normal_imgs:
        img = cv2.imread(str(image_path))
        img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))
        if img.shape[2]==1:
            img = np.dstack([img, img, img])
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img/255.0
        img  = img.astype(np.float32)
        normal_images.append(img)
        normal_labels.append(0)

    for image_path in pneumonia_imgs:
        img = cv2.imread(str(image_path))
        img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))
        if img.shape[2]==1:
            img = np.dstack([img, img, img])
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img/255.0
        img  = img.astype(np.float32)
        normal_images.append(img)
        normal_labels.append(1)
    return np.array(normal_images),np.array(normal_labels)

#Function for garbage collection
def clean(x):
    x=0
    gs.collect()
#Ready all 3 sets of data
normal_train_path,pneumonia_train_path = get_imgs_path('train')
train_images,train_labels=get_images(normal_train_path,pneumonia_train_path)
print("Shape of train_images is {}".format(train_images.shape))
print("Shape of train_labels is {}".format(train_labels.shape))

normal_val_path,pneumonia_val_path = get_imgs_path('val')
val_images,val_labels=get_images(normal_val_path,pneumonia_val_path)
print("Shape of val_images is {}".format(val_images.shape))
print("Shape of val_labels is {}".format(val_labels.shape))


normal_test_path,pneumonia_test_path = get_imgs_path('test')
test_images,test_labels=get_images(normal_test_path,pneumonia_test_path)
print("Shape of test_images is {}".format(test_images.shape))
print("Shape of test_labels is {}".format(test_labels.shape))
f = plt.figure(figsize=(20,20))
for i in range(1,17):
    ax = f.add_subplot(4,4,i)
    if i>=9:
        i += 1341
        ax.set_title('Pneumonia')
    else:
        ax.set_title('Normal')
    plt.imshow(train_images[i,:,:,:])
    ax.axis('off')
#define focal loss function which works best for imbalanced classification
#Refer to original paper here https://arxiv.org/pdf/1708.02002.pdf
from keras import backend as K
def focal_loss(alpha=0.25,gamma=2.0):
    def focal_crossentropy(y_true, y_pred):
        bce = K.binary_crossentropy(y_true, y_pred)
        
        y_pred = K.clip(y_pred, K.epsilon(), 1.- K.epsilon())
        p_t = (y_true*y_pred) + ((1-y_true)*(1-y_pred))
        
        alpha_factor = 1
        modulating_factor = 1

        alpha_factor = y_true*alpha + ((1-alpha)*(1-y_true))
        modulating_factor = K.pow((1-p_t), gamma)

        # compute the final loss and return
        return K.mean(alpha_factor*modulating_factor*bce, axis=-1)
    return focal_crossentropy
#opt = tf.keras.optimizers.Adadelta(lr=0.0001)
#es = EarlyStopping(patience=5)
#ckpt = ModelCheckpoint(filepath='best_model_todate', save_best_only=True, save_weights_only=True)
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=16,activation='relu',kernel_size=(3,3),padding='same',input_shape=(IMG_SIZE,IMG_SIZE,3)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    tf.keras.layers.Conv2D(filters=32,activation='relu',padding='same',kernel_size=(3,3)),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    tf.keras.layers.Conv2D(filters=64,activation='relu',padding='same',kernel_size=(3,3)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    
    tf.keras.layers.Conv2D(filters=128,activation='relu',padding='same',kernel_size=(3,3)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    tf.keras.layers.Conv2D(filters=256,activation='relu',padding='same',kernel_size=(3,3)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256,activation='relu'),
    tf.keras.layers.Dropout(0.6),
    tf.keras.layers.Dense(1,activation='sigmoid')
])
model.compile(loss=focal_loss(),optimizer='adadelta',metrics=['accuracy'])
class_weights = class_weight.compute_class_weight('balanced',np.unique(train_labels),train_labels)
class_weights = dict(enumerate(class_weights))
history1 =  model.fit(train_images, train_labels.astype(np.float32), batch_size=16, epochs=50,
                      validation_data=(val_images,val_labels.astype(np.float32)),class_weight=class_weights, 
                      verbose=1)
# plot the model loss and accuracy
train_loss = history1.history['loss']
train_acc = history1.history['accuracy']

valid_loss = history1.history['val_loss']
valid_acc = history1.history['val_accuracy']

x = [(i+1) for i in range(len(train_loss))]

f,ax = plt.subplots(1,2, figsize=(12,5))
ax[0].plot(x, train_loss)
ax[0].plot(x, valid_loss)
ax[0].set_title("Loss plot")
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("loss")
ax[0].legend(['train', 'valid'])


ax[1].plot(x, train_acc)
ax[1].plot(x, valid_acc)
ax[1].set_title("Accuracy plot")
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("acc")
ax[1].legend(['train', 'valid'])

plt.show()
pred = np.round(model.predict(test_images,steps=test_steps))
print("Precision:{:.4f}\nRecall score:{:.4f}\n".format(precision_score(test_labels,pred),recall_score(test_labels,pred)))
print("Confusion Matrix:\n",sns.heatmap(confusion_matrix(test_labels,pred), annot=True, annot_kws={"size": 16}, cmap='YlGnBu', fmt='d'))