import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras import applications
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dropout, Flatten, Dense
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
import requests
from PIL import Image
from io import BytesIO
import cv2
import pandas as pd
import numpy as np
import os
translate = {"cane": "Dog", "cavallo": "Horse", "elefante": "Elephant", "farfalla": "Butterfly", "gallina": "Chicken", "gatto": "Cat", "mucca": "Cow", "pecora": "Sheep", "scoiattolo": "Squirrel", "ragno": "Spider"}
##### Creating Dataframe of images' path #####
# Class 2 has 1446 imges only. To maintain uniformty of counts among different classes, choose only first 1400 images.
foldernames = os.listdir('/kaggle/input/animals10/raw-img/')
files, files2, target, target2 = [], [], [], []

for i, folder in enumerate(foldernames):
    filenames = os.listdir("/kaggle/input/animals10/raw-img/" + folder);
    count = 0
    for file in filenames:
        if count < 1400:
            files.append("/kaggle/input/animals10/raw-img/" + folder + "/" + file)
            target.append(translate[folder])
        else:
            files2.append("/kaggle/input/animals10/raw-img/" + folder + "/" + file)
            target2.append(translate[folder])
        count += 1

df = pd.DataFrame({'Filepath':files, 'Target':target})
dft = pd.DataFrame({'Filepath':files2, 'Target':target2})
train, test = train_test_split(df, test_size=0.2857, random_state = 0)
#del files, files2, target, target2, df
datagen = ImageDataGenerator(rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        rotation_range=30, 
        width_shift_range=0.1,
        height_shift_range=0.1,
        samplewise_center = True,
        )

datagen_test = ImageDataGenerator(rescale=1./255, samplewise_center = True)

train_flow = datagen.flow_from_dataframe(train, x_col = 'Filepath', y_col = 'Target', target_size=(224, 224), interpolation = 'lanczos', validate_filenames = False)
test_flow = datagen_test.flow_from_dataframe(test, x_col = 'Filepath', y_col = 'Target', target_size=(224, 224), interpolation = 'lanczos', validate_filenames = False)
model = applications.VGG16(weights='imagenet', include_top = False, input_shape=(224, 224, 3))

#model.get_layer('block1_conv1').trainable = False
#model.get_layer('block1_conv2').trainable = False
#model.get_layer('block2_conv1').trainable = False
#model.get_layer('block2_conv2').trainable = False

flat1 = tf.keras.layers.Flatten()(model.output)
dropout1 = Dropout(0.1)(flat1)
class1 = tf.keras.layers.Dense(256, activation='relu')(dropout1)
dropout2 = Dropout(0.1)(class1)
output = tf.keras.layers.Dense(10, activation='softmax')(dropout2)
model = Model(inputs = model.inputs, outputs = output)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', factor = 0.2, patience = 1, verbose = 1, min_delta=0.0001, min_lr = 1e-8)

model.compile(loss = 'categorical_crossentropy', optimizer = optimizers.SGD(lr=1e-3, momentum=0.9), metrics = ['accuracy'])
model.summary()
history = model.fit_generator(train_flow, epochs = 10, validation_data = test_flow, callbacks=[ModelCheckpoint('VGG16.model', monitor='val_acc'), reduce_lr])
loss_vals = history.history['loss']
val_loss_vals = history.history['val_loss']
epochs = range(1, len(history.history['accuracy'])+1)

f, ax = plt.subplots(nrows=1,ncols=2,figsize=(18,5))
    
ax[0].plot(epochs, loss_vals,  marker='o', label='Training Loss')
ax[0].plot(epochs, val_loss_vals, marker='o', color = 'green', label='Validation Loss')
ax[0].set_title('Training & Validation Loss')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Loss')
ax[0].legend(loc='best')
ax[0].grid(True)

acc_vals = history.history['accuracy']
val_acc_vals = history.history['val_accuracy']

ax[1].plot(epochs, acc_vals, marker='o', label='Training Accuracy')
ax[1].plot(epochs, val_acc_vals, marker='o', color = 'green', label='Validation Accuracy')
ax[1].set_title('Training & Validation Accuracy')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Accuracy')
ax[1].legend(loc='best')
ax[1].grid(True)

plt.show()
f.savefig('LossAndAccuracy.eps', format='eps')
plt.close()
del loss_vals, val_loss_vals, epochs, acc_vals, val_acc_vals
test2_flow = datagen_test.flow_from_dataframe(dft, x_col = 'Filepath', y_col = 'Target', target_size=(224, 224), interpolation = 'lanczos', validate_filenames = False)
score = model.evaluate(test2_flow)
##### Append your urls of animals here for prediction #####
img_url = ['https://pbs.twimg.com/profile_images/378800000532546226/dbe5f0727b69487016ffd67a6689e75a.jpeg',
           'https://i.ytimg.com/vi/UwtTSqTbWzg/maxresdefault.jpg',
           'https://images.unsplash.com/photo-1547399300-7613d8f5f8f1?ixlib=rb-1.2.1&w=1000&q=80',
           'https://www.theartofdoingstuff.com/wp-content/uploads/2014/06/Studio_BlkCpMrnHn_8771_L2.jpg',
           'https://www.k9rl.com/wp-content/uploads/2017/01/Tibetan-Spaniel-dog.jpg',
           'https://static.boredpanda.com/blog/wp-content/uploads/2016/06/I-found-freedom-with-horses-576d2d0804976__880.jpg',
           'https://th.bing.com/th/id/OIP.tOb0fbG7VgjMf2Mj7Sa9igHaEK?pid=Api&rs=1',
           'https://pixfeeds.com/images/topic/2779/1200-2779-butterflies-photo1.jpg',
           'https://th.bing.com/th/id/OIP.GmYkDxsD--csxHFes-lWFwHaDS?pid=Api&rs=1',
           'https://resize.hswstatic.com/w_1024/gif/banana-spider.jpg',
           'https://www.nationalgeographic.com/content/dam/animals/2018/10/waq-animal-caches/01-waq-animal-caches-nationalgeographic_1902487.ngsversion.1539338405615.adapt.1900.1.jpg',
           'https://upload.wikimedia.org/wikipedia/commons/a/ac/Bombay_femelle.JPG',
           'https://th.bing.com/th/id/OIP.1L_D_VUm0lIxk_g1-5uxrgHaEi?pid=Api&rs=1',
           'https://upload.wikimedia.org/wikipedia/commons/thumb/6/69/7_month_old_Suffolk_Ram_Lamb.JPG/1200px-7_month_old_Suffolk_Ram_Lamb.JPG',
           'https://www.guidedogs.org/wp-content/uploads/2019/11/website-donate-mobile.jpg',
           'https://kids.sandiegozoo.org/sites/default/files/2017-09/animal-hero-spiders.jpg',
           'https://media4.s-nbcnews.com/j/newscms/2014_11/241886/140310-smart-elephants-347_5f910b48b8c1a85fc892e9c3d9cc2d00.fit-760w.jpg',
           'https://rollingharbourlife.files.wordpress.com/2013/05/squirrel-nyc-3.jpg',
           'https://media.mnn.com/assets/images/2015/04/butterfly.jpg.990x0_q80_crop-smart.jpg',
           'https://i.imgur.com/drfirIW.jpg'
          ]
img_get = []
!mkdir Downloaded_Images
for i in range(len(img_url)):
    Image.open(BytesIO(requests.get(img_url[i]).content)).save('/kaggle/working/Downloaded_Images/image' + str(i) + '.png')
    
# Load Images to array
new_images = []
for i in range(len(img_url)):
    temp = np.array(tf.keras.preprocessing.image.load_img( '/kaggle/working/Downloaded_Images/image' + str(i) + '.png', color_mode = "rgb", target_size = (224, 224), interpolation="lanczos")) / 255.0
    new_images.append(temp)
new_images = np.array(new_images)
# Store predictions in out
predictions = model.predict(new_images)
pred_ints = predictions.argmax(axis = -1)
out = []
for x in pred_ints:
    labels = train_flow.class_indices
    for label, val in labels.items():
        if val == x:
            out.append(label)
##### Print images and their predicitons #####

rows,cols = ((len(new_images) - 1) // 5 ) + 1,5
temp_axis = rows * 5
fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(20,temp_axis))
for i in range(len(new_images)):
    temp = 'Prediction: ' + out[i]
    axes[i//cols, i%cols].set_title(temp)
    axes[i//cols, i%cols].axis('off')
    axes[i//cols, i%cols].imshow(new_images[i])
fig.savefig('New_Images_Prediction.eps', format='eps')