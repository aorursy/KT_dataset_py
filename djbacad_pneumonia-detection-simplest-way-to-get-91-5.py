import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
np.random.seed(823)
tf.random.set_seed(188)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Conv2D, MaxPool2D, Flatten, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from sklearn.model_selection import train_test_split
datagen_train = ImageDataGenerator(rescale = 1./255, #1/255 to scale/normalize the pixel values
                                   shear_range = 0.2, #shear intensity
                                   rotation_range = 5, #random rotation (0 to 100 degrees)
                                   zoom_range = 0.2, #random zooming
                                   width_shift_range=0.1,  #random horizontal shifting
                                   height_shift_range=0.1)  #random vertical shifting 
                                   #horizontal_flip=True  #random horizontal flipping
                                   #vertical_flip=True)  #random vertical flipping)

training_set = datagen_train.flow_from_directory('../input/chest-xray-pneumonia/chest_xray/train',
                                                 target_size = (300, 300),
                                                 batch_size = 32,
                                                 shuffle = True,
                                                 seed = 823,
                                                 color_mode = 'grayscale',
                                                 class_mode = 'binary')
datagen_val = ImageDataGenerator(rescale = 1./255)
val_set = datagen_val.flow_from_directory('../input/chest-xray-pneumonia/chest_xray/val',
                                            target_size = (300, 300),
                                            batch_size = 32,
                                            shuffle = True,
                                            seed = 823,
                                            color_mode = 'grayscale',
                                            class_mode = 'binary')
datagen_test = ImageDataGenerator(rescale = 1./255)
test_set = datagen_test.flow_from_directory('../input/chest-xray-pneumonia/chest_xray/test',
                                            target_size = (300, 300),
                                            batch_size = 52,
                                            shuffle = False,
                                            seed = 823,
                                            color_mode = 'grayscale',
                                            class_mode = 'binary')
fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(1,6,figsize=(30, 10))
img_normal1 = load_img('../input/chest-xray-pneumonia/chest_xray/train/NORMAL/IM-0122-0001.jpeg')
img_normal2 = load_img('../input/chest-xray-pneumonia/chest_xray/train/NORMAL/IM-0152-0001.jpeg')
img_normal3 = load_img('../input/chest-xray-pneumonia/chest_xray/train/NORMAL/IM-0119-0001.jpeg')
img_normal4 = load_img('../input/chest-xray-pneumonia/chest_xray/train/NORMAL/IM-0145-0001.jpeg')
img_normal5 = load_img('../input/chest-xray-pneumonia/chest_xray/train/NORMAL/IM-0151-0001.jpeg')
img_normal6 = load_img('../input/chest-xray-pneumonia/chest_xray/train/NORMAL/IM-0166-0001.jpeg')
ax1.imshow(img_normal1)
ax2.imshow(img_normal2)
ax3.imshow(img_normal3)
ax4.imshow(img_normal4)
ax5.imshow(img_normal5)
ax6.imshow(img_normal6)
print("Normal Chest X-Rays")
plt.show()
fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(1,6,figsize=(30, 10))
img_normal1 = load_img('../input/chest-xray-pneumonia/chest_xray/train/PNEUMONIA/person1004_bacteria_2935.jpeg')
img_normal2 = load_img('../input/chest-xray-pneumonia/chest_xray/train/PNEUMONIA/person1007_bacteria_2938.jpeg')
img_normal3 = load_img('../input/chest-xray-pneumonia/chest_xray/train/PNEUMONIA/person1009_virus_1694.jpeg')
img_normal4 = load_img('../input/chest-xray-pneumonia/chest_xray/train/PNEUMONIA/person1000_bacteria_2931.jpeg')
img_normal5 = load_img('../input/chest-xray-pneumonia/chest_xray/train/PNEUMONIA/person1018_virus_1706.jpeg')
img_normal6 = load_img('../input/chest-xray-pneumonia/chest_xray/train/PNEUMONIA/person1020_bacteria_2951.jpeg')
ax1.imshow(img_normal1)
ax2.imshow(img_normal2)
ax3.imshow(img_normal3)
ax4.imshow(img_normal4)
ax5.imshow(img_normal5)
ax6.imshow(img_normal6)
print("Chest X-Rays of people diagnosed with Pneumonia")
plt.show()
np.random.seed(823)
tf.random.set_seed(188)

#Initialize
cnn = tf.keras.models.Sequential()

#First Convolutional Layer
cnn.add(Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[300, 300, 1]))
cnn.add(MaxPool2D(pool_size=2, strides=2))


#Second Convolutional Layer
cnn.add(Conv2D(filters = 64, kernel_size=3, activation='relu'))
cnn.add(MaxPool2D(pool_size=2, strides=2))


#Third Convolutional Layer
cnn.add(Conv2D(filters = 128, kernel_size=3, activation='relu'))
cnn.add(MaxPool2D(pool_size=2, strides=2))

#Flatten Layer
cnn.add(Flatten())

#Full Connection
cnn.add(Dense(units=256, activation='relu'))
cnn.add(Dropout(0.1))

#Output Layer
cnn.add(Dense(units=1, activation='sigmoid'))

#Compile
cnn.compile(optimizer = 'adam', 
            loss = 'binary_crossentropy', 
            metrics = ['accuracy'])

#Summary
cnn.summary()
cnn.fit(x = training_set, validation_data = val_set, epochs = 9, shuffle = False)
cnn_lossacc = pd.DataFrame(cnn.history.history)
plt.rcParams['figure.figsize'] = (12, 8)
sns.set_style('white')
cnn_lossacc.plot()
test_set.reset()
predictions_test = (cnn.predict(test_set,verbose=True) > 0.5).astype("int32")
print('Testing Accuracy: ',(cnn.evaluate(test_set))[1]*100, '%')
labels = (training_set.class_indices)
labels = dict((i,j) for j,i in labels.items())
print(labels)
from sklearn.metrics import classification_report,confusion_matrix
y_true = np.array([0] * 234 + [1] * 390) #234 Normal, #390 Pneumonia
data = confusion_matrix(y_true, predictions_test)
df_cm = pd.DataFrame(data, columns=np.unique(y_true), index = np.unique(y_true))
df_cm.index.name = 'Predicted'
df_cm.columns.name = 'Actual'
plt.figure(figsize = (10,7))
sns.set(font_scale=1.5)
ax = sns.heatmap(df_cm,cmap = 'Greens', annot=True,fmt = '.5g',annot_kws={"size": 16})# font size
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(['NORMAL', 'PNEUMONIA']); ax.yaxis.set_ticklabels(['NORMAL', 'PNEUMONIA']);
#For cmaps:
#https://matplotlib.org/3.1.1/gallery/color/colormap_reference.html
print("Classification Report: ")
print(classification_report(y_true,predictions_test))