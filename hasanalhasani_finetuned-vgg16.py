import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import os
import scipy.misc
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
import seaborn as sns
import random
import cv2
train_dir = '../input/intel-image-classification/seg_train/seg_train'
test_dir = '../input/intel-image-classification/seg_test/seg_test'
labels = sorted(os.listdir(train_dir))
labels
vgg16 = tf.keras.applications.VGG16(include_top=False, input_shape = (150, 150,3))
vgg16.summary()
model1 = tf.keras.models.Sequential()
for layer in vgg16.layers:
    model1.add(layer)
model1.summary()
for layer in model1.layers[:15]:
    layer.trainable = False
model1.summary()
model1.add(tf.keras.layers.Flatten())
model1.add(tf.keras.layers.Dense(6, activation = 'softmax'))
model1.summary()

model1.compile(optimizer = 'adam',
             loss = 'categorical_crossentropy',
             metrics = ['accuracy'])

img_aug_tr = ImageDataGenerator(rescale=1./255)
train_gen = img_aug_tr.flow_from_directory(train_dir,
                                       target_size=(150,150),
                                       class_mode='categorical',
                                       batch_size = 16)
history1 = model1.fit_generator(train_gen, epochs = 10)
tst_aug = ImageDataGenerator(rescale=1./255)
test_gen = tst_aug.flow_from_directory(test_dir,
                                      target_size = (150,150),
                                      class_mode = 'categorical',
                                      shuffle = False,
                                      batch_size = 16)
predictions1 = model1.predict_generator(test_gen)

y_pred1 = np.argmax(predictions1, axis=1)
y_pred1 = [labels[p] for p in y_pred1]
y_true = [labels[t] for t in test_gen.classes]
cm1 = confusion_matrix(y_pred = y_pred1, y_true = y_true, labels = labels, normalize = 'true')
ax= plt.subplot()
sns.heatmap(cm1,xticklabels=labels,yticklabels=labels, annot=True)
model1.evaluate(x = test_gen)

plt.figure(figsize=(12, 12))
fns = os.listdir('../input/intel-image-classification/seg_pred/seg_pred')
for i in range(9):
    filename = random.choice(fns)
    #img = load_img('../input/intel-image-classification/seg_pred/seg_pred/'+filename, target_size=(150, 150))
    #x = np.expand_dims(img, axis=0)
    #x = x/255
    img = load_img('../input/intel-image-classification/seg_pred/seg_pred/{}'.format(str(filename)))
    x = np.expand_dims(img, axis = 0)
    x = x / 255
    category1 = model1.predict(x)
    category1 = np.argmax(category1, axis=1)
    category1 = labels[int(category1)]

    plt.subplot(3, 3, i+1)
    plt.imshow(img)
    plt.xlabel("VGG16:"'(' + "{}".format(category1)+")")
plt.tight_layout()
plt.show()
