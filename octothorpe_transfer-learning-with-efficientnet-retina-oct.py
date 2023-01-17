!pip install efficientnet

import os
import cv2
import time
import glob
import itertools
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import efficientnet.tfkeras as efn

from collections import Counter
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.callbacks import *
from sklearn.metrics import multilabel_confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# sneak at the data

train_path = r'../input/kermany2018/OCT2017 /train/'
val_path = r'../input/kermany2018/OCT2017 /val/'
test_path = r'../input/kermany2018/OCT2017 /test/'

classes = ['CNV', 'DME', 'DRUSEN', 'NORMAL']


cnv_examples = glob.glob(val_path + 'CNV/*')
dme_examples = glob.glob(val_path + 'DME/*')
drusen_examples = glob.glob(val_path + 'DRUSEN/*')
normal_examples = glob.glob(val_path + 'NORMAL/*')

examples = cnv_examples[:4] + dme_examples[:4] + drusen_examples[:4] + normal_examples[:4]

fig=plt.figure(figsize=(15, 15))
columns = 4
rows = 4
for i in range(columns*rows):
    img = plt.imread(examples[i])
    ax = fig.add_subplot(rows, columns, i+1)
    if i%4==0:
        plt.ylabel(classes[int(i/4)], fontsize=16)
    plt.imshow(img, cmap='jet')
plt.show()


# Imbalance Graph

total_cnv_samples = len(glob.glob(train_path + 'CNV/*'))
total_dme_samples = len(glob.glob(train_path + 'DME/*'))
total_drusen_samples = len(glob.glob(train_path + 'DRUSEN/*'))
total_normal_samples = len(glob.glob(train_path + 'NORMAL/*'))

sample_distribution = [total_cnv_samples, total_dme_samples, total_drusen_samples, total_normal_samples]

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(classes, sample_distribution)
plt.show()
# Initialize the parameters

nh = nw = 150
nc = 3
n_classes = 4

epochs = 10
batch_size = 16

acc_thres = 0.99

learning_rate = 0.001
beta_1 = 0.9
beta_2 = 0.999
train_datagen = ImageDataGenerator(rescale=1./255)
train_data = train_datagen.flow_from_directory(train_path,
                                               target_size=(nh, nw),
                                               class_mode='categorical',
                                               batch_size=batch_size,
                                               shuffle=True
                                              )

# Class balancing
counter = Counter(train_data.classes)                          
max_val = float(max(counter.values()))  
class_weights = {class_id : max_val/num_images for class_id, num_images in counter.items()}                     
print(class_weights)

# It does not increase the samples but assign weights to each class which is passed to model.fit to avoid any bias via unbalanced data

cls_wghts = np.fromiter(class_weights.values(), dtype=float)
print(cls_wghts)

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(classes, sample_distribution*cls_wghts)
plt.show()
# Callbacks

def scheduler(epoch, lr):
    if epoch == 1:
        return lr
    elif epoch<=3:
        lr = lr**1.1
    else :
        lr /= 2

    return lr


lr_scheduler = LearningRateScheduler(scheduler) 

class Callbacks(Callback):
#     def on_train_batch_end(self, batch, logs=None):
#         print('For Batch {} : Loss = {:7.2f}, Acc = {:7.2f}'.format(batch,
#                                                                    logs['loss'],
#                                                                    logs['acc']))
    
    def on_epoch_end(self, epoch, logs=None):
        if logs['acc']>acc_thres:
            self.model.stop_training = True
            
            
# tensorboard = TensorBoard(log_dir=r'../logs/retinalOCT_{}'.format(int(time.time())), histogram_freq=1, write_graph=True)

csvlogger = CSVLogger(filename=r'training_{}.log'.format(int(time.time())))


# learning rate ploy

lr_y = []
ep_x = []
lr = learning_rate

for i in range(1, epochs+1):
    lr = scheduler(i, lr)
    lr_y.append(lr)
    ep_x.append(i)

print(lr_y)

plt.plot(ep_x, lr_y, 'ro--')
plt.xlabel('EPOCHS')
plt.ylabel('Learning Rate')
plt.show()

def model_efn():
    input_shape = (150, 150, 3)
    classes = 4
    
    model = efn.EfficientNetB7(weights='imagenet', input_shape=input_shape, pooling='max', include_top=False)
    x = model.output
    output = Dense(classes, activation='softmax')(x)

    return Model(inputs=model.input, outputs=output)
model = model_efn()
model.compile(loss="categorical_crossentropy",
             optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2),
             metrics=['acc']
             )


history = model.fit(x=train_data,
                    steps_per_epoch=int(80000//batch_size),
                    epochs=epochs,
                    verbose=1,
                    shuffle=True,
                    class_weight=class_weights,
                    callbacks=[Callbacks(),
                               csvlogger,
                               lr_scheduler]
                   )

model.save_weights('retina_oct_model.h5')
plt.figure(1)
plt.plot(history.history['loss'])
plt.legend(['Training Loss'])
plt.title('Loss')
plt.xlabel('epoch')
plt.figure(2)
plt.plot(history.history['acc'])
plt.legend(['training Accuracy'])
plt.title('Accuracy')
plt.xlabel('epoch')
plt.show()
# test_datagen = ImageDataGenerator(rescale=1./255)
# test_generator = test_datagen.flow_from_directory(test_path,
#                                                   target_size=(150, 150),
#                                                   batch_size=1,
#                                                   class_mode='categorical',
#                                                   shuffle=False
#                                                  ) 

# probabilities = model.predict_generator(test_generator, 968)

test_images_path = glob.glob(test_path + '*/*.jpeg')
x_test = []
y_test = []
for i in range(len(test_images_path)):
    img = cv2.imread(test_images_path[i])
    img = cv2.resize(img, (150,150))
    img = np.array(img/255.0)
    x_test.append(img)
    if 'CNV' in test_images_path[i]:
        y_test.append(0)
    elif 'DME' in test_images_path[i]:
        y_test.append(1)
    elif 'DRUSEN' in test_images_path[i]:
        y_test.append(2)
    elif 'NORMAL' in test_images_path[i]:
        y_test.append(3)
y_test = np.array(y_test)
x_test = np.array(x_test)
print(x_test.shape, y_test.shape)

# Confusion Matrix

y_test_cat = tf.keras.utils.to_categorical(y_test)
loss_and_metrics = model.evaluate(x_test, y_test_cat)

y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred,axis=1)

labels = ('CNV', 'DME', 'DRUSEN', 'NORMAL')

y_actu = pd.Series(y_test, name='Actual')
y_pred = pd.Series(y_pred, name='Predicted')
df_confusion = pd.crosstab(y_actu, y_pred)

df_conf_norm = df_confusion / df_confusion.sum(axis=1)
print(df_confusion)
print(df_conf_norm)

plt.figure(figsize=(20, 20))
plt.matshow(df_confusion, cmap=plt.get_cmap('Blues'), fignum=1)  # imshow
plt.colorbar()
tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels,fontsize=16, rotation=60)
plt.yticks(tick_marks, labels, fontsize=16)
thresh = 0.6

for i in range(n_classes):
    for j in range(n_classes):
        plt.text(i, j, "{:0.2f}%".format(df_conf_norm[i][j] * 100),
                 horizontalalignment='center',
                 color='white' if df_conf_norm[i][j] > thresh else 'black',
                fontsize = 16)

# plt.tight_layout()
plt.ylabel(df_confusion.index.name, fontsize=16)
plt.xlabel(df_confusion.columns.name,fontsize=16)
plt.show()
