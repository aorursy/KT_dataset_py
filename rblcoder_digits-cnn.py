import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))
#https://www.kaggle.com/yassineghouzam/introduction-to-cnn-keras-0-997-top-6

img_rows, img_cols = 28, 28

X =  pd.read_csv("../input/digit-recognizer/train.csv")

y = X['label']

X = X.drop(['label'],axis = 1)

X = X.values.reshape(-1, 28, 28, 1).astype('float32')/255

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42,stratify=y)

# X_train = X_train.values.reshape(X_train.shape[0], img_rows, img_cols, 1).astype('float32')/255

# X_test =  X_test.values.reshape(X_test.shape[0], img_rows, img_cols, 1).astype('float32')/255

test = pd.read_csv("../input/digit-recognizer/test.csv")

sample = pd.read_csv("../input/digit-recognizer/sample_submission.csv")

print(sample.shape)
y[:10]
sample.head()
test.head()
test.shape
X_unseen = test.values.reshape(-1, 28, 28, 1).astype('float32')/255
X_unseen.shape
X.shape
IMAGE_HT_WID = 28
#https://keras.io/examples/cifar10_cnn_tfaugment2d/

import tensorflow as tf

from tensorflow import keras
keras.backend.backend()


#https://towardsdatascience.com/easy-image-classification-with-tensorflow-2-0-f734fee52d13

print(tf.__version__)

IMG_SHAPE = (IMAGE_HT_WID, IMAGE_HT_WID, 1)

#https://hackernoon.com/tf-serving-keras-mobilenetv2-632b8d92983c

# base_model = tf.keras.applications.MobileNetV2(

#         include_top=False,

#         weights='imagenet',

#         input_shape=(IMAGE_HT_WID, IMAGE_HT_WID, 3),

#         pooling='avg')



 

# base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,

#                                                include_top=False, 

#                                                weights='imagenet')

#base_model.trainable = False

#https://machinelearningmastery.com/how-to-improve-deep-learning-model-robustness-by-adding-noise/



model = tf.keras.Sequential([

    tf.keras.layers.Conv2D(32, (3, 3), activation=tf.nn.tanh, input_shape=IMG_SHAPE,padding='same'),

    tf.keras.layers.GaussianNoise(0.1),

    #tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(32, (3, 3), activation=tf.nn.tanh),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Dropout(.25),

    

    tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.tanh,padding='same'),

#     #tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.tanh),

    tf.keras.layers.BatchNormalization(),

     tf.keras.layers.MaxPooling2D(2, 2),

     tf.keras.layers.Dropout(.25),

    

    tf.keras.layers.Conv2D(128, (3, 3), activation=tf.nn.tanh,padding='same'),

    #tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(128, (3, 3), activation=tf.nn.tanh),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Dropout(.25),



    

    tf.keras.layers.Flatten(),

    #tf.keras.layers.GlobalMaxPooling2D(),

    tf.keras.layers.Dense(512, activation=tf.nn.relu),

    tf.keras.layers.Dropout(.5), 

    tf.keras.layers.Dense(10, activation='softmax')

    

    

 

])

#https://www.dlology.com/blog/how-to-use-keras-sparse_categorical_crossentropy/

# model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.001), 

#              # loss='categorical_crossentropy', 

#               loss='sparse_categorical_crossentropy',

#               metrics=['sparse_categorical_accuracy'])

                     #  ,tf.keras.metrics.Precision(),tf.keras.metrics.Recall()



#https://keras.io/getting-started/sequential-model-guide/

sgd = tf.keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd, metrics=['sparse_categorical_accuracy'])







model.summary()
len(model.trainable_variables)
EPOCHS=32
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, 

                                                                    patience=2, verbose=2, mode='auto',

                                                                    min_lr=1e-6)
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 

                                            # min_delta=0, 

                                             patience=4, verbose=2, mode='auto',

                                             baseline=None, restore_best_weights=True)
history = model.fit(X, y, batch_size=50, epochs=EPOCHS, verbose = 2, callbacks = [reduce_lr,early_stop], validation_split=0.25,

                    shuffle=True)
history.history
import matplotlib.pyplot as plt

acc = history.history['sparse_categorical_accuracy']

val_acc = history.history['val_sparse_categorical_accuracy']



epochs = range(len(acc))



plt.plot(epochs, acc, 'r', label='Training accuracy')

plt.plot(epochs, val_acc, 'b', label='Validation accuracy')

plt.title('Training and validation accuracy')

plt.legend(loc=0)

plt.figure()





plt.show()
pred_X = model.predict(X, verbose = 2)
pred_X.shape
pred_X[:1]
pred_X[:1].shape
pred_X[:,0].shape
pred_X[:,0]
#https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/

from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder()
ohe_encoded_y = ohe.fit_transform(y.values.reshape(-1, 1))
ohe_encoded_y[0].toarray()
ohe_encoded_y[:,0].toarray().ravel().shape
pred_X.ravel()
#https://www.dlology.com/blog/simple-guide-on-how-to-generate-roc-plot-for-keras-classifier/

from sklearn.metrics import roc_curve, auc

fpr = dict()

tpr = dict()

roc_auc = dict()

fpr["micro"], tpr["micro"], thresholds = roc_curve(ohe_encoded_y.toarray().ravel(), pred_X.ravel())

roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
roc_auc["micro"]
np.mean(thresholds)
np.histogram(thresholds)
#https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py

import matplotlib.pyplot as plt

plt.plot(fpr['micro'], tpr['micro'], color='darkorange',

         label='ROC curve (area = %0.2f)' % roc_auc['micro'])

plt.plot([0, 1], [0, 1], color='navy', linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic micro')

plt.legend(loc="lower right")

plt.show()
np.argmax(pred_X,axis=1)
#https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html#sklearn.metrics.precision_recall_fscore_support

from sklearn.metrics import precision_recall_fscore_support

precision_recall_fscore_support(y, np.argmax(pred_X,axis=1), average='macro')
precision_recall_fscore_support(y, np.argmax(pred_X,axis=1), average='micro')
precision_recall_fscore_support(y, np.argmax(pred_X,axis=1), average='weighted')
precision_recall_fscore_support(y, np.argmax(pred_X,axis=1), average=None)
precision_recall_fscore_support(y, np.argmax(pred_X,axis=1), average=None, beta=2)
precision_recall_fscore_support(y, np.argmax(pred_X,axis=1), average=None, beta=0.5)
#https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html#sklearn.metrics.classification_report

from sklearn.metrics import classification_report

print(classification_report(y, np.argmax(pred_X,axis=1)))
pred = model.predict(X_unseen, verbose = 2)
pred.shape
pred[:1]
results=pd.DataFrame({"ImageId":sample['ImageId'],

                      "Label":np.argmax(pred,axis=1)})
results['Label'].value_counts()
results.to_csv("submission.csv",index=False)
#https://www.kaggle.com/xhlulu/eda-simple-keras-cnn-k-mnist

train_images = np.load('../input/kuzushiji/kmnist-train-imgs.npz')['arr_0']

test_images = np.load('../input/kuzushiji/kmnist-test-imgs.npz')['arr_0']

train_labels = np.load('../input/kuzushiji/kmnist-train-labels.npz')['arr_0']

test_labels = np.load('../input/kuzushiji/kmnist-test-labels.npz')['arr_0']

train_images = train_images / 255.0

test_images = test_images / 255.0

x_train = np.expand_dims(train_images, axis=-1)

x_test = np.expand_dims(test_images, axis=-1)

y_train = train_labels

y_test = test_labels
np.unique(y_test)
hist = model.fit(x_train, y_train,

          batch_size=50,

          epochs=EPOCHS,

          verbose=1,

          validation_data=(x_test, y_test))
import matplotlib.pyplot as plt

acc = hist.history['sparse_categorical_accuracy']

val_acc = hist.history['val_sparse_categorical_accuracy']



epochs = range(len(acc))



plt.plot(epochs, acc, 'r', label='Training accuracy')

plt.plot(epochs, val_acc, 'b', label='Validation accuracy')

plt.title('Training and validation accuracy')

plt.legend(loc=0)

plt.figure()





plt.show()
score = model.evaluate(x_test, y_test, verbose=0)

score
#https://keras.io/models/model/

model.metrics_names
kpred_x_test = model.predict(x_test, verbose = 2)
print(classification_report(y_test, np.argmax(kpred_x_test,axis=1)))