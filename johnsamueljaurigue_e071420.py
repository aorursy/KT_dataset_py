import os
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import math 
import itertools

from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
!ls ../input/utensils/utensils

train_dataset = '../input/utensils/utensils/utensils_train.csv'
test_dataset = '../input/utensils/utensils/utensils_test.csv'
df = pd.read_csv(train_dataset)
df.head()
ab = pd.read_csv(test_dataset)
ab.head()
train = pd.read_csv('../input/utensils/utensils/utensils_train.csv')
J=train.iloc[:,1:].values 
S=train.iloc[:,0].values 
S[:10]
X_train, X_test, Y_train, Y_test = train_test_split(J, S, test_size = 0.10, random_state=42) 
y = df['Label'].values
y[:5]
y_encoder = OneHotEncoder(sparse=None)
y_encoded = y_encoder.fit_transform(y.reshape(-1, 1))
y_encoded[:5]
c = ab['Label'].values
c[:5]
c_encoder = OneHotEncoder(sparse=None)
c_encoded = c_encoder.fit_transform(y.reshape(-1, 1))
c_encoded[:5]
X = df.drop('Label', axis=1).values
X = X.reshape(-1, 28, 28, 1)
D = ab.drop('Label', axis=1).values
D = D.reshape(-1, 28, 28, 1)
for i in range(10):
    plt.imshow(X[i].reshape(28, 28))
    plt.show()
    print('Label:', y[i])
input_ = tf.keras.Input((28, 28, 1))
conv1 = tf.keras.layers.Conv2D(8, (3, 3), activation='relu')(input_)
conv2 = tf.keras.layers.Conv2D(8, (3, 3), activation='relu')(conv1)


# first name : JOHN Max pooling
mp1 = tf.keras.layers.MaxPool2D((2,2))(conv2)

conv3 = tf.keras.layers.Conv2D(8, (3, 3), activation='relu')(mp1)
conv4 = tf.keras.layers.Conv2D(8, (3, 3), activation='relu')(conv3)

#NXN Jaurigue 5 vowel + 1 
conv5 = tf.keras.layers.Conv2D(8, (6, 6), activation='relu')(conv4)

# first name : JOHN Max pooling
mp2 = tf.keras.layers.MaxPool2D((2,2))(conv5)

fl = tf.keras.layers.Flatten()(mp2)

# two dense layer
dense1 = tf.keras.layers.Dense(8, activation='relu')(fl)
dense2 = tf.keras.layers.Dense(8, activation='relu')(dense1)

output = tf.keras.layers.Dense(3, activation='softmax')(dense2)

model = tf.keras.Model(inputs=input_, outputs=output)
model.summary()
model.compile('adam', 'categorical_crossentropy', 'accuracy')
early_stop = tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)
MODEL_PATH = 'checkpoints/model_at_{epoch:02d}.mdl'
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(MODEL_PATH)
tf.keras.utils.plot_model(model)
hst = model.fit(X, y_encoded, batch_size=4, epochs=20, validation_split=0.2, callbacks=[early_stop, model_checkpoint], verbose=1)
!ls checkpoints/
predictions = model.predict(D)
for i in range(10):
    plt.imshow(D[i].reshape(28, 28))
    plt.show()
    print('Prediction:', predictions[i])
for i in range(10):
    plt.imshow(D[i].reshape(28, 28))
    plt.show()
    print('Prediction:',np.argmax(predictions[i]))
plt.plot(hst.history['loss'])
plt.plot(hst.history['val_loss'])
plt.title("Model Loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Train', 'Test'])
plt.show()
plt.plot(hst.history['accuracy'])
plt.plot(hst.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['Train','Test'])
plt.show()
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

Y_pred = model.predict(X)
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
Y_true = np.argmax(y_encoded,axis = 1) 
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
plot_confusion_matrix(confusion_mtx, classes = range(3))

knn = KNeighborsClassifier(n_neighbors=22)
knn.fit(X_train, Y_train)
knnpred = knn.predict(X_test)
print('', classification_report(Y_test, knnpred))