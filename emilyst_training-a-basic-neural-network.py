import pandas as pd
import numpy as np
import random 
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

import tensorflow as tf
from tensorflow import keras

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

train = pd.read_json('../input/train.json')
train.head()
plt.figure(figsize=[10, 7])
for i in range(1, 7):
    plt.subplot(2, 3, i)
    which_row = random.randint(1, len(train.is_turkey))
    sns.heatmap(data = np.matrix(train.audio_embedding[which_row]).T)
    plt.xlabel("Second of Video"); plt.ylabel("Audio Encoding Value");
    if train.is_turkey[which_row] == 0:
        plt.title("Not a Turkey")
    else:
        plt.title("Is a Turkey")
plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.suptitle('Audio Encoding Heatmaps');
def PrepareData(audio_data,vid_duration):
    X = np.ones((len(audio_data), 1280+1) )*0.5
    for i in range(0, len(audio_data)):
        unrolled_matrix = np.matrix(audio_data[i]).getA1()
        X[i, 0:len(unrolled_matrix)] = unrolled_matrix/255.0
        X[i, len(unrolled_matrix)] = vid_duration[i]/10.0
    return X

X = PrepareData(train.audio_embedding, train.end_time_seconds_youtube_clip - train.start_time_seconds_youtube_clip)

trainX, valX, trainy, valy = train_test_split(X, train.is_turkey, test_size=0.4)
valX, testX, valy, testy = train_test_split(valX, valy, test_size = 0.5)
print("Number of Training Examples:", len(trainy))
print("Number of Validation Examples:", len(valy))
print("Number of Test Set Examples:", len(testy))
model = keras.Sequential([
    keras.layers.Dense(15, activation = tf.nn.relu, input_shape = (1281,)), 
    keras.layers.Dense(40, activation = tf.nn.relu ),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', 
                  metrics = ['acc'])
print(model.summary())


history = model.fit(trainX, trainy, epochs=12, validation_data=(valX, valy))

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc)+1)

plt.figure()
plt.plot(epochs, acc, 'k-', label = 'Training accuracy')
plt.plot(epochs, val_acc, 'b--', label = 'Validation accuracy')
plt.legend()
plt.title('Model Accuracy');

pred = model.predict(valX)

print("\nFinal Validation Set Scores:")
print("ROC_AUC score:", roc_auc_score(y_score=pred, y_true=valy))
print("Accuracy Score:", accuracy_score(pred>0.5, valy))

print("\nTotal # of Turkeys in Validation Set:", sum(valy))
print("Sum of Predictions on Validation Set:", sum(pred))
pred = (model.predict(testX)) 
print("\nROC_AUC score:", roc_auc_score(y_score=pred, y_true=testy))
print("Accuracy Score:", accuracy_score(pred>0.5, testy))
test = pd.read_json('../input/test.json')
X_test = PrepareData(test.audio_embedding,test.end_time_seconds_youtube_clip - test.start_time_seconds_youtube_clip)

pred = model.predict(X_test) 
d = {'vid_id':np.array(test.vid_id), 'is_turkey':pred[:, 0]}
result = pd.DataFrame(data = d)
result.to_csv("Predictions.csv", index=False)
result.head(10)