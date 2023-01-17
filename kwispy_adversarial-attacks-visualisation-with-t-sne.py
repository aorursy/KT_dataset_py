from sklearn import datasets
from sklearn.manifold import TSNE

import foolbox
import keras
import numpy as np
from keras.applications.resnet50 import ResNet50
import pandas as pd
from keras.datasets import mnist
import matplotlib.pyplot as plt
%matplotlib inline
from keras.models import load_model
import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
from sklearn.model_selection import train_test_split


import keras
from keras.datasets import mnist
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.layers import Activation, Dense


def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(64, input_dim=784, kernel_initializer='normal', activation='relu'))
    model.add(Dense(10, kernel_initializer='normal', activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
def baseline_model1():
    # building a linear stack of layers with the sequential model
    model = Sequential()
    model.add(Dense(512, input_shape=(64,)))
    model.add(Activation('relu'))                            
    model.add(Dropout(0.2))

    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(10))
    model.add(Activation('softmax'))
    # compiling the sequential model
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    return model
def baseline_model2():

    input_shape=(8,8,1)
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    return model
digits = datasets.load_digits()
# Take the first 500 data points: it's hard to see 1500 points
X_train = digits.data[:1000]/255.
X_test = digits.data[1000:]/255.
y_train = digits.target[:1000]
y_test = digits.target[1000:]
tsne = TSNE(n_components=2, random_state=0)
X_2d = tsne.fit_transform(X_train)
target_ids = range(len(digits.target_names))

from matplotlib import pyplot as plt
plt.figure(figsize=(6, 5))
colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', '#ff66cc', 'orange', 'purple'
for i, c, label in zip(target_ids, colors, digits.target_names):
    plt.scatter(X_2d[y_train == i, 0], X_2d[y_train == i, 1], c=c, label=label)
plt.legend()
plt.show()
model = baseline_model1()
model.fit(X_train.reshape(1000,64), np_utils.to_categorical(y_train), validation_data=(X_test.reshape(len(X_test),64), np_utils.to_categorical(y_test)), epochs=50, batch_size=200)
scores = model.evaluate(X_test.reshape(len(X_test),64), np_utils.to_categorical(y_test), verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))
gmodel = foolbox.models.KerasModel(model, bounds=(0, 1))
attackF = foolbox.attacks.FGSM(gmodel)
model.predict_classes(X_test[0].reshape(1,64))
plt.imshow(X_test[0].reshape(8,8))
plt.title("{} predict {} ".format(str(y_test[0]), model.predict_classes(X_test[0].reshape(1,64))))
plt.show()
np_utils.to_categorical(y_test)[0]
advs= attackF(X_test[10].reshape(64),1)
advs=[]
for img, label in zip(X_test.reshape(len(X_test), 64), y_test):
    adversarialM = attackF(img, label)
    advs.append(adversarialM)

np.asarray(advs).shape
advs_=np.asarray(advs).reshape(len(advs),64)
X_test_advs=np.append(X_test.reshape(len(X_test),64), advs_.reshape(len(advs),64), axis=0)
X_train_advs=np.append(X_train.reshape(len(X_train),64), advs_.reshape(len(advs),64), axis=0)

X_test_advs.shape
y_test_advs=np.append(y_test, 10*np.ones((797,1)))
y_train_advs=np.append(y_train, 10*np.ones((797,1)))
X_all_advs= np.append(digits.data.reshape(digits.data.shape[0],64)/255., advs_.reshape(len(advs),64), axis=0)
y_all_advs=np.append(digits.target, 10*np.ones((797,1)))

print(X_test_advs.shape)
np.unique(y_test_advs)

tsne = TSNE(n_components=2, random_state=0)
X_2d_test_advs = tsne.fit_transform(X_test_advs)
X_2d_train_advs = tsne.fit_transform(X_train_advs)
target_ids = range(11)

plt.figure(figsize=(20, 15))
colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple', '#00ff00'
for i, c, label in zip(range(11), colors, range(11)):
    plt.scatter(X_2d_test_advs[y_test_advs == i, 0], X_2d_test_advs[y_test_advs == i, 1], c=c, label=label)
plt.title("t-sne X_test + adversarials trained on x_test ")
plt.legend()
plt.show()
target_ids = range(11)

plt.figure(figsize=(20, 15))
colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple', '#00ff00'
for i, c, label in zip(range(11), colors, range(11)):
    plt.scatter(X_2d_train_advs[y_train_advs == i, 0], X_2d_train_advs[y_train_advs == i, 1], c=c, label=label)
plt.legend()
plt.title("t-sne X_train + adversarials trained on x_test ")
plt.show()
predict_before=model.predict_classes(X_test[1].reshape(1,64))
predict_after=model.predict_classes(advs_[1].reshape(1,64))
print("real value {} predict before {} predict after {}".format(y_test[0],predict_before, predict_after))
plt.imshow(advs[0].reshape(8,8))
tsne = TSNE(n_components=2, random_state=0)
X_2d_all_advs = tsne.fit_transform(X_all_advs)

target_ids = range(11)

plt.figure(figsize=(20, 15))
colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple', '#00ff00'
for i, c, label in zip(range(11), colors, range(11)):
    plt.scatter(X_2d_all_advs[y_all_advs == i, 0], X_2d_all_advs[y_all_advs == i, 1], c=c, label=label)
plt.legend()
plt.title("t-sne X_all + adversarials trained on x_test ")
plt.show()
predict_before=model.predict_classes(X_test.reshape(len(X_test),64)).reshape(len(X_test),1)
predict_after=model.predict_classes(advs_.reshape(len(advs_),64)).reshape(len(advs_),1)
predictions=np.append(predict_before,predict_after, axis=1)
predictions_=np.append(predictions,np.ones((len(predictions),1)), axis=1)
df_prediction=pd.DataFrame(predictions_, columns=["before","after","ones"], dtype=int)
df_prediction.head()
grouped = df_prediction.groupby(by=["after"])["ones"].sum().astype(int)
plt.plot(grouped.values)
grouped.values
grouped = df_prediction.groupby(by=["before","after"])["ones"].sum().astype(int)
print(type(grouped))
grouped_=grouped.sort_values(ascending=False).to_frame()
grouped_.head()
#grouped["after"].agg(np.size)
#plt.scatter(grouped.index,grouped.values)
#grouped.groups
#df_prediction.groupby(by=["before","after"]).groups
grouped_.head(20)
grouped2=grouped_.reset_index().values

mat=np.zeros((10,10))
for idxbefore, idxafter, value in grouped2 :
    mat[idxbefore,idxafter]=value
import seaborn as sns; sns.set()
ax = sns.heatmap(mat)
plt.title("horizontal: before, vertical: after")
plt.show()
predict_before=model.predict_classes(X_test.reshape(len(X_test),64))
predict_after=model.predict_classes(np.array(advs).reshape(X_test.shape[0],64))
#y_test_advs2=np.append(y_test, predict_after.reshape(predict_after))
y_train_after_advs2=np.append(y_train, 10+predict_after.reshape(predict_after.shape[0],1))
y_train_before_advs2=np.append(y_train, 10+predict_before.reshape(predict_before.shape[0],1))
np.unique(y_train_after_advs2)
def plot_tsne(target, y_advs ,legend ):
    target_ids = range(20)

    #plt.figure(figsize=(6, 5))
    colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple', '#98FB98', '#90EE90','#00FA9A', '#00FF7F','#ADFF2F','#7FFF00', '#7CFC00','#00FF00','#32CD32','#9ACD32'
    for i, c, label in zip(range(10), colors, range(10)):
        plt.scatter(X_2d_train_advs[y_advs == i, 0], X_2d_train_advs[y_advs == i, 1], c=c, label=label)

    plt.scatter(X_2d_train_advs[y_advs == target, 0], X_2d_train_advs[y_advs == target, 1], c='#00ff00', label=str(target-10))

    plt.legend()
    plt.title("{} {} ".format(legend, target-10))
    #plt.show()
plt.figure(figsize=(20,50))
for i in range(10,20,1):
    plt.subplot(5,2,i-9)
    plot_tsne(i,y_train_after_advs2, "t-sne adversarials targeted on")
    #plt.plot(range(10))
    
plt.show()
plt.figure(figsize=(20,50))
for i in range(10,20,1):
    plt.subplot(5,2,i-9)
    plot_tsne(i,y_train_before_advs2, "t-sne adversarials from " )
    #plt.plot(range(10))
    
plt.show()
